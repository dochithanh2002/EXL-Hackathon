"""Deterministic tools for the Documentation agent (agent/agents/documentation-agent.md).

The agent reverse-engineers purpose, methodology, variables, preprocessing, assumptions and limitations from the
legacy source and writes them to output/<model>/logs/documentation/extract.json. These tools establish the facts,
verify the agent's extract and render the model card. The agent is never allowed to write a statistic.

    mrf doc-facts   <model>    FACTS:  formula variables, factor levels, coefficients + p-values, holdout performance
                               from the captured R predictions, data schema (no rows), REQUIRED disclosures,
                               numbered source listing -> logs/documentation/
    mrf doc-verify  <model>    VERIFY: checks V-01..V-05 on extract.json (+ schema, V-06); bounded by
                               agent_limits.max_doc_iterations (agent/config/thresholds.yaml), then ESCALATE
    mrf doc-render  <model>    RENDER: inventory_<id>.json + MODEL_CARD_<id>.md (numbers from code, narrative from
                               the verified extract) -> reports/documentation/; status AWAITING_HUMAN_APPROVAL

Verification checks (the agent cannot pass a card that fails these):
  V-01 variable coverage   documented inputs == formula variables; target == model response
  V-02 factor levels       documented levels == levels captured from the fitted R object
  V-03 grounding           every cited line exists and is non-blank; every step cites a line; every variable
                           cites a line that contains its name
  V-04 disclosures         every REQUIRED disclosure (D-01 protected input, D-02 unseeded split, D-03 non-convergence,
                           D-04 category merges, D-05 non-significant terms, D-06 capping) is covered by a limitation that cites its id and names it
  V-05 no invented numbers decimals and percentages in narrative must appear verbatim in source or documents
  V-06 schema              extract.json must match the DocExtract schema (agent/tools/schemas.py)
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from agent.tools.paths import INVENTORY, Model, Stage, now, rel, sha, thresholds
from agent.tools.schemas import DocExtract

# ====================================================================== FACTS
def strip_console_output(text: str) -> str:
    """G-11 for supporting documents: drop printed R console output ('## ...') and blank lines."""
    return "\n".join(l for l in text.splitlines() if not l.startswith("##") and l.strip())


def formula_variables(formula: str, all_columns: list[str]) -> tuple[str, list[str]]:
    lhs, rhs = formula.split("~", 1)
    rhs = re.sub(r"\s+", "", rhs)
    vars_: list[str] = []
    for tok in re.split(r"[+*:]", rhs):
        tok = re.sub(r"^(I|log|factor|as\.factor|offset)\((.*)\)$", r"\2", tok)
        if tok in ("", "1", "0", "-1"):
            continue
        if tok == ".":
            vars_ += [c for c in all_columns if c != lhs.strip() and c not in vars_]
        elif tok not in vars_:
            vars_.append(tok)
    return lhs.strip(), vars_


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = y.astype(int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    ranks = pd.Series(p).rank().to_numpy()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    order = np.argsort(-p, kind="mergesort")
    tpr, fpr = np.cumsum(y[order]) / n1, np.cumsum(1 - y[order]) / n0
    return {"n": int(len(y)), "event_rate": round(float(y.mean()), 4), "auc": round(float(auc), 4),
            "gini": round(float(2 * auc - 1), 4), "ks": round(float(np.max(tpr - fpr)), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4), "mean_predicted": round(float(p.mean()), 4)}


def data_profile(csv: Path) -> dict:
    raw = pd.read_csv(csv)
    return {"n_rows": int(len(raw)), "columns": [
        {"name": c, "dtype": str(raw[c].dtype), "n_unique": int(raw[c].nunique()), "n_missing": int(raw[c].isna().sum()),
         **({"levels": sorted(raw[c].dropna().unique().tolist())} if raw[c].nunique() <= 12 else {})}
        for c in raw.columns]}


SYNONYMS = {"sex": ["sex", "gender"], "marriage": ["marriage", "marital"], "education": ["education"], "age": ["age"],
            "drivage": ["age"], "gender": ["sex", "gender"]}


def protected_match(variable: str, protected: list[str]) -> str | None:
    """The protected attribute a model variable is, or is derived from (e.g. DrivAgeGLM <- DrivAge)."""
    v = variable.lower()
    for p in protected:
        if v == p.lower() or v.startswith(p.lower()):
            return p
    return None


def required_disclosures(src: str, meta: dict, coefs: pd.DataFrame, variables: list[str], protected: list[str]) -> dict:
    """Deterministic triggers. Each value: {why, keywords (any must appear in the limitation text), lines?}."""
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    d: dict[str, dict] = {}
    for v in variables:
        p = protected_match(v, protected)
        if p:
            src_txt = "" if p.lower() == v.lower() else f" (derived from '{p}')"
            d[f"D-01:{v}"] = {"why": f"protected / sensitive attribute '{v}'{src_txt} is a direct model input (Equality Act 2010, Consumer Duty)",
                              "keywords": sorted(set(SYNONYMS.get(p.lower(), [p.lower()]) + [v.lower()]))}
    if re.search(r"\bsample\s*\(|createDataPartition|sample\.split|runif\s*\(", code) and not re.search(r"set\.seed", code):
        d["D-02"] = {"why": "random train/test split with no seed in the legacy source: results are not reproducible",
                     "keywords": ["seed", "reproduc"]}
    if not meta.get("converged", True):
        d["D-03"] = {"why": "the fitted model did not converge", "keywords": ["converg"]}
    recode = [i for i, l in enumerate(src.splitlines(), 1)
              if re.search(r"\[.*(%in%|==).*\]\s*(\[\[.*\]\])?\s*<-", l.split("#")[0])]
    if recode:
        d["D-04"] = {"why": "category codes are merged / recoded before fitting; the merge changes what the model learns",
                     "keywords": ["merge", "recod", "group"], "lines": recode}
    caps = [i for i, l in enumerate(src.splitlines(), 1) if re.search(r"\bp(min|max)\s*\(", l.split("#")[0])]
    if caps:
        d["D-06"] = {"why": "values are capped / floored before fitting (pmin/pmax); capping a response or exposure changes "
                            "what the model estimates", "keywords": ["cap", "truncat", "floor"], "lines": caps}
    alpha = thresholds()["documentation"]["significance_level"]
    insig = coefs[(coefs["p_value"] > alpha) & (coefs["term"] != "(Intercept)")]["term"].tolist()
    if insig:
        d["D-05"] = {"why": f"terms not significant at the {alpha:g} level in the final model: {insig}", "keywords": ["significan"]}
    return d


# ====================================================================== VERIFY
NUM_RE = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])|(?<![\w.])\d+(?:\.\d+)?\s?%")


def _narrative(ex: dict) -> list[tuple[str, str]]:
    out = [(k, ex[k]) for k in ("purpose", "intended_use", "out_of_scope_use", "methodology", "target_definition", "variable_selection")]
    out += [(f"variables[{v['name']}]", v["description"] + " " + v["transformation"]) for v in ex["variables"]]
    out += [(f"preprocessing[{s['order']}]", s["description"] + " " + s["author_rationale"]) for s in ex["preprocessing"]]
    out += [(f"assumptions[{i}]", a) for i, a in enumerate(ex["assumptions"])]
    out += [(f"limitations[{i}]", l["text"]) for i, l in enumerate(ex["limitations"])]
    out += [(f"discrepancies[{i}]", t) for i, t in enumerate(ex["code_vs_docs_discrepancies"])]
    return out


def verify(ex: dict, facts: dict, src: str, evidence_text: str) -> list[dict]:
    viol: list[dict] = []
    lines = src.splitlines()
    nl = len(lines)

    def v(check, msg):
        viol.append({"check": check, "message": msg})

    # V-01 variable coverage
    inputs = [x["name"] for x in ex["variables"] if x["role"] == "input"]
    targets = [x["name"] for x in ex["variables"] if x["role"] == "target"]
    want = facts["model"]["input_variables"]
    if missing := [x for x in want if x not in inputs]:
        v("V-01", f"formula variables not documented: {missing}")
    if extra := [x for x in inputs if x not in want]:
        v("V-01", f"documented inputs that are not in the model formula: {extra}")
    if targets != [facts["model"]["response"]]:
        v("V-01", f"target must be exactly ['{facts['model']['response']}'], got {targets}")
    # V-02 factor levels
    xl = facts["model"]["factor_levels"]
    for x in ex["variables"]:
        if x["name"] in xl and [str(l) for l in x["levels"]] != [str(l) for l in xl[x["name"]]]:
            v("V-02", f"{x['name']}: documented levels {x['levels']} != captured R levels {xl[x['name']]}")
        if x["name"] in xl and x["kind"] not in ("categorical", "binary"):
            v("V-02", f"{x['name']} is a factor in the fitted R model but documented as '{x['kind']}'")
        if x["name"] not in xl and x["role"] == "input" and x["kind"] in ("categorical", "binary"):
            v("V-02", f"{x['name']} is numeric in the fitted R model but documented as '{x['kind']}'")
    # V-03 grounding
    def bad_lines(ls):
        return [n for n in ls if not (1 <= n <= nl) or not lines[n - 1].strip()]
    for x in ex["variables"]:
        if b := bad_lines(x["source_lines"]):
            v("V-03", f"variable {x['name']}: invalid or blank source lines {b}")
        if not any(1 <= n <= nl and x["name"].lower() in lines[n - 1].lower() for n in x["source_lines"]):
            v("V-03", f"variable {x['name']}: no cited line contains the variable name")
    for s in ex["preprocessing"]:
        if not s["source_lines"]:
            v("V-03", f"preprocessing step {s['order']} cites no source line")
        elif b := bad_lines(s["source_lines"]):
            v("V-03", f"preprocessing step {s['order']}: invalid or blank source lines {b}")
    for i, l in enumerate(ex["limitations"]):
        if b := bad_lines(l["source_lines"]):
            v("V-03", f"limitation {i}: invalid or blank source lines {b}")
    # V-04 disclosures
    for did, spec in facts["required_disclosures"].items():
        cov = [l for l in ex["limitations"] if did in l["disclosure_ids"]]
        if not cov:
            v("V-04", f"required disclosure {did} not covered: {spec['why']}")
            continue
        if not any(any(k in l["text"].lower() for k in spec["keywords"]) for l in cov):
            v("V-04", f"limitation citing {did} does not name the issue (expected one of {spec['keywords']})")
        if spec.get("lines") and not any(set(l["source_lines"]) & set(spec["lines"]) for l in cov):
            v("V-04", f"limitation citing {did} must cite at least one of lines {spec['lines']}")
    # V-05 no invented numbers
    for where, text in _narrative(ex):
        for num in NUM_RE.findall(text):
            if num.replace(" ", "") not in evidence_text.replace(" ", ""):
                v("V-05", f"{where}: number '{num}' does not appear in the source or documents (numbers come from the factory)")
    return viol


# ====================================================================== RUN
def numbered(src: str) -> str:
    return "\n".join(f"{i:>4} | {l}" for i, l in enumerate(src.splitlines(), 1))


def _tbl(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |" for r in rows])




def _card(mdl, r, facts, ex) -> str:
    m, g = facts["model"], r["governance (human-owned, from model.yaml)"]
    test = r["performance_reproduced"].get("test", {})
    train = r["performance_reproduced"].get("train", {})
    sev = {"High": 0, "Medium": 1, "Low": 2}
    lims = sorted(ex["limitations"], key=lambda l: sev[l["severity"]])
    coef_rows = [[c["term"], f"{c['estimate']:.6g}", f"{c['std_error']:.3g}", f"{c['z']:.2f}",
                  (f"{c['p_value']:.3g}" if c['p_value'] >= 1e-16 else "<1e-16") + (" (n.s.)" if c["p_value"] > 0.05 and c["term"] != "(Intercept)" else "")] for c in r["coefficients"]]
    perf_rows = [[k, train.get(k, ""), test.get(k, "")] for k in ("n", "event_rate", "auc", "gini", "ks", "brier", "mean_predicted")]
    v_rows = [[x["iteration"], x.get("file", ""), x["verdict"], "; ".join(f"{y['check']}: {y['message']}" for y in x["violations"]) or "-"]
              for x in r["provenance"]["verification"]]
    return f"""# Model Card: {r['model_id']} {r['model_name']}

**Status: {r['status']}.** Generated by the Documentation agent on {r['provenance']['generated_utc']}.
Narrative sections are reverse-engineered by the Documentation agent ({r['provenance']['documented_by']}) and passed deterministic checks V-01..V-05.
Every number in this card is computed by code from the captured R run; the agent is not allowed to write statistics.

## 1. Identification
{_tbl(['Field', 'Value'], [['Model id', r['model_id']], ['Source', f"`{r['source']['script']}` ({r['source']['language']}, sha256 `{r['source']['sha256'][:12]}`)"],
 ['Supporting documents', ', '.join(f'`{d}`' for d in r['source']['supporting_docs']) or '-'],
 *[[k.replace('_', ' ').capitalize(), ', '.join(v) if isinstance(v, list) else '; '.join(f"{a}: {b}" for a, b in v.items()) if isinstance(v, dict) else v] for k, v in g.items()]])}

## 2. Purpose and use
- **Purpose:** {r['purpose']}
- **Intended use:** {r['intended_use']}
- **Out of scope:** {r['out_of_scope_use']}

## 3. Methodology
{r['methodology']['summary']}

{_tbl(['Item', 'Value'], [['R object class', ', '.join(m['class'])], ['Family / link', f"{m['family']} / {m['link']}"], ['Formula', f"`{m['formula']}`"],
 ['Coefficients', m['n_coefficients']], ['Converged', f"{m['converged']} ({m['iterations']} IRLS iterations)"],
 ['Deviance (null → residual)', f"{m['null_deviance']:.1f} → {m['deviance']:.1f}"], ['AIC', f"{m['aic']:.1f}"],
 ['Train / test rows', f"{facts['split']['n_train']} / {facts['split']['n_test']}"]])}

**Target.** {r['target']['definition']}

**Variable selection.** {r['methodology']['variable_selection']}

## 4. Input variables
{_tbl(['Variable', 'Kind', 'Description', 'Levels', 'Transformation', 'Protected', 'Source lines'],
  [[x['name'], x['kind'], x['description'], ', '.join(f"{l}={x['level_meaning'].get(l, '?')}" for l in x['levels']) or '-',
    x['transformation'], '**yes**' if x['protected'] else '', ', '.join(map(str, x['source_lines']))] for x in r['inputs']])}

## 5. Data treatment (in execution order)
{_tbl(['#', 'Step', 'Columns', 'Author rationale', 'Assumption?', 'Lines'],
  [[s['order'], s['description'], ', '.join(s['columns']), s['author_rationale'] or '-', 'yes' if s['is_assumption'] else '', ', '.join(map(str, s['source_lines']))] for s in r['preprocessing']])}

Data: `{r['data']['path']}` (sha256 `{r['data']['sha256'][:12]}`), {r['data']['n_rows']} rows × {r['data']['n_columns']} columns, {r['data']['missing_values']} missing values.

## 6. Coefficients (captured from the R run)
{_tbl(['Term', 'Estimate', 'SE', 'z', 'p-value'], coef_rows)}

## 7. Performance (reproduced from the R run's own predictions)
{_tbl(['Metric', 'Train', 'Test'], perf_rows) if perf_rows and test else '_Not computed for this model family._'}

The split was regenerated by the capture harness with seed {facts['split']['seed_used_by_harness']}; see limitations for reproducibility.

## 8. Assumptions
{chr(10).join(f'- {a}' for a in r['assumptions'])}

## 9. Limitations and risks
{_tbl(['Severity', 'Limitation', 'Disclosure', 'Lines'], [[l['severity'], l['text'], ', '.join(l['disclosure_ids']) or '-', ', '.join(map(str, l['source_lines'])) or '-'] for l in lims])}

## 10. Code vs. documentation discrepancies
{chr(10).join(f'- {t}' for t in r['code_vs_docs_discrepancies']) or '- None found.'}

## 11. Open questions for the model owner
{chr(10).join(f'- {q}' for q in r['open_questions']) or '- None.'}

## 12. Provenance and verification
{_tbl(['Iteration', 'Extract', 'Verify', 'Violations'], v_rows)}

Required disclosures detected by code: {', '.join(r['provenance']['required_disclosures']) or 'none'}.
Input hashes are re-checked before rendering. Full trace: `{rel(mdl.logs / 'audit.jsonl')}`.

## 13. Human review
- [ ] Purpose, intended use and governance fields confirmed by the model owner
- [ ] Every limitation reviewed; severities agreed
- [ ] Open questions answered or accepted
- [ ] Approve: `/mrf:approve {mdl.slug}` (target documentation)
"""


# ====================================================================== AGENT-FACING STEPS
def _evidence(m: Model) -> tuple[str, dict, str]:
    src = m.script.read_text()
    docs = {str(d.relative_to(m.input)): strip_console_output(d.read_text(errors="replace")) for d in m.docs}
    return src, docs, src + "\n" + "\n".join(docs.values())


def doc_facts(m: Model) -> int:
    Stage(m, "intake").require("scan")
    st = Stage(m, "documentation")
    ref = m.reference
    meta = json.loads((ref / "meta.json").read_text())
    coefs = pd.read_csv(ref / "coefficients.csv")
    coefs["z"] = coefs["estimate"] / coefs["std_error"]
    coefs["p_value"] = [math.erfc(abs(z) / math.sqrt(2)) for z in coefs["z"]]
    response, variables = formula_variables(meta["formula"], list(meta["column_classes_after_preprocessing"]))
    pred = pd.read_csv(ref / "predictions.csv")
    perf = {s: binary_metrics(g["y"].to_numpy(), g["pd"].to_numpy()) for s, g in pred.groupby("set")} \
        if meta["family"] == "binomial" else {}
    src, docs, _ = _evidence(m)
    protected = m.cfg["data"].get("protected_attributes", [])
    f = {
        "model": {"class": meta["class"], "family": meta["family"], "link": meta["link"], "formula": re.sub(r"\s+", " ", meta["formula"]),
                  "response": response, "input_variables": variables, "factor_levels": meta.get("xlevels") or {},
                  "contrasts": meta.get("contrasts") or {}, "n_coefficients": len(coefs), "converged": meta["converged"],
                  "iterations": meta["iterations"], "deviance": meta["deviance"], "null_deviance": meta["null_deviance"], "aic": meta["aic"],
                  "r_version": meta["r_version"], "r_packages_loaded": meta["packages"]},
        "column_classes_after_preprocessing": meta["column_classes_after_preprocessing"],
        "split": {"n_train": meta["n_train"], "n_test": meta["n_test"], "seed_used_by_harness": meta["seed"]},
        "protected_attributes_in_data": protected,
        "protected_attributes_used_as_inputs": [v for v in variables if protected_match(v, protected)],
        "data_profile": data_profile(m.data),
    }
    f["required_disclosures"] = required_disclosures(src, meta, coefs, variables, protected)
    (st.dir / "facts.json").write_text(json.dumps(f, indent=2, default=str))
    coefs.to_csv(st.dir / "coefficients.csv", index=False)
    (st.dir / "performance.json").write_text(json.dumps(perf, indent=2))
    (st.dir / "source_numbered.txt").write_text(numbered(src))
    for name, txt in docs.items():
        stem = name.replace("/", "__")
        (st.dir / f"doc__{stem if stem.endswith('.txt') else stem + '.txt'}").write_text(txt)
    st.state = {"stages": {}, "iterations": []}
    st.mark("facts", n_variables=len(variables), disclosures=list(f["required_disclosures"]))
    print(json.dumps({"facts": rel(st.dir / "facts.json"), "numbered_source": rel(st.dir / "source_numbered.txt"),
                      "supporting_docs_stripped": [rel(p) for p in sorted(st.dir.glob("doc__*.txt"))],
                      "input_variables": variables, "target": response,
                      "required_disclosures": {k: v["why"] + (f" (lines {v['lines']})" if v.get("lines") else "")
                                               for k, v in f["required_disclosures"].items()},
                      "write_extract_to": rel(st.dir / "extract.json")}, indent=1))
    return 0


def doc_verify(m: Model) -> int:
    st = Stage(m, "documentation")
    st.require("facts")
    path = st.dir / "extract.json"
    final = st.dir / "extract_final.json"
    if st.done("verify"):
        if final.exists() and path.exists() and sha(path) == sha(final):
            print("extract already verified and unchanged; run `mrf doc-render`")
            return 0
        # the agent changed extract.json after a PASS: verify it again (counts as a new iteration, same limit)
        for k in ("verify", "render"):
            st.state["stages"].pop(k, None)
        st.state.pop("status", None)
        st.audit("reverify", reason="extract.json changed after a passing verify")
    if not path.exists():
        raise SystemExit(f"{rel(path)} not found: the Documentation agent must write it first")
    hist = st.state.setdefault("iterations", [])
    it = len(hist) + 1
    max_it = m.limits["max_doc_iterations"]
    shutil.copyfile(path, st.dir / f"extract_{it}.json")
    try:
        ex = DocExtract.model_validate_json(path.read_text()).model_dump()
        facts = json.loads((st.dir / "facts.json").read_text())
        src, _, evidence = _evidence(m)
        viol = verify(ex, facts, src, evidence)
    except ValidationError as e:
        viol = [{"check": "V-06", "message": f"schema: {err['loc']}: {err['msg']}"} for err in e.errors()[:15]]
    hist.append({"iteration": it, "file": rel(st.dir / f"extract_{it}.json"), "violations": viol, "verdict": "PASS" if not viol else "FAIL"})
    st.save()
    st.audit("verify", iteration=it, verdict=hist[-1]["verdict"], n_violations=len(viol))
    if not viol:
        shutil.copyfile(path, st.dir / "extract_final.json")
        st.mark("verify", iterations=it)
        print(json.dumps({"verdict": "PASS", "iteration": it}, indent=1))
        return 0
    if it >= max_it:
        st.escalate("verify", [f"{x['check']}: {x['message']}" for x in viol], f"Documentation escalation: {m.id}")
        print(json.dumps({"verdict": "ESCALATED", "violations": viol, "file": rel(st.dir / "ESCALATION.md")}, indent=1))
        return 4
    print(json.dumps({"verdict": "FAIL", "iteration": it, "iterations_left": max_it - it, "violations": viol}, indent=1))
    return 1


def doc_render(m: Model, documented_by: str = "") -> int:
    st = Stage(m, "documentation")
    st.require("verify")
    if changed := m.changed_inputs():
        st.escalate("render", [f"inputs changed since intake: {changed}"], f"Documentation escalation: {m.id}")
        print(f"ESCALATED: inputs changed since intake: {changed}")
        return 4
    cfg, gov = m.cfg, m.cfg.get("governance", {})
    facts = json.loads((st.dir / "facts.json").read_text())
    ex = DocExtract.model_validate_json((st.dir / "extract_final.json").read_text()).model_dump()   # fills optional fields
    perf = json.loads((st.dir / "performance.json").read_text())
    coefs = pd.read_csv(st.dir / "coefficients.csv")
    hashes = m.input_hashes()
    out = m.reports / "documentation"
    out.mkdir(parents=True, exist_ok=True)
    mid, mm = m.id, facts["model"]
    documented_by = documented_by or "Documentation agent (agent/agents/documentation-agent.md)"
    record = {
        "model_id": mid, "model_name": cfg.get("model_name", mid), "record_version": 1,
        "status": "DRAFT - AWAITING HUMAN APPROVAL",
        "governance (human-owned, from model.yaml)": {k: gov.get(k, "NOT PROVIDED - owner to supply") for k in
                                                      ("owner", "developer", "domain", "business_use", "materiality_tier",
                                                       "regulatory_framework", "last_independent_validation", "data_source")},
        "source": {"language": cfg["legacy"]["language"], "script": rel(m.script), "sha256": hashes["legacy_script"],
                   "supporting_docs": [rel(d) for d in m.docs], "r_version": mm["r_version"]},
        "purpose": ex["purpose"], "intended_use": ex["intended_use"], "out_of_scope_use": ex["out_of_scope_use"],
        "methodology": {"summary": ex["methodology"], "class": mm["class"], "family": mm["family"], "link": mm["link"],
                        "formula": mm["formula"], "variable_selection": ex["variable_selection"]},
        "target": {"name": mm["response"], "definition": ex["target_definition"]},
        "inputs": [{**{k: x[k] for k in ("name", "kind", "description", "levels", "level_meaning", "transformation", "source_lines")},
                    "r_class": facts["column_classes_after_preprocessing"].get(x["name"]),
                    "protected": x["name"] in facts["protected_attributes_used_as_inputs"]}
                   for x in ex["variables"] if x["role"] == "input"],
        "preprocessing": ex["preprocessing"],
        "data": {"path": rel(m.data), "sha256": hashes["data"], "n_rows": facts["data_profile"]["n_rows"],
                 "n_columns": len(facts["data_profile"]["columns"]), **facts["split"],
                 "missing_values": sum(c["n_missing"] for c in facts["data_profile"]["columns"])},
        "protected_attributes_used_as_inputs": facts["protected_attributes_used_as_inputs"],
        "coefficients": coefs[["term", "estimate", "std_error", "z", "p_value"]].to_dict("records"),
        "performance_reproduced": perf,
        "assumptions": ex["assumptions"], "limitations": ex["limitations"],
        "code_vs_docs_discrepancies": ex["code_vs_docs_discrepancies"], "open_questions": ex["open_questions"],
        "provenance": {"generated_utc": now(), "documented_by": documented_by, "verification": st.state["iterations"],
                       "input_hashes": hashes, "required_disclosures": list(facts["required_disclosures"])},
    }
    (out / f"inventory_{mid}.json").write_text(json.dumps(record, indent=2, default=str))
    (out / f"MODEL_CARD_{mid}.md").write_text(_card(m, record, facts, ex))
    INVENTORY.mkdir(parents=True, exist_ok=True)
    idx_f = INVENTORY / "inventory_index.json"
    idx = json.loads(idx_f.read_text()) if idx_f.exists() else {"models": {}}
    idx["models"].setdefault(mid, {}).update({
        "model_name": record["model_name"], "input": rel(m.input), "output": rel(m.out), "status": record["status"],
        "language": cfg["legacy"]["language"], "model_type": f"{mm['class'][0]} {mm['family']}/{mm['link']}",
        "protected_inputs": record["protected_attributes_used_as_inputs"],
        "n_limitations_high": sum(l["severity"] == "High" for l in ex["limitations"]),
        "record": rel(out / f"inventory_{mid}.json"), "model_card": rel(out / f"MODEL_CARD_{mid}.md"), "updated_utc": now()})
    idx_f.write_text(json.dumps(idx, indent=2))
    st.state["status"] = "AWAITING_HUMAN_APPROVAL"
    st.mark("render", output=rel(out))
    from agent.tools import tracking
    st.state["mlflow_run_id"] = tracking.log_run(
        m, st, "agent_documentation",
        tags={"model_id": mid, "stage": "documentation", "status": st.state["status"], "documented_by": documented_by},
        params={"legacy_script": rel(m.script), "supporting_docs": record["source"]["supporting_docs"],
                "required_disclosures": list(facts["required_disclosures"])},
        metrics={"verify_iterations": len(st.state["iterations"]), "verify_pass": 1.0, "n_inputs": len(record["inputs"]),
                 "n_limitations": len(ex["limitations"]), "n_limitations_high": sum(l["severity"] == "High" for l in ex["limitations"]),
                 "n_open_questions": len(ex["open_questions"]), "n_required_disclosures": len(facts["required_disclosures"]),
                 "test_auc": perf.get("test", {}).get("auc")},
        artifacts={"documentation": out, "facts": st.dir / "facts.json"})
    st.save()
    print(f"RENDERED -> {rel(out / f'MODEL_CARD_{mid}.md')}  status AWAITING_HUMAN_APPROVAL")
    return 0


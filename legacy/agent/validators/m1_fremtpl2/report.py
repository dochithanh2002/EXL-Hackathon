"""Assemble the M1 independent validation report (Markdown + self-contained HTML) from the JSON outputs.

Inputs: reports/m1/{parity,validation,fairness}/*.json and the Documentation Agent's record
(output/m1_fremtpl2/reports/documentation/inventory_M1.json). Run by the Validation agent.
"""
from __future__ import annotations

import base64
import json
import re
from datetime import date

import markdown
import pandas as pd

from agent.validators.m1_fremtpl2 import common as c

M1 = c.OUT
ICON = {"GREEN": "🟢 Green", "AMBER": "🟠 Amber", "RED": "🔴 Red"}
f3 = lambda x: f"{x:.3f}"


LABELS = {"freq_psi": "PSI of predicted frequency", "max_csi": "Any rating-factor CSI", "ae_dev": "Overall A/E deviation from 1",
          "loss_increase_rel": "Poisson loss increase vs validation (rel.)", "max_age_band_ae_dev": "Max A/E deviation, any driver-age band"}

def trigger_table(extra: dict | None = None) -> str:
    """Re-validation triggers rendered from agent/config/thresholds.yaml (monitoring section)."""
    rows = ["| Metric | Watch (amber) | Re-validate (red) |", "|---|---|---|"]
    for k, v in c.M.thresholds("monitoring").items():
        if isinstance(v, dict):
            op = "<" if v.get("direction") == "lower_is_worse" else "≥"
            rows.append(f"| {LABELS.get(k, k)} | {op} {v['amber']} | {op} {v['red']} |")
    return "\n".join(rows) + "\n\n_Thresholds from `agent/config/thresholds.yaml` (monitoring)._"


def build_md() -> str:
    P = json.loads((M1 / "parity" / "parity_results.json").read_text())
    V = json.loads((M1 / "validation" / "validation_results.json").read_text())
    F = json.loads((M1 / "fairness" / "fairness_results.json").read_text())
    inv_path = c.M.reports / "documentation" / "inventory_M1.json"
    INV = json.loads(inv_path.read_text())
    G = INV["governance (human-owned, from model.yaml)"]; DS = G.get("data_source", {}) if isinstance(G.get("data_source"), dict) else {}
    CH = V["champion"]
    pp, cal, st, sp = V["predictive_power"], V["calibration"], V["stability"], V["specification"]
    dec = pd.read_csv(M1 / "validation" / "ae_by_decile_test.csv")
    lr = sp["lr_drop_tests"]
    mis = F["bands_mispriced"]
    ratings = [
        ("Reproducibility & Python rebuild", "GREEN" if P["parity_pass"] else "RED",
         f"Validator re-check **{'PASS' if P['parity_pass'] else 'FAIL'}** ({sum(x['pass'] for x in P['checks'])}/{len(P['checks'])}); agent needed {P['migration_iterations']} iterations. Seeded split."),
        ("Predictive power", pp["rag"], f"Test Poisson loss {pp['loss_test_x100']:.3f} vs {pp['loss_test_null_x100']:.3f} for a constant frequency; Lorenz Gini {f3(pp['gini_test'])}"),
        ("Calibration & back-test", cal["rag"], f"A/E {cal['ae_test']:.3f} on test; slope {cal['calibration_slope']:.2f}; {cal['deciles_flagged']} of 10 deciles flagged; exposure elasticity {cal['exposure_elasticity']:.2f} (offset assumes 1)"),
        ("Stability", st["rag"], f"PSI {st['psi_freq_learn_vs_test']:.5f}, max CSI {st['max_csi']:.5f}; no out-of-time data"),
        ("Model specification", sp["rag"], f"Poisson GBM challenger loss {sp['challenger_gbm_loss_test_x100']:.3f} ({-sp['champion_minus_challenger_loss']:+.3f}); AreaGLM not significant (p = {lr['AreaGLM']['p']:.2f})"),
        ("Fairness: driver age (statistical)", F["statistical_rag"], f"{len(mis)} age band(s) mispriced on test: " + "; ".join(f"{m['band']} A/E {m['ae']:.2f}" for m in mis)),
        ("Fairness: legal / conduct", F["legal_rag"], "Age is a rating factor: permitted only with documented actuarial evidence (Equality Act Sch.3 para 20A)"),
    ]
    findings = [
        ("F1", "High", f"Claims are not proportional to exposure: freeing the log-exposure coefficient gives {cal['exposure_elasticity']:.2f} (SE {cal['exposure_elasticity_se']:.3f}) instead of the 1.0 the offset imposes. Short-period policies are under-priced and full-year policies over-priced relative to their risk.",
         "Investigate exposure recording (mid-term cancellations, renewals); test exposure as a covariate or a calibrated exposure curve. Re-validate.", "Developer", "Before next tariff"),
        ("F2", "Medium", f"The youngest drivers (18-20) are charged {mis[0]['relativity']:.1f}x the 41-50 frequency but claim at A/E {mis[0]['ae']:.2f} on test (p = {mis[0]['p_value']:.3f}, {mis[0]['exposure']:.0f} policy-years). 51-70 is also over-predicted (A/E {mis[-1]['ae']:.2f})." if mis else "No age band is significantly mispriced.",
         "Document the actuarial evidence for each age band; review the 18-20 relativity with credibility weighting; monitor A/E by age band quarterly.", "Model owner", "30 days"),
        ("F3", "Medium", f"A Poisson GBM on the raw rating factors reaches test loss {sp['challenger_gbm_loss_test_x100']:.3f} vs {pp['loss_test_x100']:.3f} (Gini {f3(sp['challenger_gbm_gini_test'])} vs {f3(pp['gini_test'])}): the GLM leaves material signal (non-linearity, interactions).",
         "Consider finer bands / interactions (e.g. driver age × bonus-malus) while keeping an interpretable GLM tariff.", "Developer", "Next redevelopment"),
        ("F4", "Medium", "Response and exposure are capped before fitting (ClaimNb at 4, Exposure at 1 year) and treated as data errors without investigation (model card D-06).",
         "Quantify the capped records and their claims; confirm the treatment with the data owner.", "Data owner", "30 days"),
        ("F5", "Low", f"{sp['n_nonsignificant_terms']} coefficients are not significant at 5%; AreaGLM adds nothing beyond DensityGLM (LR {lr['AreaGLM']['LR']:.2f}, p = {lr['AreaGLM']['p']:.2f}).",
         "Group sparse VehBrand / Region levels; drop or justify AreaGLM.", "Developer", "Next redevelopment"),
        ("F6", "Low", "freMTPL2 has no time dimension, so out-of-time stability and drift cannot be assessed from development data; the data is French, not a UK book.",
         "Monitor PSI/CSI and A/E on live batches (monitoring triggers below).", "Model owner", "Ongoing"),
    ]
    age_rows = "\n".join(f"| {a['age_band']} | {a['exposure']:,.0f} | {a['actual']:.0f} | {a['expected']:.1f} | {a['ae']:.3f} ({a['ci_lo']:.2f}–{a['ci_hi']:.2f}) | {a['p_value']:.3f} | {a['flag']} | {a['premium_relativity_vs_41_50']:.2f} |"
                         for a in F["ae_by_age_band"])
    dec_rows = "\n".join(f"| {int(r.band)} | {r.exposure:,.0f} | {r.actual:.0f} | {r.expected:.1f} | {r.ae:.3f} | {r.p_value:.3f} | {r.flag} |" for r in dec.itertuples())
    par_rows = "\n".join(f"| {x['check']} | {x['value']:.2e} | {x['tolerance']:.0e} | {'✅' if x['pass'] else '❌'} |" for x in P["checks"])
    lr_rows = "\n".join(f"| {k} | {v['LR']:.1f} | {v['df']} | {v['p']:.2g} |" for k, v in lr.items())
    csi_rows = "\n".join(f"| {k} | {v:.5f} |" for k, v in st["csi_learn_vs_test"].items())
    rp = pp["repeated_split_loss_x100"]
    return f"""# Independent Model Validation Report: M1 French Motor TPL Claim Frequency

| | |
|---|---|
| **Model** | {INV['model_name']} (`M1`), {INV['methodology']['family']}/{INV['methodology']['link']} GLM with log-exposure offset |
| **Source** | {INV['source']['language']} `{INV['source']['script']}`; developer: {G['developer']} |
| **Model under validation** | `{CH['module']}` (sha256 `{CH['module_sha256'][:12]}`), generated by the Migration Agent ({CH.get('migrated_by') or 'agent'}); migration status {CH['migration_status']} |
| **Documentation** | `output/m1_fremtpl2/reports/documentation/MODEL_CARD_M1.md` (Documentation Agent, {INV['status']}) |
| **Use** | {G['business_use']} |
| **Framework** | {', '.join(G['regulatory_framework'])} |
| **Report date** | {date.today().isoformat()} |
| **Overall outcome** | **{ICON[V['overall_rag']]}: approved for continued use subject to F1 (exposure) and F2 (young-driver relativities)** |

## 1. Executive summary

- **The migration is exact.** The agent-generated Python module reproduces R to {max(x['value'] for x in P['checks'] if 'coef max rel' in x['check']):.0e} on coefficients and {max(x['value'] for x in P['checks'] if 'expected claims' in x['check']):.0e} on all 678,013 expected claim counts; the validator confirmed it independently.
- **The model ranks risk.** Test Poisson loss {pp['loss_test_x100']:.3f} against {pp['loss_test_null_x100']:.3f} for a flat frequency, Lorenz Gini {f3(pp['gini_test'])}. Across {rp['n']} random splits the loss is {rp['mean']:.2f} ± {rp['sd']:.2f}.
- **Overall level is right, the exposure assumption is not (F1).** A/E on test is {cal['ae_test']:.3f}, but claims scale with exposure to the power {cal['exposure_elasticity']:.2f}, not 1.
- **Age is used as a rating factor (F2).** It is statistically justified (LR {F['lr_test_driver_age']['LR']:.0f} on {F['lr_test_driver_age']['df']} df), but the youngest band is over-charged on test.
- **A GBM challenger is materially better (F3)**, so the GLM is under-specified.

| Area | Rating | Evidence |
|---|---|---|
""" + "\n".join(f"| {n} | {ICON[r]} | {t} |" for n, r, t in ratings) + f"""

## 2. Scope and data

- **Data:** {DS.get('dataset', '')}, {DS.get('licence', '')}. {DS.get('population', '')}; {DS.get('observation_window', '')}.
- **Target:** {INV['target']['definition']}
- **Split:** random 90/10 (set.seed(100)), {INV['data']['n_train']:,} learning and {INV['data']['n_test']:,} test policies, frozen in `{CH['reference_dir']}/split.csv`.
- **Rating factors ({len(INV['inputs'])}):** {', '.join(i['name'] for i in INV['inputs'])}.
- **Model card:** {len(INV['limitations'])} limitations ({sum(l['severity'] == 'High' for l in INV['limitations'])} high), {len(INV['open_questions'])} open questions for the owner.

## 3. Reproducibility and Python rebuild (parity)

The Migration Agent ran the legacy R script unmodified, froze its outputs and generated `{CH['module']}` (parity in {P['migration_iterations']} iterations: the first candidate failed at runtime because pandas 3 reads text as `str`, the repair fixed the quote handling). The validator re-ran that exact module with its own code (`agent/validators/m1_fremtpl2/parity.py`). **Result: {'PASS' if P['parity_pass'] else 'FAIL'}.** A fully converged statsmodels fit agrees to {P['independent_crosscheck_statsmodels'][0]['value']:.0e}.

| Check | Observed | Tolerance | Pass |
|---|---|---|---|
{par_rows}

## 4. Predictive power

| | Learn | Test |
|---|---|---|
| Poisson deviance loss (×100) | {pp['loss_learn_x100']:.4f} | {pp['loss_test_x100']:.4f} |
| Constant-frequency loss (×100) | | {pp['loss_test_null_x100']:.4f} |
| Lorenz Gini | {f3(pp['gini_learn'])} | {f3(pp['gini_test'])} |

- Deviance explained on test: {100 * pp['deviance_explained_test']:.1f}% (typical for claim frequency).
- Repeated random 90/10 splits ({rp['n']}): loss {rp['mean']:.3f} ± {rp['sd']:.3f} (range {rp['min']:.2f}–{rp['max']:.2f}); this test split is a favourable one.
- Published case study GLM1: learn {pp['published_case_study_glm1']['loss_learn_x100']}, test {pp['published_case_study_glm1']['loss_test_x100']} (different split).

![Lorenz](validation/lorenz.png)

## 5. Calibration and back-test

- **A/E (test):** {cal['ae_test']:.4f}. **Calibration slope** of the risk ranking: {cal['calibration_slope']:.3f} (SE {cal['calibration_slope_se']:.3f}).
- **Exposure elasticity:** {cal['exposure_elasticity']:.3f} (SE {cal['exposure_elasticity_se']:.3f}); the offset forces 1.0 (F1).

| Decile | Exposure | Actual | Expected | A/E | Poisson p | Flag |
|---|---|---|---|---|---|---|
{dec_rows}

![A/E by decile](validation/ae_deciles.png)

## 6. Stability

- PSI of predicted frequency learn → test {st['psi_freq_learn_vs_test']:.5f}; coefficient variation across splits is {st['coef_split_sd_over_model_se']['median']:.2f}× the model SE (median), well inside sampling error.
- {st['note']}

| Rating factor | CSI learn → test |
|---|---|
{csi_rows}

## 7. Specification

| Drop | LR | df | p |
|---|---|---|---|
{lr_rows}

- Poisson GBM challenger (raw factors): test loss {sp['challenger_gbm_loss_test_x100']:.4f}, Gini {f3(sp['challenger_gbm_gini_test'])}.
- {sp['n_nonsignificant_terms']} of the GLM's coefficients are not significant at 5%.

## 8. Fairness: driver age

Age is a protected characteristic. In UK insurance it may be used for pricing only where the differences rest on relevant and reliable actuarial data. The test is therefore **calibration within each age band** (is the band's price supported by its claims?) and the size of the relativities.

| Age band | Exposure (years) | Actual | Expected | A/E (95% CI) | p | Flag | Frequency relativity vs 41-50 |
|---|---|---|---|---|---|---|---|
{age_rows}

![A/E by driver age](fairness/ae_by_driver_age.png)

- Driver age is strongly significant (LR {F['lr_test_driver_age']['LR']:.0f}, p < 1e-100), which supports its use in principle.
- Non-age factors predict young drivers (≤ 25) with AUC {F['proxy_auc_young_driver']:.2f} (mainly bonus-malus), so removing age would not remove its effect.
- {F['legal_note']}

## 9. Findings and actions

| ID | Severity | Finding | Required action | Owner | Due |
|---|---|---|---|---|---|
""" + "\n".join(f"| {i} | {s} | {t} | {a} | {o} | {d} |" for i, s, t, a, o, d in findings) + """

## 10. Ongoing monitoring and re-validation triggers

""" + trigger_table() + """

Monitoring runs these checks on every scoring batch (`/mrf:monitor m1_fremtpl2`, results in `monitoring/`).

## 11. Limitations of this validation

- No out-of-time sample; the monitoring batches are simulated from the test data.
- Claim severity is out of scope: this validates the frequency component only.
- Region could proxy for other protected characteristics (e.g. ethnicity); not testable with this data.

## Appendix: reproduce

In Claude Code: `/mrf:run m1_fremtpl2` (or in Codex: the `run` skill). Thresholds: `agent/config/thresholds.yaml`.
"""


CSS = """:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#f4f3f0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--card:#232321}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--card:#232321}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 80px}h1{font-size:28px;line-height:1.25}h2{font-size:20px;margin:40px 0 12px;padding-top:12px;border-top:1px solid var(--line)}
code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:13px}pre{background:var(--card);padding:12px;border-radius:8px;overflow-x:auto}
.table-wrap{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--ink2);background:var(--card)}td{font-variant-numeric:tabular-nums}img{max-width:100%;border-radius:8px;border:1px solid var(--line);margin:12px 0;background:#fcfcfb}"""


def main():
    md_text = build_md()
    (M1 / "validation_report_M1.md").write_text(md_text)
    inline = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)",
                    lambda m: f"![{m.group(1)}](data:image/png;base64,{base64.b64encode((M1 / m.group(2)).read_bytes()).decode()})", md_text)
    body = markdown.markdown(inline, extensions=["tables", "fenced_code"]).replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    (M1 / "validation_report_M1.html").write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 Validation Report</title><style>{CSS}</style></head><body><main>{body}</main></body></html>')


if __name__ == "__main__":
    main()
    print("wrote", M1 / "validation_report_M1.md", "and .html")

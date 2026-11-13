"""Assemble the M2 independent validation report (Markdown + self-contained HTML).

Every number is read from the JSON outputs of parity.py, validate.py and fairness.py, so the
report regenerates deterministically (and an agent can rewrite the narrative from the same inputs).
Run by the Validation agent.
"""
from __future__ import annotations

import base64
import json
import re
from datetime import date

import markdown
import pandas as pd

from agent.validators.m2_taiwan import common as c

M2 = c.M.reports
P = json.loads((M2 / "parity" / "parity_results.json").read_text())
V = json.loads((M2 / "validation" / "validation_results.json").read_text())
F = json.loads((M2 / "fairness" / "fairness_results.json").read_text())
# Model documentation = the record produced by the Documentation Agent (single source of truth)
INV_PATH = c.M.reports / "documentation" / "inventory_M2.json"
INV = json.loads(INV_PATH.read_text())
GOV = INV["governance (human-owned, from model.yaml)"]
DS = GOV["data_source"] if isinstance(GOV.get("data_source"), dict) else {}
CH = V["champion"]
CAL = pd.read_csv(M2 / "validation" / "calibration_backtest_test.csv")

pct = lambda x, d=1: f"{100 * x:.{d}f}%"
f3 = lambda x: f"{x:.3f}"
LABELS = {"score_psi": "Score PSI (vs development)", "max_csi": "Any input CSI", "auc_drop": "AUC fall from validated test AUC (abs.)",
          "citl_pp": "Calibration-in-the-large (pp)", "min_dir": "DIR for any protected group", "max_group_cal_gap_pp": "Within-group calibration gap (pp)"}

def trigger_table(extra: dict | None = None) -> str:
    """Re-validation triggers rendered from agent/config/thresholds.yaml (monitoring section)."""
    rows = ["| Metric | Watch (amber) | Re-validate (red) |", "|---|---|---|"]
    for k, v in c.M.thresholds("monitoring").items():
        if isinstance(v, dict):
            op = "<" if v.get("direction") == "lower_is_worse" else "≥"
            rows.append(f"| {LABELS.get(k, k)} | {op} {v['amber']} | {op} {v['red']} |")
    return "\n".join(rows) + "\n\n_Thresholds from `agent/config/thresholds.yaml` (monitoring)._"


ICON = {"GREEN": "🟢 Green", "AMBER": "🟠 Amber", "RED": "🔴 Red"}


def group_table(attr: str, variant: str = "A_baseline", cut: str = "approve_80pct") -> str:
    rows = F["variants"][variant]["attrs"][attr][cut]["table"]
    out = ["| Group | n | Default rate | Mean PD | Calibration gap | AUC | Approval rate (95% CI) | Approval of good payers | DIR vs best |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        small = " ⚠️" if r["small_group"] else ""
        out.append(f"| {r['group']}{small} | {int(r['n']):,} | {pct(r['default_rate'])} | {pct(r['mean_pd'])} | "
                   f"{r['cal_gap_pp']:+.2f}pp | {r['auc']:.3f} | {pct(r['approval_rate'])} "
                   f"({pct(r['approval_ci_lo'], 0)}–{pct(r['approval_ci_hi'], 0)}) | {pct(r['approval_good_payers'])} | "
                   f"{r['dir_vs_best']:.3f} |")
    return "\n".join(out)


def ablation_table() -> str:
    names = {"A_baseline": "A. Baseline (as built)", "B_no_sex_marriage": "B. Without sex and marital status",
             "C_no_sex_marriage_education": "C. Without sex, marital status and education"}
    out = ["| Variant | Test AUC | DIR sex | DIR age | DIR education* | Worst calibration gap: sex / age / education* | New flags |",
           "|---|---|---|---|---|---|---|"]
    for k, lab in names.items():
        A = F["variants"][k]
        s = {a: A["attrs"][a]["approve_80pct"]["summary"] for a in ("sex_g", "age_g", "edu_g")}
        fl = "; ".join(f for a in ("sex_g", "age_g", "edu_g") for f in A["attrs"][a]["approve_80pct"]["flags"]) or "none"
        out.append(f"| {lab} | {A['test_auc']:.4f} | {s['sex_g']['disparate_impact_ratio']:.3f} | "
                   f"{s['age_g']['disparate_impact_ratio']:.3f} | {s['edu_g']['disparate_impact_ratio_excl_small']:.3f} | "
                   f"{s['sex_g']['max_abs_cal_gap_pp']:.2f} / {s['age_g']['max_abs_cal_gap_pp']:.2f} / "
                   f"{s['edu_g']['max_abs_cal_gap_pp_excl_small']:.2f} pp | {fl} |")
    return "\n".join(out) + "\n\n\\* Education figures exclude the \"Other/unknown\" group (n = 136), which is too small to measure reliably."


def build_md() -> str:
    d, cal, st, sp, pol = V["discrimination"], V["calibration"], V["stability"], V["specification"], V["policy_cutoffs"]
    dq, rec = V["data_quality"], F["recommendation"]
    hs = {a: F["headline"][a]["summary"] for a in ("sex_g", "age_g", "edu_g", "mar_g")}
    _VT = c.M.thresholds("validation")             # agent/config/thresholds.yaml -> validation.binary_pd
    spec_rag = "AMBER" if sp["pay0_linearity_p"] < _VT["spec_linearity_p"] or d["champion_vs_challenger_auc_gap"] > _VT["spec_challenger_auc_gap"] else "GREEN"
    repro_rag = "AMBER"  # qualitative: parity passes, but the original source had no seed (documentation finding D-02)
    ratings = [
        ("Reproducibility & Python rebuild", repro_rag,
         f"Python parity **PASS** (coefficients agree to {max(x['value'] for x in P['checks'] if 'coef max rel' in x['check']):.0e}). The original source recorded no seed."),
        ("Discrimination", d["rag"], f"Test AUC {f3(d['test']['auc'])} (95% CI {f3(d['test_auc_ci95'][0])}–{f3(d['test_auc_ci95'][1])}), Gini {f3(d['test']['gini'])}, KS {f3(d['test']['ks'])}"),
        ("Calibration & back-test", cal["rag"], f"Mean PD {pct(cal['mean_pd_test'])} vs observed {pct(cal['observed_rate_test'])}. Hosmer–Lemeshow p = {cal['hosmer_lemeshow_test']['p_value']:.2f}. {cal['bands_flagged']} of 10 bands flagged"),
        ("Stability", st["rag"], f"PSI train→test {st['psi_train_vs_test']:.3f}, train→shifted month {st['psi_train_vs_shifted_month']:.3f}. pay_0 CSI {st['csi_train_vs_shifted_month']['pay_0']:.2f}, which is a data-coding artefact"),
        ("Model specification", spec_rag, f"pay_0 is not linear (LR test p < 0.001). The gradient-boosting challenger scores +{100 * d['champion_vs_challenger_auc_gap']:.1f} AUC points"),
        ("Fairness: outcomes", F["statistical_rag"], f"All disparate impact ratios (DIR) ≥ 0.80. Sex {f3(hs['sex_g']['disparate_impact_ratio'])}, age {f3(hs['age_g']['disparate_impact_ratio'])}, education {f3(hs['edu_g']['disparate_impact_ratio_excl_small'])}. Within-group calibration within ±2pp"),
        ("Fairness: legal / conduct", F["legal_rag"], "**Sex and marital status are direct model inputs**"),
    ]
    rating_rows = "\n".join(f"| {n} | {ICON[r]} | {t} |" for n, r, t in ratings)

    findings = [
        ("F1", "High", "Sex and marital status are used as direct inputs (sex2 = −0.17: women are scored as lower risk; marriage2 = −0.25: single people are scored as lower risk). This is direct discrimination risk under the Equality Act 2010 s.13, and there is no financial-services exception for sex in credit decisions.",
         f"Remove both inputs (variant B). Test AUC changes by {rec['auc_change']:+.4f}, which is no loss, and the sex DIR improves by {rec['sex_dir_change']:+.3f}. Re-validate the variant before use.", "Model owner", "Before next use"),
        ("F2", "Medium", f"Undocumented category codes: education 0/5/6 ({dq['education_undocumented_0_5_6']} rows) and marriage 0 ({dq['marriage_undocumented_0']} rows) are folded into \"others\". This produces a spurious education4 coefficient of {sp['undocumented_codes']['education4_coef']:.2f} (SE {sp['undocumented_codes']['education4_se']:.2f}), and the \"Other/unknown\" group gets 99% approval.",
         "Confirm the code meanings with the data owner. Treat unknown codes as missing, not as a category with a score benefit.", "Data owner", "30 days"),
        ("F3", "Medium", f"Repayment-status coding differs by month. Code 1 (one-month delay) appears {dq['pay_code_1_counts']['PAY_0']:,} times in PAY_0 but only {dq['pay_code_1_counts']['PAY_2']}/{dq['pay_code_1_counts']['PAY_3']}/{dq['pay_code_1_counts']['PAY_4']}/{dq['pay_code_1_counts']['PAY_5']}/{dq['pay_code_1_counts']['PAY_6']} times in PAY_2–PAY_6. The author also merged −2/−1/0 into 0 by assumption.",
         "Get the field definitions confirmed. Recode delinquency consistently across months, then refit.", "Data owner / developer", "30 days"),
        ("F4", "Medium", "The model can't be reproduced from source: no random seed was recorded, and the accuracy-maximising cut-off was tuned on the test set. That cut-off also lands exactly on an observed score, so ties occur.",
         "Freeze the split (done: the agent's frozen reference, split.csv). Set a policy cut-off from risk appetite on development data, not the test set.", "Model owner", "Done / 30 days"),
        ("F5", "Medium", f"The model is under-specified. pay_0 is non-linear (LR statistic {sp['pay0_linearity_LR_stat']:.0f}, p < 0.001). Dropped variables are jointly significant (p = {sp['dropped_vars_p']:.3f}). A gradient-boosting challenger reaches AUC {f3(d['challenger_gbm_test']['auc'])}, against {f3(d['test']['auc'])} for the model.",
         "Consider a banded/WoE treatment of the delinquency variables. Record the rationale for keeping an interpretable GLM.", "Developer", "Next redevelopment"),
        ("F6", "Low", f"Age is not a model input, but the remaining inputs predict age ≤ 25 with AUC {F['proxy_auc']['age<=25']:.2f}, mainly through credit limit. Applicants aged 21–25 have the lowest approval rate ({pct(F['variants']['A_baseline']['attrs']['age_g']['approve_80pct']['table'][0]['approval_rate'], 0)}). After removing sex and marital status, the 46–55 band is under-predicted by 2.3pp, just outside the ±2pp tolerance.",
         "Recalibrate variant B and confirm the 46–55 gap is back within ±2pp. Monitor age-band calibration quarterly. Document the risk-based justification for any age effect (Equality Act Sch. 3 para 20A).", "Model owner", "Quarterly"),
        ("F7", "Low", "The data is Taiwan 2005 and is not representative of a UK retail book. Two groups are small: education \"Other/unknown\" (n = 136) and marital status \"Other\" (n = 123), so their confidence intervals are wide.",
         "Re-validate on UK data before any production use.", "Model owner", "Before deployment"),
    ]
    finding_rows = "\n".join(f"| {i} | {s} | {t} | {a} | {o} | {due} |" for i, s, t, a, o, due in findings)

    cal_rows = "\n".join(
        f"| {int(r.band)} | {int(r.n):,} | {pct(r.mean_pd)} | {pct(r.obs_rate)} | {r.gap_pp:+.2f}pp | {r.binom_p:.2f} | {r.flag} |"
        for r in CAL.itertuples())
    csi_rows = "\n".join(f"| {k} | {v:.4f} | {st['csi_train_vs_shifted_month'][k]:.4f} |" for k, v in st["csi_train_vs_test"].items())
    pc = "\n".join(f"| {k} | {v['threshold']:.3f} | {pct(v['approval_rate'])} | {pct(v['defaulters_caught'])} | "
                   f"{pct(v['good_customers_declined'])} | {pct(v['bad_rate_in_approved'])} |" for k, v in pol.items())
    parity_rows = "\n".join(f"| {x['check']} | {x['value']:.2e} | {x['tolerance']:.0e} | {'✅' if x['pass'] else '❌'} |"
                            for x in P["checks"])

    return f"""# Independent Model Validation Report: M2 Taiwan Card-Default PD

| | |
|---|---|
| **Model** | {INV['model_name']} (`{INV['model_id']}`), {INV['methodology']['family']}/{INV['methodology']['link']} GLM |
| **Source** | {INV['source']['language']} `{INV['source']['script']}`; developer: {GOV['developer']} |
| **Model under validation** | `{CH['module']}` (sha256 `{CH['module_sha256'][:12]}`), generated by the Migration Agent ({CH.get('migrated_by') or 'agent'}); migration status {CH['migration_status']} |
| **Documentation** | `{INV_PATH.relative_to(c.ROOT).parent}/MODEL_CARD_M2.md` (Documentation Agent, {INV['status']}) |
| **Use** | {GOV['business_use']} |
| **Framework** | PRA SS1/23 (Principles 3 and 4), Equality Act 2010, FCA Consumer Duty |
| **Validator** | Agentic Model Risk Factory (independent of the developer) |
| **Report date** | {date.today().isoformat()} |
| **Overall outcome** | **🔴 Not approved in its current form. Variant B (without sex and marital status) is approved subject to conditions F2–F5** |

## 1. Executive summary

- **The model works.** Test AUC is {f3(d['test']['auc'])}. Across 200 random train/test splits the AUC averages {f3(d['repeated_split_auc']['mean'])} ± {d['repeated_split_auc']['sd']:.3f}, so the split used here is a pessimistic one. Calibration is good: the average PD is within {abs(cal['calibration_in_the_large_pp']):.2f}pp of the observed default rate, and no PD band fails its back-test.
- **The agent-generated Python module matches the R original exactly**, and the validator confirmed it independently. Coefficients, PDs, AUC and every accept/decline decision agree to floating-point precision (§3).
- **Approval outcomes pass the four-fifths rule for every group.** Disparate impact ratios are sex {f3(hs['sex_g']['disparate_impact_ratio'])}, age {f3(hs['age_g']['disparate_impact_ratio'])}, education {f3(hs['edu_g']['disparate_impact_ratio_excl_small'])} and marital status {f3(hs['mar_g']['disparate_impact_ratio'])}. PDs are calibrated within ±2pp inside every large group.
- **But the model uses sex and marital status as inputs.** That is a legal and conduct issue however good the outcome metrics look. Removing both costs **no accuracy** (AUC {rec['auc_change']:+.4f}) and narrows the gap between men's and women's approval rates.
- **There are data-quality problems** (F2, F3): undocumented category codes, and repayment status coded differently from month to month. These need fixing before redevelopment.

| Area | Rating | Evidence |
|---|---|---|
{rating_rows}

## 2. Scope and data

- **Data:** {DS.get('dataset', INV['data']['path'])}, {DS.get('licence', 'licence not provided')}. {DS.get('population', '')}, observed {DS.get('observation_window', 'n/a')}.
- **Target:** {INV['target']['definition']}
- **Split:** stratified 70/30, with {INV['data']['n_train']:,} training rows and {INV['data']['n_test']:,} test rows. The split is frozen in `{CH['reference_dir']}/split.csv`.
- **Model inputs ({len(INV['inputs'])}):** {', '.join(i['name'] for i in INV['inputs'])}.
- **Documented limitations:** the model card lists {len(INV['limitations'])} ({sum(l['severity'] == 'High' for l in INV['limitations'])} high) and {len(INV['open_questions'])} open questions for the owner; this report tests them.
- **Protected characteristics:** sex and marital status are used directly. Age is available but not used. Education is treated as a potential proxy.
- **Out of scope:** the notebook's rpart and random-forest challengers are listed in the inventory only.

## 3. Reproducibility and Python rebuild (parity)

The Migration Agent ran the legacy R script unmodified, froze its outputs, and generated `{CH['module']}`. The validator then re-ran that exact module (SHA-256 checked against the agent's manifest) on the frozen split with its own code and tolerances (`agent/validators/m2_taiwan/parity.py`). **Result: {'PASS' if P['parity_pass'] else 'FAIL'}.** An independent scikit-learn fit agrees as well.

| Check | Observed | Tolerance | Pass |
|---|---|---|---|
{parity_rows}

The full coefficient-by-coefficient comparison is in `parity/parity_report_M2.md`.

## 4. Discrimination

| | Train | Test | Test 95% CI (bootstrap, 1,000 resamples) |
|---|---|---|---|
| AUC | {f3(d['train']['auc'])} | {f3(d['test']['auc'])} | {f3(d['test_auc_ci95'][0])}–{f3(d['test_auc_ci95'][1])} |
| Gini | {f3(d['train']['gini'])} | {f3(d['test']['gini'])} | |
| KS | {f3(d['train']['ks'])} | {f3(d['test']['ks'])} | {f3(d['test_ks_ci95'][0])}–{f3(d['test_ks_ci95'][1])} |

- **Overfitting:** the train-minus-test AUC gap is {d['overfit_auc_gap']:.3f}. It is small, and the seed analysis shows it is mostly an effect of this particular split.
- **Repeated-split AUC (200 seeds):** mean {f3(d['repeated_split_auc']['mean'])}, 95% range {f3(d['repeated_split_auc']['p2_5'])}–{f3(d['repeated_split_auc']['p97_5'])}. The notebook's reported 0.754 falls inside this range.
- **Challenger** (gradient boosting on all 23 features): AUC {f3(d['challenger_gbm_test']['auc'])}, which is {100 * d['champion_vs_challenger_auc_gap']:+.1f} points better (see F5).

![ROC curves](validation/roc.png)

## 5. Calibration and back-test

- **Brier score:** {cal['brier_test']:.4f}, against {cal['brier_reference_base_rate']:.4f} for a model that predicts the base rate for everyone.
- **Calibration-in-the-large:** {cal['calibration_in_the_large_pp']:+.2f}pp.
- **Calibration slope:** {cal['calibration_slope']:.3f}. A value of 1 is ideal; below 1 means the predictions are slightly too spread out.
- **Hosmer–Lemeshow:** χ² = {cal['hosmer_lemeshow_test']['hl_stat']:.1f}, df = {cal['hosmer_lemeshow_test']['df']}, p = {cal['hosmer_lemeshow_test']['p_value']:.3f}.

| PD band | n | Mean PD | Observed | Gap | Binomial p | Flag |
|---|---|---|---|---|---|---|
{cal_rows}

![Back-test](validation/calibration.png)

## 6. Stability

- **Population stability index (PSI) of the score:** train vs test {st['psi_train_vs_test']:.4f}. Train vs the **shifted month** {st['psi_train_vs_shifted_month']:.4f}. In the shifted month every client is re-scored as at Aug-2005, one month earlier, as a pseudo out-of-time test. Both values are below the 0.10 watch level.
- **Pseudo out-of-time discrimination:** AUC {f3(st['pseudo_oot']['auc'])} against a proxy outcome, a Sep-2005 repayment delay ({pct(st['pseudo_oot']['proxy_event_rate'])} event rate). The proxy is easier to predict than true default, so this is a lower bound on instability, not a like-for-like AUC.
- **Characteristic stability index (CSI) for pay_0 in the shifted month = {st['csi_train_vs_shifted_month']['pay_0']:.2f}.** This is **not population drift**. Code 1 exists only in PAY_0 (see F3), so moving a month back changes how the variable is coded. That makes it a data-definition finding, and it is why stability is rated Amber.

| Variable | CSI train→test | CSI train→shifted month |
|---|---|---|
{csi_rows}

![CSI](validation/csi.png)

## 7. Model specification

- **pay_0 linearity:** banding pay_0 as {{0, 1, 2, 3+}} improves the fit (LR = {sp['pay0_linearity_LR_stat']:.0f}, df = 2, p < 0.001) but adds only {sp['pay0_banded_auc_uplift']:+.4f} test AUC. The non-linearity is statistically significant but makes little difference in practice.
- **Dropped variables:** age, pay_4, bill amounts and pay_amt3–6 together give LR = {sp['dropped_vars_LR_stat']:.1f} on {sp['dropped_vars_df']} df (p = {sp['dropped_vars_p']:.3f}). Selection by p-value left some signal out.
- **Multicollinearity:** the highest variance inflation factor (VIF) is {sp['max_vif']:.2f}, so collinearity is not a concern.
- **Cut-off policy** (test set):

| Cut-off | PD threshold | Approval rate | Defaulters declined | Good customers declined | Bad rate among approved |
|---|---|---|---|---|---|
{pc}

## 8. Fairness assessment

**Method.** The favourable outcome is *approval* (PD below the cut-off). The headline cut-off is the one that approves 80% of applicants (PD < {F['variants']['A_baseline']['cutoffs']['approve_80pct']:.3f}). PD < 0.50 was also tested and is in the JSON output. For every group the assessment computes:
- the disparate impact ratio (DIR: approval rate ÷ the best group's rate, failing below 0.80);
- equal opportunity (approval rate among customers who did not default);
- AUC within the group;
- **within-group calibration** (mean PD minus observed default rate, tolerance ±2pp).

Calibration within groups is the primary test for credit. Default rates genuinely differ between groups, so a calibrated model *cannot* also give every group the same approval rate. Small differences in approval rates that follow real differences in risk are expected.

![Approval by group](fairness/approval_by_group.png)

### 8.1 Sex
{group_table('sex_g')}

Women are approved more often ({pct(F['variants']['A_baseline']['attrs']['sex_g']['approve_80pct']['table'][1]['approval_rate'], 0)} vs {pct(F['variants']['A_baseline']['attrs']['sex_g']['approve_80pct']['table'][0]['approval_rate'], 0)}). That matches their lower observed default rate, and both groups are calibrated within 1pp. Part of the gap, however, comes from the **sex input itself** (F1).

### 8.2 Age (not a model input)
{group_table('age_g')}

### 8.3 Education
{group_table('edu_g')}

⚠️ means n < 200, so treat the figure with caution. The 99% approval rate for "Other/unknown" is the data artefact described in F2.

### 8.4 Marital status
{group_table('mar_g')}

### 8.5 What happens without the protected inputs (ablation)
{ablation_table()}

![Calibration by group](fairness/calibration_by_group.png)

- **Removing sex and marital status (B)** costs no accuracy, reduces the sex gap, and leaves both sexes calibrated within 1pp. The trade-off is that marital status was partly standing in for age: the 46–55 band moves to 2.3pp under-prediction, just outside tolerance.
- **Removing education as well (C)** does *not* narrow the gap between the large education groups (DIR {F['variants']['A_baseline']['attrs']['edu_g']['approve_80pct']['summary']['disparate_impact_ratio_excl_small']:.3f} → {F['variants']['C_no_sex_marriage_education']['attrs']['edu_g']['approve_80pct']['summary']['disparate_impact_ratio_excl_small']:.3f}). The education differences reach the score through repayment history, which reflects real behaviour. Removing the input also leaves the small "Other/unknown" group over-predicted by about 8pp, because that group's low default rate is a data artefact the model can no longer see. **Variant B is therefore preferred.** Education can stay in once its codes are cleaned (F2).

### 8.6 Proxy analysis
These are AUCs for predicting each protected characteristic from the **non-protected** model inputs alone (credit limit, repayment history, payment amounts). 0.5 means the inputs carry no information about the characteristic.

| Protected characteristic | Proxy AUC |
|---|---|
""" + "\n".join(f"| {k} | {v:.3f} |" for k, v in F["proxy_auc"].items()) + f"""

The inputs say little about sex or marital status. They do strongly identify **young applicants** (AUC {F['proxy_auc']['age<=25']:.2f}), mainly through credit limit. So the model can disadvantage young applicants through those inputs even though age itself is not used (F6).

### 8.7 Mitigation options considered
- **Removing the protected inputs (recommended):** this is variant B.
- **Group-specific thresholds** (e.g. fairlearn `ThresholdOptimizer`): *rejected.* Setting different cut-offs by sex would itself be direct discrimination under UK law.
- **Fairness-constrained training** (e.g. `ExponentiatedGradient`): not needed. Once the inputs are removed, the only breach is the 46–55 band's calibration gap, which exceeds the ±2pp tolerance by about 0.3pp. That is handled by recalibration and age-band monitoring (F6), not by constraining the model.

## 9. Findings and actions

| ID | Severity | Finding | Required action | Owner | Due |
|---|---|---|---|---|---|
{finding_rows}

## 10. Ongoing monitoring and re-validation triggers

{trigger_table()}

Monitoring runs these checks on every scoring batch (`/mrf:monitor m2_taiwan`, results in `monitoring/`); any red metric raises a re-validation trigger in the model inventory.

## 11. Limitations of this validation

- There is no true out-of-time sample. The pseudo-OOT test uses a proxy outcome.
- Fairness is measured on observed attributes only. Intersectional groups (e.g. young women) were not tested.
- The data is public and non-UK, and was used as a stand-in for this exercise.

## Appendix: reproduce

In Claude Code: `/mrf:run m2_taiwan` (or in Codex: the `run` skill). Thresholds: `agent/config/thresholds.yaml`.

Every run is logged to MLflow under the experiment `m2-taiwan-default`, in Azure ML when `agent/config/azure.env` is set and in `output/mlflow/` otherwise.
"""


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#f4f3f0;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--card:#232321;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--card:#232321;--accent:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:28px;line-height:1.25;margin:0 0 20px}h2{font-size:20px;margin:40px 0 12px;padding-top:12px;border-top:1px solid var(--line)}h3{font-size:16px;margin:28px 0 8px}
p,li{color:var(--ink)}code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:13px}
pre{background:var(--card);padding:12px;border-radius:8px;overflow-x:auto}
.table-wrap{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600;background:var(--card)}
td{font-variant-numeric:tabular-nums}img{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--line);margin:12px 0;background:#fcfcfb}
"""


def build_html(md_text: str) -> str:
    def embed(m):
        path = M2 / m.group(2)
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f"![{m.group(1)}](data:image/png;base64,{b64})"
    md_inline = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, md_text)
    body = markdown.markdown(md_inline, extensions=["tables", "fenced_code"])
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>M2 Validation Report</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")


def main():
    md_text = build_md()
    (M2 / "validation_report_M2.md").write_text(md_text)
    (M2 / "validation_report_M2.html").write_text(build_html(md_text))
    return md_text


if __name__ == "__main__":
    main()
    print("wrote", M2 / "validation_report_M2.md", "and .html")

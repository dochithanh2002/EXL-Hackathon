# Monitoring Report: M2

_Generated 2026-11-13T16:41:56+00:00 by `agent/validators/m2_taiwan/monitor.py`. Champion `output/m2_taiwan/model/m2_taiwan_pd.py` (sha256 `9d9c4e58426b`). Baseline = development (train) scores; validated test AUC 0.7402._

**Re-validation required: YES**

| Metric | Amber / red | 2005-09 cohort | 2005-08 re-score |
|---|---|---|---|
| score_psi | 0.1 / 0.25 | 🟢 0.0016 | 🟢 0.0826 |
| max_csi | 0.1 / 0.25 | 🟢 0.0017 | 🔴 0.6175 |
| auc_drop | 0.03 / 0.05 | 🟢 0.0000 | ⚪ n/a |
| citl_pp | 1.0 / 2.0 | 🟢 0.2429 | ⚪ n/a |
| min_dir | 0.85 / 0.8 | 🟢 0.9069 | 🟢 0.9121 |
| max_group_cal_gap_pp | 1.5 / 2.0 | 🟠 1.5899 | ⚪ n/a |

| Overall | | 🟠 **AMBER** | 🔴 **RED** |

## Characteristic stability (CSI vs development)

| Input | 2005-09 cohort | 2005-08 re-score |
|---|---|---|
| limit_bal | 0.0011 | 0.0011 |
| pay_0 | 0.0003 | 0.6175 |
| pay_2 | 0.0001 | 0.0006 |
| pay_3 | 0.0000 | 0.0073 |
| pay_5 | 0.0000 | 0.0000 |
| pay_6 | 0.0005 | 0.0005 |
| pay_amt1 | 0.0013 | 0.0072 |
| pay_amt2 | 0.0017 | 0.0455 |
| sex | 0.0000 | 0.0000 |
| education | 0.0002 | 0.0002 |
| marriage | 0.0003 | 0.0003 |

## Batches

### 2005-09 cohort: AMBER

Simulated batch: held-out test clients scored as at Sep-2005, observed default outcome.

- max_group_cal_gap_pp = 1.5899 (AMBER; amber/red at 1.5/2.0)

### 2005-08 re-score: RED

Simulated batch: the same clients re-scored with inputs shifted one month back (Mar-2005 repayment status is unobserved and carried). Default outcome not yet observed.

- max_csi = 0.6175 (RED; amber/red at 0.1/0.25), driven by pay_0

## Notes

- DIR is computed at the development 80%-approval cut-off for groups with n ≥ 200.
- Performance metrics (AUC drop, calibration) need observed outcomes; batches without them are rated on stability and outcome fairness only.
- A red rating means re-validation is required. It does not diagnose the cause: the validator triages it (for example, validation finding F3 explains a pay_0 shift as a data-coding artefact).

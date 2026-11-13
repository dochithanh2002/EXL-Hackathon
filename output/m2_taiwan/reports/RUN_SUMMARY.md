# Run summary: M2 Taiwan Card-Default Logistic PD (Traditional Score)

_Generated 2026-11-13T16:41:58Z by `mrf summary m2_taiwan`. Input `input/m2_taiwan`, output `output/m2_taiwan`.
Thresholds: `agent/config/thresholds.yaml`. Full audit trail: `output/m2_taiwan/logs/audit.jsonl`._

| Stage | Result | Artefact |
|---|---|---|
| Document | AWAITING_HUMAN_APPROVAL | `output/m2_taiwan/reports/documentation/MODEL_CARD_M2.md` |
| Migrate | PASS in 1 iteration(s); AWAITING_HUMAN_APPROVAL | `output/m2_taiwan/model/m2_taiwan_pd.py` |
| Validate: parity re-check | PASS 14/14 | `output/m2_taiwan/reports/parity/parity_report_M2.md` |
| Validate: performance | overall AMBER (discrimination GREEN, calibration GREEN, stability AMBER); test AUC 0.7402 | `output/m2_taiwan/reports/validation_report_M2.html` |
| Validate: fairness | overall RED (statistical GREEN, legal RED); recommended: B_no_sex_marriage | `output/m2_taiwan/reports/fairness/fairness_results.json` |
| Monitor | re-validation required: True; 2005-09 cohort AMBER, 2005-08 re-score RED | `output/m2_taiwan/reports/monitoring/MONITORING_REPORT.md` |

**Waiting on a human:** model card, migrated module.

## Deterministic steps run by the Validation and Monitoring agents
| Command | Exit | Seconds |
|---|---|---|
| `Rscript agent/validators/m2_taiwan/seed_sensitivity.R input/m2_taiwan/data/taiwan_default.csv output/m2_taiwan/reports/validation` | 0 | 14.6 |
| `python -m agent.validators.m2_taiwan.parity` | 0 | 3.5 |
| `python -m agent.validators.m2_taiwan.validate` | 0 | 12.0 |
| `python -m agent.validators.m2_taiwan.fairness` | 0 | 9.9 |
| `python -m agent.validators.m2_taiwan.report` | 0 | 1.5 |
| `python -m agent.validators.m2_taiwan.monitor` | 0 | 3.0 |

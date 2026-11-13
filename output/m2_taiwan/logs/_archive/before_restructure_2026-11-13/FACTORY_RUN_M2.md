# Factory run: M2

_Started 2026-11-13T15:59:48Z, finished 2026-11-13T15:59:48Z. Command: `python -m src.factory legacy_models/m2_taiwan/migration.yaml`._

| Stage | Result | Artefact |
|---|---|---|
| 1 Discover | done | `reports/inventory/DISCOVERY.md` |
| 2 Document | APPROVED (verify iterations 1) | `runs/M2/doc/output/MODEL_CARD_M2.md` |
| 3 Migrate | PASS in 1 iteration(s); APPROVED | `runs/M2/output/m2_taiwan_pd.py` |
| 4a Parity re-check | PASS 14/14 | `reports/m2/parity/parity_report_M2.md` |
| 4b Validation | overall AMBER (discrimination GREEN, calibration GREEN, stability AMBER); test AUC 0.7402 | `reports/m2/validation_report_M2.html` |
| 4c Fairness | overall RED (statistical GREEN, legal RED); recommended: B_no_sex_marriage | `reports/m2/fairness/fairness_results.json` |
| 5 Monitor | re-validation required: True; 2005-09 cohort AMBER, 2005-08 re-score RED | `reports/m2/monitoring/MONITORING_REPORT.md` |

**Waiting on a human:** nothing.

## LLM usage
| Agent | Calls (live / cached) | Input tokens | Output tokens | Cost USD |
|---|---|---|---|---|
| Migration | 3 (0 / 3) | 0 | 0 | 0 |
| Documentation | 1 (0 / 1) | 0 | 0 | 0 |

Token counts are recorded only for live calls on the `anthropic` or `claude-cli` backend; cached replays cost nothing.

## Steps
| Command | Exit | Seconds |
|---|---|---|
| `python -m r2py_agent discover .` | 0 | 0.4 |
| `python -m r2py_agent document legacy_models/m2_taiwan/migration.yaml` | 0 | 0.3 |
| `python -m r2py_agent run legacy_models/m2_taiwan/migration.yaml` | 0 | 0.3 |
| `Rscript src/m2/seed_sensitivity.R` | 0 | 10.5 |
| `python -m src.m2.parity` | 0 | 2.1 |
| `python -m src.m2.validate` | 0 | 6.3 |
| `python -m src.m2.fairness` | 0 | 5.9 |
| `python -m src.m2.report` | 0 | 1.2 |
| `python -m src.m2.monitor` | 0 | 2.6 |

# Factory run: M1

_Started 2026-11-13T15:56:14Z, finished 2026-11-13T15:58:47Z. Command: `python -m src.factory legacy_models/m1_fremtpl2/migration.yaml`._

| Stage | Result | Artefact |
|---|---|---|
| 1 Discover | done | `reports/inventory/DISCOVERY.md` |
| 2 Document | APPROVED (verify iterations 1) | `runs/M1/doc/output/MODEL_CARD_M1.md` |
| 3 Migrate | PASS in 2 iteration(s); APPROVED | `runs/M1/output/m1_fremtpl2_freq.py` |
| 4a Parity re-check | PASS 11/11 | `reports/m1/parity/parity_report_M1.md` |
| 4b Validation | overall AMBER (predictive_power GREEN, calibration AMBER, stability AMBER, specification AMBER); test Poisson deviance loss x100 30.9109 | `reports/m1/validation_report_M1.html` |
| 4c Fairness | overall AMBER (statistical AMBER, legal AMBER); recommended: keep driver age with documented actuarial justification; monitor A/E by age band | `reports/m1/fairness/fairness_results.json` |
| 5 Monitor | re-validation required: True; test cohort GREEN, young-driver growth RED | `reports/m1/monitoring/MONITORING_REPORT.md` |

**Waiting on a human:** nothing.

## LLM usage
| Agent | Calls (live / cached) | Input tokens | Output tokens | Cost USD |
|---|---|---|---|---|
| Migration | 4 (4 / 0) | 0 | 0 | 0 |
| Documentation | 1 (1 / 0) | 0 | 0 | 0 |

Token counts are recorded only for live calls on the `anthropic` or `claude-cli` backend; cached replays cost nothing.

## Steps
| Command | Exit | Seconds |
|---|---|---|
| `python -m r2py_agent discover .` | 0 | 0.6 |
| `python -m r2py_agent document legacy_models/m1_fremtpl2/migration.yaml --backend manual` | 0 | 0.3 |
| `python -m r2py_agent run legacy_models/m1_fremtpl2/migration.yaml --backend manual` | 0 | 0.3 |
| `python -m src.m1.parity` | 0 | 19.2 |
| `python -m src.m1.validate` | 0 | 120.9 |
| `python -m src.m1.report` | 0 | 1.1 |
| `python -m src.m1.monitor` | 0 | 12.1 |

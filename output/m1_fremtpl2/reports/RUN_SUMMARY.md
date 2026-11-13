# Run summary: M1 French Motor TPL Claim Frequency (Poisson GLM1)

_Generated 2026-11-13T16:52:45Z by `mrf summary m1_fremtpl2`. Input `input/m1_fremtpl2`, output `output/m1_fremtpl2`.
Thresholds: `agent/config/thresholds.yaml`. Full audit trail: `output/m1_fremtpl2/logs/audit.jsonl`._

| Stage | Result | Artefact |
|---|---|---|
| Document | AWAITING_HUMAN_APPROVAL | `output/m1_fremtpl2/reports/documentation/MODEL_CARD_M1.md` |
| Migrate | PASS in 1 iteration(s); AWAITING_HUMAN_APPROVAL | `output/m1_fremtpl2/model/m1_fremtpl2_freq.py` |
| Validate: parity re-check | PASS 11/11 | `output/m1_fremtpl2/reports/parity/parity_report_M1.md` |
| Validate: performance | overall AMBER (predictive_power GREEN, calibration AMBER, stability AMBER, specification AMBER); test Poisson deviance loss x100 30.9109 | `output/m1_fremtpl2/reports/validation_report_M1.html` |
| Validate: fairness | overall AMBER (statistical AMBER, legal AMBER); recommended: keep driver age with documented actuarial justification; monitor A/E by age band | `output/m1_fremtpl2/reports/fairness/fairness_results.json` |
| Monitor | re-validation required: True; test cohort GREEN, young-driver growth RED | `output/m1_fremtpl2/reports/monitoring/MONITORING_REPORT.md` |

**Waiting on a human:** model card, migrated module.

## Deterministic steps run by the Validation and Monitoring agents
| Command | Exit | Seconds |
|---|---|---|
| `python -m agent.validators.m1_fremtpl2.parity` | 0 | 17.7 |
| `python -m agent.validators.m1_fremtpl2.validate` | 0 | 87.8 |
| `python -m agent.validators.m1_fremtpl2.report` | 0 | 0.9 |
| `python -m agent.validators.m1_fremtpl2.monitor` | 0 | 9.6 |

# Intake brief: M2 Taiwan Card-Default Logistic PD (Traditional Score)

_Generated 2026-11-13T16:30:31Z by `mrf intake`. Deterministic; no LLM involved._

| | |
|---|---|
| Legacy script | `input/m2_taiwan/model.R` (R) |
| R model | `default.payment.next.month ~ limit_bal + sex + education + marriage +      pay_0 + pay_2 + pay_3 + pay_5 + pay_6 + pay_amt1 + pay_amt2`; binomial/logit; 15 coefficients; converged True |
| Train / test rows | 21001 / 8999 (split frozen in `output/m2_taiwan/model/reference/split.csv`) |
| Data | `input/m2_taiwan/data/taiwan_default.csv`: 30000 rows, schema in `output/m2_taiwan/logs/intake/scan.json` (never read data rows, G-11) |
| Protected attributes | SEX, MARRIAGE, AGE, EDUCATION |
| Allowed Python packages | numpy, pandas, statsmodels, scipy |
| Target module | `m2_taiwan_pd.py` |

**Triggered MUST rules** (the plan must cite every one, G-13): GOV-01, R-EST-01, R-EST-02, R-FAC-01, R-FAC-02, R-FAC-03, R-FAC-04, R-IDX-01, R-IDX-02, R-IO-01, R-IO-02, R-IO-03, R-PKG-01, R-PRED-01, R-REC-01, R-REC-02, R-RNG-01

**Files**
- R capture (facts): `output/m2_taiwan/model/reference/meta.json`, `coefficients.csv`
- Rulebook: `agent/config/rulebook.yaml` · thresholds: `agent/config/thresholds.yaml` (read-only for agents)
- Migration agent writes: `output/m2_taiwan/logs/migration/analysis.json`, `output/m2_taiwan/logs/migration/plan.json`, `output/m2_taiwan/logs/migration/work/candidate.py`
- Documentation agent writes: `output/m2_taiwan/logs/documentation/extract.json`

# Intake brief: M1 French Motor TPL Claim Frequency (Poisson GLM1)

_Generated 2026-11-13T16:47:05Z by `mrf intake`. Deterministic; no LLM involved._

| | |
|---|---|
| Legacy script | `input/m1_fremtpl2/model.R` (R) |
| R model | `ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM +      VehBrand + VehGas + DensityGLM + Region + AreaGLM`; poisson/log; 49 coefficients; converged True |
| Train / test rows | 610212 / 67801 (split frozen in `output/m1_fremtpl2/model/reference/split.csv`) |
| Data | `input/m1_fremtpl2/data/freMTPL2freq.csv`: 678013 rows, schema in `output/m1_fremtpl2/logs/intake/scan.json` (never read data rows, G-11) |
| Protected attributes | DrivAge |
| Allowed Python packages | numpy, pandas, statsmodels, scipy |
| Target module | `m1_fremtpl2_freq.py` |

**Triggered MUST rules** (the plan must cite every one, G-13): GOV-01, R-EST-02, R-EST-03, R-FAC-01, R-FAC-02, R-FAC-04, R-IO-01, R-IO-03, R-NA-01, R-PRED-01, R-REC-02, R-RNG-01

**Files**
- R capture (facts): `output/m1_fremtpl2/model/reference/meta.json`, `coefficients.csv`
- Rulebook: `agent/config/rulebook.yaml` · thresholds: `agent/config/thresholds.yaml` (read-only for agents)
- Migration agent writes: `output/m1_fremtpl2/logs/migration/analysis.json`, `output/m1_fremtpl2/logs/migration/plan.json`, `output/m1_fremtpl2/logs/migration/work/candidate.py`
- Documentation agent writes: `output/m1_fremtpl2/logs/documentation/extract.json`

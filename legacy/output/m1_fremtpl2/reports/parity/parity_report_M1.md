# M1 Parity Re-check: R `glm` (Poisson, offset) → agent-generated Python module

_Generated 2026-11-13T16:50:05+00:00 by `agent/validators/m1_fremtpl2/parity.py`_

**Result: ✅ PASS** (11/11). Module `output/m1_fremtpl2/model/m1_fremtpl2_freq.py` sha256 `fda45c0fbca2682b`; the agent reached parity in 1 iterations.

## Validator checks

| Check | Observed | Tolerance | Pass |
|---|---|---|---|
| artefact: module SHA-256 differs from agent manifest | 0 | 0 | ✅ |
| term names/order differ from R model.matrix | 0 | 0 | ✅ |
| split: rows with different learn/test membership | 0 | 0 | ✅ |
| response: rows with different ClaimNb | 0 | 0 | ✅ |
| coef max abs diff | 7.283e-14 | 1.000e-08 | ✅ |
| coef max rel diff | 9.870e-13 | 1.000e-06 | ✅ |
| std error max rel diff | 7.242e-14 | 1.000e-06 | ✅ |
| expected claims max abs diff (all 678,013 rows) | 6.195e-14 | 1.000e-10 | ✅ |
| residual deviance abs diff | 1.327e-08 | 1.000e-06 | ✅ |
| AIC-equivalent: deviance + 2k abs diff | 1.327e-08 | 1.000e-06 | ✅ |
| test Poisson loss abs diff | 1.066e-14 | 1.000e-10 | ✅ |

## Independent cross-check (statsmodels, fully converged)

| Check | Observed | Tolerance | Pass |
|---|---|---|---|
| statsmodels coef max rel diff vs R | 7.009e-08 | 1.000e-04 | ✅ |
| statsmodels expected claims max abs diff vs R | 7.079e-09 | 1.000e-06 | ✅ |

## Agent-reported parity

| Check | Observed | Tolerance | Pass |
|---|---|---|---|
| G-05 coefficient names: missing + extra | 0 | 0 | ✅ |
| coefficient max rel diff | 9.870e-13 | 1.000e-06 | ✅ |
| std error max rel diff | 7.242e-14 | 1.000e-06 | ✅ |
| rows: identical set | 0 | 0 | ✅ |
| response values: rows differing | 0 | 0 | ✅ |
| split membership: rows differing | 0 | 0 | ✅ |
| prediction NA pattern: rows differing | 0 | 0 | ✅ |
| prediction max abs diff (rows predicted by both) | 6.195e-14 | 1.000e-08 | ✅ |
| poisson deviance abs diff | 0 | 1.000e-06 | ✅ |

## Coefficients

| Term | R | Python | rel diff |
|---|---|---|---|
| (Intercept) | -3.98532768849 | -3.98532768849 | 1.83e-14 |
| VehPowerGLM5 | 0.181808701385 | 0.181808701385 | 3.83e-14 |
| VehPowerGLM6 | 0.224300771427 | 0.224300771427 | 1.08e-14 |
| VehPowerGLM7 | 0.130252320271 | 0.130252320271 | 6.82e-15 |
| VehPowerGLM8 | -0.0790582223747 | -0.0790582223747 | 1.21e-13 |
| VehPowerGLM9 | 0.209341020689 | 0.209341020689 | 1.86e-14 |
| VehAgeGLM1 | 1.18037448858 | 1.18037448858 | 1.43e-14 |
| VehAgeGLM3 | -0.20391719745 | -0.20391719745 | 5.43e-14 |
| DrivAgeGLM1 | 0.0722855541888 | 0.0722855541887 | 7.04e-13 |
| DrivAgeGLM2 | -0.298615203106 | -0.298615203106 | 4.48e-14 |
| DrivAgeGLM3 | -0.431093071145 | -0.431093071145 | 4.52e-14 |
| DrivAgeGLM4 | -0.294931028351 | -0.294931028351 | 3.65e-14 |
| DrivAgeGLM6 | -0.0662845761121 | -0.0662845761121 | 1.84e-14 |
| DrivAgeGLM7 | 0.00493223822783 | 0.00493223822783 | 4.29e-13 |
| BonusMalusGLM | 0.0228868108614 | 0.0228868108614 | 2.47e-14 |
| VehBrandB10 | 0.0150923826993 | 0.0150923826993 | 7.77e-14 |
| VehBrandB11 | 0.0901053181143 | 0.0901053181143 | 1.62e-14 |
| VehBrandB12 | 0.123229166282 | 0.123229166282 | 5.92e-14 |
| VehBrandB13 | 0.0331937757491 | 0.0331937757491 | 7.94e-14 |
| VehBrandB14 | -0.113572121088 | -0.113572121088 | 1.45e-14 |
| VehBrandB2 | 0.0031070095067 | 0.00310700950669 | 3.18e-13 |
| VehBrandB3 | 0.0120613963556 | 0.0120613963556 | 1.80e-13 |
| VehBrandB4 | 0.00438223738989 | 0.00438223738989 | 2.31e-13 |
| VehBrandB5 | 0.0843957377331 | 0.0843957377331 | 6.33e-14 |
| VehBrandB6 | -0.0054390535014 | -0.0054390535014 | 2.64e-13 |
| VehGasRegular | 0.0712017459368 | 0.0712017459368 | 3.10e-14 |
| DensityGLM | 0.0324274952279 | 0.0324274952279 | 3.04e-13 |
| RegionR11 | -0.139026326185 | -0.139026326185 | 1.28e-14 |
| RegionR21 | 0.0541130636902 | 0.0541130636902 | 2.01e-14 |
| RegionR22 | -0.0141131971502 | -0.0141131971502 | 4.14e-14 |
| RegionR23 | -0.127511964908 | -0.127511964908 | 2.63e-14 |
| RegionR25 | -0.0462113635586 | -0.0462113635586 | 6.16e-15 |
| RegionR26 | -0.0612420674399 | -0.0612420674399 | 4.19e-14 |
| RegionR31 | -0.181097460546 | -0.181097460546 | 4.02e-14 |
| RegionR41 | -0.307319576925 | -0.307319576925 | 4.15e-15 |
| RegionR42 | -0.065068426724 | -0.065068426724 | 1.24e-14 |
| RegionR43 | -0.178732751275 | -0.178732751275 | 4.81e-15 |
| RegionR52 | -0.081423176997 | -0.081423176997 | 5.28e-15 |
| RegionR53 | 0.0276429477085 | 0.0276429477085 | 3.39e-15 |
| RegionR54 | -0.116383284195 | -0.116383284195 | 4.81e-14 |
| RegionR72 | -0.161388958481 | -0.161388958481 | 4.47e-15 |
| RegionR73 | -0.203729054913 | -0.203729054913 | 9.40e-15 |
| RegionR74 | 0.102224486318 | 0.102224486318 | 2.17e-15 |
| RegionR82 | 0.00921726670615 | 0.00921726670615 | 3.89e-13 |
| RegionR83 | -0.375441382828 | -0.375441382828 | 5.86e-14 |
| RegionR91 | -0.119592122245 | -0.119592122245 | 3.25e-15 |
| RegionR93 | -0.0929945126586 | -0.0929945126586 | 3.10e-14 |
| RegionR94 | 0.0169190150448 | 0.0169190150448 | 3.65e-13 |
| AreaGLM | 0.0164654726225 | 0.0164654726226 | 9.87e-13 |

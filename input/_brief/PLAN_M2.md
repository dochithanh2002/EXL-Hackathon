# M2 – Taiwan Card-Default Logistic Regression: Work Plan

_Drafted 2026-11-13. Deadline: 14 Nov 2026, 23:59 Irish time._

| Field | Value |
|---|---|
| ID | M2 |
| Model | Taiwan card-default logistic regression (PD, next-month default) |
| Domain / source language | Credit / R |
| Reference code | Kaggle `netzone/default-credit-and-credit-score-prediction` |
| Data | UCI #350 "Default of Credit Card Clients" (same as Kaggle `uciml/default-of-credit-card-clients-dataset`), 30,000 rows, 23 features |
| Role | Document it, write a **fairness report by sex, age and education**, and keep it as the **backup model to rebuild** if M1 parity fails |

---

## 0. Environment (done)

- `.venv/` (Python 3.13) has pandas, scikit-learn, statsmodels, fairlearn, mlflow 3.18, azure-ai-ml, azureml-mlflow and kaggle. Pinned versions are in `requirements.lock.txt`.
- R is installed at `/usr/local/bin/Rscript` and runs the original model locally to produce reference outputs.
- MLflow is set up in `src/common/tracking.py`:
  - when `config/azure.env` has `AZURE_SUBSCRIPTION_ID`, runs log to the **Azure ML workspace**;
  - otherwise they log to local `sqlite:///mlflow.db`. Smoke test passed for experiment `m2-taiwan-default`.
- Azure: `scripts_setup_azure.sh` creates the resource group and the ML workspace (uksouth).
- Data is in `data/taiwan_default/` (raw `.xls` plus `taiwan_default.csv`).

**Waiting on the user:**
1. `az login`, then copy `config/azure.env.example` to `config/azure.env` and fill in the subscription ID. Then run `./scripts_setup_azure.sh`.
2. A Kaggle API token (`~/.kaggle/kaggle.json`), needed to `kaggle kernels pull netzone/default-credit-and-credit-score-prediction -p ref_code/taiwan_default`. Without it, download the notebook by hand from the Kaggle page.

## 1. Data facts (profiled)

- Default rate is **22.1%**. There are no nulls.
- **Sex:** male (1) n=11,888, default rate 24.2%; female (2) n=18,112, default rate 20.8%.
- **Education:** 1 grad school (19.2%), 2 university (23.7%), 3 high school (25.2%). Codes 0, 4, 5 and 6 (n=468) are undocumented, and their default rates are very low (≈6%). Group them as "other/unknown" and record this as a data-quality finding.
- **Age bands:** ≤25 26.7%, 26–35 19.8%, 36–45 21.8%, 46–55 24.9%, 55+ 26.5%. The curve is U-shaped.
- **Marriage:** code 0 is undocumented (n=54). In `PAY_*`, codes -2 and 0 are undocumented. Both go in the data-quality log.

## 2. Discover & Document (inventory record)

Read the netzone R notebook and extract:
- the preprocessing (recodes, factor encoding, train/test split and seed), the formula, and whether protected attributes (SEX, AGE, EDUCATION, MARRIAGE) are **model inputs**;
- the reported metrics, which are the baseline we need to reproduce.

Deliverables:
- `reports/m2/inventory_M2.json` with the fields owner, purpose, source language, inputs, target, algorithm, data lineage, intended use, limitations, PRA SS1/23 tier, last validation date and status.
- `reports/m2/model_card_M2.md`.

## 3. Reproduce in R, then rebuild in Python (backup parity)

1. `src/m2/run_reference.R` runs the notebook's `glm(..., family=binomial)` on a fixed split and writes the coefficients, the test-set predictions and the AUC to `reports/m2/r_reference/`.
2. `src/m2/rebuild.py` refits with `statsmodels.Logit`, which is unpenalised like R `glm`, on the same split by row ID. It uses identical factor encoding and reference levels.
3. `src/m2/parity.py` passes when coefficients agree within 1e-6 absolute and predictions within 1e-8. It also compares AUC and Gini. Every result is logged to MLflow.

The time box is about 2 hours. If M1 parity already passes, M2 parity is only a supporting extra.

## 4. Validate (performance, calibration, stability)

`src/m2/validate.py` logs all of the following to MLflow:
- **Discrimination:** AUC, Gini and KS on train and test, with 1,000 bootstrap resamples for confidence intervals.
- **Calibration:** a decile reliability table, the Hosmer–Lemeshow test, and the Brier score.
- **Stability:** PSI between train and test score distributions, and CSI per feature. The data has no date column, so use the `PAY_0` → `PAY_6` repayment-history months as a pseudo out-of-time test. Concretely, rebuild the features shifted one month back, score them, and compare.
- **Back-test:** a predicted vs actual default rate per score band.

## 5. Fairness report (the core M2 deliverable)

`src/m2/fairness.py` uses `fairlearn.MetricFrame`. The groups are sex, age band and education (1/2/3/other). Marriage is an optional extra.

| Metric | Why it matters |
|---|---|
| Selection (approval) rate and the **disparate impact ratio** (4/5ths rule) | Approval disparity |
| Statistical parity difference | Same idea, as a difference |
| TPR and FPR gaps (**equalised odds**) | Error-rate parity |
| AUC per group | Whether the model ranks equally well across groups |
| **Calibration within group** (predicted vs observed PD) | The key test for credit. Unequal default base rates make equal approval rates and equal calibration incompatible, so the report has to discuss this trade-off. |

- Use a fixed threshold of PD 0.5, plus a cut-off set so the approval rate matches policy (for example 80%).
- **Ablation:** refit without SEX, AGE and EDUCATION and compare performance and the fairness gaps. This shows whether removing the attributes achieves "fairness through unawareness" or whether proxies (`LIMIT_BAL`, `PAY_*`) carry the signal.
- Optional mitigation: `fairlearn` ThresholdOptimizer or ExponentiatedGradient, showing the AUC cost against the reduction in the gaps.
- Regulatory framing: UK Equality Act 2010 protected characteristics (sex and age; education is a proxy risk), FCA Consumer Duty, and PRA SS1/23 Principle 3 (model development) and Principle 4 (independent validation).

Output: `reports/m2/fairness_report_M2.md` (and HTML), with charts.

## 6. Monitor

`src/m2/monitor.py` handles the re-validation triggers: PSI > 0.25, AUC falling more than 5 points from baseline, a disparate impact ratio below 0.8 for any group, or a calibration gap above 2 percentage points. It writes the results to `reports/m2/monitoring_M2.json` and the MLflow run tags.

## 7. Agent hooks (GenAI track)

The documentation and validation narrative should be **agent-generated** from MLflow artefacts and the inventory JSON. The same pipeline runs M1, M2 and M3, so keep every script parameterised by model ID.

## Order of work and time

| # | Step | Time |
|---|---|---|
| 1 | Get the R notebook (Kaggle token) and read it | 0.5 h |
| 2 | Reproduce in R and write the inventory record | 1 h |
| 3 | Python rebuild and parity | 1.5 h |
| 4 | Validation metrics | 1.5 h |
| 5 | Fairness report and ablation | 2 h |
| 6 | Monitoring, Azure ML logging, report polish | 1 h |

## Layout

```
data/taiwan_default/          raw + csv
ref_code/taiwan_default/      netzone R notebook
src/common/tracking.py        MLflow (Azure ML or local)
src/m2/                       run_reference.R, rebuild.py, parity.py, validate.py, fairness.py, monitor.py
reports/m2/                   inventory, model card, validation + fairness reports
config/azure.env              (gitignored) Azure IDs
```

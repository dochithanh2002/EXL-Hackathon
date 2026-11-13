# Challenge 4 – Agentic Model Risk Factory: Dataset & Model Candidates

_Compiled 2026-11-10 from Kaggle and Hugging Face searches, plus GitHub for SAS code._

> **Verification update (2026-11-10):** The SAS repos listed for HMEQ do **not** use HMEQ. Kaggle `uciml/german-credit` has **no target column**. `sklearn PoissonRegressor` is penalised by default, so use `statsmodels`. See `PROJECT_PLAN.md` §2 for the corrected sources and final scope.

## Challenge summary (from EXL_Trinity_Data_AI_Hackathon_2026.pptx, slide 9)

**Client context:** A Tier-1 UK retail bank has about 250 credit, fraud and pricing models in production, built in SAS, R and Python. Under **PRA SS1/23** each one needs current documentation and independent validation. Today the documents are out of date and each validation takes 8–12 weeks.

**Starter data & tools:** Public credit-risk datasets (e.g. Kaggle) and EXL sample model code. Azure ML and MLflow are preferred.

**What to build:**
1. Document **3–5 sample models** and build an inventory record for each.
2. Use AI agents to **rebuild one SAS or R model in Python, with proof of parity**.
3. Auto-generate a **validation report** on performance, stability and fairness.

**Suggested approach:** Discover → Document → Redevelop → Validate (back-test, calibration, stability, fairness) → Monitor (drift, re-validation triggers)

**Tracks:** GenAI & Agents, Predictive Analytics, Responsible AI

**Judging:** Business Impact 30%, Technical 25%, Innovation 20%, Responsible AI 15%, Presentation 10%.

**Submission:** Due 14 Nov 2026, 23:59 Irish time. Only public or synthetic data may be used.

---

## Candidate datasets

| # | Domain | Dataset | Existing R / SAS model code | Fairness attributes | Notes |
|---|---|---|---|---|---|
| 1 | **Credit (probability of default)** | Kaggle `uciml/german-credit` (1k rows). HF mirror: `AiresPucrs/german-credit-data` | R notebooks: `quinn126/predicting-german-credit-risk-logit-reg-xgboost`, `ewelinak/credit-risk-modeling-with-logistic-regression`, `piyushbhatt16/predicting-the-credit-risk-using-logit-regression` | ✅ age, sex/marital status, foreign worker | The standard fairness benchmark. Small and fast to iterate on. |
| 2 | **Credit (SAS-native)** | Kaggle `ajay1735/hmeq-data` (HMEQ, 5.9k rows) | HMEQ is SAS's own sample credit-scoring dataset. GitHub SAS code: `Ikram-Awan/-SAS-Credit-Risk-Scorecard` (WoE/IV and logistic regression), `arielseidman/SAS-Credit-Risk-Engine`, `anilgithub22/SAS_Credit_Risk`. R notebook: `chongchong33/predicting-customer-risk-by-home-equity-loan` | ⚠️ job type only | **Best fit for the "SAS model" story.** |
| 3 | **Pricing (motor insurance)** | Kaggle `floser/french-motor-claims-datasets-fremtpl2freq` (678k policies) and `floser/fremtpl2sev`. HF mirror: `mabilton/fremtpl2` | R notebooks: `floser/glm-neural-nets-and-xgboost-for-insurance-pricing` (Poisson frequency GLM), `floser/glm-nn-embeddings-and-gbm-for-claim-frequency`. Python reference: GitHub `lorentzenchr/Tutorial_freMTPL2` | ⚠️ driver age, region | The industry-standard pricing GLM. Large enough for stability and drift tests. |
| 4 | **Fraud** | Kaggle `mlg-ulb/creditcardfraud` (284k transactions, 0.17% fraud) | R notebooks: `gpreda/credit-card-fraud-detection-with-rf-auc-0-93`, `chrisbow/credit-card-fraud-prediction-in-r`, `atharvaingle/credit-card-fraud-detection-with-r-sampling`, `nschneider/gbm-vs-xgboost-vs-lightgbm` | ❌ features are anonymised (PCA) | Useful for validating extreme class imbalance. Has a `Time` column for out-of-time tests. |
| 5 | **Credit (scorecard)** | Kaggle competition `GiveMeSomeCredit` (150k rows) | R notebooks: `abhishekgiri/give-me-some-credit-logistic-regression`, `brynning/top-1-give-me-some-credit-lgbm-model-in-r` | ✅ age | Large enough to calculate a population stability index (PSI) across age or income bands. |
| 6 | Credit (card default) | Kaggle `uciml/default-of-credit-card-clients-dataset` (Taiwan, 30k rows) | R notebooks: `netzone/default-credit-and-credit-score-prediction`, `sourabh24293/randomforest-for-default-of-credit-card-clients` | ✅ sex, age, education, marital status | Strong second choice for fairness. |
| 7 | Credit (large) | Kaggle competition `home-credit-default-risk` | R notebook: `kailex/tidy-xgb-all-tables-0-796` (573 votes) | ✅ gender, age | Probably too large for the remaining time. |

### Other options (lower priority)
- `laotse/credit-risk-dataset`: credit risk, with a few R notebooks.
- `wordsforthewise/lending-club`: 1.3 GB, and the R notebooks are mostly EDA.
- `kartik2112/fraud-detection`: synthetic card fraud with gender and age. There is an R notebook: `shirsamukherjee/credit-card-fraud-detection-project-using-r`.
- Competition `ieee-fraud-detection`: R LightGBM notebooks (`psystat/ieee-extensive-eda-lgb-with-r`, `duykhanh99/lightgbm-starter-with-r-0-9493-lb`).
- Competition `porto-seguro-safe-driver-prediction`: insurance, with R notebooks (`captcalculator/logistic-regression-and-roc-curve-primer`).
- `mirichoi0218/insurance`: a simple medical-premium pricing dataset with R linear-regression notebooks.

Hugging Face mostly mirrors the Kaggle datasets and adds no SAS or R model code.

---

## Recommended plan

### Model inventory (5 models)
| ID | Model | Domain | Source language |
|---|---|---|---|
| M1 | German Credit logistic PD | Credit | R |
| M2 | HMEQ WoE scorecard | Credit | SAS |
| M3 | freMTPL2 Poisson frequency GLM | Pricing | R |
| M4 | ULB credit-card fraud RF/GBM | Fraud | R |
| M5 | Taiwan card-default model | Credit | R |

### Model to rebuild in Python (pick one)
- **Option A: freMTPL2 Poisson GLM (R `glm`) → Python `statsmodels` / `sklearn PoissonRegressor`.** GLMs fit deterministically, so you can show coefficients and predictions match to about 1e-6. That makes a very strong parity proof, and pricing is a less common choice among teams.
- **Option B: HMEQ WoE + logistic scorecard (SAS) → Python.** This is closest to the bank's real legacy-SAS situation. Students can run real SAS for free with **SAS OnDemand for Academics** to produce reference outputs.

### Fairness section of the validation report
Use German Credit (M1) or Taiwan Default (M5), because both have sex and age attributes.

---

## Commands

```bash
# Datasets
kaggle datasets download -d uciml/german-credit -p data/german_credit --unzip
kaggle datasets download -d ajay1735/hmeq-data -p data/hmeq --unzip
kaggle datasets download -d floser/french-motor-claims-datasets-fremtpl2freq -p data/fremtpl2 --unzip
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ccfraud --unzip
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset -p data/taiwan_default --unzip

# Reference R code (Kaggle notebooks)
kaggle kernels pull floser/glm-neural-nets-and-xgboost-for-insurance-pricing -p ref_code/fremtpl2
kaggle kernels pull ewelinak/credit-risk-modeling-with-logistic-regression -p ref_code/german_credit
kaggle kernels pull chongchong33/predicting-customer-risk-by-home-equity-loan -p ref_code/hmeq
kaggle kernels pull chrisbow/credit-card-fraud-prediction-in-r -p ref_code/ccfraud
kaggle kernels pull netzone/default-credit-and-credit-score-prediction -p ref_code/taiwan_default

# Reference SAS code (GitHub)
gh repo clone Ikram-Awan/-SAS-Credit-Risk-Scorecard ref_code/hmeq_sas_scorecard
gh repo clone arielseidman/SAS-Credit-Risk-Engine ref_code/sas_credit_engine
```

> Note: I found these notebooks and repos through search but haven't downloaded or run any of them. Check that each one runs and has a suitable licence before you use it.

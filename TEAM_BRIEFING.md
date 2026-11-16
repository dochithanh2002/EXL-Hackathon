# R-Pythonizer — Team Briefing

A plain-language guide to the whole project: what problem we solve, how the platform works, how we prove the
Python model is correct, and what the judges are likely to ask. Written so that someone with no coding, modelling
or risk-management background can read it once and explain it to someone else.

Facts come from `PLAN.md`, `README.md`, `platform/` and `legacy/`. Anything that is an estimate or an assumption
is marked **[estimate]** or **[to confirm]**. Do not present those as measured results.

---

## Contents

0. [The project in one minute](#0-the-project-in-one-minute)
1. [Context: legacy models in the risk industry](#1-context-legacy-models-in-the-risk-industry)
2. [Our approach: why R, and which models](#2-our-approach-why-r-and-which-models)
3. [How we prove the translation is correct](#3-how-we-prove-the-translation-is-correct) ← the most important section
4. [How the workflow runs, and how edge cases are covered](#4-how-the-workflow-runs-and-how-edge-cases-are-covered)
5. [How to use the platform](#5-how-to-use-the-platform)
6. [Business value](#6-business-value)
7. [Architecture and AI safety design](#7-architecture-and-ai-safety-design)
8. [Results so far](#8-results-so-far)
9. [Limitations: be honest about these](#9-limitations-be-honest-about-these)
10. [Roadmap](#10-roadmap)
11. [Likely judge questions, with answers](#11-likely-judge-questions-with-answers)
12. [Suggested presentation outline](#12-suggested-presentation-outline)
13. [Glossary](#13-glossary)
14. [Numbers cheat sheet](#14-numbers-cheat-sheet)

---

## 0. The project in one minute

Banks and insurers run many old statistical models written in the programming language **R**. Companies want to
move them to **Python**, which is now the industry standard. Moving a model by hand is slow, and the biggest risk
is that the new Python version gives **slightly different answers** without anyone noticing. In a bank, a
different answer can mean a different loan decision or a different insurance price.

**R-Pythonizer** is a platform where:

1. A data scientist uploads the dataset and the R code.
2. AI agents read the R code, work out which model it builds, and write the Python version.
3. **Plain code, not AI,** runs both versions on the same data and checks that every number matches, typically to
   about 12 decimal places.
4. The platform also checks whether the model is fit for business use: accuracy, stability, fairness and so on.
5. A human approves. The platform then produces a validation report and a package anyone can re-run to check
   the result.

**The key idea: AI does the writing, deterministic code does the judging, and a human signs off.**

---

## 1. Context: legacy models in the risk industry

### 1.1 What is a "model" here?

A model is a formula learned from historical data that makes a prediction. Examples:

| Business question | What the model predicts | Typical model type |
|---|---|---|
| Will this customer default on their loan? | A probability (e.g. 3%) | Logistic regression (a "binomial GLM") |
| How many claims will this car policy have next year? | A count rate (e.g. 0.07 claims/year) | Poisson GLM |
| How much will a claim cost? | An amount (e.g. €2,400) | Gamma GLM |
| How much money will we lose if they default? | A percentage or amount | Linear regression, GLM |
| What will default rates be next quarter in a recession? | A time series | ARIMA, regression |

Banks use these models to approve loans, set interest rates, price insurance, set aside capital (the money a bank
must keep in reserve) and run regulatory stress tests.

### 1.2 Why so many of them are written in R

For roughly 15–20 years, R was the main language of statisticians, actuaries and credit-risk teams. Many models
still running today were built in R, often by people who have since moved on. These are the **legacy models**.

### 1.3 Why companies want to move them to Python

- **Talent**: most new data scientists and engineers know Python, not R.
- **One stack**: production systems, data platforms and machine-learning tools are mostly Python.
- **Maintainability**: R scripts written years ago are often long, undocumented and hard to change.
- **Cost**: keeping two languages means two sets of skills, tools, servers and security reviews.

### 1.4 Why moving them is risky

The same calculation can give different numbers in R and Python, even when the code "looks" the same:

- **Different defaults.** R's `glm` fits a plain logistic regression. Python's popular scikit-learn
  `LogisticRegression` quietly adds a penalty (regularisation) unless you switch it off, so the coefficients
  never match.
- **Different data reading.** R's `read.csv` treats only the text `NA` as missing. Python's pandas also treats
  empty cells, `null`, `N/A`, `nan` and others as missing. This changes which rows get dropped, and so changes
  the model.
- **Different column names.** R silently renames columns that contain spaces or symbols
  (`default payment next month` becomes `default.payment.next.month`).
- **Different categories.** R and Python can choose a different "reference" category for a variable like
  region, which changes every coefficient.
- **Different algorithms inside.** Both fit a GLM with the same method (IRLS), but they start from different
  points and stop at different moments. The answer agrees to about 6 decimal places instead of 12.
- **Different random numbers.** R and Python generate different "random" train/test splits even with the same
  seed.

A tiny difference in a coefficient can change a credit score near a cut-off, and so flip a loan decision.

### 1.5 Regulation: why "probably the same" is not enough

Regulators require banks to manage **model risk**: the risk of losses from a model that is wrong or misused. In
the UK, the Prudential Regulation Authority's **SS1/23** (Supervisory Statement 1/23, model risk management
principles) expects banks to:

- keep an inventory of every model;
- document how each model works;
- validate models independently before use, and again after material changes;
- monitor models over time;
- have humans accountable for approving models.

**Rewriting a model in another language is a model change.** The bank must be able to show evidence that the
new version behaves like the old one. R-Pythonizer is built to produce that evidence automatically.

### 1.6 What real legacy R code looks like

Public R code and in-house scripts are rarely "one clean model". Typical patterns:

| Source | Typical content |
|---|---|
| Kaggle notebooks, blogs, tutorials | 3–5 different models compared (logistic, random forest, xgboost, neural net), plus lots of charts |
| Academic or course code | One model type, but several versions (`glm1`, `glm2`, `glm3`), with the last one chosen |
| Package documentation examples | Short, one model |
| In-house bank or insurer scripts | A main model plus a comparison model, reporting, exports; edited by many people over years |

Our four test models show this range:

| | Dataset | Business problem | R code | Models inside |
|---|---|---|---|---|
| **M1** | freMTPL2 (French motor insurance), 678,013 rows | Claim frequency | 1,126 lines | Poisson GLM, plus keras, xgboost and catboost |
| **M2** | Taiwan credit card default, 30,000 rows | Probability of default | medium | Binomial GLM (uses caret to split) |
| **M3** | ULB credit card fraud | Fraud detection | — | Random forest |
| **M4** | GiveMeSomeCredit, 150,000 rows | Probability of default | 57 lines | One binomial GLM |

---

## 2. Our approach: why R, and which models

### 2.1 Why R → Python

- It is the most common real-world migration in risk teams. R was the statisticians' language; Python is the
  engineering standard.
- R's statistical functions have well-defined, documented behaviour. That lets us check the Python version against
  it precisely.

### 2.2 Which models we support, and why

The platform supports **classical statistical models**: the kinds of models the Python library **statsmodels**
can reproduce. It does **not** support machine-learning models such as random forests, xgboost or neural networks,
and it never uses scikit-learn.

The support list (`platform/envs/support_matrix.yaml`) has three levels:

| Level | Meaning | Models |
|---|---|---|
| **Proven** | A test runs real R and real Python and they match to ~1e-12 | GLM binomial (default prediction), GLM Poisson with exposure offset (claim frequency) |
| **Expected exact, not yet proven** | Should match exactly; no test fixture yet | Linear regression (`lm`), other GLM families (Gamma, gaussian, quasi), negative binomial, multinomial logit, ordinal logit, quantile regression, GLS |
| **Approximate** | The two languages use different algorithms, so answers are close, not identical | Penalised regression (glmnet), GAM, mixed models (lme4/nlme), robust regression, Cox survival, ARIMA |

A model not on the list can still be attempted. The data scientist must explicitly accept the risk that parity
is unproven. If the AI finds no faithful Python equivalent (as with M3's random forest), it marks the model
**unsupported** and stops. It never silently swaps in a different model.

### 2.3 Why focus on statistical models? Not because risk only uses these

Banks and insurers do use machine learning (gradient boosting in particular), and that is growing. We chose this
scope deliberately:

1. **It is where most legacy R code is.** Credit scorecards are almost always logistic regression. Insurance
   pricing is classically a Poisson GLM for frequency times a Gamma GLM for severity. Loss and stress-test models
   are regressions and time series.
2. **Regulators favour explainable models.** Statistical models give a coefficient per variable ("each extra €1k
   of income lowers default odds by x%") with standard errors and significance tests. SS1/23 expects models to be
   understood and explained.
3. **We can prove the translation exactly.** These models are deterministic mathematics: the same data always
   gives the same answer, so we can demand agreement to 12 decimal places. Boosting and neural nets involve
   randomness, multi-threading and GPUs. Even two runs in the same language can differ, so exact proof is not
   possible.
4. **Depth over breadth.** A full, provable pipeline for the most common model family is more convincing than a
   shallow one for everything.

### 2.4 Why scikit-learn is excluded

`PLAN.md` records this as a decision without stating the reason. Our explanation: scikit-learn gives no standard
errors or statistical tests, which risk teams need. Its defaults also differ from R in ways that silently change
results (for example the default penalty in `LogisticRegression`). statsmodels is designed to mirror classical
statistics, and therefore R.

---

## 3. How we prove the translation is correct

> This is the section judges care about most. Learn the logic, not just the words.

### 3.1 What we prove, and what we don't

We do **not** prove that every R command is equivalent to every Python command. That is not possible in general:
the two languages compute things in different internal steps.

We **do** prove that **the final outputs are the same**: every coefficient, every standard error, every
prediction for every row, and the fit statistics and metrics. This is measured on the same data rows, within
tolerances a human agreed in advance, and checked by plain code rather than AI.

Analogy: two kitchens can use different stoves and different stirring techniques. We don't inspect the
technique. We taste the soup: same ingredients in, same dish out, measured precisely.

### 3.2 Step 1: freeze an "answer key" from R

**What `baseline.R` is**: the data scientist uploads a full R file, which often does many things (charts, three
different models, comparisons). The AI "Analyst" copies out **only the lines needed to build the one target
model**: read the data, clean it, split it, fit the model. That short script is `baseline.R`.

> Analogy: a chef's notebook has 100 pages (pho, bread, desserts, shopping notes). You need to recreate the pho
> exactly, so you copy only the pho recipe onto one sheet. The notebook is the original R file; the single sheet
> is `baseline.R`.

Why we need it:
- The platform's R environment does not include packages like keras or xgboost, so the full file would crash.
- Training neural nets can take hours; the target model takes seconds.
- The platform needs to know exactly which model, train set and test set to check.
- It must read the data from a known location so every row can be traced.

Even M4, which is already one clean model, needs a `baseline.R`: almost the whole file, with small changes
(read the data from `inputs/`, name the model object, and no test set because M4's test file has no answers).

The data scientist sees `baseline.R` side by side with the original, as a diff, before confirming.

The platform then runs `baseline.R` in real R and records the coefficients, standard errors, predictions for
every row, fit statistics, and **the exact row numbers that went into the training and test sets**. This becomes
the **frozen baseline**, the answer key. It cannot be changed afterwards.

### 3.3 Step 2: remove every difference that isn't the translation

Before comparing, we eliminate everything except the translation itself:

| Possible source of difference | How we remove it |
|---|---|
| Random train/test split | Python never re-splits. It receives R's exact row numbers and must return exactly those rows. |
| Different data file | Every uploaded file is fingerprinted (SHA-256 hash) and checked before each run. |
| Different package versions | One pinned R environment and one pinned Python environment, shared by all projects. Every run records an environment hash. |
| AI copying R's answers | The Translator AI never sees R's coefficient values, only their names. |
| AI "cheating" with code tricks | Static guardrails block forbidden imports, network, file writing, shell commands, randomness, `eval`, scikit-learn. |

### 3.4 Step 3: compare outputs with plain code

The comparison is done by `platform/backend/controlplane/runner/parity.py`: standard-library Python, no AI.

**The rule for every number:**

```
|Python value − R value|  ≤  atol + rtol × |R value|
```

- `atol` (absolute tolerance) allows a tiny fixed difference. Default 1e-8 (0.00000001).
- `rtol` (relative tolerance) allows a tiny percentage difference. Default 1e-6 (0.0001%).
- The data scientist confirms or changes these in the specification before any work starts.

**What is checked:**

1. Both sides **converged**: the fitting algorithm finished properly.
2. **Coefficient names** match exactly, matched by name, not by order.
3. Every **coefficient estimate** is within tolerance.
4. Every **standard error** is within tolerance.
5. Every **prediction**, for every row in every set (train and test), is within tolerance.
6. **Missing values**: missing on both sides is a match. Missing or infinite on one side only is a **failure**.
7. **Fit statistics**: deviance, null deviance, dispersion, AIC, log-likelihood, etc.
8. **Metrics** (AUC, deviance, Brier score, …), computed by **the same code** for both sides from each side's
   predictions.

A required check that cannot be computed is a **failure**, never silently skipped.

**Why matching outputs is strong evidence**: standard errors depend on the entire data matrix, the weights and the
model's dispersion. If every coefficient, every standard error and every prediction on hundreds of thousands of
rows match to ~1e-12, it is extremely unlikely that a preprocessing mistake slipped through. Real example from
testing: a scaling bug (`ddof=0` instead of `ddof=1`, a one-character difference) shifted coefficients by only
3.3e-6. It was still caught, because the tolerance was 1e-6.

### 3.5 Step 4: a rulebook of known R-vs-Python traps

`platform/backend/controlplane/rulebook.yaml` lists known differences between R and Python and how to handle each.
Many rules are marked `verified`: they were proven by running real R. Examples:

- `read.csv` renames columns (`make.names`) and treats only `NA` as missing.
- How R builds categories and chooses reference levels.
- **R-EST-02**: R's GLM starts its fitting algorithm from a specific point (`mustart`). Starting Python from the
  same point improves standard-error agreement from ~1e-6 to ~1e-12–1e-15.
- R's numerical safety limits (clamping extreme values in the logistic link, etc.).

The AI agents can look up the rulebook when they need it; it is not pasted into every prompt. **The rulebook
helps the AI get it right; the parity check is what decides pass or fail.**

### 3.6 Step 5: when numbers don't match, diagnose precisely

If parity fails, the "Diagnostician" AI receives a filtered view:
- which checks failed, and by how much (in tolerance units, not raw values);
- a **design comparison**: rows used in fitting on each side, the statistics of every column, reference levels,
  weights and offsets.

It **never** sees R's actual coefficient values, so it cannot pass the answers to the Translator. It classifies
the problem:

| Category | Who handles it |
|---|---|
| Migration defect (Python bug) | Back to the Translator to fix |
| Specification problem | Data scientist decides |
| Not reproducible in Python | Data scientist decides |
| Model fitness issue (model itself is weak) | Data scientist decides |

Up to **3 repair attempts** by default, then it escalates to the human.

### 3.7 Step 6: independent re-checks

- **Parity fixtures**: synthetic datasets with a known-good Python candidate, run through real R and Python in
  the automated test suite (`platform/backend/tests/fixtures/parity/`). A model type counts as "proven" only once
  it has one.
- **`verify.py` in the deliverables bundle**: the same comparison engine ships with the result, so an auditor can
  re-run the proof on their own machine without the platform.

### 3.8 Weak point: be ready for this question

Parity proves **Python = `baseline.R`**. It does **not** prove **`baseline.R` = the original model**.

If the AI's extract left out a step (for example "remove customers under 18"), both R and Python would compute
on the wrong data and still match. Today the code checks only the **form** of `baseline.R`: it reads the dataset
from the right path, uses only allowed packages, and creates the named model objects. Faithfulness to the
original is protected by **the data scientist reviewing the side-by-side diff before confirming**.

Planned improvements:
1. When the environment allows, run the full original script once and compare its model with `baseline.R`'s
   model.
2. When it doesn't, let the data scientist upload coefficients from their own run of the original as a
   reference.
3. Static comparison of formula, family, row filters and split between the original and `baseline.R`.

Second limitation: parity is shown **on this dataset**. On new data with unusual edge cases (new categories,
extreme values) the two could diverge. The re-runnable bundle helps; stress-testing with synthetic variations is
on the roadmap.

---

## 4. How the workflow runs, and how edge cases are covered

### 4.1 The seven steps

The order is **fixed in code** (`workflow/machine.py`). The AI cannot skip, reorder or approve steps.

| # | Step (UI name) | Who acts | What happens | Gate to move on |
|---|---|---|---|---|
| 1 | **Intake** | Data scientist | Upload the dataset, other data files, the R code and documents | Files fingerprinted (SHA-256) |
| 2 | **Model Specification** | Analyst AI + data scientist chat | AI reads the code and data profile and drafts the `ModelSpec` (see 4.2) and `baseline.R` | Data scientist confirms spec version N |
| 3 | **R Reference Results** | Code only | Runs `baseline.R` in real R and freezes the answer key | R run succeeds |
| 4 | **Python Migration & Comparison** | Translator AI, then code | AI writes Python; code runs it and checks parity and business fitness | Data scientist accepts or rejects |
| 5 | **Discrepancy Analysis** (only after a failure or rejection) | Diagnostician AI | Diagnoses and sends repair instructions back to step 4 | Max 3 retries, then human |
| 6 | **Model Validation Report** | Reporter AI + code | Report built from recorded evidence only | Every claim cites evidence |
| 7 | **Code & Reproducibility Package** | Code only | Bundle: Python code, environment files, results, report, `verify.py` | Download |

### 4.2 The specification (`ModelSpec`): the contract

Before any translation, the Analyst AI drafts a specification, which the data scientist confirms. It states:

- the target model, the R function and the Python implementation, plus the **parity tier** (exact, approximate
  or unsupported);
- family, link and formula: the mathematical form of the model;
- **explicit semantics**: what the target means and how it is coded (e.g. "1 = customer defaulted"), weights,
  offsets (e.g. exposure in years for insurance), what a prediction means and in what units;
- **tolerances** per quantity;
- **business assessment**: intended use, population, decision threshold, acceptance criteria (e.g. minimum AUC
  0.65, maximum PSI 0.25, minimum fairness impact ratio 0.8);
- **validation plan**: which business-fitness checks to run and why (see 4.4);
- `baseline.R`.

The specification changes only through a new version confirmed by the human. A new version restarts the work
from step 3 or 4.

### 4.3 Four AI agents, tightly limited

| Agent | Sees | Produces |
|---|---|---|
| **Analyst** | R code, column names, data types, category lists, summary statistics. **Never data rows.** | `ModelSpec` + `baseline.R` |
| **Translator** | Spec, `baseline.R`, R's coefficient **names** (never values), data profile, rulebook | One Python module with three fixed functions: `prepare`, `fit`, `predict` |
| **Diagnostician** | Filtered failure details, design comparison, the candidate code, earlier attempts. **Never R's numbers.** | `Diagnosis`: category, root cause, evidence, repair instructions |
| **Reporter** | An evidence catalogue built by code | Report narrative in which **every number is a placeholder** (`{{id.field}}`) filled in by code, so it cannot invent numbers |

Each agent has read-only tools and exactly one "submit" tool, whose input must match a strict schema. An invalid
submission is sent back with the error, for a limited number of retries. Agents have **no** shell, write or edit
tools.

### 4.4 Beyond "same numbers": is the model actually any good?

A perfect translation of a bad model is still a bad model. After parity passes, code runs business-fitness checks:

| Check | Plain-English question |
|---|---|
| Baseline comparison | Does the model beat a naive guess (a constant)? |
| Calibration | When it predicts 5% default, do about 5% actually default? |
| Discrimination / decision performance | Does it rank risky customers above safe ones? (AUC, Gini, KS) |
| Subgroup performance | Does it work equally well across segments (region, age band, …)? |
| Stability | Does the population shift between periods? (PSI) Are results stable across data folds? |
| Sensitivity | If one input moves a little, does the prediction move sensibly? |
| Assumptions | Statistical assumptions: overdispersion, multicollinearity (VIF), separation, … |
| Fairness | Are favourable outcomes spread fairly across protected groups? (four-fifths rule: impact ratio ≥ 0.8) |
| Sample suitability | Enough data and enough events per estimated parameter? |

The result is a **business-fitness verdict**: *criteria met*, *criteria not met*, or *not established*.
**Migration accepted ≠ model fit for use**; the platform keeps the two separate.

Default criteria come from the team's earlier Model Risk Factory thresholds (`envs/criteria_defaults.yaml`).
For binary models: AUC ≥ 0.65, PSI ≤ 0.25, impact ratio ≥ 0.8. For count models: PSI ≤ 0.25, calibration deviation
≤ 10%.

### 4.5 Edge cases and how they are covered

**Data and R semantics**
| Edge case | Coverage |
|---|---|
| Columns with spaces or symbols | Rulebook R-IO-01: reproduce R's `make.names` exactly |
| Which values count as missing | Rulebook R-IO-02: R's exact `NA` rules |
| Category reference levels and order | Rulebook factor rules; design comparison shows mismatches |
| R's `merge()` scrambles row identity (found in M1) | `capture.R` carries original row numbers through joins; a join that duplicates or adds rows fails loudly |
| A "test" file with no answers (found in M4) | Analyst instructions and schema require the test set to be held-out rows of the dataset |
| Event coding (is 1 "default" or "no default"?) | Stated in the spec; R capture refuses a coding that doesn't match what R predicts |
| Grouped data, weights, offsets | Spec states type and normalisation; carried into fitting and metrics |
| Missing / infinite predictions | Both-missing = match; one-sided = fail |
| Models with no iterative fitting | Convergence recorded as "not applicable" with a reason, never shown as "passed" |
| Families where R and Python define likelihood differently (Gamma, quasi) | Those fit statistics are excluded and the exclusion is recorded |

**Process and failures**
| Edge case | Coverage |
|---|---|
| Model type has no faithful Python version | Marked unsupported, stops at step 2 (M3) |
| `baseline.R` needs a package not installed | Flagged at step 2; human drops it or approves a new environment version. Never silent. |
| R baseline fails | Human can retry, or send the error to the Analyst to fix the spec |
| AI submits invalid output | Returned with the validation error; bounded retries |
| Python candidate fails to run | Recorded as a failed validation, which goes to diagnosis |
| Repair loop doesn't converge | After 3 attempts, escalates to the human, who can raise the limit |
| Spec changes after approval | Approval is bound to hashes of spec, baseline, code and result; any change voids it |
| User cancels a step | Process group killed, AI session aborted, retry possible |
| Server crashes mid-run | Running jobs marked "interrupted" on restart; resume or retry. Tested by killing the server mid-run. |
| MLflow (tracking server) down | MLflow only mirrors the record; jobs don't depend on it. Queued and retried. |
| Error messages leaking data values | Messages to AI are redacted (quoted text and decimals removed) |
| Small subgroups leaking information | Subgroup metrics withheld from AI below a minimum size |

---

## 5. How to use the platform

### 5.1 Setup (once)

Requirements: R 4.6.2, `uv` (Python package manager), Node 22.19+, and a C/C++/Fortran compiler.

```bash
./platform/bootstrap.sh
```

This builds the pinned R and Python environments, the agent service and the UI.

### 5.2 Start

```bash
./platform/dev.sh
```

| Service | Address |
|---|---|
| Web UI (development) | http://localhost:5173 |
| Control plane API (also serves the built UI) | http://localhost:8000 |
| Agent service | localhost:8100 |
| MLflow (experiment tracking) | http://localhost:5000 |

### 5.3 Configure the AI models

In the UI open **Settings**:
1. Add a model endpoint: name, base URL, API type, API key, model ids.
2. Press **Test connection**.
3. Assign a model, with a reasoning level, to each role: Analyst, Translator, Diagnostician, Reporter.

Keys are encrypted at rest and never sent back to the browser.

### 5.4 Run a project

1. **Projects → New**. Upload the dataset CSV, any other data CSVs, the R file and any documents.
2. **Model Specification**: chat with the Analyst. If the R file has several models, **say which one you want**.
   Review the drafted spec and the `baseline.R` diff. Ask "which steps did you drop and why?" if unsure. Confirm.
3. **R Reference Results**: runs automatically. Check the baseline report.
4. **Python Migration & Comparison**: review side-by-side R and Python code, the differences table (coloured by
   tolerance), and the business-fitness checks. Accept, or reject with a comment.
5. **Discrepancy Analysis** (if needed): read the diagnosis. Let the repair continue, or talk to the Analyst to
   change the spec.
6. **Model Validation Report**: generated automatically.
7. **Code & Reproducibility Package**: download files individually or as one bundle. Anyone can re-run the proof
   with the bundle's README (`uv sync`, then `verify.py`).

**One model per project.** To migrate two models from one file, create two projects with the same uploads.

### 5.5 Tests and deployment

```bash
cd platform/backend && uv run pytest
```

Linux server: `docker compose -f deploy/compose.yaml up --build` from `platform/` (HTTPS with login via the
Caddy compose file; see `README.md`). The Docker image is written but **not yet built on a Linux host**.

---

## 6. Business value

### 6.1 Time and people

- **Manual migration and validation of one model takes a team weeks [estimate, to confirm with industry
  sources]**: understand old code, rewrite, reconcile numbers, write validation documentation, get sign-off.
- **Measured on our platform**: in the first end-to-end run, M2 took 30 minutes, M4 18 minutes and M1
  (1,126-line notebook) 64 minutes, including one automatic repair each for M1 and M4. In the later M4 benchmark
  (6.3), active time was **1.5 to 22 minutes** depending on the AI model. These times include the AI's thinking;
  the human only reviewed and approved.
- The human's role shifts from **doing** to **reviewing and approving**: the right use of an expert's time.

### 6.2 Quality and risk

- **Numbers that match to ~12 decimals**, proven, not "looks about right".
- **Bugs that humans miss are caught automatically** (e.g. the `ddof` scaling error, a 0.0003% difference).
- **Audit-ready evidence**: every run, version, approval and hash is recorded. The report cites evidence. The
  bundle re-runs the proof independently. This maps directly to SS1/23 expectations: documentation, independent
  validation, human accountability.
- **No silent model substitution**: unsupported models are flagged, not faked.

### 6.3 Measured cost: the M4 live benchmark

The team ran M4 once on each of four AI models. Each run was a fresh project on the deployed local service, with
the same M4 R notebook, training CSV and unlabeled scoring CSV, and every role set to "medium" reasoning. Costs are
**estimates from public price lists**, not invoices.

| AI model | Outcome | Active time¹ | Wall time² | Total tokens | Cached share of input | Est. cost |
|---|---|---:|---:|---:|---:|---:|
| gpt-6.1-terra | done | 1m 24s | 4m 50s | 68,305 | 56% | **$0.045** |
| deepseek-flash | done | 2m 43s | 6m 07s | 189,827 | 77% | **$0.023** |
| gpt-6.1-sol | done | 3m 30s | 4m 38s | 108,678 | 45% | **$0.184** |
| grok-4.7 | done | 21m 37s | 23m 13s | 644,032 | 44% | **$1.268** |

¹ Time the jobs actually ran (AI plus R and Python). ² Also includes review and waiting at gates.

**Every run reached the same quality bar:**

| AI model | Spec confirmations | Migration approvals | Parity checks | Common checks³ | Bundle re-run |
|---|---:|---:|---:|---:|---|
| gpt-6.1-terra | 1 | 1 | 25/25 | 23/23 | exit 0, 2.9s |
| deepseek-flash | 1 | 1 | 26/26 | 23/23 | exit 0, 3.1s |
| gpt-6.1-sol | 1 | 1 | 23/23 | 23/23 | exit 0, 2.6s |
| grok-4.7 | 1 | 1 | 27/27 | 23/23 | exit 0, 3.7s |

³ Each agent picked slightly different extra metrics, so the parity totals differ. For a fair comparison every
approved output was re-checked against its own frozen R baseline with the same four metrics (AUC,
actual/expected, log loss, Brier) at the default tolerances.

**Where the tokens go** (share of total tokens, by step):

| AI model | Analyst (spec) | Translator (migration) | Diagnostician | Reporter |
|---|---:|---:|---:|---:|
| gpt-6.1-terra | 43% | 33% | – | 23% |
| deepseek-flash | 35% | 35% | – | 29% |
| gpt-6.1-sol | 56% | 28% | – | 16% |
| grok-4.7 | 47% | 26% | 17% | 9% |

Intake, the R baseline and packaging use **zero tokens**. They are plain code and take seconds: R baseline 3–4s,
packaging under 0.5s.

**What the safety net caught during the benchmark** (good stories for the pitch):
- **Grok's first Python candidate hid required fit statistics inside a wrapper object.** Parity failed, the
  Diagnostician found the cause, and the second candidate passed after returning the native result.
- **Grok set tolerances that were too loose, twice**, including after the first correction. The spec had to be
  revised again. This is limitation 6 in section 9 ("tolerances are human-set") happening for real and being stopped.
- **Sol made an unsupported "outcome leakage" claim** in the spec and needed one revision.

All of these revision and repair jobs are included in the token and cost totals above.

**Headline for the pitch:** *"Migrating and validating a credit-risk model cost between 2 cents and $1.27 of AI
usage, and every AI model we tried reached the same verified result."*

**Caveats to state honestly:**
- One run per model, on **M4 only**, the simplest model (57 lines, one GLM). M1's 1,126-line notebook will cost
  more; its repair loop alone took about 45 minutes of AI time in an earlier run.
- It compares AI models with each other, **not** our design with a naive "one agent translates everything"
  approach, so there is still no "X% fewer tokens than the alternative" number.
- Prices are public list rates. DeepSeek is at off-peak rates; Grok at US rates for requests under 200k input
  tokens.
- Projects could overlap across stages, and cache hits, spec revisions and agent choices affect the comparison.

### 6.4 Choosing AI models: what the benchmark suggests

The platform lets you assign a different AI model to each role (Settings). The benchmark supports doing that
deliberately:

| AI model | Verdict on M4 |
|---|---|
| **gpt-6.1-terra** | Best balance: cheap, fastest, fewest tokens, **no revisions or repairs** |
| **deepseek-flash** | Cheapest in dollars, but almost 3× terra's tokens (lots of reasoning output). Cheap because of very low unit prices and off-peak rates |
| **gpt-6.1-sol** | About 4× terra's cost, one spec revision, no better result on M4 |
| **grok-4.7** | Most expensive and slowest, most revisions and repairs, but still reached the verified result thanks to the safety net |

**Recommended rule of thumb:**
- **Analyst: use a strong model.** It uses the most tokens (35–56%), and its output (the `baseline.R` slice, the
  tolerances, the business criteria) is the part **not** proven by code. A mistake there is a real mistake.
- **Translator and Reporter: a cheaper model is fine.** Parity checks guard the Translator, and code-filled
  placeholders guard the Reporter. A weak model costs an extra repair round, not a wrong answer.
- **Diagnostician: a mid-range model.** It only runs after a failure, and a good diagnosis shortens the repair
  loop.

**The key business point:** because a deterministic check decides pass/fail, **the choice of AI model affects
cost and speed, not correctness**. A bank can use the cheapest model its policy approves, or a self-hosted one,
without lowering trust in the result.

### 6.5 Why our design uses fewer AI tokens (lower cost)

The benchmark (6.3) gives absolute costs, but we have **not measured token savings against an alternative
approach** [to measure before presenting a percentage]. The design choices that reduce token use are real and can
be explained:

| Design choice | Why it saves tokens |
|---|---|
| **Code does the checking, not AI** | Parity, metrics, validation and the report's numbers are plain code: zero tokens, and the same answer every time. Many "AI agent" designs ask the model to check its own work, which costs tokens and is unreliable. |
| **AI sees data profiles, never data rows** | A profile (column names, types, summary stats) is the same small size whether the dataset has 1,000 rows or 678,000. Prompt cost does not grow with data size. It also keeps customer data out of the AI. |
| **`baseline.R` slice** | The Translator works from the short slice, not the full notebook (M1: 1,126 lines with four model families). |
| **Rulebook on demand** | The AI reads the rulebook through a tool only when needed, instead of carrying it in every prompt. |
| **Structured outputs with schemas** | One `submit` call with a strict schema, so no long free-text back-and-forth. Invalid output is caught immediately with a precise error. |
| **Filtered context for diagnosis** | The Diagnostician gets differences in tolerance units and a design summary, not full outputs or logs. |
| **Bounded retries** | Max 3 repair attempts by default, so cost is capped per model. |
| **Reporter writes claims only** | Numbers are placeholders filled by code, so the AI writes short claims, not full tables. |

Suggested way to get a comparison number: the platform already records token counts per model call (that is
how the benchmark was produced). Run the same models with a single general AI agent asked to "translate this R
script and check it", and compare tokens, cost and whether the result actually matches R.

### 6.6 Scalability

- One pinned environment shared by all projects: no per-project setup.
- Each model is an independent project, so many can run in parallel.
- Works with any AI provider (OpenAI-, Anthropic- and Google-style APIs), so the bank can use its approved
  vendor.
- Runs locally or on a Linux server, so data can stay inside the bank.

### 6.7 Who benefits

| Stakeholder | Benefit |
|---|---|
| Data scientists | Hours instead of weeks; focus on judgement, not rewriting |
| Model risk / validation teams | Standardised, evidence-backed validation packs |
| IT / engineering | Retire R infrastructure; one Python stack |
| Regulators / auditors | Reproducible proof, full audit trail, human approval recorded |
| Business | Faster model updates, lower operational risk |

---

## 7. Architecture and AI safety design

### 7.1 Components

```
Web UI (React) ──HTTP/live stream──> Control plane (Python, FastAPI)
                                       ├─ workflow state machine + approvals (SQLite database)
                                       ├─ runner: runs R and Python steps in isolated processes
                                       ├─ validator: parity + business-fitness checks (plain code)
                                       ├─ MLflow mirror ──> MLflow server (experiment tracking)
                                       └─ HTTP ──> Agent service (Node.js, Pi agent SDK)
                                                     └─ agent tools call back into the control plane
```

- **Control plane**: the brain. It owns all workflow, execution and approval logic. Every step change and job is
  a database row.
- **Agent service**: deliberately small (~220 lines). It only runs an AI role and returns schema-valid output;
  it holds no workflow state.
- **Web UI**: project list, a seven-step stepper, and a chat panel.
- **MLflow**: an industry-standard tracker. One experiment per project, with nested runs for baseline,
  translation, validation, diagnosis and report. It is a mirror only.

### 7.2 Safety principles (useful phrases for the pitch)

1. **"AI proposes, code disposes."** AI writes specs, code and narratives; deterministic code decides pass/fail.
2. **"Humans approve, never the AI."** Approvals are bound to hashes; any change voids them.
3. **"Least information."** Each agent sees only what its job needs: no data rows, no R answers for the
   Translator or Diagnostician.
4. **"Least privilege."** Agents have read-only tools plus one submit tool. No shell, no file writing.
5. **"Sandboxed execution."** Each step runs in a fresh folder, with credentials scrubbed, CPU, memory and time
   limits, and read-only data. In Docker, steps run as a separate unprivileged user.
6. **"Guardrails on generated code."** Import allow-list; no network, randomness, `eval` or file writes.
7. **"No invented numbers."** Report numbers are filled in from recorded evidence by code.

Honest caveat: in local (non-Docker) mode, process isolation is **not a security boundary**; the static
guardrails are the main protection. In Docker, step code still has network access.

### 7.3 Relation to the earlier "Model Risk Factory" (`legacy/`)

The team first built the **Agentic Model Risk Factory**, driven from Claude Code with slash commands. It covered
documentation, migration, validation and **monitoring** for M1 and M2, and its results are in `legacy/`.
R-Pythonizer is the rebuild as a proper web platform. It reuses the factory's R capture harness, guardrails,
parity logic, rulebook and validation thresholds. Monitoring (drift on new data batches) existed in the factory
but is not part of the new platform yet.

---

## 8. Results so far

End-to-end run on 2026-11-14 with a live AI model for all four agents. The data scientist's replies were
scripted.

| Model | Specification | Outcome | Repairs | Minutes | Re-run `verify.py` |
|---|---|---|---|---|---|
| M1 freMTPL2 | Poisson GLM, log-exposure offset | **Done** | 1 (a coefficient off by 1.7e-5 vs 1e-6 tolerance) | 64 | Pass, 21 checks |
| M2 Taiwan | Binomial GLM | **Done** | 0 | 30 | Pass |
| M3 ULB fraud | Random forest | **Stopped at step 2: unsupported** (correct behaviour) | – | 3 | – |
| M4 GiveMeSomeCredit | Binomial GLM, no test set | **Done** | 1 (data-type error in `prepare()`) | 18 | Pass, 19 checks |

Precision achieved (M4): coefficients max relative difference **1.1e-12**, standard errors **1.1e-12**,
predictions max absolute difference **2e-13**.

That first run found two real problems, both fixed: R's `merge()` losing row identity (M1), and an unlabeled file
being treated as a test set (M4).

Test suite: about 240 backend tests passing (2 skipped for macOS/root reasons). The UI and agent service
type-check and build.

**Later M4 benchmark across four AI models** (details in 6.3): all four reached "done" with one spec
confirmation and one approval each, passed 23–27 parity checks (23/23 on a common check set), and re-ran the bundle
with exit 0. Cost ranged from $0.02 to $1.27. The check counts are higher than the first run's 19 because the
current validation workflow and the agents' chosen metrics add checks.

**Not yet verified**: the newest validation workflow (the Analyst proposing a validation plan, report template
v3) against a live AI; the Docker image on Linux; clicking through every interactive path in a real browser.

---

## 9. Limitations: be honest about these

1. **`baseline.R` faithfulness is human-reviewed, not machine-proven** (section 3.8).
2. **Parity is shown on the given dataset**, not on all possible data.
3. **Only two model types are fully proven** (binomial and Poisson GLM). Others are expected-exact or approximate.
4. **No machine-learning models** (random forest, boosting, neural nets).
5. **One model per project**: a notebook with four models needs several projects, and the ML ones cannot be done.
6. **Tolerances are human-set.** Loosening them too far would weaken the proof; the human confirmation is the
   control.
7. **Approximate-tier models** may not reach the precision a data scientist expects.
8. **Local mode is not a sandbox**; Docker is the security boundary, and step code still has network access there.
9. **Error-message redaction is heuristic**: it keeps integers, which are usually row numbers but could be data.
10. **Evaluation simplifications**: cross-validation reuses preprocessing fitted on all rows; the constant
    comparator ignores offsets. Both are noted in the evaluation.
11. **Monitoring** is not in the new platform yet.
12. **M1's repair loop is slow** (about 45 minutes of AI time).

---

## 10. Roadmap

Ordered from cheapest and highest value to hardest:

1. **Prove more "expected exact" pairs**: add parity fixtures for `lm`, Gamma GLM, negative binomial, etc. Tests
   only, no architecture change.
2. **Verify `baseline.R` against the original**: run the full original script where possible, or accept
   user-uploaded reference coefficients.
3. **Robustness fuzzing**: run R and Python on synthetic variations (extra missing values, rare categories,
   extreme values), not just one dataset.
4. **Extend the cost benchmark**: done for M4 on four AI models (6.3). Next, repeat on M1 and M2, run several
   times per model, and compare with a naive single-agent approach.
5. **Bring back monitoring** (drift on new scoring batches) from the Model Risk Factory.
6. **Machine-learning models**: xgboost and catboost share one C++ core between R and Python, so near-identical
   results are possible. R's keras calls Python's TensorFlow underneath. This needs new packages, prediction-only
   parity (no coefficients) and approximate tolerances.
7. Build and test the Docker image on Linux; add network isolation for step code.

---

## 11. Likely judge questions, with answers

### Problem and scope

**Q: Why not just ask ChatGPT to translate the R code?**
A: An LLM can write plausible Python, but nobody can tell whether it gives the same numbers. In a regulated bank,
"looks right" is not evidence. Our AI writes the code, but plain deterministic code proves the outputs match to
~1e-12. The AI never sees R's answers, so it cannot fake a match. A human then approves, and the proof is
re-runnable.

**Q: Why only GLMs?**
A: Not only GLMs: we support classical statistical models (linear regression, GLMs, quantile, multinomial,
mixed models, survival, ARIMA…) at different confidence levels. Binomial and Poisson GLMs are fully proven
because they are the backbone of credit scoring and insurance pricing. We chose depth and provability first.

**Q: Banks use xgboost now. Isn't this outdated?**
A: Most legacy R in risk is statistical, and regulators favour explainable models. ML models are random and
multi-threaded, so exact parity is impossible; they need a different, approximate proof. It is on our roadmap,
and xgboost/catboost are good candidates because R and Python share the same engine.

**Q: Why exclude scikit-learn?**
A: It doesn't produce standard errors or statistical tests, and its defaults (e.g. built-in penalty in logistic
regression) silently differ from R. statsmodels mirrors classical statistics.

**Q: What happens with a model you can't support?**
A: The Analyst marks it unsupported with the reason and the project stops at step 2. M3 (random forest) did
exactly that in 3 minutes. We never silently substitute a different model.

### The proof

**Q: How do you know the Python is correct if R and Python compute differently inside?**
A: We don't compare the internal steps; we compare every output: coefficients, standard errors, every prediction,
fit statistics and metrics, on exactly the same rows, within tolerances agreed in advance. Matching standard errors
to 12 decimals across hundreds of thousands of rows leaves almost no room for a hidden difference.

**Q: What tolerance do you use, and who decides?**
A: Default absolute 1e-8 and relative 1e-6 for every quantity. The data scientist confirms or changes them in the
specification before work starts. Results on M4 were around 1e-12, a million times tighter than required.

**Q: Couldn't the AI just copy R's numbers into the Python code?**
A: No. The Translator sees R's coefficient names, never their values. The Diagnostician also never sees R's values.
Guardrails block reading anything other than the dataset. And the coefficients must come from actually fitting
the model on R's training rows.

**Q: Isn't the random train/test split different between R and Python?**
A: Python never splits. It receives R's exact row numbers and must return exactly those rows, so the observations
are identical by construction.

**Q: What if the AI extracts the wrong part of the R script into `baseline.R`?**
A: Fair point; this is our main known gap. Parity proves Python equals `baseline.R`, not that `baseline.R` equals
the original. Today the data scientist reviews a side-by-side diff before confirming. Next we plan to run the
original script, where possible, and compare its model with the slice automatically.

**Q: Does it work on new data?**
A: Parity is demonstrated on the provided dataset. The bundle's `verify.py` lets anyone re-run the proof on
other data. Synthetic stress testing is on the roadmap.

**Q: Give an example of a bug it caught.**
A: A standardisation using `ddof=0` instead of `ddof=1` (population vs sample standard deviation) shifted
coefficients by 3.3e-6. A human would never notice, but it exceeded the 1e-6 tolerance. The Diagnostician
identified it and the Translator fixed it automatically.

### Workflow and governance

**Q: Can the AI approve its own work?**
A: No. Only the human approves, and the approval is tied to hashes of the spec, baseline, code and validation
result. Change any of them and the approval is void.

**Q: How does this fit SS1/23?**
A: Model inventory (one project = one model record), documentation (spec and report), independent validation
(deterministic checks plus business-fitness checks), human accountability (bound approvals), reproducibility
(pinned environments plus `verify.py`), and an audit trail (every transition in the database and MLflow).

**Q: What if the server crashes, or the user cancels?**
A: Every job is a database row. On restart, running jobs are marked interrupted and can be resumed. Cancel kills
the process and aborts the AI session. Both are tested, including killing the server mid-run.

**Q: Does customer data go to the AI provider?**
A: No data rows. Agents see column names, types, category lists and summary statistics. Error messages are
redacted before reaching the AI, and small subgroup statistics are withheld.

**Q: Is the model "good" just because it migrated correctly?**
A: No, and we keep these separate. After parity, the platform checks calibration, discrimination, stability,
fairness, sample size and more, and gives a separate business-fitness verdict.

### Business

**Q: How much does it save?**
A: Measured: on M4, AI usage cost $0.02–$1.27 per full migration and validation, with 1.5–22 minutes of active
time depending on the AI model; earlier end-to-end runs took 18–64 minutes per model. The human only reviews.
Manual migration plus validation typically takes weeks [estimate, cite a source if you use it]. We have not yet
benchmarked token use against a naive single-agent approach.

**Q: Which AI model should a bank use?**
A: It matters for cost and speed, not correctness: all four models we tested reached the same verified result,
because code decides pass/fail. We recommend a strong model for the Analyst, whose spec and slice are not
code-proven, and cheaper models for the Translator and Reporter, which parity and placeholders guard. On M4,
gpt-6.1-terra was the best balance ($0.045, 1.5 minutes, no repairs).

**Q: Did an AI ever try to get around the checks?**
A: In the benchmark, one model's Python hid required fit statistics inside a wrapper. Parity failed, the
Diagnostician found it, and the fix passed. The same model also proposed tolerances that were too loose, and
they were rejected until corrected. The safety net works on real agent mistakes, not just in theory.

**Q: Is $1.27 not expensive for one model?**
A: That was the slowest model, with two spec revisions and one repair, all included. The same task cost $0.02–$0.18
on the other three. Either way it is negligible next to analyst time.

**Q: Will bigger models like M1 cost much more?**
A: Yes, more. The benchmark is on M4, the simplest case (57 lines). M1's 1,126-line notebook took 64 minutes
end to end earlier, with about 45 minutes of AI time in its repair loop. We have not yet benchmarked M1's cost
across models.

**Q: Can a bank run it on-premises with its own AI provider?**
A: Yes. It runs locally or in Docker on Linux, and works with OpenAI-, Anthropic- or Google-style APIs, including
self-hosted endpoints with those interfaces.

**Q: What's the hardest part you solved?**
A: Making "same model" precise: row identity through R's data manipulation (e.g. `merge()`), R's exact CSV and
category semantics, matching R's GLM starting point to get standard errors to 1e-12, and keeping the AI unable to
cheat.

---

## 12. Suggested presentation outline

1. **Hook (30s)**: "A 0.0003% difference in one coefficient. Would your team notice? Our platform did."
2. **Problem (1 min)**: legacy R models in banks; migration is a regulated model change (SS1/23); "looks the same"
   isn't evidence.
3. **Solution (1 min)**: R-Pythonizer: AI writes, code proves, human approves.
4. **How it works (2 min)**: seven-step workflow diagram; four agents and what each can't see.
5. **The proof (2 min)**: frozen R answer key, same rows, outputs compared within tolerance, results at 1e-12;
   the `ddof` bug story.
6. **Beyond parity (1 min)**: business-fitness checks and verdict.
7. **Demo (2–3 min)**: M4 (fastest, 18 min, so use a pre-recorded or completed project): spec chat, `baseline.R`
   diff, differences table, report, bundle `verify.py`.
8. **Results (30s)**: M1/M2/M4 done, M3 correctly refused.
9. **Business value (1 min)**: the benchmark table ($0.02–$1.27 per model, same 23/23 result on four AI
   models), audit-ready evidence, per-role model choice, on-prem.
10. **Limitations and roadmap (1 min)**: own the gaps (slice faithfulness, ML models) before judges raise them.
11. **Close**: "Trustworthy AI for regulated model migration."

Tip: say the limitations yourselves. Judges respect a team that knows exactly where its proof ends.

---

## 13. Glossary

| Term | Plain meaning |
|---|---|
| **R / Python** | Programming languages. R is traditional in statistics; Python is today's industry standard. |
| **Legacy model** | An old model still in use, often poorly documented. |
| **Model risk** | Risk of loss from a wrong or misused model. |
| **SS1/23** | UK PRA supervisory statement on model risk management principles. |
| **PRA** | Prudential Regulation Authority, the UK bank regulator. |
| **GLM** | Generalised Linear Model, a family of statistical models (logistic, Poisson, Gamma…). |
| **Logistic regression / binomial GLM** | Predicts a probability of a yes/no event (e.g. default). |
| **Poisson GLM** | Predicts counts or rates (e.g. claims per year). |
| **Offset / exposure** | Adjusts for how long something was observed (a policy held 6 months vs 12). |
| **Coefficient** | The weight the model gives each variable. |
| **Standard error (SE)** | How uncertain a coefficient is. |
| **Prediction** | The model's output for one row (e.g. 3% default probability). |
| **Parity** | R and Python giving the same outputs within tolerance. |
| **Tolerance (atol / rtol)** | The maximum allowed difference: absolute / relative. |
| **1e-12** | 0.000000000001 (twelve decimal places). |
| **Baseline** | R's frozen results: the answer key. |
| **`baseline.R`** | The minimal R script that builds only the target model. |
| **Candidate** | The Python code the Translator AI submits. |
| **ModelSpec** | The confirmed specification (contract) of what the model is and how to check it. |
| **Convergence** | The fitting algorithm finished properly. |
| **IRLS** | The iterative algorithm both R and Python use to fit GLMs. |
| **Reference level** | The baseline category others are compared to (e.g. region "North"). |
| **Train / test set** | Data used to fit the model / held-out data used to check it. |
| **AUC / Gini / KS** | How well the model ranks risky above safe (higher is better; AUC 0.5 = random). |
| **Calibration** | Whether predicted rates match observed rates. |
| **PSI** | Population Stability Index: how much the data has shifted (>0.25 = big shift). |
| **Impact ratio / four-fifths rule** | Fairness test: the lowest group's favourable rate should be ≥ 80% of the highest's. |
| **VIF** | Variance inflation factor: detects variables that are too correlated. |
| **Overdispersion** | More variation in counts than a Poisson model assumes. |
| **Separation** | A variable perfectly predicts the outcome, which breaks logistic regression. |
| **statsmodels** | Python library for classical statistics; our target. |
| **scikit-learn** | Python machine-learning library; deliberately not used. |
| **Guardrails** | Automatic rules blocking unsafe or forbidden code. |
| **SHA-256 hash** | A file fingerprint; any change gives a different hash. |
| **Pinned environment / lockfile** | Exact package versions frozen, so results are reproducible. |
| **MLflow** | Industry tool for tracking experiments and model runs. |
| **Agent** | An AI model given a role, tools and an output format. |
| **Deterministic** | Same input always gives the same output (plain code, not AI). |
| **Token** | The unit AI providers charge by (roughly ¾ of a word). |

---

## 14. Numbers cheat sheet

| Fact | Value |
|---|---|
| Workflow steps | 7 |
| AI agents | 4 (Analyst, Translator, Diagnostician, Reporter) |
| Default tolerances | atol 1e-8, rtol 1e-6 |
| Achieved precision (M4) | coefficients 1.1e-12, SE 1.1e-12, predictions 2e-13 |
| Default repair limit | 3 attempts |
| Proven model pairs | 2 (binomial GLM, Poisson GLM with offset) |
| Listed model pairs | 20 (9 expected exact incl. the 2 proven, 11 approximate) |
| M1 / M2 / M4 end-to-end time (first run) | 64 / 30 / 18 minutes |
| M4 benchmark: AI cost per model | $0.02 (deepseek-flash) to $1.27 (grok-4.7); gpt-6.1-terra $0.045 |
| M4 benchmark: total tokens | 68k (terra) to 644k (grok) |
| M4 benchmark: active time | 1m 24s (terra) to 21m 37s (grok) |
| M4 benchmark: quality | All 4 AI models done, 23/23 common checks, bundle exit 0 |
| Share of tokens used by the Analyst | 35–56% |
| Cached share of input | 44–77% |
| M3 | Correctly refused (unsupported) in 3 minutes |
| M4 compute time (150k rows: baseline, candidate, diagnostics, evaluation) | ~14 seconds |
| `verify.py` checks | 21 (M1), 19 (M4, first run); 23–27 (M4 benchmark) |
| Backend tests | ~240 passing, 2 skipped |
| Dataset sizes | M1 678,013 · M2 30,000 · M4 150,000 rows |
| Default business criteria (binary) | AUC ≥ 0.65, PSI ≤ 0.25, impact ratio ≥ 0.8 |
| Default business criteria (count) | PSI ≤ 0.25, calibration deviation ≤ 0.10 |
| Pinned versions | R 4.6.2, Node ≥ 22.19, Pi SDK 1.5.2 |

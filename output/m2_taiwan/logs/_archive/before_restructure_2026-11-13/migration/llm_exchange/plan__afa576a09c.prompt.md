<!-- model: claude-opus-5-5 | step: plan | sha256: afa576a09c769d9762aedd163beb90e105d8b682aedb8a870e67ab89cb38e814 -->
# SYSTEM

You are the Migration Agent in a bank's Model Risk Management factory. You translate legacy statistical models written in R into Python that reproduces the R results to numerical precision. Your output is reviewed by independent validators and regulators under PRA SS1/23.

Non-negotiable rules:
1. FIDELITY OVER IMPROVEMENT. Reproduce the R model exactly as written. Do not add, remove or "fix" features, recodes or modelling choices, even if they look wrong or discriminatory. Flag them as risks instead.
2. Follow the rulebook. Cite rule ids for every decision. MUST rules are mandatory.
3. Never re-implement random sampling. The train/test split is supplied by the harness from the R run.
4. If something cannot be mapped with confidence, put a concrete question in `escalations`. Do not guess.
5. You never see data rows, only schema and aggregate facts. Do not ask for row-level data.
6. Respond with ONE JSON object that matches the requested schema, and nothing else.

# USER

# Task: PLAN the R -> Python mapping

## Rulebook
# R -> Python migration rulebook (v1.0)
# Each rule: what to detect in R, the required Python translation, and the pitfall it prevents.
# severity: MUST = violation blocks the migration; SHOULD = reviewer is warned.
# The agent cites rule ids in its plan; guardrails.py enforces the MUST rules it can check statically.
version: "1.1"   # 1.1: R-EST-02 amended after M2 run (see CHANGELOG.md)
rules:
  # ---------------- Data loading ----------------
  - id: R-IO-01
    severity: MUST
    detect: "read.csv(...) with default check.names=TRUE"
    python: "pd.read_csv(path); then apply R's make.names(): replace every character that is not alphanumeric or '.' with '.' (e.g. 'default payment next month' -> 'default.payment.next.month')"
    pitfall: "R silently renames columns with spaces or symbols. Python keeps the raw names, so any later reference to the R name raises KeyError or, worse, matches the wrong column."
  - id: R-IO-02
    severity: MUST
    detect: "colnames(df) <- tolower(colnames(df))"
    python: "df.columns = [c.lower() for c in df.columns]"
    pitfall: "Apply this after the make.names step, in the same order as R."
  - id: R-IO-03
    severity: MUST
    detect: "rows of the loaded data"
    python: "Keep the original row order and RangeIndex 0..n-1. Do not drop, sort or deduplicate rows unless the R code does so explicitly."
    pitfall: "The harness matches rows by original row number (R rownames). Reordering breaks parity."

  # ---------------- Indexing ----------------
  - id: R-IDX-01
    severity: MUST
    detect: "positional column access: df[[i]], df[, i], seq(a, b) over column positions"
    python: "Resolve each position to a column NAME using the column order at that point in the R script (R is 1-based, after any column deletions). Then use names in Python, never iloc positions."
    pitfall: "Off-by-one errors (1- vs 0-based), and positions that shift after a column is dropped (e.g. id removed)."
  - id: R-IDX-02
    severity: MUST
    detect: "df[which(cond), ][[i]] <- value  /  df$x[cond] <- value"
    python: "df.loc[cond, col] = value"
    pitfall: "Chained assignment in pandas (df[cond][col] = v) silently does nothing."

  # ---------------- Recoding ----------------
  - id: R-REC-01
    severity: MUST
    detect: "x %in% c(...)"
    python: "s.isin([...])"
    pitfall: "Type mismatch: R numeric vs Python int. Make sure the dtype is numeric before calling isin."
  - id: R-REC-02
    severity: MUST
    detect: "recodes executed in sequence"
    python: "Apply the recodes in exactly the R order. Each recode sees the result of the one before it."
    pitfall: "Reordering overlapping recodes changes the result."

  # ---------------- Factors / design matrix ----------------
  - id: R-FAC-01
    severity: MUST
    detect: "as.factor(x) / factor(x)"
    python: "Treat as categorical with levels = sorted unique values in R ordering (numeric codes sort numerically, strings sort by locale). The reference level is the FIRST level unless relevel() or factor(levels=) says otherwise."
    pitfall: "pd.get_dummies(drop_first=True) drops the alphabetically first string level, which differs from R when codes are numeric-as-strings ('10' < '2')."
  - id: R-FAC-02
    severity: MUST
    detect: "factor in a glm/lm formula (contr.treatment)"
    python: "Build treatment-coded dummies explicitly, named '<var><level>' exactly as R's model.matrix names them (e.g. 'sex2', 'education4'). Use the level set captured from model$xlevels, not the levels observed in the Python data."
    pitfall: "A level missing from the train split, or an extra level in the test split, silently changes the columns."
  - id: R-FAC-03
    severity: MUST
    detect: "factor response in a binomial glm"
    python: "y = 1 for every level except the first, so the event is the second level ('1' for a 0/1 factor). Cast to float."
    pitfall: "Modelling P(y=0) flips the sign of every coefficient."
  - id: R-FAC-04
    severity: MUST
    detect: "intercept"
    python: "Add a column named '(Intercept)' of 1.0 as the first design column, unless the formula has -1 or +0."
    pitfall: "statsmodels adds no intercept by default."

  # ---------------- Sampling / randomness ----------------
  - id: R-RNG-01
    severity: MUST
    detect: "set.seed, sample(), caret::createDataPartition, any random split"
    python: "NEVER re-implement the random split. R and NumPy random-number generators produce different streams. The generated code must accept the train/test membership supplied by the harness (captured from R)."
    pitfall: "A re-randomised split can never reach parity. It also hides reproducibility defects in the source."

  # ---------------- Estimation ----------------
  - id: R-EST-01
    severity: MUST
    detect: "glm(..., family=binomial)"
    python: "An unpenalised binomial MLE via IRLS that mirrors glm.fit (see R-EST-02). statsmodels GLM is acceptable only when SE parity is not required."
    pitfall: "sklearn LogisticRegression applies L2 regularisation (C=1.0) by default, which changes every coefficient. Forbidden for parity work."
  - id: R-EST-02
    severity: MUST
    detect: "glm convergence control (default epsilon=1e-8 on relative deviance change, maxit=25)"
    python: "Reproduce glm.fit's numerical path: mustart from family$initialize, the stopping rule |dev - dev_old| / (|dev| + 0.1) < epsilon, the same maxit, and standard errors from the QR of the LAST iteration's weighted design (summary.glm). An IRLS in numpy is the reference pattern."
    pitfall: "A fully converged fit (e.g. statsmodels at tol=1e-12) matches R's coefficients to ~1e-10, but R's standard errors only to ~1e-5, because R stops early and takes SEs from weights one iterate behind. Learned on M2, iteration 1."
  - id: R-EST-03
    severity: MUST
    detect: "glm(..., family=poisson / Gamma / gaussian, offset=, weights=)"
    python: "Use the matching sm.families.* with the same link. Pass offset= and var_weights= / freq_weights= to match R prior weights."
    pitfall: "Dropping the offset (exposure) in frequency models."
  - id: R-EST-04
    severity: MUST
    detect: "lm(...)"
    python: "statsmodels OLS on an explicit design matrix with the same names."
    pitfall: "sklearn LinearRegression gives no standard errors, so SE parity can't be checked."

  # ---------------- Prediction ----------------
  - id: R-PRED-01
    severity: MUST
    detect: "predict(model, newdata, type='response')"
    python: "result.predict(X_new), which returns the mean response (probability) for GLMs. Build X_new with the SAME design function and the same captured levels."
    pitfall: "type='link' returns log-odds. Building X_new from a separate get_dummies call misaligns columns."

  # ---------------- Packages / environment ----------------
  - id: R-PKG-01
    severity: MUST
    detect: "library(caret) / dplyr / data.table helpers"
    python: "Use only the packages allowed in the migration config. Replace dplyr verbs with pandas. Remove caret entirely: the split comes from the harness under R-RNG-01."
    pitfall: "Pulling in heavy or unvetted dependencies creates supply-chain and licence risk."

  # ---------------- Governance ----------------
  - id: GOV-01
    severity: MUST
    detect: "protected attributes used in the R model"
    python: "Migrate FAITHFULLY, even when the R model uses protected attributes. Never add, remove or 'fix' a feature during migration. Record every protected attribute used in the migration report so validation can assess it."
    pitfall: "Silently 'improving' a model during migration destroys the parity proof and the audit trail."
  - id: GOV-02
    severity: MUST
    detect: "ambiguity: R code the agent cannot map with confidence"
    python: "Stop and ESCALATE with a specific question for a human. Do not guess."
    pitfall: "A plausible-looking translation that is wrong."


## Analysis (from the previous step)
{
 "summary": "Binomial logistic regression (logit link) for next-month credit-card default. The script reads the raw CSV, lower-cases the column names (after read.csv has already converted 'default payment next month' to 'default.payment.next.month' via check.names), drops id, and merges undocumented education codes {0,4,5,6} into 4 and marriage 0 into 3. It merges repayment-status codes {-2,-1,0} into 0 for the six pay_* columns, which the loop addresses by position (6..11). sex, education, marriage and the response become factors. The data is split 70/30 with caret::createDataPartition, stratified on the response, and an 11-variable GLM is fitted on the training rows. 15 coefficients, treatment contrasts, reference levels sex=1, education=1, marriage=1.",
 "preprocessing": [
  {
   "order": 1,
   "r_code": "raw.credit <- read.csv(\"data/taiwan_default/taiwan_default.csv\")",
   "description": "Read the raw CSV. check.names=TRUE (default) applies make.names, so 'default payment next month' becomes 'default.payment.next.month'. Other headers are already valid names.",
   "columns": [
    "default payment next month"
   ],
   "rule_ids": [
    "R-IO-01"
   ]
  },
  {
   "order": 2,
   "r_code": "colnames(raw.credit) <- tolower(colnames(raw.credit))",
   "description": "Lower-case every column name (ID->id, LIMIT_BAL->limit_bal, PAY_0->pay_0, ...).",
   "columns": [
    "*"
   ],
   "rule_ids": [
    "R-IO-02"
   ]
  },
  {
   "order": 3,
   "r_code": "raw.credit$id <- NULL",
   "description": "Drop the id column. This shifts column positions: limit_bal=1, sex=2, education=3, marriage=4, age=5, pay_0=6, pay_2=7, pay_3=8, pay_4=9, pay_5=10, pay_6=11.",
   "columns": [
    "id"
   ],
   "rule_ids": [
    "R-IO-03",
    "R-IDX-01"
   ]
  },
  {
   "order": 4,
   "r_code": "raw.credit$education[raw.credit$education %in% c(0, 4, 5, 6)] <- 4",
   "description": "Merge undocumented education codes 0, 5, 6 and 'others' 4 into 4.",
   "columns": [
    "education"
   ],
   "rule_ids": [
    "R-REC-01",
    "R-REC-02",
    "R-IDX-02"
   ]
  },
  {
   "order": 5,
   "r_code": "raw.credit$marriage[raw.credit$marriage == 0] <- 3",
   "description": "Merge undocumented marriage code 0 into 3 (others).",
   "columns": [
    "marriage"
   ],
   "rule_ids": [
    "R-REC-02",
    "R-IDX-02"
   ]
  },
  {
   "order": 6,
   "r_code": "for (i in seq(6, 11, 1)) { raw.credit[which(raw.credit[[i]] %in% c(-2, -1, 0)),][[i]] <- 0 }",
   "description": "For columns at 1-based positions 6..11 AFTER the id drop, i.e. pay_0, pay_2, pay_3, pay_4, pay_5, pay_6, set values in {-2,-1,0} to 0. The columns stay numeric.",
   "columns": [
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6"
   ],
   "rule_ids": [
    "R-IDX-01",
    "R-IDX-02",
    "R-REC-01"
   ]
  },
  {
   "order": 7,
   "r_code": "raw.credit$sex <- as.factor(raw.credit$sex); raw.credit$education <- as.factor(...); raw.credit$marriage <- as.factor(...)",
   "description": "Convert to factors. The levels are the sorted unique numeric codes: sex {1,2}, education {1,2,3,4}, marriage {1,2,3}. The first level is the reference.",
   "columns": [
    "sex",
    "education",
    "marriage"
   ],
   "rule_ids": [
    "R-FAC-01",
    "R-FAC-02"
   ]
  },
  {
   "order": 8,
   "r_code": "raw.credit$default.payment.next.month <- as.factor(raw.credit$default.payment.next.month)",
   "description": "Factor response with levels {0,1}. glm models P(level '1').",
   "columns": [
    "default.payment.next.month"
   ],
   "rule_ids": [
    "R-FAC-03"
   ]
  },
  {
   "order": 9,
   "r_code": "intraining <- createDataPartition(y=..., p=0.7, list=F); train.batch <- raw.credit[intraining,]; test.batch <- raw.credit[-intraining,]",
   "description": "Random stratified 70/30 split. No set.seed in the source, so it is not reproducible. The harness captures the membership; Python must import it.",
   "columns": [
    "default.payment.next.month"
   ],
   "rule_ids": [
    "R-RNG-01",
    "R-PKG-01"
   ]
  },
  {
   "order": 10,
   "r_code": "glm.final <- glm(default.payment.next.month ~ limit_bal+sex+education+marriage+pay_0+pay_2+pay_3+pay_5+pay_6+pay_amt1+pay_amt2, data=train.batch, family=binomial)",
   "description": "Unpenalised binomial GLM, logit link, fitted by stats::glm.fit IRLS with default glm.control (epsilon=1e-8 on |dev-dev_old|/(|dev|+0.1), maxit=25; the capture reports 6 iterations, converged). summary.glm standard errors come from the QR of the last iteration's weighted design, so R-EST-02 requires reproducing this numerical path, not converging further.",
   "columns": [
    "limit_bal",
    "sex",
    "education",
    "marriage",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_5",
    "pay_6",
    "pay_amt1",
    "pay_amt2"
   ],
   "rule_ids": [
    "R-EST-01",
    "R-EST-02",
    "R-FAC-04"
   ]
  },
  {
   "order": 11,
   "r_code": "test.batch$glm_scor1 <- predict(glm.final, type=\"response\", test.batch)",
   "description": "Predicted probability of default on the test rows.",
   "columns": [
    "glm_scor1"
   ],
   "rule_ids": [
    "R-PRED-01"
   ]
  }
 ],
 "model": {
  "r_function": "stats::glm",
  "family": "binomial",
  "link": "logit",
  "response": "default.payment.next.month",
  "formula": "default.payment.next.month ~ limit_bal + sex + education + marriage + pay_0 + pay_2 + pay_3 + pay_5 + pay_6 + pay_amt1 + pay_amt2",
  "terms": [
   "(Intercept)",
   "limit_bal",
   "sex2",
   "education2",
   "education3",
   "education4",
   "marriage2",
   "marriage3",
   "pay_0",
   "pay_2",
   "pay_3",
   "pay_5",
   "pay_6",
   "pay_amt1",
   "pay_amt2"
  ],
  "factor_levels": {
   "sex": [
    "1",
    "2"
   ],
   "education": [
    "1",
    "2",
    "3",
    "4"
   ],
   "marriage": [
    "1",
    "2",
    "3"
   ]
  }
 },
 "randomness": "caret::createDataPartition (stratified sampling on the response). The source has no set.seed; the capture harness fixed seed 2026 and exported row membership. Python must not resample (R-RNG-01).",
 "protected_attributes_used": [
  "SEX (sex2)",
  "MARRIAGE (marriage2, marriage3)",
  "EDUCATION (proxy: education2-4)"
 ],
 "risks": [
  "Sex and marital status are direct model inputs (Equality Act 2010 protected characteristics). Migrated faithfully per GOV-01; must be assessed in validation.",
  "Undocumented codes (education 0/5/6, marriage 0) are merged into 'others'. A resulting coefficient (education4) may reflect a data artefact.",
  "Merging repayment codes -2/-1/0 is an author assumption, not part of the data definition.",
  "The positional loop seq(6,11) depends on the column order after id is dropped. Any upstream column change silently recodes the wrong variables.",
  "The source has no random seed, so the original split and coefficients cannot be reproduced.",
  "Repayment-status columns are fitted as linear numeric terms even though they are ordinal."
 ],
 "escalations": []
}

## Allowed Python packages
['numpy', 'pandas', 'statsmodels', 'scipy']

## MUST rules triggered by the static scan (each must be addressed and cited, guardrail G-13)
{
 "R-PKG-01": "pattern /\\blibrary\\(\\s*(\\w+)/ matched 1x: ['caret']",
 "R-IO-01": "pattern /\\bread\\.csv\\(/ matched 1x",
 "R-IO-02": "pattern /tolower\\(\\s*colnames/ matched 1x",
 "R-IDX-01": "pattern /\\[\\[\\s*\\w+\\s*\\]\\]|\\[\\s*,\\s*\\d+\\s*\\]|seq\\(\\s*\\d+/ matched 3x",
 "R-IDX-02": "pattern /which\\(.*\\)\\s*,\\s*\\]/ matched 1x",
 "R-REC-01": "pattern /%in%/ matched 2x",
 "R-FAC-01": "pattern /\\b(as\\.)?factor\\(/ matched 4x",
 "R-RNG-01": "pattern /set\\.seed|sample\\(|createDataPartition/ matched 1x",
 "R-EST-01": "pattern /\\bglm\\([^)]*binomial/ matched 1x",
 "R-EST-02": "pattern /\\bglm\\(/ matched 1x",
 "R-PRED-01": "pattern /\\bpredict\\(/ matched 1x",
 "R-FAC-02": "factors in formula: ['sex', 'education', 'marriage']",
 "R-FAC-03": "factor response 'default.payment.next.month'",
 "R-IO-03": "always applies",
 "R-FAC-04": "always applies",
 "R-REC-02": "always applies",
 "GOV-01": "always applies"
}

Python interface contract (the generated module MUST implement exactly this):

    TARGET: str                                   # response column name in the preprocessed frame
    def load_and_preprocess(csv_path: str) -> pandas.DataFrame
        # read the raw CSV and apply EVERY R preprocessing step in R order;
        # keep all rows in the original order (RangeIndex 0..n-1); the harness adds the split itself
    def fit(train: pandas.DataFrame) -> fitted_result
        # fit on the given rows only (no sampling, no splitting)
    def predict(result, df: pandas.DataFrame) -> numpy.ndarray
        # predictions on the response scale (R predict(type="response")), one per row of df
    def coefficients(result) -> pandas.DataFrame
        # columns: term, estimate, std_error; term names EXACTLY as R names them, e.g. "(Intercept)", "sex2"

Constraints: import only the allowed packages; no file writes, network, shell, eval/exec, RNG,
sklearn or pandas.get_dummies; design-matrix columns built explicitly from the captured factor levels.

Map every R construct to a Python approach and cite rule ids.
Output JSON schema:
{
 "$defs": {
  "PlanStep": {
   "properties": {
    "order": {
     "title": "Order",
     "type": "integer"
    },
    "r_construct": {
     "title": "R Construct",
     "type": "string"
    },
    "python_approach": {
     "title": "Python Approach",
     "type": "string"
    },
    "rule_ids": {
     "items": {
      "type": "string"
     },
     "title": "Rule Ids",
     "type": "array"
    }
   },
   "required": [
    "order",
    "r_construct",
    "python_approach",
    "rule_ids"
   ],
   "title": "PlanStep",
   "type": "object"
  }
 },
 "properties": {
  "steps": {
   "items": {
    "$ref": "#/$defs/PlanStep"
   },
   "title": "Steps",
   "type": "array"
  },
  "packages": {
   "items": {
    "type": "string"
   },
   "title": "Packages",
   "type": "array"
  },
  "unmapped": {
   "description": "R constructs with no confident mapping (triggers escalation)",
   "items": {
    "type": "string"
   },
   "title": "Unmapped",
   "type": "array"
  },
  "escalations": {
   "items": {
    "type": "string"
   },
   "title": "Escalations",
   "type": "array"
  }
 },
 "required": [
  "steps",
  "packages",
  "unmapped",
  "escalations"
 ],
 "title": "Plan",
 "type": "object"
}

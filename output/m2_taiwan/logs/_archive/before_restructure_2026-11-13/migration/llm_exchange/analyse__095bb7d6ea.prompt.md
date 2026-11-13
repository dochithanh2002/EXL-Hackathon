<!-- model: claude-opus-5-5 | step: analyse | sha256: 095bb7d6ea06781931ac4d7bcee30a8c6a92a1cf06ea916bad17429f87b88c54 -->
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

# Task: ANALYSE the legacy R model

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


## Legacy R source
```r
# ---------------------------------------------------------------------------
# LEGACY MODEL (as received) - Taiwan credit-card default "Traditional Score"
# Source: Kaggle notebook netzone/default-credit-and-credit-score-prediction (v2),
#         Apache-2.0. Code reproduced from the published notebook; lines the
#         notebook described but did not print are marked [reconstructed].
# This file is the INPUT to the R->Python migration agent. Do not edit.
# ---------------------------------------------------------------------------
library(caret)

raw.credit <- read.csv("data/taiwan_default/taiwan_default.csv")
colnames(raw.credit) <- tolower(colnames(raw.credit))

# id column is the client identifier, not important -> removed
raw.credit$id <- NULL

# For education, we will merge 0, 4, 5, 6 to 4 (others)        [reconstructed]
raw.credit$education[raw.credit$education %in% c(0, 4, 5, 6)] <- 4
# For marriage, we will merge 0 to 3 (others)                  [reconstructed]
raw.credit$marriage[raw.credit$marriage == 0] <- 3

# Assume -2, -1 and 0 have the same meaning. Merge them to 0.   (verbatim)
for (i in seq(6, 11, 1)) {
  raw.credit[which(raw.credit[[i]] %in% c(-2, -1, 0)),][[i]] <- 0
}

# convert to factor (sex, education, marriage, target)         [reconstructed]
raw.credit$sex <- as.factor(raw.credit$sex)
raw.credit$education <- as.factor(raw.credit$education)
raw.credit$marriage <- as.factor(raw.credit$marriage)
raw.credit$default.payment.next.month <- as.factor(raw.credit$default.payment.next.month)

# training (70%) and testing (30%)                              (verbatim)
intraining <- createDataPartition(y=raw.credit$default.payment.next.month, p=0.7, list=F)
train.batch <- raw.credit[intraining,]
test.batch <- raw.credit[-intraining,]

# Traditional Score using Logistic Regression                   (verbatim)
glm.final <- glm(default.payment.next.month ~ limit_bal+sex+education+marriage+
                   pay_0+pay_2+pay_3+pay_5+pay_6+pay_amt1+pay_amt2, data=train.batch,
                 family=binomial)

test.batch$glm_scor1 <- predict(glm.final, type="response", test.batch)

```

## Static scan facts (deterministic)
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

## Dynamic facts captured by running the R script (model object metadata)
{
 "r_version": "R version 4.6.1 (2026-06-24)",
 "seed": 2026,
 "packages": [
  "MASS",
  "Matrix",
  "ModelMetrics",
  "R6",
  "RColorBrewer",
  "Rcpp",
  "S7",
  "caret",
  "class",
  "cli",
  "codetools",
  "compiler",
  "data.table",
  "dplyr",
  "farver",
  "foreach",
  "future",
  "future.apply",
  "generics",
  "ggplot2",
  "globals",
  "glue",
  "gower",
  "grid",
  "gtable",
  "hardhat",
  "ipred",
  "iterators",
  "jsonlite",
  "lattice",
  "lava",
  "lifecycle",
  "listenv",
  "lubridate",
  "magrittr",
  "nlme",
  "nnet",
  "pROC",
  "parallel",
  "parallelly",
  "pillar",
  "pkgconfig",
  "plyr",
  "prodlim",
  "purrr",
  "recipes",
  "reshape2",
  "rlang",
  "rpart",
  "scales",
  "splines",
  "stats4",
  "stringi",
  "stringr",
  "survival",
  "tibble",
  "tidyselect",
  "timeDate",
  "timechange",
  "tools",
  "vctrs",
  "withr"
 ],
 "class": [
  "glm",
  "lm"
 ],
 "formula": "default.payment.next.month ~ limit_bal + sex + education + marriage +      pay_0 + pay_2 + pay_3 + pay_5 + pay_6 + pay_amt1 + pay_amt2",
 "family": "binomial",
 "link": "logit",
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
 "response": "default.payment.next.month",
 "xlevels": {
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
 },
 "contrasts": {
  "sex": "contr.treatment",
  "education": "contr.treatment",
  "marriage": "contr.treatment"
 },
 "n_train": 21001,
 "n_test": 8999,
 "converged": true,
 "iterations": 6,
 "null_deviance": 22196.2617795277,
 "deviance": 18587.8388471168,
 "aic": 18617.8388471168,
 "control": {
  "epsilon": 1e-08,
  "maxit": 25
 },
 "column_classes_after_preprocessing": {
  "limit_bal": "integer",
  "sex": "factor",
  "education": "factor",
  "marriage": "factor",
  "age": "integer",
  "pay_0": "numeric",
  "pay_2": "numeric",
  "pay_3": "numeric",
  "pay_4": "numeric",
  "pay_5": "numeric",
  "pay_6": "numeric",
  "bill_amt1": "integer",
  "bill_amt2": "integer",
  "bill_amt3": "integer",
  "bill_amt4": "integer",
  "bill_amt5": "integer",
  "bill_amt6": "integer",
  "pay_amt1": "integer",
  "pay_amt2": "integer",
  "pay_amt3": "integer",
  "pay_amt4": "integer",
  "pay_amt5": "integer",
  "pay_amt6": "integer",
  "default.payment.next.month": "factor"
 }
}

## Raw data schema (no rows, guardrail G-11)
{
 "n_rows": 30000,
 "columns": [
  {
   "name": "ID",
   "dtype": "int64",
   "n_unique": 30000
  },
  {
   "name": "LIMIT_BAL",
   "dtype": "int64",
   "n_unique": 81
  },
  {
   "name": "SEX",
   "dtype": "int64",
   "n_unique": 2,
   "levels": [
    1,
    2
   ]
  },
  {
   "name": "EDUCATION",
   "dtype": "int64",
   "n_unique": 7,
   "levels": [
    0,
    1,
    2,
    3,
    4,
    5,
    6
   ]
  },
  {
   "name": "MARRIAGE",
   "dtype": "int64",
   "n_unique": 4,
   "levels": [
    0,
    1,
    2,
    3
   ]
  },
  {
   "name": "AGE",
   "dtype": "int64",
   "n_unique": 56
  },
  {
   "name": "PAY_0",
   "dtype": "int64",
   "n_unique": 11,
   "levels": [
    -2,
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8
   ]
  },
  {
   "name": "PAY_2",
   "dtype": "int64",
   "n_unique": 11,
   "levels": [
    -2,
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8
   ]
  },
  {
   "name": "PAY_3",
   "dtype": "int64",
   "n_unique": 11,
   "levels": [
    -2,
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8
   ]
  },
  {
   "name": "PAY_4",
   "dtype": "int64",
   "n_unique": 11,
   "levels": [
    -2,
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8
   ]
  },
  {
   "name": "PAY_5",
   "dtype": "int64",
   "n_unique": 10,
   "levels": [
    -2,
    -1,
    0,
    2,
    3,
    4,
    5,
    6,
    7,
    8
   ]
  },
  {
   "name": "PAY_6",
   "dtype": "int64",
   "n_unique": 10,
   "levels": [
    -2,
    -1,
    0,
    2,
    3,
    4,
    5,
    6,
    7,
    8
   ]
  },
  {
   "name": "BILL_AMT1",
   "dtype": "int64",
   "n_unique": 22723
  },
  {
   "name": "BILL_AMT2",
   "dtype": "int64",
   "n_unique": 22346
  },
  {
   "name": "BILL_AMT3",
   "dtype": "int64",
   "n_unique": 22026
  },
  {
   "name": "BILL_AMT4",
   "dtype": "int64",
   "n_unique": 21548
  },
  {
   "name": "BILL_AMT5",
   "dtype": "int64",
   "n_unique": 21010
  },
  {
   "name": "BILL_AMT6",
   "dtype": "int64",
   "n_unique": 20604
  },
  {
   "name": "PAY_AMT1",
   "dtype": "int64",
   "n_unique": 7943
  },
  {
   "name": "PAY_AMT2",
   "dtype": "int64",
   "n_unique": 7899
  },
  {
   "name": "PAY_AMT3",
   "dtype": "int64",
   "n_unique": 7518
  },
  {
   "name": "PAY_AMT4",
   "dtype": "int64",
   "n_unique": 6937
  },
  {
   "name": "PAY_AMT5",
   "dtype": "int64",
   "n_unique": 6897
  },
  {
   "name": "PAY_AMT6",
   "dtype": "int64",
   "n_unique": 6939
  },
  {
   "name": "default payment next month",
   "dtype": "int64",
   "n_unique": 2,
   "levels": [
    0,
    1
   ]
  }
 ]
}

Produce the analysis. Resolve every positional column reference to a column NAME (R-IDX-01).
List preprocessing steps in R execution order. Use the captured `terms` and `xlevels` for the model.
Output JSON schema:
{
 "$defs": {
  "ModelDef": {
   "properties": {
    "r_function": {
     "title": "R Function",
     "type": "string"
    },
    "family": {
     "title": "Family",
     "type": "string"
    },
    "link": {
     "title": "Link",
     "type": "string"
    },
    "response": {
     "title": "Response",
     "type": "string"
    },
    "formula": {
     "title": "Formula",
     "type": "string"
    },
    "terms": {
     "description": "Coefficient names exactly as R reports them",
     "items": {
      "type": "string"
     },
     "title": "Terms",
     "type": "array"
    },
    "factor_levels": {
     "additionalProperties": {
      "items": {
       "type": "string"
      },
      "type": "array"
     },
     "title": "Factor Levels",
     "type": "object"
    }
   },
   "required": [
    "r_function",
    "family",
    "link",
    "response",
    "formula",
    "terms",
    "factor_levels"
   ],
   "title": "ModelDef",
   "type": "object"
  },
  "PreprocStep": {
   "properties": {
    "order": {
     "title": "Order",
     "type": "integer"
    },
    "r_code": {
     "description": "The exact R line(s)",
     "title": "R Code",
     "type": "string"
    },
    "description": {
     "title": "Description",
     "type": "string"
    },
    "columns": {
     "description": "Column NAMES affected (positional refs resolved)",
     "items": {
      "type": "string"
     },
     "title": "Columns",
     "type": "array"
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
    "r_code",
    "description",
    "columns",
    "rule_ids"
   ],
   "title": "PreprocStep",
   "type": "object"
  }
 },
 "properties": {
  "summary": {
   "title": "Summary",
   "type": "string"
  },
  "preprocessing": {
   "items": {
    "$ref": "#/$defs/PreprocStep"
   },
   "title": "Preprocessing",
   "type": "array"
  },
  "model": {
   "$ref": "#/$defs/ModelDef"
  },
  "randomness": {
   "description": "How the data is split / randomised in R",
   "title": "Randomness",
   "type": "string"
  },
  "protected_attributes_used": {
   "items": {
    "type": "string"
   },
   "title": "Protected Attributes Used",
   "type": "array"
  },
  "risks": {
   "items": {
    "type": "string"
   },
   "title": "Risks",
   "type": "array"
  },
  "escalations": {
   "description": "Questions a human must answer before migration can continue; empty if none",
   "items": {
    "type": "string"
   },
   "title": "Escalations",
   "type": "array"
  }
 },
 "required": [
  "summary",
  "preprocessing",
  "model",
  "randomness",
  "protected_attributes_used",
  "risks",
  "escalations"
 ],
 "title": "Analysis",
 "type": "object"
}

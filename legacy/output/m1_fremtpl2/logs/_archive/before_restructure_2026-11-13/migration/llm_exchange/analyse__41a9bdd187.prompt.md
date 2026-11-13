<!-- model: claude-opus-5-5 | step: analyse | sha256: 41a9bdd187bba6daa194dad352889e6656769679e7e632bb084baf46ac1a6280 -->
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
version: "1.2"   # 1.1: R-EST-02 amended after M2 run; 1.2: row identity and NA handling (see CHANGELOG.md)
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
    python: "Keep the index as the raw file's 0-based row position (pd.read_csv gives RangeIndex 0..n-1). If R drops rows (subsetting, na.omit, filter), drop the same rows with boolean masks and keep the surviving index labels. Never sort, reset_index, deduplicate or append rows unless the R code does so explicitly."
    pitfall: "The harness matches rows by original row number (R rownames). Reordering or reset_index breaks parity; dropping a row R kept fails G-07."

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

  - id: R-NA-01
    severity: MUST
    detect: "missing values in model inputs or response (glm/lm default na.action = na.omit)"
    python: "Fit only on training rows complete in every formula variable and the response, as R does. predict() returns NaN for any row with a missing input, exactly where R predict() returns NA. Never impute, fillna or drop those rows to obtain a prediction."
    pitfall: "Imputing (e.g. fillna(0)) gives a prediction R never made; parity compares the NA pattern row by row and fails it."

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
# LEGACY MODEL (as received) - French motor TPL claim frequency, Poisson GLM "GLM1"
# Source: reconstructed from Noll, Salzmann & Wuthrich (2018), "Case Study: French
#         Motor Third-Party Liability Claims", SSRN 3164764, Listing 2 / Section 3.
#         The original Kaggle notebook could not be retrieved; every line below follows
#         the published feature engineering and model of that case study. [reconstructed]
# Data:   freMTPL2freq (OpenML 41214, CC0), as exported by OpenML (strings in single quotes).
# This file is the INPUT to the R->Python migration agent. Do not edit.
# ---------------------------------------------------------------------------

dat <- read.csv("data/fremtpl2/freMTPL2freq.csv", quote = "\"'", stringsAsFactors = FALSE)

# claim counts above 4 and exposures above one year are treated as data errors (case study, Sec. 2)
dat$ClaimNb <- pmin(dat$ClaimNb, 4)
dat$Exposure <- pmin(dat$Exposure, 1)

# --- feature pre-processing for GLM1 (case study, Sec. 3.1) ---
# Area: ordinal A..F mapped to 1..6, used as a continuous variable
dat$AreaGLM <- as.integer(factor(dat$Area))
# VehPower: categorical, levels above 9 merged into 9
dat$VehPowerGLM <- as.factor(pmin(dat$VehPower, 9))
# VehAge: three classes [0,1), [1,10], (10,inf); class 2 is the reference
VehAgeGLM <- cbind(c(0:110), c(1, rep(2, 10), rep(3, 100)))
dat$VehAgeGLM <- as.factor(VehAgeGLM[dat$VehAge + 1, 2])
dat[, "VehAgeGLM"] <- relevel(dat[, "VehAgeGLM"], ref = "2")
# DrivAge: seven age classes 18-20, 21-25, 26-30, 31-40, 41-50, 51-70, 71+; class 5 (41-50) is the reference
DrivAgeGLM <- cbind(c(18:100), c(rep(1, 21 - 18), rep(2, 26 - 21), rep(3, 31 - 26), rep(4, 41 - 31),
                                 rep(5, 51 - 41), rep(6, 71 - 51), rep(7, 101 - 71)))
dat$DrivAgeGLM <- as.factor(DrivAgeGLM[dat$DrivAge - 17, 2])
dat[, "DrivAgeGLM"] <- relevel(dat[, "DrivAgeGLM"], ref = "5")
# BonusMalus: capped at 150, continuous
dat$BonusMalusGLM <- as.integer(pmin(dat$BonusMalus, 150))
# Density: log-transformed, continuous
dat$DensityGLM <- as.numeric(log(dat$Density))
# VehBrand and Region: categorical with references B1 and R24 (largest classes)
dat$VehBrand <- relevel(as.factor(dat$VehBrand), ref = "B1")
dat$Region <- relevel(as.factor(dat$Region), ref = "R24")
dat$VehGas <- as.factor(dat$VehGas)

# --- learning / test split: 90% / 10% (case study, Sec. 2) ---
set.seed(100)
ll <- sample(c(1:nrow(dat)), round(0.9 * nrow(dat)), replace = FALSE)
learn <- dat[ll, ]
test <- dat[-ll, ]

# --- GLM1: Poisson frequency model with log-exposure offset ---
d.glm1 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand + VehGas +
                DensityGLM + Region + AreaGLM,
              data = learn, offset = log(Exposure), family = poisson())

learn$fitGLM1 <- fitted(d.glm1)
test$fitGLM1 <- predict(d.glm1, newdata = test, type = "response")

```

## Static scan facts (deterministic)
{
 "R-IO-01": "pattern /\\bread\\.csv\\(/ matched 1x",
 "R-FAC-01": "pattern /\\b(as\\.)?factor\\(/ matched 7x",
 "R-RNG-01": "pattern /set\\.seed|sample\\(|createDataPartition/ matched 2x",
 "R-EST-02": "pattern /\\bglm\\(/ matched 1x",
 "R-EST-03": "pattern /poisson|Gamma|offset\\s*=/ matched 2x",
 "R-PRED-01": "pattern /\\bpredict\\(/ matched 1x",
 "R-FAC-02": "factors in formula: ['VehPowerGLM', 'VehAgeGLM', 'DrivAgeGLM', 'VehBrand', 'VehGas', 'Region']",
 "R-IO-03": "always applies",
 "R-FAC-04": "always applies",
 "R-REC-02": "always applies",
 "GOV-01": "always applies"
}

## Dynamic facts captured by running the R script (model object metadata)
{
 "r_version": "R version 4.6.1 (2026-06-24)",
 "seed": 100,
 "packages": [
  "compiler",
  "jsonlite"
 ],
 "class": [
  "glm",
  "lm"
 ],
 "formula": "ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM +      VehBrand + VehGas + DensityGLM + Region + AreaGLM",
 "family": "poisson",
 "link": "log",
 "terms": [
  "(Intercept)",
  "VehPowerGLM5",
  "VehPowerGLM6",
  "VehPowerGLM7",
  "VehPowerGLM8",
  "VehPowerGLM9",
  "VehAgeGLM1",
  "VehAgeGLM3",
  "DrivAgeGLM1",
  "DrivAgeGLM2",
  "DrivAgeGLM3",
  "DrivAgeGLM4",
  "DrivAgeGLM6",
  "DrivAgeGLM7",
  "BonusMalusGLM",
  "VehBrandB10",
  "VehBrandB11",
  "VehBrandB12",
  "VehBrandB13",
  "VehBrandB14",
  "VehBrandB2",
  "VehBrandB3",
  "VehBrandB4",
  "VehBrandB5",
  "VehBrandB6",
  "VehGasRegular",
  "DensityGLM",
  "RegionR11",
  "RegionR21",
  "RegionR22",
  "RegionR23",
  "RegionR25",
  "RegionR26",
  "RegionR31",
  "RegionR41",
  "RegionR42",
  "RegionR43",
  "RegionR52",
  "RegionR53",
  "RegionR54",
  "RegionR72",
  "RegionR73",
  "RegionR74",
  "RegionR82",
  "RegionR83",
  "RegionR91",
  "RegionR93",
  "RegionR94",
  "AreaGLM"
 ],
 "response": "ClaimNb",
 "response_levels": {},
 "xlevels": {
  "VehPowerGLM": [
   "4",
   "5",
   "6",
   "7",
   "8",
   "9"
  ],
  "VehAgeGLM": [
   "2",
   "1",
   "3"
  ],
  "DrivAgeGLM": [
   "5",
   "1",
   "2",
   "3",
   "4",
   "6",
   "7"
  ],
  "VehBrand": [
   "B1",
   "B10",
   "B11",
   "B12",
   "B13",
   "B14",
   "B2",
   "B3",
   "B4",
   "B5",
   "B6"
  ],
  "VehGas": [
   "Diesel",
   "Regular"
  ],
  "Region": [
   "R24",
   "R11",
   "R21",
   "R22",
   "R23",
   "R25",
   "R26",
   "R31",
   "R41",
   "R42",
   "R43",
   "R52",
   "R53",
   "R54",
   "R72",
   "R73",
   "R74",
   "R82",
   "R83",
   "R91",
   "R93",
   "R94"
  ]
 },
 "contrasts": {
  "VehPowerGLM": "contr.treatment",
  "VehAgeGLM": "contr.treatment",
  "DrivAgeGLM": "contr.treatment",
  "VehBrand": "contr.treatment",
  "VehGas": "contr.treatment",
  "Region": "contr.treatment"
 },
 "n_train": 610212,
 "n_test": 67801,
 "converged": true,
 "iterations": 6,
 "null_deviance": 201966.364055248,
 "deviance": 191651.269966774,
 "aic": 254279.763151247,
 "control": {
  "epsilon": 1e-08,
  "maxit": 25
 },
 "column_classes_after_preprocessing": {
  "IDpol": "numeric",
  "ClaimNb": "numeric",
  "Exposure": "numeric",
  "Area": "character",
  "VehPower": "integer",
  "VehAge": "integer",
  "DrivAge": "integer",
  "BonusMalus": "integer",
  "VehBrand": "factor",
  "VehGas": "factor",
  "Density": "integer",
  "Region": "factor",
  "AreaGLM": "integer",
  "VehPowerGLM": "factor",
  "VehAgeGLM": "factor",
  "DrivAgeGLM": "factor",
  "BonusMalusGLM": "integer",
  "DensityGLM": "numeric",
  "fitGLM1": "numeric"
 }
}

## Raw data schema (no rows, guardrail G-11)
{
 "n_rows": 678013,
 "columns": [
  {
   "name": "IDpol",
   "dtype": "float64",
   "n_unique": 678013
  },
  {
   "name": "ClaimNb",
   "dtype": "int64",
   "n_unique": 11,
   "levels": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    11,
    16
   ]
  },
  {
   "name": "Exposure",
   "dtype": "float64",
   "n_unique": 181
  },
  {
   "name": "Area",
   "dtype": "str",
   "n_unique": 6,
   "levels": [
    "'A'",
    "'B'",
    "'C'",
    "'D'",
    "'E'",
    "'F'"
   ]
  },
  {
   "name": "VehPower",
   "dtype": "int64",
   "n_unique": 12,
   "levels": [
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15
   ]
  },
  {
   "name": "VehAge",
   "dtype": "int64",
   "n_unique": 78
  },
  {
   "name": "DrivAge",
   "dtype": "int64",
   "n_unique": 83
  },
  {
   "name": "BonusMalus",
   "dtype": "int64",
   "n_unique": 115
  },
  {
   "name": "VehBrand",
   "dtype": "str",
   "n_unique": 11,
   "levels": [
    "'B1'",
    "'B10'",
    "'B11'",
    "'B12'",
    "'B13'",
    "'B14'",
    "'B2'",
    "'B3'",
    "'B4'",
    "'B5'",
    "'B6'"
   ]
  },
  {
   "name": "VehGas",
   "dtype": "str",
   "n_unique": 2,
   "levels": [
    "'Diesel'",
    "'Regular'"
   ]
  },
  {
   "name": "Density",
   "dtype": "int64",
   "n_unique": 1607
  },
  {
   "name": "Region",
   "dtype": "str",
   "n_unique": 22
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

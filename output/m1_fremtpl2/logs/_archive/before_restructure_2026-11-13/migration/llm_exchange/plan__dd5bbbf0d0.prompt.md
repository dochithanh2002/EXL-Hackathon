<!-- model: claude-opus-5-5 | step: plan | sha256: dd5bbbf0d018a5b8945c4acb29a89ef1a1293b8491efa01d5f37c21839e1ba26 -->
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


## Analysis (from the previous step)
{
 "summary": "Poisson GLM (log link) for motor TPL claim frequency, GLM1 of the Noll/Salzmann/Wuthrich case study. The script reads the OpenML CSV with both double and single quotes as quote characters (strings such as 'D' and 'B12' arrive without quotes), caps ClaimNb at 4 and Exposure at 1, and engineers nine features: AreaGLM (integer code of Area's alphabetical factor levels A..F -> 1..6), VehPowerGLM (factor of pmin(VehPower, 9)), VehAgeGLM (lookup: VehAge 0 -> 1, 1..10 -> 2, 11..110 -> 3; reference 2), DrivAgeGLM (lookup on DrivAge-17: 18-20 -> 1, 21-25 -> 2, 26-30 -> 3, 31-40 -> 4, 41-50 -> 5, 51-70 -> 6, 71-100 -> 7; reference 5), BonusMalusGLM (pmin(BonusMalus, 150)), DensityGLM (log Density), VehBrand (reference B1), Region (reference R24), VehGas (reference Diesel). A random 90/10 split (set.seed(100), sample without replacement) gives learn/test. glm(ClaimNb ~ ..., offset = log(Exposure), family = poisson()) is fitted on learn: 49 coefficients, treatment contrasts, the offset is the log of the CAPPED exposure and is also used by predict(type='response') on newdata. No rows are dropped and there are no missing values.",
 "preprocessing": [
  {
   "order": 1,
   "r_code": "dat <- read.csv(\"data/fremtpl2/freMTPL2freq.csv\", quote = \"\\\"'\", stringsAsFactors = FALSE)",
   "description": "Read the CSV; both \" and ' are quote characters, so single-quoted string values (Area, VehBrand, VehGas, Region) lose their quotes. Header names are already syntactic (make.names leaves them unchanged). Keep all 678,013 rows in file order.",
   "columns": [
    "*"
   ],
   "rule_ids": [
    "R-IO-01",
    "R-IO-03"
   ]
  },
  {
   "order": 2,
   "r_code": "dat$ClaimNb <- pmin(dat$ClaimNb, 4)",
   "description": "Cap the response at 4.",
   "columns": [
    "ClaimNb"
   ],
   "rule_ids": [
    "R-REC-02"
   ]
  },
  {
   "order": 3,
   "r_code": "dat$Exposure <- pmin(dat$Exposure, 1)",
   "description": "Cap exposure at 1 year (this capped value feeds the offset).",
   "columns": [
    "Exposure"
   ],
   "rule_ids": [
    "R-REC-02",
    "R-EST-03"
   ]
  },
  {
   "order": 4,
   "r_code": "dat$AreaGLM <- as.integer(factor(dat$Area))",
   "description": "Integer code of Area within its sorted factor levels (A=1 .. F=6); numeric term.",
   "columns": [
    "Area",
    "AreaGLM"
   ],
   "rule_ids": [
    "R-FAC-01"
   ]
  },
  {
   "order": 5,
   "r_code": "dat$VehPowerGLM <- as.factor(pmin(dat$VehPower, 9))",
   "description": "Cap VehPower at 9 and make a factor; levels 4..9 (numeric order), reference 4.",
   "columns": [
    "VehPower",
    "VehPowerGLM"
   ],
   "rule_ids": [
    "R-FAC-01",
    "R-FAC-02"
   ]
  },
  {
   "order": 6,
   "r_code": "VehAgeGLM <- cbind(c(0:110), c(1, rep(2, 10), rep(3, 100))); dat$VehAgeGLM <- as.factor(VehAgeGLM[dat$VehAge + 1, 2]); dat[, \"VehAgeGLM\"] <- relevel(dat[, \"VehAgeGLM\"], ref = \"2\")",
   "description": "Band VehAge: 0 -> 1, 1..10 -> 2, 11..110 -> 3 (row VehAge+1 of the lookup; outside 0..110 would be NA). Factor relevelled so 2 is the reference: levels ['2','1','3'].",
   "columns": [
    "VehAge",
    "VehAgeGLM"
   ],
   "rule_ids": [
    "R-IDX-01",
    "R-FAC-01",
    "R-FAC-02"
   ]
  },
  {
   "order": 7,
   "r_code": "DrivAgeGLM <- cbind(c(18:100), c(rep(1, 3), rep(2, 5), rep(3, 5), rep(4, 10), rep(5, 10), rep(6, 20), rep(7, 30))); dat$DrivAgeGLM <- as.factor(DrivAgeGLM[dat$DrivAge - 17, 2]); relevel(ref = \"5\")",
   "description": "Band DrivAge (row DrivAge-17 of the lookup): 18-20 -> 1, 21-25 -> 2, 26-30 -> 3, 31-40 -> 4, 41-50 -> 5, 51-70 -> 6, 71-100 -> 7. Reference 5: levels ['5','1','2','3','4','6','7'].",
   "columns": [
    "DrivAge",
    "DrivAgeGLM"
   ],
   "rule_ids": [
    "R-IDX-01",
    "R-FAC-01",
    "R-FAC-02",
    "GOV-01"
   ]
  },
  {
   "order": 8,
   "r_code": "dat$BonusMalusGLM <- as.integer(pmin(dat$BonusMalus, 150))",
   "description": "Cap BonusMalus at 150; numeric term.",
   "columns": [
    "BonusMalus",
    "BonusMalusGLM"
   ],
   "rule_ids": [
    "R-REC-02"
   ]
  },
  {
   "order": 9,
   "r_code": "dat$DensityGLM <- as.numeric(log(dat$Density))",
   "description": "Natural log of Density; numeric term.",
   "columns": [
    "Density",
    "DensityGLM"
   ],
   "rule_ids": [
    "R-REC-02"
   ]
  },
  {
   "order": 10,
   "r_code": "dat$VehBrand <- relevel(as.factor(dat$VehBrand), ref = \"B1\"); dat$Region <- relevel(as.factor(dat$Region), ref = \"R24\"); dat$VehGas <- as.factor(dat$VehGas)",
   "description": "Factors with references B1, R24 and Diesel; level order as captured in xlevels (string sort, e.g. B1, B10, B11, ..., B2).",
   "columns": [
    "VehBrand",
    "Region",
    "VehGas"
   ],
   "rule_ids": [
    "R-FAC-01",
    "R-FAC-02"
   ]
  },
  {
   "order": 11,
   "r_code": "set.seed(100); ll <- sample(c(1:nrow(dat)), round(0.9 * nrow(dat)), replace = FALSE); learn <- dat[ll, ]; test <- dat[-ll, ]",
   "description": "Random 90/10 split. Not re-implemented: membership comes from the harness.",
   "columns": [
    "*"
   ],
   "rule_ids": [
    "R-RNG-01"
   ]
  }
 ],
 "model": {
  "r_function": "glm",
  "family": "poisson",
  "link": "log",
  "response": "ClaimNb",
  "formula": "ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region + AreaGLM, offset = log(Exposure)",
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
  "factor_levels": {
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
  }
 },
 "randomness": "set.seed(100) then sample() of round(0.9*n) row numbers without replacement (learn); the rest is test. Seeded, but sample()'s algorithm depends on the R version. The harness supplies the captured membership (R-RNG-01).",
 "protected_attributes_used": [
  "DrivAge (as DrivAgeGLM)"
 ],
 "risks": [
  "The offset log(Exposure) is not a formula term and is easy to drop; it must be applied both in fitting and in predict (R-EST-03).",
  "The exposure in the offset is the capped value (pmin(Exposure, 1)).",
  "Lookup-table banding is positional (VehAge+1, DrivAge-17); out-of-range ages would give NA classes in R.",
  "Factor level orders come from relevel(); VehBrand/Region level order is R string sort (B10 before B2) and must be taken from xlevels.",
  "Driver age is a protected characteristic used as a rating factor (GOV-01: migrate faithfully, flag for validation).",
  "The response is capped at 4, which changes the Poisson likelihood for the affected policies."
 ],
 "escalations": []
}

## Allowed Python packages
['numpy', 'pandas', 'statsmodels', 'scipy']

## MUST rules triggered by the static scan (each must be addressed and cited, guardrail G-13)
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

Python interface contract (the generated module MUST implement exactly this):

    TARGET: str                                   # response column name in the preprocessed frame; the column holds
                                                  # the response on R's numeric model scale (R-FAC-03 for factors)
    def load_and_preprocess(csv_path: str) -> pandas.DataFrame
        # read the raw CSV and apply EVERY R preprocessing step in R order;
        # the index must stay the raw file's 0-based row position: drop rows only where R drops them,
        # never sort, reset_index or duplicate rows (R-IO-03); the harness adds the split itself
    def fit(train: pandas.DataFrame) -> fitted_result
        # fit on the given rows only (no sampling, no splitting)
    def predict(result, df: pandas.DataFrame) -> numpy.ndarray
        # predictions on the response scale (R predict(type="response")), one per row of df;
        # NaN exactly where R predicts NA (e.g. missing inputs under na.action), never imputed
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

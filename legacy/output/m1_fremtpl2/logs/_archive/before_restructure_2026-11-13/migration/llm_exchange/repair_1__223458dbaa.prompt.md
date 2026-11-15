<!-- model: claude-opus-5-5 | step: repair_1 | sha256: 223458dbaaf6991260f53458178983656c5def7fb953accf3ccf30f717631554 -->
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

# Task: REPAIR the Python module (iteration 1)

The previous candidate failed. Diagnose the ROOT CAUSE from the evidence, change only what is
needed, and explain the change in `change_summary`. Do NOT relax tolerances, re-split data,
or change features to force a pass (that is a guardrail violation and is detected).

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

## Allowed packages
['numpy', 'pandas', 'statsmodels', 'scipy']

## Failure evidence
{
 "stage": "runtime",
 "ok": false,
 "error_type": "ValueError",
 "error": "VehBrand: levels [\"'B1'\", \"'B10'\", \"'B11'\", \"'B12'\", \"'B13'\", \"'B14'\", \"'B2'\", \"'B3'\", \"'B4'\", \"'B5'\", \"'B6'\"] not seen by the R model",
 "traceback": "Traceback (most recent call last):\n  File \"/Users/dochithanh/Documents/Ireland Master/Trinity/EXL Hackathon/r2py_agent/candidate_runner.py\", line 49, in main\n    res = mod.fit(train)\n  File \"/Users/dochithanh/Documents/Ireland Master/Trinity/EXL Hackathon/runs/M1/candidates/candidate_1.py\", line 114, in fit\n    Xdf = design_matrix(train)\n  File \"/Users/dochithanh/Documents/Ireland Master/Trinity/EXL Hackathon/runs/M1/candidates/candidate_1.py\", line 87, in design_matrix\n    raise ValueError(f\"{v}: levels {sorted(unknown)} not seen by the R model\")\nValueError: VehBrand: levels [\"'B1'\", \"'B10'\", \"'B11'\", \"'B12'\", \"'B13'\", \"'B14'\", \"'B2'\", \"'B3'\", \"'B4'\", \"'B5'\", \"'B6'\"] not seen by the R model\n"
}

## Previous candidate
```python
"""M1 French motor TPL claim frequency: Python migration of legacy R model `legacy_models/m1_fremtpl2/model.R` (GLM1).

Generated by the R->Python Migration Agent. Faithful translation (GOV-01): driver age (DrivAgeGLM) is a rating
factor exactly as in the R source and is flagged for validation.

Estimation reproduces R's stats::glm.fit IRLS path for poisson(link = "log") with an offset (R-EST-02, R-EST-03):
mustart = y + 0.1 and eta = log(mustart) WITHOUT the offset on the first pass (as glm.fit), working response
z = (eta - offset) + (y - mu) / mu, weights sqrt(mu), stopping rule |dev - dev_old| / (|dev| + 0.1) < 1e-8,
maxit = 25, standard errors from the QR of the LAST iteration's weighted design (summary.glm, dispersion 1).
The offset is log of the CAPPED exposure and is applied again in predict(), as R's predict.glm does for newdata.

Rules applied: R-IO-01 R-IO-03 R-REC-02 R-IDX-01 R-FAC-01 R-FAC-02 R-FAC-04 R-RNG-01 R-EST-02 R-EST-03 R-PRED-01 GOV-01
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TARGET = "ClaimNb"

# Factor levels captured from the R model (d.glm1$xlevels); first level = reference after relevel() (R-FAC-01/02)
FACTOR_LEVELS = {
    "VehPowerGLM": ["4", "5", "6", "7", "8", "9"],
    "VehAgeGLM": ["2", "1", "3"],
    "DrivAgeGLM": ["5", "1", "2", "3", "4", "6", "7"],
    "VehBrand": ["B1", "B10", "B11", "B12", "B13", "B14", "B2", "B3", "B4", "B5", "B6"],
    "VehGas": ["Diesel", "Regular"],
    "Region": ["R24", "R11", "R21", "R22", "R23", "R25", "R26", "R31", "R41", "R42", "R43", "R52", "R53", "R54",
               "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"],
}
# Formula order: VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region + AreaGLM
FORMULA = ["VehPowerGLM", "VehAgeGLM", "DrivAgeGLM", "BonusMalusGLM", "VehBrand", "VehGas", "DensityGLM", "Region", "AreaGLM"]
TERMS = ["(Intercept)"] + [f"{v}{l}" if v in FACTOR_LEVELS else v
                           for v in FORMULA for l in (FACTOR_LEVELS[v][1:] if v in FACTOR_LEVELS else [None])]
GLM_EPSILON, GLM_MAXIT = 1e-8, 25          # stats::glm.control() defaults

# Lookup tables from the R source (R-IDX-01): R row k <-> Python position k-1
VEH_AGE_CLASS = np.array([1] + [2] * 10 + [3] * 100)                                 # VehAge 0..110
DRIV_AGE_CLASS = np.array([1] * 3 + [2] * 5 + [3] * 5 + [4] * 10 + [5] * 10 + [6] * 20 + [7] * 30)   # DrivAge 18..100


def _make_names(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._]", ".", name)
    return out if re.match(r"^([A-Za-z]|\.(?![0-9]))", out) else "X" + out


def _lookup(table: np.ndarray, pos: pd.Series) -> pd.Series:
    """R matrix indexing: an out-of-range row gives NA."""
    p = pos.to_numpy()
    ok = (p >= 0) & (p < len(table))
    out = np.full(len(p), np.nan)
    out[ok] = table[p[ok].astype(int)]
    return pd.Series(out, index=pos.index)


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)                                            # R-IO-03: RangeIndex = raw row position
    df.columns = [_make_names(c) for c in df.columns]                     # R-IO-01 (no-op for these headers)
    for c in df.columns[df.dtypes == object]:                             # read.csv(quote = "\"'"): drop '...'
        df[c] = df[c].str.replace(r"^'(.*)'$", r"\1", regex=True)

    df["ClaimNb"] = np.minimum(df["ClaimNb"], 4).astype(float)            # R-REC-02, R order
    df["Exposure"] = np.minimum(df["Exposure"], 1.0)
    area_levels = sorted(df["Area"].dropna().unique())                    # factor(Area): sorted levels A..F
    df["AreaGLM"] = df["Area"].map({a: i + 1 for i, a in enumerate(area_levels)})
    df["VehPowerGLM"] = np.minimum(df["VehPower"], 9).astype(int).astype(str)
    df["VehAgeGLM"] = _lookup(VEH_AGE_CLASS, df["VehAge"])
    df["DrivAgeGLM"] = _lookup(DRIV_AGE_CLASS, df["DrivAge"] - 18)       # R row DrivAge - 17
    for c in ("VehAgeGLM", "DrivAgeGLM"):
        df[c] = df[c].map(lambda v: str(int(v)) if v == v else np.nan)
    df["BonusMalusGLM"] = np.minimum(df["BonusMalus"], 150).astype(int)
    df["DensityGLM"] = np.log(df["Density"].astype(float))
    return df


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Treatment-coded design with R model.matrix names and order (R-FAC-02, R-FAC-04). NaN propagates."""
    X = pd.DataFrame(index=df.index)
    X["(Intercept)"] = 1.0
    for v in FORMULA:
        if v in FACTOR_LEVELS:
            s = df[v]
            unknown = set(s.dropna().unique()) - set(FACTOR_LEVELS[v])
            if unknown:
                raise ValueError(f"{v}: levels {sorted(unknown)} not seen by the R model")
            for lvl in FACTOR_LEVELS[v][1:]:
                X[f"{v}{lvl}"] = np.where(s.isna(), np.nan, (s == lvl).astype(float))
        else:
            X[v] = df[v].astype(float)
    return X[TERMS]


def _offset(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["Exposure"].to_numpy(dtype=float))                  # offset = log(Exposure) (R-EST-03)


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.sum(ylogy - (y - mu)))


class GLMResult:
    """Fitted R-equivalent Poisson GLM (plain class so it loads under any import mechanism)."""

    def __init__(self, params, bse, deviance, iterations, converged):
        self.params, self.bse, self.deviance = params, bse, deviance
        self.iterations, self.converged = iterations, converged


def fit(train: pd.DataFrame) -> GLMResult:
    """IRLS mirroring stats::glm.fit for poisson(log) with offset, prior weights 1. Rows come from the caller (R-RNG-01)."""
    Xdf = design_matrix(train)
    off = _offset(train)
    y = train[TARGET].to_numpy(dtype=float)
    ok = Xdf.notna().all(axis=1).to_numpy() & ~np.isnan(y) & ~np.isnan(off)   # na.action = na.omit
    X, y, off = Xdf.to_numpy(dtype=float)[ok], y[ok], off[ok]
    mu = y + 0.1                                                         # poisson()$initialize
    eta = np.log(mu)                                                     # linkfun(mustart): no offset here (glm.fit)
    dev_old = _poisson_deviance(y, mu)
    converged = False
    for it in range(1, GLM_MAXIT + 1):
        mu_eta = np.maximum(np.exp(eta), np.finfo(float).eps)            # poisson()$mu.eta
        z = (eta - off) + (y - mu) / mu_eta
        w = np.sqrt(mu_eta ** 2 / mu)                                    # sqrt(weights * mu.eta^2 / variance(mu))
        Q, R = np.linalg.qr(X * w[:, None])
        coef = np.linalg.solve(R, Q.T @ (z * w))
        eta = X @ coef + off
        mu = np.exp(eta)
        dev = _poisson_deviance(y, mu)
        if abs(dev - dev_old) / (abs(dev) + 0.1) < GLM_EPSILON:
            converged = True
            break
        dev_old = dev
    R_inv = np.linalg.inv(R)                                             # summary.glm: chol2inv(R), dispersion 1
    se = np.sqrt(np.diag(R_inv @ R_inv.T))
    return GLMResult(pd.Series(coef, index=TERMS), pd.Series(se, index=TERMS), dev, it, converged)


def predict(result: GLMResult, df: pd.DataFrame) -> np.ndarray:
    """predict(glm, newdata, type = "response"): exp(X b + log(Exposure)); NaN where an input is NA (R-PRED-01)."""
    eta = design_matrix(df).to_numpy(dtype=float) @ result.params.to_numpy() + _offset(df)
    return np.exp(eta)


def coefficients(result: GLMResult) -> pd.DataFrame:
    return pd.DataFrame({"term": result.params.index, "estimate": result.params.values, "std_error": result.bse.values})

```

Output JSON schema:
{
 "properties": {
  "python_code": {
   "title": "Python Code",
   "type": "string"
  },
  "rule_ids_applied": {
   "items": {
    "type": "string"
   },
   "title": "Rule Ids Applied",
   "type": "array"
  },
  "notes": {
   "items": {
    "type": "string"
   },
   "title": "Notes",
   "type": "array"
  },
  "change_summary": {
   "default": "",
   "description": "For repairs: what changed and why",
   "title": "Change Summary",
   "type": "string"
  }
 },
 "required": [
  "python_code",
  "rule_ids_applied",
  "notes"
 ],
 "title": "CodeCandidate",
 "type": "object"
}

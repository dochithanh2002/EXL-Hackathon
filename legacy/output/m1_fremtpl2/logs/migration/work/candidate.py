"""M1 French Motor TPL claim frequency - Poisson GLM1, migrated from R.

Legacy source: input/m1_fremtpl2/model.R (object d.glm1; Noll, Salzmann & Wuthrich 2018, GLM1).
Faithful migration: every preprocessing step, feature and reference level is reproduced as written (GOV-01).

Rule ids applied:
  R-IO-01 (make.names on columns), R-IO-04 (R quote = "\"'": strip surrounding single quotes),
  R-IO-03 (index = raw 0-based row position, never sorted or reset),
  R-REC-02 (recodes in R order), R-IDX-01 (lookup-matrix column 2 resolved by position -> class code),
  R-FAC-01 / R-FAC-02 / R-FAC-04 (explicit treatment dummies from the captured R xlevels, intercept first),
  R-RNG-01 (no split here: fit() uses only the rows supplied by the harness),
  R-EST-02 / R-EST-03 (numpy IRLS mirroring glm.fit for poisson/log with offset log(Exposure)),
  R-NA-01 (na.omit for fitting, NaN predictions where an input is missing),
  R-PRED-01 (response-scale prediction with the same design function),
  GOV-01 (DrivAge, a protected attribute, is used via DrivAgeGLM exactly as in R).
"""
import re

import numpy as np
import pandas as pd

TARGET = "ClaimNb"
OFFSET_COL = "Exposure"          # offset = log(Exposure)  (R-EST-03)
DBL_EPSILON = np.finfo(float).eps

# Factor levels exactly as captured from R (model$xlevels); first level = reference (R-FAC-01, R-FAC-02)
XLEVELS = {
    "VehPowerGLM": ["4", "5", "6", "7", "8", "9"],
    "VehAgeGLM": ["2", "1", "3"],
    "DrivAgeGLM": ["5", "1", "2", "3", "4", "6", "7"],
    "VehBrand": ["B1", "B10", "B11", "B12", "B13", "B14", "B2", "B3", "B4", "B5", "B6"],
    "VehGas": ["Diesel", "Regular"],
    "Region": ["R24", "R11", "R21", "R22", "R23", "R25", "R26", "R31", "R41", "R42", "R43",
               "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"],
}

# Formula terms in R order: ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand
#                                     + VehGas + DensityGLM + Region + AreaGLM
FORMULA_VARS = ["VehPowerGLM", "VehAgeGLM", "DrivAgeGLM", "BonusMalusGLM", "VehBrand",
                "VehGas", "DensityGLM", "Region", "AreaGLM"]

TERMS = ["(Intercept)"]
for _var in FORMULA_VARS:
    if _var in XLEVELS:
        TERMS += [_var + lvl for lvl in XLEVELS[_var][1:]]
    else:
        TERMS.append(_var)

# glm.control() defaults (R-EST-02)
EPSILON = 1e-8
MAXIT = 25


# ----------------------------------------------------------------------------------------------
# Loading and preprocessing
# ----------------------------------------------------------------------------------------------
def _make_names(names):
    """R make.names(unique = TRUE) (R-IO-01)."""
    out = []
    for name in names:
        n = re.sub(r"[^A-Za-z0-9._]", ".", str(name))
        if not re.match(r"^([A-Za-z]|\.(?![0-9]))", n):
            n = "X" + n
        out.append(n)
    seen = {}
    unique = []
    for n in out:
        if n in seen:
            seen[n] += 1
            candidate = f"{n}.{seen[n]}"
            while candidate in out or candidate in unique:
                seen[n] += 1
                candidate = f"{n}.{seen[n]}"
            unique.append(candidate)
        else:
            seen[n] = 0
            unique.append(n)
    return unique


def _strip_single_quotes(s: pd.Series) -> pd.Series:
    """R read.csv(quote = "\\"'") removes surrounding single quotes; pandas only handles '"' (R-IO-04)."""
    return s.str.replace(r"^'(.*)'$", r"\1", regex=True)


def _lookup(values: pd.Series, first_key: int, codes: np.ndarray) -> pd.Series:
    """Row lookup in an R cbind(keys, codes) matrix: column 2 (the class code) of the row for `values`.
    Values outside the table give NaN (R would stop with 'subscript out of bounds')."""
    pos = values - first_key
    ok = values.notna() & (pos >= 0) & (pos < len(codes))
    out = pd.Series(np.nan, index=values.index, dtype=object)
    idx = pos[ok].astype(int).to_numpy()
    out[ok] = [str(int(c)) for c in codes[idx]]
    return out


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    # dat <- read.csv(..., quote = "\"'", stringsAsFactors = FALSE)       (R-IO-01, R-IO-04, R-IO-03)
    dat = pd.read_csv(csv_path)
    dat.columns = _make_names(dat.columns)
    for col in dat.columns:
        if pd.api.types.is_string_dtype(dat[col]):
            dat[col] = _strip_single_quotes(dat[col])

    # dat$ClaimNb <- pmin(dat$ClaimNb, 4); dat$Exposure <- pmin(dat$Exposure, 1)   (R-REC-02)
    dat["ClaimNb"] = dat["ClaimNb"].clip(upper=4).astype(float)
    dat["Exposure"] = dat["Exposure"].clip(upper=1).astype(float)

    # dat$AreaGLM <- as.integer(factor(dat$Area)): 1-based code over the sorted levels (R-FAC-01)
    area_levels = sorted(dat["Area"].dropna().unique())
    area_code = {lvl: float(i + 1) for i, lvl in enumerate(area_levels)}
    dat["AreaGLM"] = dat["Area"].map(area_code).astype(float)

    # dat$VehPowerGLM <- as.factor(pmin(dat$VehPower, 9))
    vp = dat["VehPower"].clip(upper=9)
    dat["VehPowerGLM"] = vp.map(lambda v: np.nan if pd.isna(v) else str(int(v))).astype(object)

    # VehAgeGLM <- cbind(c(0:110), c(1, rep(2, 10), rep(3, 100)))
    # dat$VehAgeGLM <- as.factor(VehAgeGLM[dat$VehAge + 1, 2]); relevel(ref = "2")   (R-IDX-01)
    veh_age_codes = np.array([1] + [2] * 10 + [3] * 100)            # keys 0..110
    dat["VehAgeGLM"] = _lookup(dat["VehAge"], 0, veh_age_codes)

    # DrivAgeGLM <- cbind(c(18:100), c(rep(1,3), rep(2,5), rep(3,5), rep(4,10), rep(5,10), rep(6,20), rep(7,30)))
    # dat$DrivAgeGLM <- as.factor(DrivAgeGLM[dat$DrivAge - 17, 2]); relevel(ref = "5")   (R-IDX-01, GOV-01)
    driv_age_codes = np.array([1] * (21 - 18) + [2] * (26 - 21) + [3] * (31 - 26) + [4] * (41 - 31)
                              + [5] * (51 - 41) + [6] * (71 - 51) + [7] * (101 - 71))   # keys 18..100
    dat["DrivAgeGLM"] = _lookup(dat["DrivAge"], 18, driv_age_codes)

    # dat$BonusMalusGLM <- as.integer(pmin(dat$BonusMalus, 150))
    dat["BonusMalusGLM"] = dat["BonusMalus"].clip(upper=150).astype(float)

    # dat$DensityGLM <- as.numeric(log(dat$Density))
    dat["DensityGLM"] = np.log(dat["Density"].astype(float))

    # VehBrand, Region, VehGas: factors; reference levels (B1, R24, Diesel) are applied in design_matrix
    # from the captured xlevels (R-FAC-01, R-FAC-02). Values are kept as their level strings.
    return dat


# ----------------------------------------------------------------------------------------------
# Design matrix (model.matrix, contr.treatment)
# ----------------------------------------------------------------------------------------------
def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model matrix as R builds it: '(Intercept)' first, treatment dummies '<var><level>' for every
    non-reference captured level, numeric terms as is. A row with a missing value or a level unknown
    to the fit is NaN in every column (R-FAC-02, R-FAC-04, R-NA-01)."""
    cols = {"(Intercept)": np.ones(len(df))}
    bad = np.zeros(len(df), dtype=bool)
    for var in FORMULA_VARS:
        if var in XLEVELS:
            s = df[var].astype(object)
            valid = s.isin(XLEVELS[var]).to_numpy()
            bad |= ~valid
            for lvl in XLEVELS[var][1:]:
                cols[var + lvl] = (s == lvl).to_numpy().astype(float)
        else:
            x = pd.to_numeric(df[var], errors="coerce").astype(float).to_numpy()
            bad |= ~np.isfinite(x)
            cols[var] = x
    X = pd.DataFrame(cols, index=df.index)[TERMS].astype(float)
    X.loc[bad, :] = np.nan
    return X


def _offset(df: pd.DataFrame) -> np.ndarray:
    return np.log(df[OFFSET_COL].astype(float).to_numpy())


# ----------------------------------------------------------------------------------------------
# Fitting: glm.fit IRLS for poisson(link = "log") with offset (R-EST-02, R-EST-03)
# ----------------------------------------------------------------------------------------------
def _poisson_dev(y, mu):
    # poisson()$dev.resids: 2 * wt * (y * log(ifelse(y == 0, 1, y / mu)) - (y - mu))
    ratio = np.where(y == 0, 1.0, y / mu)
    return float(np.sum(2.0 * (y * np.log(ratio) - (y - mu))))


def _linkinv(eta):
    # poisson log link: pmax(exp(eta), .Machine$double.eps)
    return np.maximum(np.exp(eta), DBL_EPSILON)


def fit(train: pd.DataFrame):
    X_df = design_matrix(train)
    y_s = pd.to_numeric(train[TARGET], errors="coerce").astype(float)
    off_s = pd.Series(_offset(train), index=train.index)
    # na.omit over response, formula variables and offset (R-NA-01)
    complete = X_df.notna().all(axis=1) & y_s.notna() & np.isfinite(off_s)
    X = X_df.loc[complete].to_numpy()
    y = y_s.loc[complete].to_numpy()
    offset = off_s.loc[complete].to_numpy()

    # family$initialize: mustart <- y + 0.1 ; eta <- linkfun(mustart)
    mustart = y + 0.1
    eta = np.log(mustart)
    mu = _linkinv(eta)
    devold = _poisson_dev(y, mu)
    converged = False
    beta = None
    R = None
    it = 0
    for it in range(1, MAXIT + 1):
        mu_eta = np.maximum(np.exp(eta), DBL_EPSILON)
        var_mu = mu
        z = (eta - offset) + (y - mu) / mu_eta
        w = np.sqrt(mu_eta ** 2 / var_mu)
        Q, R = np.linalg.qr(X * w[:, None], mode="reduced")
        beta = np.linalg.solve(R, Q.T @ (z * w))
        eta = X @ beta + offset
        mu = _linkinv(eta)
        dev = _poisson_dev(y, mu)
        if abs(dev - devold) / (abs(dev) + 0.1) < EPSILON:
            converged = True
            break
        devold = dev

    # summary.glm: dispersion 1 (poisson); cov = chol2inv(R) of the LAST iteration's weighted QR
    R_inv = np.linalg.solve(R, np.eye(R.shape[0]))
    cov = R_inv @ R_inv.T
    se = np.sqrt(np.diag(cov))
    return {
        "beta": beta,
        "se": se,
        "deviance": dev,
        "iterations": it,
        "converged": converged,
        "n_obs": int(complete.sum()),
    }


# ----------------------------------------------------------------------------------------------
# Prediction and coefficients
# ----------------------------------------------------------------------------------------------
def predict(result, df: pd.DataFrame) -> np.ndarray:
    """predict(d.glm1, newdata, type = "response"): exp(X beta + log(Exposure)); NaN where any input
    is missing, never imputed (R-PRED-01, R-NA-01)."""
    X = design_matrix(df).to_numpy()
    eta = X @ result["beta"] + _offset(df)
    return _linkinv(eta)


def coefficients(result) -> pd.DataFrame:
    return pd.DataFrame({"term": TERMS, "estimate": result["beta"], "std_error": result["se"]})

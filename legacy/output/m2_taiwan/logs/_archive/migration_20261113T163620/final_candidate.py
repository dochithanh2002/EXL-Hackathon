"""M2 Taiwan Card-Default Logistic PD (Traditional Score): Python migration.

Legacy source: input/m2_taiwan/model.R (R glm, object `glm.final`, binomial/logit).
Migrated faithfully, without changing features or recodes (GOV-01). Rules applied:
  R-IO-01  make.names() on column names ('default payment next month' -> 'default.payment.next.month')
  R-IO-02  lower-case column names after make.names
  R-IO-03  index = raw 0-based row position; no rows dropped, sorted or reset
  R-IDX-01 positional loop seq(6, 11) resolved to names pay_0, pay_2, pay_3, pay_4, pay_5, pay_6
  R-IDX-02 conditional assignments via df.loc[mask, col]
  R-REC-01 / R-REC-02  %in% -> isin on numeric columns, recodes in R order
  R-FAC-01 / R-FAC-02  treatment dummies built explicitly from the captured R xlevels
  R-FAC-03 response event = second level '1'
  R-FAC-04 '(Intercept)' first design column
  R-RNG-01 / R-PKG-01  no caret, no split: the harness supplies the R train rows
  R-EST-01 / R-EST-02  unpenalised IRLS mirroring stats::glm.fit (mustart, stop rule, maxit,
                       SEs from the last iteration's weighted QR as in summary.glm)
  R-PRED-01 predict on the response scale with the same design function
Protected attributes used as model inputs (GOV-01): sex, education, marriage.
"""
import re

import numpy as np
import pandas as pd

TARGET = "default.payment.next.month"

# Captured from R model$xlevels (output/m2_taiwan/model/reference/meta.json)
FACTOR_LEVELS = {
    "sex": ["1", "2"],
    "education": ["1", "2", "3", "4"],
    "marriage": ["1", "2", "3"],
}
RESPONSE_LEVELS = ["0", "1"]

# Formula right-hand side in R order
FORMULA_VARS = ["limit_bal", "sex", "education", "marriage",
                "pay_0", "pay_2", "pay_3", "pay_5", "pay_6", "pay_amt1", "pay_amt2"]

# R-IDX-01: columns 6..11 after `raw.credit$id <- NULL`
# (1 limit_bal, 2 sex, 3 education, 4 marriage, 5 age, 6 pay_0, 7 pay_2, 8 pay_3, 9 pay_4, 10 pay_5, 11 pay_6)
PAY_STATUS_COLS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]

# glm.control() defaults
EPSILON = 1e-8
MAXIT = 25

_DBL_EPS = np.finfo(float).eps
_THRESH = -np.log(_DBL_EPS)   # R's binomial logit linkinv clamp
_MTHRESH = -_THRESH


def _make_names(name: str) -> str:
    """R make.names() for these headers: invalid characters become '.'."""
    out = re.sub(r"[^A-Za-z0-9._]", ".", name)
    if not re.match(r"^([A-Za-z]|\.(?![0-9]))", out):
        out = "X" + out
    return out


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    # read.csv(...) with check.names=TRUE (R-IO-01); RangeIndex = raw row position (R-IO-03)
    df = pd.read_csv(csv_path)
    df.columns = [_make_names(c) for c in df.columns]
    # colnames(raw.credit) <- tolower(colnames(raw.credit))  (R-IO-02)
    df.columns = [c.lower() for c in df.columns]
    # raw.credit$id <- NULL
    df = df.drop(columns=["id"])

    # education: 0, 4, 5, 6 -> 4  (R-REC-01, R-IDX-02)
    df.loc[df["education"].isin([0, 4, 5, 6]), "education"] = 4
    # marriage: 0 -> 3
    df.loc[df["marriage"] == 0, "marriage"] = 3

    # for (i in seq(6, 11, 1)): -2, -1, 0 -> 0; assigning R's double 0 makes the column numeric
    for col in PAY_STATUS_COLS:
        df[col] = df[col].astype(float)
        df.loc[df[col].isin([-2.0, -1.0, 0.0]), col] = 0.0

    # as.factor(sex / education / marriage / response): levels handled in _design / _response
    return df


def _design(df: pd.DataFrame) -> pd.DataFrame:
    """model.matrix() equivalent with contr.treatment and the captured levels."""
    X = pd.DataFrame(index=df.index)
    X["(Intercept)"] = 1.0
    for var in FORMULA_VARS:
        if var in FACTOR_LEVELS:
            codes = df[var]
            missing = codes.isna()
            as_str = codes.map(lambda v: None if pd.isna(v) else str(int(v)))
            for level in FACTOR_LEVELS[var][1:]:
                col = (as_str == level).astype(float)
                col[missing] = np.nan
                X[var + level] = col
        else:
            X[var] = df[var].astype(float)
    return X


def _response(df: pd.DataFrame) -> np.ndarray:
    y = df[TARGET]
    return (y.map(lambda v: str(int(v))) != RESPONSE_LEVELS[0]).astype(float).to_numpy()


def _linkinv(eta):
    eta = np.clip(eta, _MTHRESH, _THRESH)
    tmp = np.exp(eta)
    return tmp / (1.0 + tmp)


def _mu_eta(eta):
    opexp = 1.0 + np.exp(eta)
    return np.maximum(np.exp(eta) / (opexp * opexp), _DBL_EPS)


def _ylogy(y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(y != 0, y * np.log(np.where(y != 0, y / mu, 1.0)), 0.0)


def _deviance(y, mu):
    return float(np.sum(2.0 * (_ylogy(y, mu) + _ylogy(1.0 - y, 1.0 - mu))))


def fit(train: pd.DataFrame):
    X_df = _design(train)
    # na.omit: complete cases on formula variables and response (R-NA-01)
    complete = X_df.notna().all(axis=1) & train[TARGET].notna()
    X_df = X_df.loc[complete]
    y = _response(train.loc[complete])
    X = X_df.to_numpy(dtype=float)
    names = list(X_df.columns)

    # family$initialize for binomial (prior weights 1): mustart = (y + 0.5) / 2
    mu = (y + 0.5) / 2.0
    eta = np.log(mu / (1.0 - mu))
    devold = _deviance(y, mu)
    converged = False
    beta = None
    R_last = None
    for it in range(1, MAXIT + 1):
        varmu = mu * (1.0 - mu)
        mu_eta_val = _mu_eta(eta)
        z = eta + (y - mu) / mu_eta_val
        w = np.sqrt(mu_eta_val ** 2 / varmu)
        Q, R = np.linalg.qr(X * w[:, None])
        beta = np.linalg.solve(R, Q.T @ (z * w))
        R_last = R
        eta = X @ beta
        mu = _linkinv(eta)
        dev = _deviance(y, mu)
        if abs(dev - devold) / (abs(dev) + 0.1) < EPSILON:
            converged = True
            break
        devold = dev

    # summary.glm: dispersion 1, covariance = chol2inv(R) of the last iteration's weighted QR
    R_inv = np.linalg.inv(R_last)
    cov = R_inv @ R_inv.T
    se = np.sqrt(np.diag(cov))
    return {"terms": names, "beta": beta, "se": se, "deviance": dev,
            "iterations": it, "converged": converged}


def predict(result, df: pd.DataFrame) -> np.ndarray:
    X = _design(df)[result["terms"]].to_numpy(dtype=float)
    eta = X @ result["beta"]          # NaN propagates where any input is missing (R-NA-01)
    out = _linkinv(eta)
    out[np.isnan(eta)] = np.nan
    return out


def coefficients(result) -> pd.DataFrame:
    return pd.DataFrame({"term": result["terms"],
                         "estimate": result["beta"],
                         "std_error": result["se"]})

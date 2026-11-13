"""M1 French Motor TPL claim frequency, Poisson GLM1 - Python migration.

Legacy source: input/m1_fremtpl2/model.R (R glm object `d.glm1`, fitted on `learn`).
Reproduces the R preprocessing, model.matrix and glm.fit numerical path.

Rule ids applied (agent/config/rulebook.yaml):
  R-IO-01, R-IO-03, R-IO-04  loading, quote stripping, row identity (raw 0-based row position kept)
  R-REC-02                   recodes in exactly the R order
  R-FAC-01, R-FAC-02, R-FAC-04  factor levels from the R capture (xlevels), treatment dummies, '(Intercept)'
  R-RNG-01                   no random split here: the train rows are supplied by the harness
  R-EST-02, R-EST-03         IRLS mirroring glm.fit (poisson/log, offset = log(Exposure), eps 1e-8, maxit 25)
  R-NA-01, R-PRED-01         NaN predictions where R predicts NA; response scale incl. offset
  GOV-01                     DrivAge (protected attribute) migrated faithfully, flagged for validation
"""
import numpy as np
import pandas as pd

TARGET = "ClaimNb"

# Factor levels exactly as captured from R (d.glm1$xlevels); the first level is the reference (R-FAC-01/02).
XLEVELS = {
    "VehPowerGLM": ["4", "5", "6", "7", "8", "9"],
    "VehAgeGLM": ["2", "1", "3"],
    "DrivAgeGLM": ["5", "1", "2", "3", "4", "6", "7"],
    "VehBrand": ["B1", "B10", "B11", "B12", "B13", "B14", "B2", "B3", "B4", "B5", "B6"],
    "VehGas": ["Diesel", "Regular"],
    "Region": ["R24", "R11", "R21", "R22", "R23", "R25", "R26", "R31", "R41", "R42", "R43",
               "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"],
}

# Formula terms in R order; a term is either numeric or a factor in XLEVELS.
FORMULA_TERMS = ["VehPowerGLM", "VehAgeGLM", "DrivAgeGLM", "BonusMalusGLM", "VehBrand", "VehGas",
                 "DensityGLM", "Region", "AreaGLM"]

# R coefficient names in order (d.glm1 coefficients; = design-matrix columns, R-FAC-02/R-FAC-04).
TERMS = ["(Intercept)"]
for _t in FORMULA_TERMS:
    TERMS += [f"{_t}{lev}" for lev in XLEVELS[_t][1:]] if _t in XLEVELS else [_t]

EPSILON = 1e-8   # glm.control() default (R-EST-02)
MAXIT = 25


def _make_names(name):
    """R make.names() for read.csv(check.names = TRUE) (R-IO-01)."""
    out = "".join(c if (c.isalnum() or c in "._") else "." for c in str(name))
    if not out or out[0].isdigit() or (out[0] == "." and len(out) > 1 and out[1].isdigit()):
        out = "X" + out
    return out


def _band(values, lower, upper, edges, labels):
    """Lookup-matrix recode: values outside [lower, upper] give NaN, as an out-of-range R index does."""
    v = values.to_numpy(dtype=float)
    out = np.full(v.shape, np.nan, dtype=object)
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        out[(v >= lo) & (v < hi)] = lab
    out[(v < lower) | (v > upper) | np.isnan(v)] = np.nan
    return pd.Series(out, index=values.index, dtype=object)


def load_and_preprocess(csv_path):
    # read.csv(..., quote = "\"'", stringsAsFactors = FALSE)  (R-IO-01, R-IO-03, R-IO-04)
    dat = pd.read_csv(csv_path)                       # RangeIndex = raw 0-based row position
    dat.columns = [_make_names(c) for c in dat.columns]
    for col in dat.columns:
        if pd.api.types.is_string_dtype(dat[col]) or dat[col].dtype == object:
            dat[col] = dat[col].astype("string").str.strip().str.strip("'\"").astype(object)

    # Caps, in R order (R-REC-02)
    dat["ClaimNb"] = np.minimum(dat["ClaimNb"], 4)
    dat["Exposure"] = np.minimum(dat["Exposure"], 1)

    # AreaGLM <- as.integer(factor(Area)): 1-based code in sorted level order (A..F -> 1..6)
    area_levels = sorted(dat["Area"].dropna().unique())
    dat["AreaGLM"] = dat["Area"].map({lev: i + 1 for i, lev in enumerate(area_levels)}).astype(float)

    # VehPowerGLM <- as.factor(pmin(VehPower, 9))
    dat["VehPowerGLM"] = np.minimum(dat["VehPower"], 9).astype(int).astype(str)

    # VehAgeGLM: 0 -> 1, 1..10 -> 2, 11..110 -> 3 (table covers 0..110), reference "2"
    dat["VehAgeGLM"] = _band(dat["VehAge"], 0, 110, [0, 1, 11, 111], ["1", "2", "3"])

    # DrivAgeGLM: 18-20,21-25,26-30,31-40,41-50,51-70,71-100 -> 1..7 (table covers 18..100), reference "5"
    # GOV-01: DrivAge is a protected attribute; reproduced exactly as the legacy model uses it.
    dat["DrivAgeGLM"] = _band(dat["DrivAge"], 18, 100, [18, 21, 26, 31, 41, 51, 71, 101],
                              ["1", "2", "3", "4", "5", "6", "7"])

    # BonusMalusGLM <- as.integer(pmin(BonusMalus, 150)); DensityGLM <- log(Density)
    dat["BonusMalusGLM"] = np.minimum(dat["BonusMalus"], 150).astype(float)
    dat["DensityGLM"] = np.log(dat["Density"].astype(float))

    # VehBrand (ref B1), Region (ref R24), VehGas: kept as strings, coded against XLEVELS in _design()
    return dat


def design_matrix(df):
    """model.matrix equivalent: '(Intercept)' first, then each term in formula order (R-FAC-02, R-FAC-04).

    Columns are exactly TERMS. A row with any missing or unmappable input (level not in the captured
    xlevels) is all-NaN, so that it is excluded from fitting and predicted as NaN (R-NA-01).
    """
    cols = {"(Intercept)": np.ones(len(df))}
    complete = np.ones(len(df), dtype=bool)
    for term in FORMULA_TERMS:
        if term in XLEVELS:
            s = df[term].astype(object)
            levels = XLEVELS[term]
            complete &= s.isin(levels).to_numpy()
            for lev in levels[1:]:
                cols[f"{term}{lev}"] = (s == lev).to_numpy(dtype=float)
        else:
            x = df[term].to_numpy(dtype=float)
            complete &= ~np.isnan(x)
            cols[term] = x
    X = pd.DataFrame(cols, index=df.index)[TERMS]
    X.loc[~complete, :] = np.nan
    return X


def fit(train):
    """glm(ClaimNb ~ ..., offset = log(Exposure), family = poisson()) via an IRLS mirroring glm.fit (R-EST-02/03)."""
    X_df = design_matrix(train)
    complete = ~X_df.isna().any(axis=1).to_numpy()
    y_all = train[TARGET].to_numpy(dtype=float)
    off_all = np.log(train["Exposure"].to_numpy(dtype=float))
    complete &= ~np.isnan(y_all) & ~np.isnan(off_all)          # na.omit (R-NA-01)

    X = X_df.to_numpy()[complete]
    y = y_all[complete]
    offset = off_all[complete]
    terms = list(X_df.columns)

    def deviance(mu):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(y > 0, y * np.log(y / mu), 0.0)
        return float(np.sum(2.0 * (r - (y - mu))))

    # poisson()$initialize: mustart = y + 0.1; eta = linkfun(mustart) (no offset, as glm.fit does)
    mu = y + 0.1
    eta = np.log(mu)
    dev_old = deviance(mu)
    converged = False
    beta = None
    R = None
    for it in range(1, MAXIT + 1):
        z = (eta - offset) + (y - mu) / mu             # working response (mu.eta = mu for log link)
        w = np.sqrt(mu)                                # sqrt(weights * mu.eta^2 / variance)
        Q, R = np.linalg.qr(X * w[:, None])
        beta = np.linalg.solve(R, Q.T @ (z * w))
        eta = X @ beta + offset
        mu = np.exp(eta)
        dev = deviance(mu)
        if abs(dev - dev_old) / (abs(dev) + 0.1) < EPSILON:
            converged = True
            break
        dev_old = dev

    # summary.glm: dispersion 1 for poisson; covariance from the QR of the last iteration's weighted design
    R_inv = np.linalg.solve(R, np.eye(R.shape[0]))
    cov = R_inv @ R_inv.T
    se = np.sqrt(np.diag(cov))
    return {"terms": terms, "beta": beta, "se": se, "converged": converged, "iterations": it,
            "deviance": dev}


def predict(result, df):
    """predict(d.glm1, newdata, type = 'response'), offset log(Exposure) evaluated in newdata (R-PRED-01)."""
    X_df = design_matrix(df)
    complete = ~X_df.isna().any(axis=1).to_numpy()
    offset = np.log(df["Exposure"].to_numpy(dtype=float))
    X = X_df[result["terms"]].to_numpy()
    with np.errstate(invalid="ignore", over="ignore"):
        pred = np.exp(X @ result["beta"] + offset)
    pred[~complete | np.isnan(offset)] = np.nan       # NA exactly where R predicts NA, never imputed
    return pred


def coefficients(result):
    return pd.DataFrame({"term": result["terms"], "estimate": result["beta"], "std_error": result["se"]})

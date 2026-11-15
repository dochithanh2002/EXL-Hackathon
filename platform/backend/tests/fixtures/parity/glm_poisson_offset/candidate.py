"""Known-good candidate for the Poisson GLM fixture with an exposure offset (M1-shaped claim frequency model). Each
step cites the rulebook rule it follows."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

RESPONSE = "ClaimNb"
POWER_LEVELS = [6, 4, 5, 7, 8, 9]          # R-FAC-01 then R-FAC-07: sorted levels of pmin(VehPower, 9), "6" first


def prepare(path, sets):
    df = pd.read_csv(path, keep_default_na=False, na_values=["NA"])          # R-IO-02
    df.index = pd.RangeIndex(1, len(df) + 1)                                # R-IO-03
    areas = sorted(df["Area"].unique())                                     # C-locale order of single letters
    df["AreaGLM"] = df["Area"].map({a: i + 1 for i, a in enumerate(areas)})  # R-FAC-06: 1-based level codes
    df["VehPowerGLM"] = np.minimum(df["VehPower"], 9)                       # R-VEC-01: integer literal 9 in R is a
    df["VehAgeGLM"] = np.minimum(df["VehAge"], 12).astype(float)            # double, so the result is double
    df["DrivAgeGLM"] = np.minimum(df["DrivAge"], 80).astype(float)
    df["BonusMalusGLM"] = np.trunc(np.minimum(df["BonusMalus"], 150))       # R-VEC-02: as.integer truncates
    df["DensityGLM"] = np.log(df["Density"].astype(float))                  # R-VEC-03
    return {name: df.loc[rows] for name, rows in sets.items()}


def design(frame):
    X = pd.DataFrame({"(Intercept)": 1.0}, index=frame.index)              # R-FAC-04
    for level in POWER_LEVELS[1:]:                                          # R-FAC-02
        X[f"VehPowerGLM{level}"] = (frame["VehPowerGLM"] == level).astype(float)
    for col in ["VehAgeGLM", "DrivAgeGLM", "BonusMalusGLM", "DensityGLM", "AreaGLM"]:
        X[col] = frame[col].astype(float)
    return X


class RPoisson(sm.families.Poisson):
    def starting_mu(self, y):                                               # R-EST-02: glm.fit's mustart
        return y + 0.1


def fit(train):
    return sm.GLM(train[RESPONSE].astype(float), design(train), family=RPoisson(),
                  offset=np.log(train["Exposure"])).fit()                   # R-EST-03


def predict(result, frame):
    return pd.Series(np.asarray(result.predict(design(frame), offset=np.log(frame["Exposure"]))),
                     index=frame.index)

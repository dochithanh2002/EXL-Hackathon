"""Known-good M4 candidate: a binomial GLM on the standardised Give Me Some Credit training file."""
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

RESPONSE = "SeriousDlqin2yrs"


def make_names(names):
    """R's make.names(unique = TRUE) for the names in this file."""
    out = []
    for n in names:
        n = re.sub(r"[^A-Za-z0-9._]", ".", n)
        out.append(n if re.match(r"^([A-Za-z]|\.(?![0-9]))", n) else "X" + n)
    return out


def prepare(path, sets):
    df = pd.read_csv(path, keep_default_na=False, na_values=["NA"])
    df.columns = make_names(["" if c.startswith("Unnamed") else c for c in df.columns])
    df.index = pd.RangeIndex(1, len(df) + 1)
    df = df.drop(columns=[df.columns[0], "MonthlyIncome", "NumberOfDependents"])
    df[RESPONSE] = (df[RESPONSE] != 0).astype(float)
    predictors = [c for c in df.columns if c != RESPONSE]
    df[predictors] = (df[predictors] - df[predictors].mean()) / df[predictors].std(ddof=1)
    return {name: df.loc[rows] for name, rows in sets.items()}


def design(frame):
    X = frame.drop(columns=[RESPONSE])
    X.insert(0, "(Intercept)", 1.0)
    return X


def fit(train):
    return sm.GLM(train[RESPONSE], design(train), family=sm.families.Binomial()).fit()


def predict(result, frame):
    return pd.Series(np.asarray(result.predict(design(frame))), index=frame.index)

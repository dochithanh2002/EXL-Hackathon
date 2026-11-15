"""Known-good candidate for the binomial GLM fixture (M2-shaped credit default model). Each step cites the rulebook
rule it follows."""
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

RESPONSE = "default.payment.next.month"
LEVELS = {"sex": [1, 2], "education": [1, 2, 3, 4], "marriage": [1, 2, 3]}    # R-FAC-01: numeric sort order
NUMERIC = ["limit_bal", "age", "pay_0", "bill_amt1"]


def make_names(names):
    """R-IO-01: make.names for these headers (no reserved words or duplicates occur)."""
    out = []
    for n in names:
        m = re.sub(r"[^A-Za-z0-9._]", ".", n)
        out.append(m if re.match(r"^([A-Za-z]|\.(?![0-9]))", n) else "X" + m)
    return out


def prepare(path, sets):
    df = pd.read_csv(path, keep_default_na=False, na_values=["NA"])          # R-IO-02
    df.columns = [c.lower() for c in make_names(list(df.columns))]          # R-IO-05: lower-case after make.names
    df.index = pd.RangeIndex(1, len(df) + 1)                                # R-IO-03
    df = df.drop(columns="id")                                              # R-DF-01
    df["education"] = df["education"].astype(float)                         # R-REC-01: a double assignment
    df.loc[df["education"].isin([0, 4, 5, 6]), "education"] = 4.0
    df["marriage"] = df["marriage"].astype(float)                           # R-REC-04
    df.loc[df["marriage"].eq(0) & df["marriage"].notna(), "marriage"] = 3.0
    df[RESPONSE] = (df[RESPONSE] == 1).astype(float)                        # R-FAC-03: the second level is the event
    return {name: df.loc[rows] for name, rows in sets.items()}


def design(frame):
    """R-FAC-02 / R-FAC-04: R's model.matrix columns, in the frame's column order (y ~ ., R-FAC-05)."""
    X = pd.DataFrame({"(Intercept)": 1.0}, index=frame.index)
    for col in ["limit_bal", "sex", "education", "marriage", "age", "pay_0", "bill_amt1"]:
        if col in LEVELS:
            for level in LEVELS[col][1:]:
                X[f"{col}{level}"] = (frame[col] == level).astype(float)
        else:
            X[col] = frame[col].astype(float)
    return X


def fit(train):
    return sm.GLM(train[RESPONSE], design(train), family=sm.families.Binomial()).fit()   # R-EST-01 / R-EST-02


def predict(result, frame):
    return pd.Series(np.asarray(result.predict(design(frame))), index=frame.index)        # R-PRED-01

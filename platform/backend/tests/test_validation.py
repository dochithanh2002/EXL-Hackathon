import json
import math

import pytest

from controlplane.validation import _auc, compare, kpi
from tests.test_schemas import spec as make_spec

COEFS = [("(Intercept)", 0.5, 0.1), ("x", -1.25, 0.02)]
PREDS = [(1, "train", 1.0, 0.8), (2, "train", 0.0, 0.3), (3, "train", 1.0, 0.6), (4, "test", 0.0, 0.4),
         (5, "test", 1.0, float("nan"))]
FIT = {"nobs": 3, "df_residual": 1, "deviance": 2.5, "null_deviance": 3.8, "dispersion": 1, "pearson_chi2": 2.9,
       "log_lik": -1.25, "aic": 6.5, "bic": 2.5 + 2 * math.log(3)}
PY_FIT = {"nobs": 3.0, "df_resid": 1.0, "deviance": 2.5, "null_deviance": 3.8, "scale": 1.0, "pearson_chi2": 2.9,
          "llf": -1.25}


def num(x: float) -> str:
    return "" if math.isnan(x) else repr(x)


def write(d, coefs, preds, fit, meta=None, r=True):
    d.mkdir()
    head = "term,estimate,std_error" + (",statistic,p_value" if r else "")
    (d / "coefficients.csv").write_text("\n".join([head] + [f'"{t}",{num(e)},{num(s)}' + (",0,0" if r else "")
                                                            for t, e, s in coefs]))
    head = "row,set,y,link,response" if r else "row,set,y,response"
    (d / "predictions.csv").write_text("\n".join([head] + [f'{row},"{s}",{num(y)},' + ("0," if r else "") + num(p)
                                                           for row, s, y, p in preds]))
    (d / "fit.json").write_text(json.dumps(fit))
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta))
    return d


def pair(tmp_path, coefs=COEFS, preds=PREDS, py_fit=PY_FIT, r_fit=FIT, family="binomial", link="logit",
         cls=("glm", "lm")):
    r = write(tmp_path / "r", COEFS, PREDS, r_fit, {"family": family, "link": link, "class": list(cls)})
    py = write(tmp_path / "py", coefs, preds, py_fit, r=False)
    return r, py


def checks(v) -> dict:
    return {c.name: c for c in v.checks}


def test_identical_output_passes_and_missing_values_match_missing_values(tmp_path):
    v = compare(*pair(tmp_path), make_spec(kpis=["auc", "log_loss", "actual_expected", "aic"]))
    assert v.passed, [c for c in v.checks if not c.passed]
    c = checks(v)
    assert c["fit aic rel diff"].value == 0 and c["fit bic rel diff"].value == 0     # parameters counted like R
    assert v.kpis["train"]["auc"] == {"r": 1.0, "python": 1.0}
    assert v.kpis["test"]["auc"] == {"r": None, "python": None}
    assert "auc is not defined for the test set's response; not compared" in v.notes
    assert "kpi aic train abs diff" in c


def test_each_kind_of_difference_fails_its_own_check(tmp_path):
    preds = [list(p) for p in PREDS]
    preds[0][3] = 0.8 + 1e-6                      # prediction drift
    preds[1][2] = 1.0                             # response differs
    preds[3][3] = float("nan")                    # NA where R predicts
    coefs = [("(Intercept)", 0.5, 0.1), ("x", -1.25, 0.03)]
    v = compare(*pair(tmp_path, coefs=coefs, preds=[tuple(p) for p in preds[:-1]]), make_spec(kpis=[]))
    failed = {c.name for c in v.checks if not c.passed}
    assert failed == {"std error max rel diff", "rows only on one side", "response values differing",
                      "prediction NA pattern differing", "prediction max abs diff"}
    assert v.differences["rows_only_in_r"] == [("test", 5)]
    assert v.differences["worst_terms"][0]["term"] == "x"
    assert v.differences["prediction_na_differs_rows"] == [{"set": "test", "row": 4, "r_is_na": False}]


def test_one_sided_missing_coefficient_fails_and_serialises(tmp_path):
    coefs = [("(Intercept)", 0.5, 0.1), ("x", float("nan"), float("nan"))]
    v = compare(*pair(tmp_path, coefs=coefs), make_spec(kpis=[]))
    c = checks(v)
    assert not c["coefficient max rel diff"].passed and c["coefficient max rel diff"].value == math.inf
    body = json.loads(v.model_dump_json())                   # strict JSON: non-finite values become null
    assert [x["value"] for x in body["checks"] if x["name"] == "coefficient max rel diff"] == [None]


def test_missing_python_statistic_fails_and_unmatched_likelihoods_are_not_compared(tmp_path):
    v = compare(*pair(tmp_path, py_fit={k: v for k, v in PY_FIT.items() if k != "pearson_chi2"}), make_spec(kpis=[]))
    assert checks(v)["fit pearson_chi2 rel diff"].value is None and not v.passed

    (tmp_path / "gamma").mkdir()
    v = compare(*pair(tmp_path / "gamma", family="Gamma", link="inverse"), make_spec(kpis=[]))
    assert "fit log_lik rel diff" not in checks(v)
    assert any("Gamma likelihood" in n for n in v.notes)


def test_lm_statistics_are_read_from_ols_results(tmp_path):
    r_fit = {"nobs": 3, "df_residual": 1, "rss": 0.5, "sigma": math.sqrt(0.5), "r_squared": 0.9,
             "adj_r_squared": 0.8, "log_lik": -1.0, "aic": 8.0, "bic": 2 + 3 * math.log(3)}
    py_fit = {"nobs": 3.0, "df_resid": 1.0, "ssr": 0.5, "scale": 0.5, "rsquared": 0.9, "rsquared_adj": 0.8,
              "llf": -1.0}
    v = compare(*pair(tmp_path, r_fit=r_fit, py_fit=py_fit, family="gaussian", link="identity", cls=("lm",)),
                make_spec(kpis=[]))
    assert v.passed, [c for c in v.checks if not c.passed]
    assert {"fit sigma rel diff", "fit aic rel diff", "fit r_squared rel diff"} <= set(checks(v))


def test_auc_counts_ties_as_half():
    assert _auc([(1, 0.5), (0, 0.5)]) == 0.5
    assert _auc([(1, 0.9), (0, 0.1), (1, 0.4), (0, 0.4)]) == pytest.approx(0.875)
    assert _auc([(2, 0.9), (0, 0.1)]) is None and _auc([(1, 0.9)]) is None


PAIRS = [(1.0, 0.9), (0.0, 0.1), (1.0, 0.4), (0.0, 0.4)]


@pytest.mark.parametrize("name, family, value", [
    ("auc", "binomial", 0.875),
    ("gini", "binomial", 0.75),
    ("rmse", "binomial", math.sqrt((0.01 + 0.01 + 0.36 + 0.16) / 4)),
    ("mae", "binomial", (0.1 + 0.1 + 0.6 + 0.4) / 4),
    ("actual_expected", "binomial", 2 / 1.8),
    ("log_loss", "binomial", -(math.log(0.9) + math.log(0.9) + math.log(0.4) + math.log(0.6)) / 4),
    ("deviance", "gaussian", 0.01 + 0.01 + 0.36 + 0.16),
    ("deviance", "binomial", -2 * (math.log(0.9) + math.log(0.9) + math.log(0.4) + math.log(0.6))),
    ("deviance", "quasipoisson", 2 * (math.log(1 / 0.9) + math.log(1 / 0.4) - 0.2)),
])
def test_kpis(name, family, value):
    assert kpi(name, family, PAIRS) == pytest.approx(value)

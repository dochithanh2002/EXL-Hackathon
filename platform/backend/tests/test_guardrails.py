import pytest

from controlplane import candidate
from tests.test_baseline import FIXTURES
from tests.test_translation import POISSON_PY

GOOD = POISSON_PY


def test_known_good_candidates_pass():
    assert candidate.check(GOOD) == []
    assert candidate.check((FIXTURES / "m4" / "candidate.py").read_text()) == []


@pytest.mark.parametrize("code, message", [
    ("import os", "module 'os' is not available"),
    ("import socket", "module 'socket' is not available"),
    ("import sklearn.linear_model", "module 'sklearn' is not available"),
    ("from numpy import random", "randomness"),
    ("import numpy.random as r", "randomness"),
    ("x = np.random.normal()", "randomness"),
    ("def f(df):\n    return df.sample(10)", "'sample()': randomness"),
    ("x = open('/etc/passwd').read()", "'open' is not allowed"),
    ("x = eval('1')", "'eval' is not allowed"),
    ("x = pd.eval('1')", "dynamic code"),
    ("evaluate = pd.eval", "dynamic code"),
    ("from pandas import eval as evaluate", "dynamic code"),
    ("x = pd.read_csv('/etc/passwd')", "only the dataset may be read"),
    ("reader = pd.read_csv", "dataset readers must be called directly"),
    ("from pandas import read_csv as reader\nx = reader('/etc/passwd')", "only the dataset may be read"),
    ("x = getattr(np, 'ran' + 'dom')", "'getattr' is not allowed"),
    ("x = ().__class__.__bases__", "attribute '__class__' is not allowed"),
    ("x = __builtins__", "'__builtins__' is not allowed"),
    ("x = pd.read_csv('https://example.com/x.csv')", "URLs are not allowed"),
    ("x = pd.read_pickle('x.pkl')", "only the dataset may be read"),
    ("def f(df):\n    df.to_csv('out.csv')", "writing files is not allowed"),
    ("def f(r):\n    r.save('fit.pickle')", "writing files is not allowed"),
    ("from . import helper", "relative imports are not allowed"),
    ("x = sm.datasets.webuse('auto')", "network"),
])
def test_violations_are_reported_with_their_line(code, message):
    errors = candidate.check(GOOD + "\n" + code + "\n")
    assert any(message in e for e in errors), errors
    assert all(e.startswith("line ") for e in errors)


def test_the_interface_must_be_complete():
    errors = candidate.check(GOOD.replace("def predict(result, frame)", "def forecast(result, frame)"))
    assert errors == ["the interface function 'predict' is not defined at module level"]
    assert "'fit' must take 1 positional argument(s)" in candidate.check(GOOD.replace("def fit(train)", "def fit()"))[0]


def test_dataset_reader_requires_the_prepare_argument():
    assert candidate.check(GOOD.replace('pd.read_csv(path)', 'pd.read_csv(filepath_or_buffer=path)')) == []
    assert any("only the dataset may be read" in e for e in candidate.check(
        GOOD.replace('pd.read_csv(path)', 'pd.read_csv("unrelated.csv")')))


def test_syntax_errors_stop_the_check():
    [error] = candidate.check("def prepare(:\n")
    assert error.startswith("line 1: syntax error")

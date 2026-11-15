"""The Python candidate: its fixed interface and the static guardrails it must pass before it runs.

Guardrails are evolved from `legacy/agent/tools/guardrails.py`. In local mode they are the main protection against
candidate code (PLAN.md section 6): imports limited to the stable Python environment, no dynamic code or
introspection, no file access other than reading the project's data files, no network, no randomness."""
from __future__ import annotations

import ast
import re

INTERFACE = """The candidate is one Python module that defines three functions:

prepare(path: str, sets: dict[str, pandas.Index]) -> dict[str, pandas.DataFrame]
    Read the dataset with pandas.read_csv(path), passing prepare's path argument directly, and reproduce baseline.R's
    data preparation. `sets` maps
    each set R formed ("train", and "test" if R has one) to the rows R put in it, as 1-based data-row numbers of the
    dataset file (pandas.read_csv's default index + 1). Return one frame per set, indexed by exactly those row
    numbers. Each frame holds the columns the model uses and the response column RESPONSE, encoded as the spec's
    semantics say: for a binary response 1 for the event (the spec's event coding) and 0 otherwise; for a grouped
    binomial response the proportion of successes; for class probabilities the class label as a string; otherwise
    the numeric value. Never sample, shuffle or re-split: the sets come from R.

fit(train: pandas.DataFrame) -> fitted results
    Fit the model on the train frame exactly as R does, applying the spec's weights (by type: frequency, sampling
    or precision), offset and grouped-binomial trials as R does. statsmodels results work as they are: their `params`
    and `bse`, indexed by R's coefficient names in R's order, are the coefficients compared with R, and `converged`
    reports convergence. For any other results object, also define coefficients(result) -> pandas.DataFrame with
    columns estimate and std_error indexed by R's coefficient names, and give the object a boolean `converged`
    unless the estimator is closed form. Never use scikit-learn.

predict(result, frame: pandas.DataFrame) -> pandas.Series | pandas.DataFrame
    Predictions in the spec's prediction units, matching R's r_predict expression, indexed like `frame`: a Series for
    scalar and time_indexed outputs, a DataFrame with one column per prediction key (named exactly as the spec's
    keys) for class probabilities and quantiles. NaN wherever R's prediction is NA. Never impute or drop rows to
    obtain a prediction."""

DATA_INTERFACE = """This project has other data files, which baseline.R reads next to the dataset. prepare therefore
takes a third argument, data: dict[str, str], mapping each of their file names to its path:

prepare(path: str, sets: dict[str, pandas.Index], data: dict[str, str]) -> dict[str, pandas.DataFrame]
    Read each with pandas.read_csv(data["<file name>"]) inside prepare, and join or aggregate it into the dataset
    as baseline.R does. The rows and their numbers still come from the dataset."""

CONTRACT = {"prepare": 2, "fit": 1, "predict": 2}
ENV_PACKAGES = frozenset({"numpy", "pandas", "scipy", "statsmodels", "formulaic", "patsy"})
STDLIB = frozenset({"__future__", "math", "typing", "dataclasses", "functools", "itertools", "collections", "operator",
                    "re", "warnings"})
BUILTINS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "exit", "quit", "globals",
            "locals", "vars", "getattr", "setattr", "delattr", "memoryview"}
CALLS = {
    **dict.fromkeys(["eval", "exec", "query"], "dynamic code is not allowed"),
    **dict.fromkeys(["sample", "shuffle", "permutation", "default_rng", "seed", "rvs", "train_test_split"],
                    "randomness is not allowed; the sets come from R"),
    **dict.fromkeys(["system", "popen", "spawn", "fork"], "shell access is not allowed"),
    **dict.fromkeys(["webuse", "get_rdataset", "read_html", "read_sql", "read_sql_query", "read_sql_table"],
                    "network and database access are not allowed"),
    **dict.fromkeys(["read_pickle", "read_parquet", "read_feather", "read_excel", "read_json", "read_xml",
                     "read_hdf", "read_stata", "read_sas", "read_spss", "read_orc", "read_clipboard", "load",
                     "loadtxt", "genfromtxt", "fromfile"], "only the data files may be read, with read_csv or read_table"),
    **dict.fromkeys(["to_csv", "to_pickle", "to_parquet", "to_feather", "to_excel", "to_json", "to_xml", "to_hdf",
                     "to_sql", "to_stata", "to_clipboard", "save", "savetxt", "savez", "savez_compressed", "tofile",
                     "write", "write_text", "write_bytes"], "writing files is not allowed"),
}
ALLOWED_DUNDERS = {"__init__", "__name__", "__doc__"}
URL = re.compile(r"^\s*[A-Za-z][A-Za-z0-9+.-]*://")


def check(module: str, packages: frozenset[str] = ENV_PACKAGES, data: bool = False) -> list[str]:
    """Every guardrail violation in the module, as `line N: message`; empty if it may run. With `data`, the project
    has other data files: prepare takes their paths as a third argument and may read them."""
    try:
        tree = ast.parse(module)
    except SyntaxError as e:
        return [f"line {e.lineno}: syntax error: {e.msg}"]
    errors: list[str] = []

    def add(node: ast.AST | None, msg: str) -> None:
        errors.append(f"line {node.lineno}: {msg}" if node is not None else msg)

    allowed = packages | STDLIB
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    prepare = functions.get("prepare")
    prepare_args = prepare.args.posonlyargs + prepare.args.args if prepare else []
    dataset_arg = prepare_args[0].arg if prepare_args else None
    data_arg = prepare_args[2].arg if data and len(prepare_args) > 2 else None
    prepare_nodes = set(ast.walk(prepare)) if prepare else set()
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    readers = {"read_csv", "read_table"}
    aliases = {a.asname or a.name: a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               for a in n.names}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            paths = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                add(node, "relative imports are not allowed")
            paths = [f"{node.module or ''}.{a.name}" for a in node.names]
        else:
            paths = []
        for path in paths:
            parts = path.split(".")
            if parts[0] == "sklearn":
                continue                                   # reported once below, with the reason
            if parts[0] not in allowed:
                add(node, f"module '{parts[0]}' is not available; allowed: {sorted(allowed)}")
            elif "random" in parts:
                add(node, "randomness is not allowed; the sets come from R")
            elif parts[-1] in CALLS:
                add(node, f"'{parts[-1]}()': {CALLS[parts[-1]]}")

        member = None
        if isinstance(node, ast.Attribute):
            member = node.attr
        elif isinstance(node, ast.Name):
            member = aliases.get(node.id, node.id)
        if member in CALLS and isinstance(node, ast.Attribute):
            add(node, f"'{member}()': {CALLS[member]}")
        if member in readers:
            call = parents.get(node)
            if not isinstance(call, ast.Call) or call.func is not node:
                add(node, "dataset readers must be called directly with prepare's path argument")
            else:
                path = call.args[0] if call.args else next(
                    (k.value for k in call.keywords if k.arg == "filepath_or_buffer"), None)
                dataset = isinstance(path, ast.Name) and path.id == dataset_arg
                other = (data_arg is not None and isinstance(path, ast.Subscript) and isinstance(path.value, ast.Name)
                         and path.value.id == data_arg)
                if node not in prepare_nodes or not (dataset or other):
                    add(node, "only the data files may be read; pass prepare's path argument directly"
                        + (", or data[<file name>]" if data else ""))

        if isinstance(node, ast.Name) and node.id in BUILTINS | {"__builtins__"}:
            add(node, f"'{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr == "random":
                add(node, "randomness is not allowed; the sets come from R")
            elif node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
                add(node, f"attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = aliases.get(fn.id, fn.id) if isinstance(fn, ast.Name) else None
            if name in CALLS:
                add(node, f"'{name}()': {CALLS[name]}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and URL.match(node.value):
            add(node, "URLs are not allowed")

    for node in ast.walk(tree):
        names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                 else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        if any(n.split(".")[0] == "sklearn" for n in names):
            add(node, "scikit-learn is excluded: implement the estimator with statsmodels, scipy or numpy")
    for name, nargs in (CONTRACT | {"prepare": 3} if data else CONTRACT).items():
        f = functions.get(name)
        if f is None:
            add(None, f"the interface function '{name}' is not defined at module level")
        elif len(f.args.posonlyargs) + len(f.args.args) < nargs:
            add(f, f"'{name}' must take {nargs} positional argument(s)")
    return errors

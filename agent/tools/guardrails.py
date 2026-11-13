"""Guardrails for the R->Python migration agent.

Static guardrails (run on every generated candidate BEFORE it is executed):
  G-01  imports limited to the config's allowed packages + a small stdlib allow-list
  G-02  forbidden APIs: penalised/unsuitable estimators, any RNG / re-splitting, shell, network,
        dynamic code execution, file writes
  G-03  the interface contract is implemented (functions + TARGET and TERMS constants)
  G-04  syntax is valid Python
Runtime guardrails (enforced by the agent / candidate runner):
  G-05  feature fidelity: Python coefficient names == R coefficient names (GOV-01)
  G-06  sandbox: separate interpreter (-I), stripped environment (no credentials), timeout, run dir as cwd
  G-07  row identity: load_and_preprocess keeps raw-row index labels in order and every row R used
  G-08  bounded repair loop (limits.max_repair_iterations), then escalate to a human
  G-09  human approval gate: output stays AWAITING_HUMAN_APPROVAL until `approve` is recorded
  G-10  every LLM response is validated against a typed schema
  G-11  data minimisation: the LLM sees column names, dtypes and level sets only, never data rows
  G-12  immutable inputs: config, tolerances and legacy script are hashed at intake and re-verified at the end
  G-13  plan coverage: every MUST rule triggered by the static R scan is cited in the plan
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

STDLIB_OK = {"__future__", "math", "typing", "dataclasses", "functools", "itertools", "re", "pathlib", "json", "warnings"}

FORBIDDEN_CALLS = {
    # estimators that break parity with R glm/lm (R-EST-01, R-EST-04)
    "LogisticRegression": "R-EST-01 sklearn LogisticRegression is L2-penalised by default",
    "LinearRegression": "R-EST-04 sklearn LinearRegression has no standard errors",
    "LogisticRegressionCV": "R-EST-01 penalised estimator",
    # randomness / re-splitting (R-RNG-01)
    "train_test_split": "R-RNG-01 re-randomising the split is forbidden",
    "sample": "R-RNG-01 random sampling is forbidden",
    "shuffle": "R-RNG-01 random shuffling is forbidden",
    "permutation": "R-RNG-01 random permutation is forbidden",
    "default_rng": "R-RNG-01 RNG use is forbidden",
    "seed": "R-RNG-01 RNG seeding is forbidden",
    # dynamic code / side effects
    "eval": "G-02 dynamic code execution", "exec": "G-02 dynamic code execution",
    "compile": "G-02 dynamic code execution", "__import__": "G-02 dynamic import",
    "system": "G-02 shell access", "popen": "G-02 shell access",
    "to_csv": "G-02 file writes are not allowed", "to_pickle": "G-02 file writes are not allowed",
    "get_dummies": "R-FAC-01/02 get_dummies level order differs from R; build dummies explicitly from captured levels",
}
FORBIDDEN_MODULES = {"sklearn", "os", "sys", "subprocess", "socket", "requests", "urllib", "http", "shutil",
                     "pickle", "random", "importlib", "ctypes"}
CONTRACT = {"load_and_preprocess": 1, "fit": 1, "predict": 2, "coefficients": 1, "design_matrix": 1}
CONSTANTS = ("TARGET", "TERMS")


@dataclass
class GuardrailReport:
    passed: bool
    violations: list[dict] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    def add(self, gid: str, msg: str, line: int | None = None):
        self.violations.append({"guardrail": gid, "message": msg, "line": line})
        self.passed = False


def static_check(code: str, allowed_packages: list[str]) -> GuardrailReport:
    rep = GuardrailReport(passed=True)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        rep.add("G-04", f"syntax error: {e.msg}", e.lineno)
        return rep
    rep.checks.append("G-04 syntax")

    allowed = set(allowed_packages) | STDLIB_OK
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                root = n.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    rep.add("G-02", f"forbidden module '{n}'", node.lineno)
                elif root not in allowed:
                    rep.add("G-01", f"module '{n}' not in allowed packages {sorted(allowed_packages)}", node.lineno)
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else None
            if name in FORBIDDEN_CALLS:
                rep.add("G-02", f"forbidden call '{name}()': {FORBIDDEN_CALLS[name]}", node.lineno)
            if name == "open":
                mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if any(ch in str(mode) for ch in "wax+"):
                    rep.add("G-02", "file writes are not allowed", node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == "random" and isinstance(node.value, ast.Name) \
                and node.value.id in {"np", "numpy"}:
            rep.add("G-02", "R-RNG-01 numpy.random is forbidden", node.lineno)
    rep.checks += ["G-01 imports", "G-02 forbidden APIs"]

    funcs = {n.name: len(n.args.args) for n in tree.body if isinstance(n, ast.FunctionDef)}
    for fname, nargs in CONTRACT.items():
        if fname not in funcs:
            rep.add("G-03", f"contract function '{fname}' missing")
        elif funcs[fname] < nargs:
            rep.add("G-03", f"'{fname}' must take at least {nargs} argument(s)")
    assigned = {t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
    for const in CONSTANTS:
        if const not in assigned:
            rep.add("G-03", f"module-level constant {const} missing (TARGET = response column, TERMS = R coefficient names in order)")
    rep.checks.append("G-03 interface contract")
    return rep


def plan_coverage(plan_rule_ids: set[str], triggered: dict[str, str]) -> GuardrailReport:
    """G-13: each MUST rule triggered by the static scan must be cited somewhere in the plan."""
    rep = GuardrailReport(passed=True, checks=["G-13 plan coverage"])
    for rid, why in triggered.items():
        if rid not in plan_rule_ids:
            rep.add("G-13", f"plan does not address {rid} (triggered by: {why})")
    return rep

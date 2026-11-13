# Rulebook & harness changelog

Each entry records a lesson from a real migration run, the evidence, and the change.
A rulebook change requires model-risk sign-off; runs record the rulebook hash at intake (G-12).

## v2.1 (2026-11-13), first end-to-end run driven by the agents (M2, /mrf:run flow)
The four agents ran M2 from scratch from their md files only (no access to earlier work). Result: migration PASS
in 1 iteration (coef 3e-14, SE 5e-15, PD 9e-15), documentation verify PASS in 1, validation AMBER / fairness RED,
monitoring re-validation trigger: identical to the earlier run. Issues they found, and the fixes:
- **Validator used functions outside the contract** (`TERMS`, `design_matrix`, `fit().deviance`): validation crashed
  on a valid module whose `fit()` returns a dict. Contract extended with `TERMS` + public `design_matrix(df)`
  (G-03 static and a runtime check in `candidate_runner.py`); validators recompute deviance from `predict()`
  (`champion.fitted_stats`). Red-team mutants now use a frozen fixture (`agent/tests/fixtures/`), not the latest module.
- **Documentation tools:** `doc-render` crashed when optional fields were omitted (now renders the schema-validated
  extract); `doc-verify` ignored an edited extract after a PASS (now re-verifies; counts as an iteration).
- **`mrf status`** lists `stale_results_for_older_module` when result files were produced for another module hash.
- **Migration report** trace shows only the current run; `reset` is audited under the stage it reset.
- **thresholds.yaml:** two rules still hard-coded in the M2 validator moved there (stability cap when score PSI is
  GREEN; specification linearity p / challenger AUC gap).
- **Rulebook 1.4:** R-IO-01 describes make.names exactly (keeps '_'); R-EST-02 adds glm.fit's numerical guards;
  R-NA-01 always triggers for glm/lm (na.omit is the default even without missing values today).
- Agent md files: escalation path, numeric TARGET, `--by` format, evidence = listed supporting docs only, never
  pipe tool output (hides exit codes), how to report a failed validation.

## v2.0 (2026-11-13), agents driven from Claude Code / Codex; three-folder layout
- **Entry point is a slash command**, not a Python command: the `mrf` plugin (`agent/.claude-plugin/`) exposes
  `/mrf:run`, `/mrf:new-model`, `/mrf:status`, `/mrf:approve`, `/mrf:monitor` (`agent/skills/`). Codex uses the same
  skills via `.agents/skills/` and `AGENTS.md`.
- **The user picks the model.** The repository-wide `discover` scan was removed; `/mrf:run` lists `input/` and asks.
- **One md file per agent** (`agent/agents/`): documentation, migration, validation, monitoring. Each states its
  rules and the exact tool commands it must run. The in-code LLM layer (`llm.py`, `prompts.py`, API/manual
  backends) was removed: the agent running the skill does the reasoning.
- **Stage tools are separate commands** (`agent/bin/mrf`): `intake`, `doc-facts|doc-verify|doc-render`,
  `check-analysis|check-plan|test-candidate|package`, `validate`, `monitor`, `summary`, `approve`, `reset`, `schema`.
  Iteration limits, schemas, parity and hashes are still enforced by code; exit 4 = escalated.
- **All thresholds in `agent/config/thresholds.yaml`**: parity tolerances (agent and validator), agent limits,
  documentation significance level, validation / fairness RAG, monitoring triggers, by `model_type`. Removed from
  `migration.yaml` and from the validator code; reports render trigger tables from the file. Hashed under G-12.
- **Layout:** `agent/` (agents, skills, tools, validators, config), `input/<model>/` (model.R, model.yaml replacing
  migration.yaml + inventory.yaml, data/, docs/), `output/<model>/` (logs/, model/, reports/) + `output/inventory/`,
  `output/mlflow/`. One audit trail per model (`logs/audit.jsonl`). Re-runs archive, never delete.
- Legacy R scripts: only the data path line changed (now relative to the model folder).
- Both models re-run end to end with the new tools from the agents' earlier work files: results unchanged
  (M2 test AUC 0.7402, parity 14/14; M1 test loss x100 30.9109, parity PASS). Earlier logs, including M1's failed
  first candidate, are kept in `output/<model>/logs/_archive/before_restructure_2026-11-13/`.

## v1.2 (2026-11-13), portability review on non-M2 models
Found by running the harness on three models outside M2 (lm on a continuous response, a binomial glm with
missing inputs, a script that filters rows) with correct hand-written Python. All three failed on harness defects.
- **Capture (`r_capture.R`): response written on the model's numeric scale.** It was `as.integer(...)`, which
  truncated continuous responses (22.8 became 22) and turned text factor labels into NA. Factor/character/logical
  binomial responses are now the event indicator (1 = any level except the model's first); numeric responses are
  kept at 17 digits. `meta.json` gains `response_levels`. Capture now stops if train/test rownames are not unique
  original row numbers (e.g. a tibble reset them), instead of mis-mapping rows silently.
- **G-07 changed from "every raw row" to "row identity".** The old rule made any script that drops rows
  (subset, `na.omit`, filter) impossible to migrate: keeping every row failed "rows missing from the R split",
  dropping them failed G-07. The index must now be the raw 0-based row position, unique and in order; every row
  R used must be present. Raw row count is parsed with pandas (quoted newlines are counted correctly).
- **Parity compares missing values instead of skipping them.** `max()` skipped NaN, so a candidate that imputed
  rows R predicted as NA passed the prediction check; and the deviance was NaN, so a correct candidate always
  failed. New check `prediction NA pattern: rows differing`; the max abs diff and deviance use rows predicted by both.
- **Deviance check uses the captured family** (binomial, poisson, Gamma, gaussian, inverse.gaussian, quasi-*),
  not binomial for every model. Unknown families skip the check and say so in the evidence.
- **Rulebook:** R-IO-03 rewritten for row identity; new MUST rule R-NA-01 (NA handling), triggered by missing
  values in the raw data or NA handling in the R code. The interface contract states both.
- **Cache note:** the rulebook and contract are part of every prompt, so M2's cached LLM answers no longer replay
  for a fresh run. M2's packaged module still passes parity under the new harness.

## v1.3 (2026-11-13), from run M1 (freMTPL2 Poisson GLM with exposure offset)
- **R-IO-04 added.** Iteration 1 failed at runtime: the candidate stripped OpenML's single quotes only from `dtype == object` columns, but pandas 3 reads text as `str`, so nothing was stripped (`'B12'` vs `B12`). The repair selected text columns with `is_string_dtype`; parity then passed in iteration 2 (coefficients 9e-13, SEs 7e-14, expected claims 6e-14, deviance 3e-11 on 678,013 rows).
- **Documentation agent:** D-01 now matches derived variables (DrivAgeGLM <- DrivAge); new disclosure **D-06** for capping (`pmin`/`pmax`), which M1 applies to the response (ClaimNb at 4) and the exposure.
- **Agents:** a no-op re-run no longer reports status RUNNING.

## Factory integration (2026-11-13)
- **MLflow:** `tracking.py` logs `agent_migration` and `agent_documentation` runs (parity values, iterations, LLM usage, artefacts, input hashes); `approve` tags the run with the decision. Failures are audited, never fatal.
- **Downstream uses agent artefacts only:** validation, fairness and monitoring load the packaged module through `src/m2/champion.py` (now `agent/validators/m2_taiwan/champion.py`) and record its SHA-256; the hand-written rebuild was removed. `src/m2/parity.py` is now the validator's independent re-check of the agent module.
- **Inventory index** entries are merged, not overwritten, so documentation, migration, validation and monitoring status live side by side.
- `inventory.yaml` gains human-owned `data_source` fields, shown in the model card and validation report.

## Documentation agent v1.0 (2026-11-13)
- **New:** `discover` (deterministic scan for model fits across R/SAS/Python) and `document` (verified inventory record and model card). Checks V-01..V-05; 12 red-team mutants in `tests/test_doc_verify.py`.
- **Design choice:** the LLM may not write statistics. Numbers in the card come from the captured R run; V-05 rejects any decimal or percentage not found verbatim in the evidence.
- **Fix during first run:** p-values were rounded to 8 decimals in the record, which showed marriage2 (p ≈ 1e-10) as "<1e-16". The record now keeps full precision.
- **Harness fix (agent.py, found on M2 re-run):** the Analyse prompt included `runtime_s` (R wall-clock time), so every re-run produced a new prompt hash and the LLM cache never replayed. It is now dropped from the prompt; a full re-run from scratch replays all 3 migration calls and the documentation call from cache.
- `approve` now takes `--target migration|doc` and records the SHA-256 of every output artefact.

## v1.1 (2026-11-13), from run M2
- **R-EST-02 amended.** Iteration 1 used statsmodels GLM at `tol=1e-12`. Coefficients matched R to 1e-10, but standard errors differed by 7.7e-6 relative (tolerance 1e-6). Root cause: R `glm.fit` stops at `epsilon=1e-8`, and `summary.glm` takes SEs from the last iteration's QR. The rule now requires reproducing glm.fit's path. With the amended rule, parity reached 3e-14 for coefficients and 5e-15 for SEs.
- **R-EST-01 updated** to point to R-EST-02's IRLS pattern. statsmodels is acceptable only when SE parity is not required.
- **Harness fix (candidate_runner.py).** The candidate module is now registered in `sys.modules` before execution. Iteration 2 failed because `@dataclass` with postponed annotations needs that registration. This was a harness defect, not a model error.
- **Packaging.** `requirements.txt` now pins only the packages the final module imports.

# Plan: agentic R → Python ML transformation platform

Status: build steps 1–7 built, plus the semantics and assessment revision (section 7); step 7's Docker image is written but not yet built on a Linux host (sections 12, 13). Date: 2026-11-14.

A Python control plane owns all workflow, execution and approval logic. A small Node service hosts Pi for the
four agents. A React UI is the data scientist's (DS) workspace. Everything runs locally as plain processes; a
Dockerfile is provided for Linux server deployment.

## 1. Decisions already made

| Topic | Decision |
|---|---|
| Agent framework | Pi SDK, `@earendil-works/pi-coding-agent` pinned to 1.5.2 (MIT, Node ≥ 22.19), embedded in-process in a Node service |
| Backend | Two processes: Python control plane (FastAPI) + Node agent service. MLflow server is a third process |
| Environments | One stable, pinned R environment and one stable, pinned Python environment, built once and shared by all projects (section 6) |
| Supported scope | Any R model the agents can implement faithfully with statsmodels, scipy or numpy; scikit-learn is excluded; known pairs are a reference, not an allowlist (section 7) |
| Where it runs | Local processes on macOS/Linux, no Docker needed. One Dockerfile + compose file for Linux deployment |
| Web UI libraries | No Pi web UI package: `@earendil-works/pi-web-ui` is stale (0.75.3, May 2026, pinned to pi-ai 0.75) and community ones are full coding cockpits. The chat panel is a plain component: the Analyst chat is request/reply over the control plane, which `@assistant-ui/react` would only wrap. shadcn/ui components are vendored in `frontend/src/components/ui` without Radix, as the UI needs no menus or dialogs |
| Network | Treated as working. Python dependency versions are pinned when the lockfile is first generated |

## 2. Research facts the design relies on

- Pi's `createAgentSession` accepts an explicit `ModelRuntime`, a restricted `tools` list, an in-memory
  `SettingsManager` and `SessionManager`, a custom `ResourceLoader` (own system prompt, no discovery of user
  config), and custom tools through extension factories (`pi.registerTool`). `session.abort()` cancels.
- Custom endpoints are providers with `baseUrl`, `apiKey`, `models[]` and an `api` of `openai-responses`,
  `openai-completions`, `anthropic-messages` or `google-generative-ai`. Keys can be set per request with
  `modelRuntime.setRuntimeApiKey`. A provider's own `apiKey` value is expanded (`$VAR`, and `!command` runs a
  shell command), so user-supplied keys go only through `setRuntimeApiKey`.
- Pi has no maintained Python RPC client and no permission system. Hence the Node service, and hence agents get
  no `bash`, `write` or `edit` tools.
- The verification endpoint in `EXL-Hackathon-26/downloads/openai.env` uses `wire_api = "responses"`, which maps
  to `openai-responses`.
- statsmodels' GLM IRLS reproduces R's `glm` to about 1e-12 in coefficients and standard errors when it starts from
  R's `mustart` (M4 binomial, a Poisson fixture with offset). From statsmodels' own start (Poisson: not R's),
  coefficients still match to about 1e-11 but standard errors only to about 1e-6. Rulebook R-EST-02 records this.
- Verification models under `EXL-Hackathon-26/downloads/`: M1 Poisson GLM with offset (1,126 lines of R, also
  trains keras/xgboost/catboost), M2 binomial GLM (caret split), M4 binomial GLM (57 lines), M3 random forest.

## 3. Architecture

```
React UI ──HTTP/SSE──> Control plane (Python, FastAPI)
                        ├─ workflow state machine, gates, approvals (SQLite)
                        ├─ schemas (Pydantic → JSON Schema, versioned)
                        ├─ runner: runs R / Python steps in the stable environments
                        ├─ validator: deterministic parity and KPIs
                        ├─ MLflow client ──> MLflow server (SQLite + local artifact dir)
                        └─ HTTP ──> Agent service (Node, Pi SDK)
                                     └─ agent tools call back into the control plane
```

- The agent service has one job: run role X with these inputs, stream events, return a schema-valid output. It
  holds no workflow state. Target size is about 300 lines of TypeScript.
- Model settings: the UI manages endpoints (name, base URL, API type, key, model ids), assigns a model and
  reasoning level per agent role, and has a "test connection" button. Keys are encrypted at rest, never sent back
  to the browser, and passed to Pi per request.
- One command starts all three processes locally (`./dev.sh`); `./bootstrap.sh` builds the two environments once.

## 4. Workflow (fixed; enforced by code, not by agents)

| # | State | Who acts | Gate to leave |
|---|---|---|---|
| 1 | Project created, dataset, other data files and R code uploaded | DS | files hashed (SHA-256) |
| 2 | Model Specification | Analyst agent + DS chat | DS confirms `ModelSpec` vN |
| 3 | R Reference Results | static code | run succeeds; baseline frozen |
| 4 | Python Migration & Comparison | Translator agent + static runner and validator | DS accepts or rejects the migration |
| 5 | Discrepancy Analysis (only after a failure or rejection) | Diagnostician agent, then back to 4 | retry limit (default 3), then escalate to DS |
| 6 | Migration Validation Report | Reporter agent | report built from recorded evidence |
| 7 | Code & Reproducibility Package | static | download bundle |

- Approvals are bound to hashes: spec version, baseline id, candidate code hash, validation result id. If any
  changes, the approval is void.
- The baseline, modelling intent and tolerances change only through a new `ModelSpec` version confirmed by the
  DS, which restarts from step 3 or 4.
- After a failed or cancelled baseline the DS can retry it, or write to the Analyst, which returns the project to
  step 2 to draft a corrected version. The Analyst sees R's error message, never the step log.
- Every transition and job is a database row. On restart, running jobs are marked interrupted and can be resumed
  or retried. Cancel kills the step's process group and aborts the Pi session.

## 5. Agents and schemas

Four fixed roles. Each has a short system prompt (goal, inputs, the one output schema, hard limits), read-only
tools scoped to the project, and one `submit_*` tool whose parameters are the JSON Schema. An invalid submission
is returned to the agent with the validation error, for a bounded number of retries. Every schema carries
`schema_version`.

| Role | Sees | Submits |
|---|---|---|
| Analyst | R code, column names, dtypes, level sets, summary statistics (no data rows) | `ModelSpec` v2: target model, R function, Python implementation and parity tier (or the limitation when none is faithful), family, link, formula, explicit semantics (response type, event meaning and coding, grouped-binomial trials, weight type and normalization, offset, prediction kind, quantity, units, keys, alignment keys and the R prediction expression), the financial assessment definition (business use, intended and sample population, evaluation-set roles, period column, decision threshold, criteria, missing information), preprocessing, split, evaluation metrics, `atol`/`rtol` per quantity, unsupported items, substitutions, and a minimal `baseline.R` slice |
| Translator | `ModelSpec`, `baseline.R`, baseline metadata, repair instructions if any | `Candidate`: one Python module against a fixed interface, notes, and limitations it cannot reproduce |
| Diagnostician | filtered validation (check statuses, differences in tolerance units, row ids, metrics, design comparison, assessment summaries, redacted errors), candidate, spec, DS comment, earlier attempts | `Diagnosis`: category (migration defect, specification, not reproducible, model fitness), root cause, evidence, repair instructions, assessment findings; all but a migration defect route to the DS |
| Reporter | an evidence catalogue built by code from recorded runs, validation results and approvals; the approved candidate and `baseline.R` | `ReportNarrative` v2: claims for fixed sections (scope, semantics, parity, the four assessment questions, skipped checks, limitations, acceptance); every claim cites catalogue ids; numbers are `{{id.field}}` placeholders filled in by code, and names with digits must be in backticks and known to the project |

The `baseline.R` slice is the minimal part of the uploaded code that produces the target model. The DS sees it
as a diff against the original before confirming. It is what keeps unrelated code (keras, xgboost, plotting)
out of the baseline run.

The Diagnostician never sees R's or the candidate's estimates and fit statistics (its instructions go to the
Translator, which must not be able to copy R's numbers), row-level predictions, the step log, or data values in error
messages (quoted text and decimals are redacted unless they are known names or literals in the candidate's code). A
repair attempt's Translator gets the previous candidate, the diagnosis and the DS's comment, and must submit a changed
module with `changes` filled in. When a diagnosis needs a spec change, the DS either resumes the repair on the
current spec or writes to the Analyst, which returns the project to step 2; after the retry limit the DS can raise it
and resume.

The candidate interface is fixed (`controlplane/candidate.py`): `prepare(path, sets)` returns one frame per set,
indexed by R's data-row numbers; `fit(train)` returns statsmodels results, whose `params` and `bse` are the
coefficients; `predict(result, frame)` returns response-scale predictions. The Translator sees R's coefficient names
but never R's estimates, and the candidate's working directory holds only the dataset and R's sets.

Data files: one CSV is the model's dataset, whose rows R's row ids trace back to; other CSVs (M1's claim severities,
M4's test file) are uploaded as data files. The agents see profiles of each, never rows. baseline.R reads them as
`inputs/<name>`; a candidate's `prepare` gets their paths as a third argument and may read only those. The bundle's
`verify.py` expects them next to the dataset and checks their hashes.

Prompt rule: state the goal, the inputs, the output contract and the limits; nothing else. Domain rules from
`legacy/agent/config/rulebook.yaml` are available to the Translator and Diagnostician as a readable reference, not
pasted into the prompt.

## 6. Stable environments and execution

Two environments, built once by `bootstrap.sh`, versioned, and shared by every project.

- **R environment**: one R version and one `renv.lock` resolved against a dated Posit Package Manager snapshot.
  Curated package set covering the supported scope: base `stats`, MASS, nnet, survival, lme4, nlme, mgcv,
  glmnet, quantreg, robustbase, forecast, plus data handling (dplyr, tidyr, data.table, readr, caret for its
  split helpers) and jsonlite.
- **Python environment**: one Python version and one `uv.lock` with statsmodels, pandas, numpy, scipy, formulaic
  (or patsy), all pinned.
- **Environment version**: a hash of the two lockfiles. Every run records it; the lockfiles are copied into the
  deliverables so a rerun elsewhere reproduces the environment.
- **A package outside the set**: flagged at step 2. The DS either lets the slice drop it or approves adding it,
  which creates a new environment version. An environment change never happens silently mid-project.

Each step runs as a subprocess in a fresh working directory holding copies of the code, with the data read-only,
a scrubbed environment (no credentials, no host variables), its own process group, and CPU-time, memory and
wall-clock limits.

- **Local mode limitation**: this isolates credentials and working files but is not a security boundary; a
  process could still read user-readable host files or use the network. Static guardrails on candidate code
  (import allow-list, no shell, no network, no RNG) are the main protection locally.
- **Linux deployment**: one Dockerfile builds Node, Python, R and both environments from the same lockfiles; a
  compose file adds MLflow and volumes. Steps run inside the container as a separate unprivileged user, so the
  container is the boundary to host files and credentials.

The R capture harness is static code, evolved from `legacy/agent/tools/r_capture.R`. It exports coefficients, standard
errors, fit statistics, predictions, the row ids of the train and test sets, design-matrix column names and
`sessionInfo()`. Python never re-splits or re-seeds: it consumes R's exported row ids, which guarantees the same
observations and evaluation splits.

## 7. Supported scope, semantics and validation

The support matrix (`support_matrix.yaml`) lists known R -> Python pairs. It is a reference, not an allowlist: a
known pair with its listed implementation fixes the tier; any other model is attempted at the tier the Analyst
proposes (exact or approximate) and is unproven, so the DS must accept the parity risk. When no faithful
implementation exists the Analyst sets the tier to unsupported and explains the limitation, which stops at step 2.
scikit-learn is excluded by the spec validator and the candidate guardrails. No model is silently substituted.

The spec states the statistical semantics explicitly, and code carries them through every step:
- event meaning and coding (`capture.R` refuses a coding that is not the outcome R predicts);
- grouped binomial data as successes and trials or a proportion with trials (observed proportion, trials as weight);
- weight type (frequency, sampling, precision) and normalization: frequency and sampling weights and trials
  weight the metrics, precision weights do not (deviance always uses prior weights, as R does);
- prediction kind (scalar, class probabilities, quantiles, time-indexed), quantity and units, keys and alignment
  keys. `capture.R` evaluates the spec's `r_predict` for any model class and exports predictions keyed by
  (set, row, key); lm and glm are read through `summary()`, other classes through `coef()`, `vcov()` and `logLik()`.
  A candidate may return any results object, defining `coefficients()` when it is not statsmodels.
  Time-indexed output currently means one value per existing data row; multiple horizons or a survival curve per
  subject are not represented by this demo contract.

Parity checks (`runner/parity.py`, shipped in bundles): convergence on both sides, coefficient names, estimates and
standard errors, alignment keys, observed values, prediction missingness and values, valid predictions per set, fit
statistics and the requested metrics. Each numeric comparison is `abs(python - r) <= atol + rtol * abs(r)` with
tolerances per quantity (and per metric). Missing on both sides and equal infinities match; one-sided missing or
non-finite values fail. A check is passed, failed or unavailable; an unavailable required check fails the
validation. Requested metrics are required by default; only `optional_metrics` exceptions declared per set in the
confirmed spec can leave an unavailable metric non-blocking. Finite mismatches still fail. R convergence uses
the estimator's status, or a confirmed `convergence.r_check` expression with a rationale and access to the fitted
model and fit warnings. A justified non-iterative exemption is recorded as unavailable, not passed. Generic
models use their Python estimator's AIC/BIC rather than inheriting the lm scale-parameter convention.

Financial model assessment (`assessment.py`), computed by code on R's and the candidate's predictions and recorded
with the validation without deciding it: sample suitability (rows, events, event rate, events per parameter),
stability (PSI against the development set and by period), calibration (observed/expected by decile, quantile
coverage) and decision performance (AUC, Gini, KS, and outcomes at the decision threshold). Each question is
assessed, partially assessed or not assessed, with what it needs. The design comparison (fitted rows,
design-matrix column statistics, reference levels, offset and weight summaries in R and Python) goes to the
Diagnostician. The report keeps migration parity and fitness for business use in separate sections.
Decision assessment stays partial without independent evaluation, computable discrimination and recall, and
agreed minimum AUC and event recall at the stated threshold. Missing output groups are listed as unassessed
calibration evidence rather than raising an exception. Grouped-binomial Brier scores are trial-weighted
individual-event scores, not squared errors against group proportions. These are screening summaries, not a
complete judgment of sample representativeness or business fitness.

Spec v1 records from earlier projects do not load under v2; those projects must be recreated.

## 8. MLflow

One experiment per project. Each baseline run, translation attempt, validation, diagnosis and report is a run
nested under a workflow parent run. Runs carry parameters (spec version, hashes, environment version, model
used), metrics (KPIs, maximum differences), artifacts (code, lockfiles, logs, differences, DS feedback) and
status, including FAILED with logs. Approvals and rejections are tags. The UI links to each run.

## 9. UI

React, Vite, TypeScript, Tailwind, shadcn/ui, TanStack Query, SSE for streaming.

- **Projects**: list, create, upload.
- **Project workspace**: left-hand stepper for the seven steps; main panel per step; chat docked on the right in
  steps 2 and 5.
- **Step views**: understanding card and spec editor with the `baseline.R` diff; baseline report; side-by-side R
  and Python code; differences table coloured by tolerance; attempt history; final report; deliverables.
- **Deliverables page**: Python code, both lockfiles, rerun instructions, validation results, final report, as
  individual downloads and one bundle.
- **Settings**: endpoints, per-role models, environment versions.

The spec is changed through the Analyst chat, not edited in a form, so every version passes the Analyst's checks
and the support matrix. Hash routes (`#/projects/<id>/<step>`) let the control plane serve the build as static
files; in development Vite proxies `/api` to it. Agent jobs stream their reply text and tool calls over SSE;
everything else polls the project, every second while a job runs.

## 10. Reuse and retirement

- Reused from this repo: capture harness, guardrails, parity logic, candidate interface contract, subprocess
  runner pattern, the R → Python rulebook.
- Reused from `EXL-Hackathon-26`: `mrf/rsplit.py` and the M1 parity tests as known-good fixtures.
- Already moved to `legacy/` (tag `legacy-v2.1`), to be deleted once the platform covers M1 and M2: the `agent/` plugin layout, per-model validators and the committed
  `output/` tree.

## 11. Repository layout

```
platform/
  backend/      FastAPI app: api/, workflow/, schemas/, runner/, validation/, tracking/, tests/
  agents/       Node service: server.ts, roles/ (one prompt + tool set per role), package.json (pinned)
  frontend/     React app
  envs/         r/renv.lock, python/pyproject.toml + uv.lock, support_matrix.yaml
  deploy/       Dockerfile, compose.yaml
  bootstrap.sh  dev.sh
```

## 12. Build order

1. Skeleton: layout, pinned dependencies, schemas, state machine with tests.
2. Stable environments and runner: `bootstrap.sh`, capture harness, MLflow logging; R baseline proven on M4.
3. Agent service: Pi SDK, endpoint settings, Analyst role and step 2 chat.
4. Translate and validate: Translator, Python runner, validator, approval gate.
5. Repair loop: Diagnostician, retry limit, escalation.
6. Report and deliverables.
7. UI polish, recovery and cancellation tests, Dockerfile and compose for Linux.

## 13. Verification

End to end on 2026-11-14 with `openai.env` (`gpt-6.1-sol`, Responses API) on M1–M4: every role (Analyst,
Translator, Diagnostician, Reporter) ran on the live model, but the DS's responses were scripted
(`platform/var/live-smoke/verify_e2e.py`): it nudges the Analyst once to draft a spec, nudges it once more toward a
statsmodels-covered model if it reports none, confirms the latest draft with the parity risk accepted, approves a
passing candidate, and forwards an R baseline failure to the Analyst. Rejection with a DS comment was not exercised
here. The first run surfaced two defects, both fixed:
- M1's code joins claim counts with `merge()`, which rebuilds row names, so the training rows could not be traced
  to the dataset. `capture.R` now masks `merge()` to carry the dataset side's row names through a temporary column
  that neither frame has; a row duplicated or added by the join still fails the capture. Regression tests cover
  both sides, column-name collisions and duplicating or row-adding joins.
- M4's Analyst named the unlabeled `cs-test.csv` scoring frame as the test set, which no dataset row can trace to.
  The Analyst prompt and `BaselineObjects.test` now say the test set must hold held-out rows of the dataset.

| Model | Spec | Outcome | Repairs | Minutes | Bundle `verify.py` |
|---|---|---|---|---|---|
| M1 freMTPL2 | Poisson GLM, log-exposure offset, severity counts merged in | done | 1 (coefficient 1.7e-5 > 1e-6) | 64 | exit 0, 21 checks |
| M2 Taiwan | binomial GLM | done | 0 | 30 | exit 0 |
| M3 ULB fraud | random forest | stopped at step 2, unsupported | – | 3 | – |
| M4 GiveMeSomeCredit | binomial GLM, no test set | done | 1 (dtype error in `prepare()`) | 18 | exit 0, 19 checks |

M1 and M4 are from the rerun after the fixes, M2 and M3 from the first run. Follow-ups: the bundle's `verify.py`
prints a harmless pandas `DtypeWarning` reading `split.csv` (`in_fit` mixes booleans and blanks), and M1's repair
loop is slow (about 45 minutes of model time). Evidence: `platform/var/live-smoke/e2e-20261114-024946/` and
`e2e-20261114-032333/` (ignored).

Semantics and assessment revision on 2026-11-14: 210 backend tests passed, two skipped; both TypeScript projects
typecheck and the UI builds. New tests cover the v2 spec validators, atol/rtol parity with missing and non-finite
values, unavailable checks, convergence and valid-prediction checks, weighted and grouped metrics, the assessment
statuses, the design comparison, R capture of `nnet::multinom` class probabilities, `quantreg::rq` quantiles,
grouped binomial trials and event coding, and a multinomial candidate end to end. The scripted end-to-end report was
rendered in the browser. Not yet run against a live LLM.

Review fixes on 2026-11-14: 223 backend tests passed, two skipped (the existing macOS/root exclusions), and both
TypeScript projects typecheck. Regressions cover required versus explicitly optional unavailable metrics in both
the control plane and standalone bundle, missing class calibration groups, independent decision assessment and
threshold recall criteria, grouped-binomial event-level Brier scores, and estimator-native multinomial AIC. A full
R quantreg::rq to scipy.optimize.linprog migration passes coefficient, prediction and pinball-loss checks using
a confirmed solver-specific convergence expression; an injected solver warning fails convergence. A justified
R convergence exemption remains visibly unavailable rather than passing. No live LLM run was performed for these
fixes; that smoke test remains pending. Existing v1 projects remain incompatible as noted above.

Step 7 built on 2026-11-14: 165 tests passed, two skipped (macOS address-space limits; the step-user test, which
needs root), and both TypeScript projects typecheck.
- Recovery: a step process now runs under a supervising launcher that kills its process group when the control
  plane dies. Before this, a killed control plane left R running (a `Sys.sleep` loop never hits the CPU limit)
  behind a job that restart recovery marked interrupted. `test_recovery.py` SIGKILLs a real control-plane process
  mid-baseline, checks that R exits, restarts, and checks that the job is interrupted, its MLflow run is KILLED (now
  closed on startup) and a retry succeeds. The test fails against the old launcher.
- Cancellation: through the API, a spinning candidate is killed within seconds (validation run KILLED, no
  validation recorded) and the retry passes; a cancelled Reporter reruns with the approval intact.
- After approval the spec cannot change: chat and confirm return 409 until the project is done; the state-machine
  test covers voiding when a new spec is confirmed.
- UI: built and rendered with headless Chrome against a live stack with scripted agents on M4 (a seeded `ddof`
  failure, diagnosis, repair, approval, report, deliverables). The interactive paths call endpoints covered by
  the tests above; clicking through them in a browser is still to do.
- Docker: one image (`deploy/Dockerfile`, pinned base images by digest, R packages from the same dated Posit
  snapshot's Linux binaries) runs the control plane, agent service and MLflow (`deploy/compose.yaml`). Steps run as
  user `step`: the control plane hands each step its working directory and gives ancestors traverse-only access,
  then takes the directory back so frozen outputs cannot be changed by later steps. Not yet built: no container
  runtime is set up on this Mac.

Step 6 verified on 2026-11-14: 159 tests passed, one skipped, and TypeScript typechecking passed. Scripted-model tests
cover rejected Reporter submissions (literal numbers, unknown ids and fields, uncited placeholders, unknown names in
code spans, missing citations of the approved validation and the approval), the rendered report, the bundle and its
manifest, the download routes, and a rerun of the extracted bundle's `verify.py`. A dry run on M4 with a scripted
Reporter reached step 7 done. The live Reporter smoke test (`platform/var/live-smoke/verify_step6.py`) also reached
step 7 done using `gpt-6.1-sol` through the configured Responses endpoint: the fixture candidate stands in for the
Translator, the DS approves it, and the live Reporter writes the report. Its nine claims were accepted on the
first submission. The extracted bundle's README commands (`uv sync`, then `verify.py` on `cs-training.csv`) passed
all 19 approval checks, including response values, fit statistics and KPIs. Maximum relative differences were
1.107e-12 for coefficients and 1.115e-12 for standard errors; prediction maximum absolute difference was 1.995e-13.
The bundle ships the same standard-library parity engine used by the control plane, matching coefficients by
name regardless of order. Packaging rejects lockfiles that differ from the approved environment; regression tests
cover both lockfiles, fit-statistic and KPI drift, response mismatches, reordered coefficients, unknown backtick
names and Unicode numeric literals. Evidence: `platform/var/live-smoke/20261114-014809/report.json`, `report.md`,
`verify.txt` and `readme-verification.json` (ignored).

Step 5 verified on 2026-11-14: 148 tests passed, one skipped, and TypeScript typechecking passed. The live M4 smoke
test (`platform/var/live-smoke/verify_step5.py`) passed both scenarios using `gpt-6.1-sol` through the configured
Responses endpoint. The Diagnostician identified the seeded `ddof=0` standardisation error (coefficient and
standard-error relative differences of 3.3e-6 against 1e-6 tolerances), and the Translator corrected it to `ddof=1`.
After a DS rejection of a passing candidate, the second repair switched to `smf.glm` while preserving R's coefficient
names. Each project used one diagnosis and one repair, kept spec v1 and its frozen baseline, and stopped awaiting DS
approval with all 19 checks passing. Both final candidates had maximum relative differences of 1.107e-12 for
coefficients and 1.115e-12 for standard errors; maximum prediction absolute difference was 1.995e-13. All child MLflow
runs finished, including the diagnoses. Local evidence: `platform/var/live-smoke/20261114-010319/report.json` and
`verification.json` in the same directory (ignored). Retry-limit and spec-change escalation remain covered by the
scripted tests; the live prerequisite for step 6 is complete.

Step 4 verified on 2026-11-14: 136 tests passed, one skipped (macOS address-space limits), and TypeScript
typechecking passed. Two live M4 Translator runs using `gpt-6.1-sol` through the configured Responses endpoint
passed on the first submission, including one after tightening the guardrails. Each started from the confirmed
M4 fixture spec, ran a fresh R baseline, and stopped awaiting DS approval with all 19 checks passing. Maximum
relative differences: coefficients 1.104e-12, standard errors 1.107e-12; maximum prediction absolute difference
1.955e-13. This verifies steps 3–4; the end-to-end deliverables checks below remain pending.

- Rejection and repair: reject one passing run with a DS comment, and seed one real failure (for example M2's
  factor reference levels) to exercise diagnosis.
- Robustness: kill the backend mid-run and resume; cancel a running step; change the spec after approval and
  confirm the approval is voided.
- Guardrails: a candidate that imports a disallowed package or opens a socket is rejected before it runs.
- Docker image: build and run the M4 flow once on a Linux host, including `test_runner.py` as root in the image.

## 14. Known risks

- M1's notebook is large and loads many heavy packages; the plan depends on the Analyst's `baseline.R` slice
  isolating the GLM cleanly.
- Approximate-tier pairs may not reach tolerances a DS expects; the UI must show the tier before work starts.
- Local mode is not a sandbox (section 6). In Docker, step code still has network access.
- Pi is moving quickly (1.5.2 on 10 Nov 2026); the version stays pinned and upgrades are deliberate.
- Error-message redaction (section 5) is heuristic: it keeps integers, which are mostly row ids and counts but
  could be integer data values.
- Gamma, inverse Gaussian and quasi-family likelihood statistics are excluded from fit-statistic parity because
  R and statsmodels handle scale differently; validation records the exclusion. An explicitly requested AIC KPI
  is still checked when R supplies it.

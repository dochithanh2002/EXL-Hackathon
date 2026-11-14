# Plan: agentic R → Python ML transformation platform

Status: build steps 1–7 built; step 7's Docker image is written but not yet built on a Linux host (sections 12, 13). Date: 2026-11-14.

A Python control plane owns all workflow, execution and approval logic. A small Node service hosts Pi for the
four agents. A React UI is the data scientist's (DS) workspace. Everything runs locally as plain processes; a
Dockerfile is provided for Linux server deployment.

## 1. Decisions already made

| Topic | Decision |
|---|---|
| Agent framework | Pi SDK, `@earendil-works/pi-coding-agent` pinned to 1.5.2 (MIT, Node ≥ 22.19), embedded in-process in a Node service |
| Backend | Two processes: Python control plane (FastAPI) + Node agent service. MLflow server is a third process |
| Environments | One stable, pinned R environment and one stable, pinned Python environment, built once and shared by all projects (section 6) |
| Supported scope | Any R model with a statsmodels counterpart; scikit-learn is not included (section 7) |
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
| 2 | Understanding and alignment | Analyst agent + DS chat | DS confirms `ModelSpec` vN |
| 3 | R baseline | static code | run succeeds; baseline frozen |
| 4 | Python translation and validation | Translator agent + static runner and validator | DS approves or rejects |
| 5 | Diagnosis and repair | Diagnostician agent, then back to 4 | retry limit (default 3), then escalate to DS |
| 6 | Final report | Reporter agent | report built from recorded evidence |
| 7 | Deliverables | static | download bundle |

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
| Analyst | R code, column names, dtypes, level sets, summary statistics (no data rows) | `ModelSpec`: target model, R function, statsmodels counterpart and support tier, family, link, formula, offset, weights, preprocessing steps, split definition, KPIs, tolerances, unsupported items, proposed substitutions, and a minimal `baseline.R` slice |
| Translator | `ModelSpec`, `baseline.R`, baseline metadata, repair instructions if any | `Candidate`: one Python module against a fixed interface, plus notes |
| Diagnostician | filtered validation (checks, relative differences, row ids, KPIs, redacted errors), candidate, spec, DS comment, earlier attempts | `Diagnosis`: root cause, evidence, repair instructions, and whether a spec change is needed (routes to the DS) |
| Reporter | an evidence catalogue built by code from recorded runs, validation results and approvals; the approved candidate and `baseline.R` | `ReportNarrative`: every claim cites catalogue ids; numbers are `{{id.field}}` placeholders filled in by code, and names with digits must be in backticks and known to the project |

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

## 7. Supported scope: everything statsmodels covers

A model is supported when statsmodels has a counterpart. Having a counterpart does not always mean the numbers
can match to tight tolerance, because some pairs use different algorithms. So there are three tiers, assigned by
the Analyst at step 2 and shown to the DS.

| Tier | R → statsmodels | Expectation |
|---|---|---|
| Exact | `lm` → OLS/WLS; `glm` (gaussian, binomial, poisson, Gamma, inverse.gaussian, quasi families) → GLM, with offsets and weights; `MASS::glm.nb` → NegativeBinomial; `nnet::multinom` → MNLogit; `MASS::polr` → OrderedModel; `quantreg::rq` → QuantReg; `nlme::gls` → GLS/GLSAR | Tight tolerances on coefficients, standard errors and predictions |
| Approximate (needs DS approval) | `glmnet` → `fit_regularized`; `mgcv::gam` → GLMGam; `lme4`/`nlme` mixed models → MixedLM; `MASS::rlm` → RLM; `survival::coxph` → PHReg; `arima`/`forecast` → ARIMA/SARIMAX | Different optimisers or penalties; tolerances agreed per case and the report labels the result approximate |
| Unsupported | `randomForest`, `rpart`, `xgboost`, `lightgbm`, `catboost`, `keras`, `e1071` and anything else without a statsmodels counterpart | Stops at step 2 with a clear message |

The tier table lives in one versioned file (`support_matrix.yaml`), is checked by code, and is extended as pairs
are proven. Tier assignments above are the starting point and each "Exact" pair needs a passing fixture before it
is advertised as such.

Validation checks, all deterministic: coefficient names match; coefficients, standard errors, predictions, fit
statistics and the agreed KPIs (AUC, Gini, actual/expected, RMSE and so on) are within the tolerances agreed at
step 2. Candidate code also passes the static guardrails reused from `legacy/agent/tools/guardrails.py`.

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

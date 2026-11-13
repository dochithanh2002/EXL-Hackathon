# Plan: agentic R → Python ML transformation platform

Status: build step 1 done (section 12). Date: 2026-11-13.

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
| Web UI libraries | No Pi web UI package: `@earendil-works/pi-web-ui` is stale (0.75.3, May 2026, pinned to pi-ai 0.75) and community ones are full coding cockpits. `@assistant-ui/react` is the candidate for the chat panel only |
| Network | Treated as working. Python dependency versions are pinned when the lockfile is first generated |

## 2. Research facts the design relies on

- Pi's `createAgentSession` accepts an explicit `ModelRuntime`, a restricted `tools` list, an in-memory
  `SettingsManager` and `SessionManager`, a custom `ResourceLoader` (own system prompt, no discovery of user
  config), and custom tools through extension factories (`pi.registerTool`). `session.abort()` cancels.
- Custom endpoints are providers with `baseUrl`, `apiKey`, `models[]` and an `api` of `openai-responses`,
  `openai-completions`, `anthropic-messages` or `google-generative-ai`. Keys can be set per request with
  `modelRuntime.setRuntimeApiKey`.
- Pi has no maintained Python RPC client and no permission system. Hence the Node service, and hence agents get
  no `bash`, `write` or `edit` tools.
- The verification endpoint in `EXL-Hackathon-26/downloads/openai.env` uses `wire_api = "responses"`, which maps
  to `openai-responses`.
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
- Model settings: the UI manages endpoints (name, base URL, API type, key, model ids, reasoning level), assigns a
  model per agent role, and has a "test connection" button. Keys are encrypted at rest, never sent back to the
  browser, and passed to Pi per request.
- One command starts all three processes locally (`./dev.sh`); `./bootstrap.sh` builds the two environments once.

## 4. Workflow (fixed; enforced by code, not by agents)

| # | State | Who acts | Gate to leave |
|---|---|---|---|
| 1 | Project created, dataset and R code uploaded | DS | files hashed (SHA-256) |
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
| Diagnostician | validation differences, candidate, spec, DS comment | `Diagnosis`: root cause, evidence, repair instructions, and whether a spec change is needed (routes to the DS) |
| Reporter | recorded runs, validation results, approvals | `ReportNarrative`: every claim cites an evidence id; numbers are injected by code |

The `baseline.R` slice is the minimal part of the uploaded code that produces the target model. The DS sees it
as a diff against the original before confirming. It is what keeps unrelated code (keras, xgboost, plotting)
out of the baseline run.

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

- End to end with `openai.env` on M4, M2 and M1, each through to deliverables.
- Rejection and repair: reject one passing run with a DS comment, and seed one real failure (for example M2's
  factor reference levels) to exercise diagnosis.
- Unsupported path: M3 (random forest) must stop at step 2 with a clear message.
- Robustness: kill the backend mid-run and resume; cancel a running step; change the spec after approval and
  confirm the approval is voided.
- Guardrails: a candidate that imports a disallowed package or opens a socket is rejected before it runs.
- Docker image: build and run the M4 flow once on a Linux host. This cannot be checked on this Mac.

## 14. Known risks

- M1's notebook is large and loads many heavy packages; the plan depends on the Analyst's `baseline.R` slice
  isolating the GLM cleanly.
- Approximate-tier pairs may not reach tolerances a DS expects; the UI must show the tier before work starts.
- Local mode is not a sandbox (section 6).
- Pi is moving quickly (1.5.2 on 10 Nov 2026); the version stays pinned and upgrades are deliberate.

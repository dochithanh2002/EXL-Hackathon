# ML transformation platform (R → Python)

An agentic platform that takes a data scientist's dataset and R model code, establishes an R baseline,
translates the model to Python (statsmodels), validates it against the baseline within agreed tolerances, and
delivers the code, environments, validation results and a final report.

Status: build steps 1–3 of [`PLAN.md`](PLAN.md) (skeleton, schemas, workflow state machine; stable
environments, step runner, R capture harness, MLflow tracking; agent service, model endpoints, Analyst and the
step 2 chat API). There is no UI yet.

```sh
./platform/bootstrap.sh                  # build the R and Python environments from their lockfiles (once)
cd platform/backend && uv run pytest     # tests; the M4 baseline test reads M4_DIR
./platform/dev.sh                        # MLflow on :5000, agent service on :8100, control plane on :8000
```

Before the Analyst can run, add a model endpoint (`POST /api/settings/endpoints`) and assign it to the role
(`PUT /api/settings/roles/analyst`); keys are encrypted with `platform/var/secret.key`, or with
`CONTROLPLANE_SECRET_KEY` if set.

Bootstrap needs R 4.6.2, uv, Node 22.19 or later and a C, C++ and Fortran toolchain. R packages come from the Posit Package
Manager snapshot recorded in `platform/envs/r/renv.lock`; `./platform/bootstrap.sh relock <date>` re-resolves it.

The previous Agentic Model Risk Factory is in [`legacy/`](legacy/README.md) as read-only reference; its working
layout is at git tag `legacy-v2.1`.

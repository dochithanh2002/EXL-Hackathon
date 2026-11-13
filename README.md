# ML transformation platform (R → Python)

An agentic platform that takes a data scientist's dataset and R model code, establishes an R baseline,
translates the model to Python (statsmodels), validates it against the baseline within agreed tolerances, and
delivers the code, environments, validation results and a final report.

Status: build steps 1–2 of [`PLAN.md`](PLAN.md) (skeleton, schemas, workflow state machine; stable
environments, step runner, R capture harness, MLflow tracking).

```sh
./platform/bootstrap.sh                  # build the R and Python environments from their lockfiles (once)
cd platform/backend && uv run pytest     # tests; the M4 baseline test reads M4_DIR
./platform/dev.sh                        # MLflow on :5000, control plane on :8000
```

Bootstrap needs R 4.6.2, uv, Node and a C, C++ and Fortran toolchain. R packages come from the Posit Package
Manager snapshot recorded in `platform/envs/r/renv.lock`; `./platform/bootstrap.sh relock <date>` re-resolves it.

The previous Agentic Model Risk Factory is in [`legacy/`](legacy/README.md) as read-only reference; its working
layout is at git tag `legacy-v2.1`.

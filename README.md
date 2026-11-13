# ML transformation platform (R → Python)

An agentic platform that takes a data scientist's dataset and R model code, establishes an R baseline,
translates the model to Python (statsmodels), validates it against the baseline within agreed tolerances, and
delivers the code, environments, validation results and a final report.

Status: build step 1 of [`PLAN.md`](PLAN.md) (skeleton, schemas, workflow state machine).

```sh
cd platform/backend && uv run pytest     # tests
./platform/dev.sh                        # control plane on http://127.0.0.1:8000
```

The previous Agentic Model Risk Factory is in [`legacy/`](legacy/README.md) as read-only reference; its working
layout is at git tag `legacy-v2.1`.

# ML transformation platform (R → Python)

An agentic platform that takes a data scientist's dataset and R model code, establishes an R baseline,
translates the model to Python (statsmodels), validates it against the baseline within agreed tolerances, and
delivers the code, environments, validation results and a final report.

Status: all seven build steps of [`PLAN.md`](PLAN.md) are built: the workflow, stable R and Python environments,
the four agents, validation, repair, the final report and deliverables, the React UI, restart recovery and
cancellation, and a Dockerfile and compose file for Linux.

```sh
./platform/bootstrap.sh                  # build the environments, the agent service and the UI (once)
cd platform/backend && uv run pytest     # tests; the M4 tests read M4_DIR
./platform/dev.sh                        # UI on :5173; MLflow :5000, agent service :8100, control plane :8000
```

In the UI, open Settings, add a model endpoint and assign a model to each agent role. Keys are encrypted with
`platform/var/secret.key`, or with `CONTROLPLANE_SECRET_KEY` if set. The control plane also serves the built UI
(`platform/frontend/dist`) on :8000.

Bootstrap needs R 4.6.2, uv, Node 22.19 or later and a C, C++ and Fortran toolchain. R packages come from the Posit Package
Manager snapshot recorded in `platform/envs/r/renv.lock`; `./platform/bootstrap.sh relock <date>` re-resolves it.

On Linux, `docker compose -f deploy/compose.yaml up --build` (from `platform/`) runs the same stack in one image,
with steps executed as a separate unprivileged user. Ports are published on 127.0.0.1 only: the API has no
authentication.

To serve it over HTTPS, point a DNS name at the server and make ports 80 and 443 reachable. Set `R2PY_DOMAIN`
to that DNS name and `R2PY_AUTH_FILE` to the absolute path of a private file outside the checkout. Put one
`username bcrypt_hash` pair per line in that file. Generate hashes with
`docker run --rm -it caddy:2.11.9-alpine caddy hash-password`. Then run from `platform/`:

```sh
docker compose -f deploy/compose.yaml -f deploy/compose.caddy.yaml up -d --build --wait
```

Caddy obtains the TLS certificate and requires the same login for the UI, API, and MLflow at `/mlflow/`.
The web UI links to project experiments and runs there. Authorized users share the same projects and settings.

The previous Agentic Model Risk Factory is in [`legacy/`](legacy/README.md) as read-only reference; its working
layout is at git tag `legacy-v2.1`.

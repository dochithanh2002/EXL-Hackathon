# One image for the control plane (with the UI), the agent service and MLflow, built from the same lockfiles as a
# local install. Build from platform/:  docker build -f deploy/Dockerfile -t r2py .   (or use compose.yaml)
FROM node:26.9.0-bookworm-slim@sha256:582460f614631b59b824ac6020533b9bf339c7fdf3a6d7db31abb6b4065f0212 AS node
FROM ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded AS uv

# R 4.6.2, the version renv.lock needs
FROM rocker/r-ver:4.6.2@sha256:66265ad70add9a7bc2bb6580f49d7328f7b98a12d5eb631197595e377b4edebf
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gfortran libcurl4-openssl-dev libssl-dev libxml2-dev zlib1g-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

# Steps run as `step`; the agent service and MLflow get their own users. The control plane runs as root with only
# the capabilities it needs to hand a step its working directory and start it as `step` (compose.yaml).
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin step \
    && useradd --system --uid 10002 --no-create-home --shell /usr/sbin/nologin agents \
    && useradd --system --uid 10003 --no-create-home --shell /usr/sbin/nologin mlflow \
    && mkdir -m 700 /data && mkdir -m 700 /mlflow && chown mlflow:mlflow /mlflow

# uv-managed Python lives outside /root so `step` can run the Python environment
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /opt/platform
COPY . .
RUN . /etc/os-release \
    && snapshot=$(grep -o '"URL": *"[^"]*"' envs/r/renv.lock | head -1 | sed 's|.*/||; s|"$||') \
    && R_REPO="https://packagemanager.posit.co/cran/__linux__/${VERSION_CODENAME}/${snapshot}" ./bootstrap.sh \
    && rm -rf /tmp/uv-cache /root/.npm /root/.cache \
    && chmod -R a+rX /opt/platform /opt/uv-python

EXPOSE 8000 8100 5000

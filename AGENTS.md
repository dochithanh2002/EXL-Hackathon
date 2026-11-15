# Agent instructions

This repository is being rebuilt as an agentic R → Python ML transformation platform. The plan is `PLAN.md`;
follow it.

- `platform/` (to be created): the new platform — `backend/` (Python, FastAPI), `agents/` (Node, Pi SDK),
  `frontend/` (React), `envs/` (pinned R and Python environments), `deploy/`.
- `legacy/`: the previous Agentic Model Risk Factory, kept as read-only reference for code to port (capture
  harness, guardrails, parity logic, rulebook). It does not run from this location; the working layout is at
  git tag `legacy-v2.1`. Ignore `legacy/AGENTS.md` and `legacy/CLAUDE.md` unless working on that tag.

Rules:
1. Do not edit `legacy/`; copy what you need into `platform/`.
2. Pin every dependency version.
3. Never commit credentials or `*.env` files, and never read dataset rows into an agent prompt.
4. Avoid unneccessary secondary labels and descriptions. Let the layout do the explaining.
5. REMEMBER: You are the owner and long-term maintainer of this codebase.

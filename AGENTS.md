# Agent instructions (Codex, Claude Code and other coding agents)

This repository is the **Agentic Model Risk Factory**: it documents, migrates (R → Python with parity proof),
validates and monitors legacy models under PRA SS1/23.

- `agent/`  the agents (`agent/agents/*.md`), the skills users invoke (`agent/skills/*/SKILL.md`), deterministic
  tools (`agent/bin/mrf`), validators, and **all thresholds** (`agent/config/thresholds.yaml`)
- `input/`  one folder per legacy model (`model.R`, `model.yaml`, `data/`, `docs/`)
- `output/` one folder per model: `logs/`, `model/`, `reports/`

## How to work here
1. When the user asks to run, document, migrate, validate or monitor a model, follow
   `agent/skills/run/SKILL.md`. Ask the user which model in `input/` to use; never scan the whole repository.
2. Each stage is done by following the matching agent file in `agent/agents/`. If you cannot spawn subagents, do
   the stages yourself, in order, exactly as those files say.
3. Every check is a command: `agent/bin/mrf <command> <model>` (`agent/bin/mrf --help`). Exit 1 = fix and retry;
   exit 4 = ESCALATED, stop and ask the user.
4. Never edit `agent/config/thresholds.yaml`, `agent/config/rulebook.yaml`, `agent/tools/`, `input/`, or frozen
   references to make a check pass. Never read data rows. Never approve on the user's behalf (`approve` skill).
5. Tests: `.venv/bin/python -m pytest` from the repository root.

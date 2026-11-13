---
name: run
description: Run the Model Risk Factory on a legacy model the user chooses from input/ - intake, documentation, R-to-Python migration with parity proof, independent validation, monitoring and a run summary - by delegating to the agents in agent/agents/. Use when the user asks to run, document, migrate, validate or monitor a model.
argument-hint: "[model folder, e.g. m2_taiwan] [stages: all|document|migrate|validate|monitor]"
---

# /mrf:run: orchestrate the factory for ONE model

You are the orchestrator. You never write model code, model cards or validation numbers yourself: the four agents
in `agent/agents/` do the work and the tools in `agent/bin/mrf` do every check. Run commands from the repository
root. Arguments: `$ARGUMENTS`

## 1. Ask which model (never scan the whole repository)
- Run `agent/bin/mrf list` to see the models in `input/` and where each one stands.
- If the arguments name a model, use it. Otherwise **ask the user** (AskUserQuestion; in Codex, ask in chat)
  which model to run, listing each model's id, name and current status. If `input/` is empty or the user wants a
  new model, hand over to the `new-model` skill.
- If the arguments do not say which stages, ask: **All stages** (recommended for a new model) /
  **Document + migrate only** / **Validate + monitor only** (needs a packaged model) / **Monitor only**.
- Show the user the plan in one line (model, stages) before starting.

## 2. Intake (deterministic)
`agent/bin/mrf intake <model>`: hashes the inputs, runs the legacy R script under the capture harness, scans it
and writes `output/<model>/logs/INTAKE_BRIEF.md`. A failure here is an input problem: show the error and stop.

## 3. Documentation + migration (two agents, independent, can run in parallel)
- Claude Code: launch both with the Agent tool in ONE message:
  - `subagent_type: "mrf:documentation-agent"`, prompt: "Document model `<model>`. Follow your instructions in
    agent/agents/documentation-agent.md. Report verdict and paths."
  - `subagent_type: "mrf:migration-agent"`, prompt: "Migrate model `<model>`. Follow your instructions in
    agent/agents/migration-agent.md. Report verdict, iterations, parity numbers and paths."
- Codex or no subagents: do the two jobs yourself, one after the other, following those md files exactly.
- If a stage was already completed (`agent/bin/mrf status <model>`), skip it and say so. If the user wants it
  redone: `agent/bin/mrf reset <model> --stage documentation|migration` (moves the stage's logs AND the agent's work files to `logs/_archive/`; nothing is deleted), then re-run.
- **ESCALATED** (exit 4 / `ESCALATION.md`): stop that branch, show the reasons, ask the user how to proceed.

## 4. Validation, then monitoring (only if the migration is packaged)
- `subagent_type: "mrf:validation-agent"` → then `subagent_type: "mrf:monitoring-agent"` (sequential; monitoring
  needs the validation baseline). Codex: follow `agent/agents/validation-agent.md` / `monitoring-agent.md`.

## 5. Summary and human gate
- `agent/bin/mrf summary <model>` → `output/<model>/reports/RUN_SUMMARY.md` and `output/inventory/INVENTORY.md`.
- Tell the user, briefly: each stage's result, overall validation and fairness RAG, re-validation triggers, and the
  files to review (model card, migration report + HUMAN_REVIEW.md, validation report HTML, monitoring report).
- Both agent outputs stay **AWAITING_HUMAN_APPROVAL**. Never approve on the user's behalf: point them to
  `/mrf:approve <model>`.

## Ground rules
- All barriers and thresholds are in `agent/config/thresholds.yaml`; nobody edits it during a run (G-12 detects it).
- Nothing is deleted: re-runs archive old logs under `output/<model>/logs/_archive/`.
- Every action lands in `output/<model>/logs/audit.jsonl`.

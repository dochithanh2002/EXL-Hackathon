---
name: monitor
description: Monitor a validated model on a new scoring batch (or the default simulated batches) and report drift and re-validation triggers.
argument-hint: "<model> [path/to/batch.csv]"
---

# /mrf:monitor

Arguments: `$ARGUMENTS`

1. If no model is given, run `agent/bin/mrf list` and ask the user which one (only validated models qualify).
2. Claude Code: launch `subagent_type: "mrf:monitoring-agent"` with the model and the batch path (if any).
   Codex: follow `agent/agents/monitoring-agent.md` yourself.
3. Then `agent/bin/mrf summary <model>` and report: per-batch RAG, breaches with thresholds, whether re-validation
   is required (and the path to `REVALIDATION_TRIGGER.md`).

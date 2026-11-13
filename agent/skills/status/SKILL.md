---
name: status
description: Show where each model in the factory stands (documentation, migration, validation, monitoring, human approvals, escalations), or the details for one model.
argument-hint: "[model folder or id]"
---

# /mrf:status

Arguments: `$ARGUMENTS`

- No model given: run `agent/bin/mrf list` and show one table row per model (id, name, documentation, migration,
  validation RAG, monitoring). Mention `output/inventory/INVENTORY.md`.
- A model given: run `agent/bin/mrf status <model>` and summarise: each stage's status, iterations, escalations
  (with reasons), human reviews, validation / fairness RAG, re-validation trigger, and
  `inputs_changed_since_intake` (non-empty means the inputs changed: a re-run is needed).
- Read-only. Suggest the next command (`/mrf:run`, `/mrf:approve`, `/mrf:monitor`) when one is obvious.

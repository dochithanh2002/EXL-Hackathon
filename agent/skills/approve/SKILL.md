---
name: approve
description: Record a HUMAN reviewer's approve/reject decision for a model's migrated module or model card (guardrail G-09). Only when the user explicitly gives the decision and their name.
argument-hint: "<model> [migration|documentation]"
disable-model-invocation: true
---

# /mrf:approve: the human approval gate (G-09)

Arguments: `$ARGUMENTS`

An agent never approves its own work. This skill only records a decision the user makes.

1. Identify the model and the target (`migration` = the Python module, `documentation` = the model card). If
   missing, ask. Run `agent/bin/mrf status <model>`: the target must be `AWAITING_HUMAN_APPROVAL`.
2. Show the user what they are approving and the checklist:
   - migration: `output/<model>/reports/migration/HUMAN_REVIEW.md` (+ `MIGRATION_REPORT.md`, the module)
   - documentation: `output/<model>/reports/documentation/MODEL_CARD_<id>.md` §13
3. Ask the user for: decision (approve / reject), reviewer name, comment. Do not fill these in yourself.
4. `agent/bin/mrf approve <model> --target <target> --reviewer "<name>" --decision <approve|reject> --comment "<comment>"`
   It records the decision with the SHA-256 of every file reviewed, in the audit trail and MLflow.
5. Re-run `agent/bin/mrf summary <model>` so the inventory shows the decision.

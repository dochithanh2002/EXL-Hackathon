---
name: new-model
description: Onboard a new legacy model into input/<name>/ - copy the R script and data, ask the user for the fields model.yaml needs (model object names, protected attributes, governance), and write model.yaml from the template. Use when the user wants to add a model to the factory.
argument-hint: "[path to the R script or folder]"
---

# /mrf:new-model: add a legacy model to `input/`

Arguments: `$ARGUMENTS`

1. Ask the user (AskUserQuestion; in Codex, in chat) for anything the arguments do not give:
   - the R script and the data file (paths), and any supporting documents (notebook output, spec);
   - a folder name (`lowercase_with_underscores`) and a model id (`M3`, ...; check `agent/bin/mrf list` for used ids);
   - the model type: `binary_pd` (logistic PD) or `poisson_frequency` (claim count with exposure). Only these two
     have threshold sets in `agent/config/thresholds.yaml`; a new type needs MRM to add one first.
2. Create `input/<name>/` with `model.R`, `data/<file>.csv`, optional `docs/`. Copy; never move the user's files.
   If the script reads data with a path, point that one line at `data/<file>.csv` (paths are relative to the model
   folder) and tell the user you changed it.
3. Read the R script (not the data rows) to propose: `model_object`, `train_object`, `test_object` (the fitted
   object and the train/test data frames), a `seed`, and the protected attributes present in the schema
   (check column names with `head -1` on the CSV only). Confirm them with the user.
4. Write `input/<name>/model.yaml` from `agent/config/model.template.yaml`. Governance fields (owner, business
   use, materiality tier, ...) must come from the user: never invent them; leave `NOT PROVIDED` if unknown.
5. Validation suite: a model of the same family can reuse an existing suite only if its columns match. Otherwise
   tell the user that `agent/validators/<name>/` must be written (copy the closest suite as a start) and set
   `validation.suite` accordingly.
6. Run `agent/bin/mrf intake <name>` to prove the R script runs under the harness, show the intake brief, and
   offer `/mrf:run <name>`.

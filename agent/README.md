# agent/: the Model Risk Factory agents

A Claude Code plugin (`mrf`) that is also usable from Codex. Agents do the reasoning; deterministic Python tools
do every check, so an agent cannot skip a step, relax a threshold or approve its own work.

```
agent/
├── .claude-plugin/plugin.json   plugin manifest (name: mrf)
├── skills/                      what humans type: /mrf:run, /mrf:new-model, /mrf:status, /mrf:approve, /mrf:monitor
├── agents/                      what each agent does, in plain words (one md file per agent)
│   ├── documentation-agent.md   model card + inventory record
│   ├── migration-agent.md       R -> Python with parity proof
│   ├── validation-agent.md      independent validation (read-only)
│   └── monitoring-agent.md      drift + re-validation triggers (read-only)
├── config/
│   ├── thresholds.yaml          EVERY barrier, tolerance and RAG threshold (one file, owned by MRM)
│   ├── rulebook.yaml            R -> Python translation rules (MUST / SHOULD)
│   ├── model.template.yaml      template for input/<model>/model.yaml
│   └── azure.env.example        Azure ML tracking (optional)
├── bin/mrf                      the tool belt the agents call (wrapper for python -m agent.tools)
├── tools/                       deterministic guardrails: capture, scan, schemas, sandbox, parity, verify, package
├── validators/                  independent validation suites, one folder per model family/dataset
└── tests/                       red-team tests for the guardrails + wiring tests
```

## Who does what

| Step | Agent (reasoning, md file) | Tool (deterministic, `agent/bin/mrf ...`) | Guardrails |
|---|---|---|---|
| Intake | orchestrator (`/mrf:run`) | `intake` hashes inputs, runs R under the capture harness, scans | G-11, G-12 |
| Document | documentation-agent writes `extract.json` | `doc-facts` → `doc-verify` (V-01..V-06, bounded) → `doc-render` | no invented numbers |
| Migrate | migration-agent writes `analysis.json`, `plan.json`, `candidate.py` | `check-analysis`, `check-plan`, `test-candidate` (bounded), `package` | G-01..G-13 |
| Validate | validation-agent reads and explains | `validate` runs the suite in `validators/<suite>/` | thresholds.yaml |
| Monitor | monitoring-agent reads and explains | `monitor [--batch FILE]` | thresholds.yaml |
| Approve | **human only** (`/mrf:approve`) | `approve` records decision + SHA-256 of what was reviewed | G-09 |

Exit codes: `0` pass · `1` fail (agent fixes and retries) · `4` ESCALATED (stop, a human decides).

## Guardrails

| ID | Guardrail | Enforced by |
|---|---|---|
| G-01 | Imports limited to the model's allowed packages | `test-candidate` (AST) |
| G-02 | Forbidden APIs: penalised sklearn estimators, RNG / re-splitting, `get_dummies`, shell, network, `eval`/`exec`, file writes | `test-candidate` (AST) |
| G-03 | Interface contract (`TARGET`, `load_and_preprocess`, `fit`, `predict`, `coefficients`) | `test-candidate` (AST) |
| G-04 | Valid Python syntax | `test-candidate` (AST) |
| G-05 | Python coefficient names equal R's | parity |
| G-06 | Sandbox: isolated interpreter, credentials stripped, timeout | `candidate_runner.py` |
| G-07 | Row identity with the raw file | `candidate_runner.py` |
| G-08 | Bounded repair loop (`agent_limits.max_repair_iterations`), then escalate | `test-candidate` |
| G-09 | Human approval before release | `approve` |
| G-10 | Agent work files validated against typed schemas (`mrf schema ...`) | `check-*`, `doc-verify` |
| G-11 | Agents see schema and level sets only, never data rows | intake brief + agent instructions |
| G-12 | model.yaml, thresholds.yaml, rulebook, script, data hashed at intake, re-checked before package/render | `package`, `doc-render` |
| G-13 | The plan cites every triggered MUST rule | `check-plan` |

## Thresholds: one file

`config/thresholds.yaml` holds parity tolerances, agent limits, the documentation significance level, validation
and fairness traffic lights, and monitoring re-validation triggers, grouped by `model_type`
(`binary_pd`, `poisson_frequency`). Validators, reports and agents all read it; nothing is hard-coded. A model's
`model.yaml` only selects the set; it cannot override it. Changing a value needs MRM sign-off and a line in
`CHANGELOG.md`; a change in the middle of a run is caught by G-12.

## Installing the plugin

- Claude Code (this repo): `claude --plugin-dir ./agent`, or add the repo as a marketplace once:
  `/plugin marketplace add ./` then `/plugin install mrf@exl-hackathon`. Skills appear as `/mrf:run`, etc.
- Codex: reads `AGENTS.md` at the repo root and the skills under `.agents/skills/` (a link to `agent/skills/`).

## Limits

- Model classes: R `glm` (binomial, Poisson, Gamma rules) and `lm`. Tree models need new rules and capture logic.
- Inputs: one CSV read by the R script; train/test keep original rownames. Missing rules: ordered factors,
  interactions, splines/`poly()`, weights, aliased terms. The static scan is regex-based.
- Validation suites are dataset-specific (`validators/m1_fremtpl2`, `validators/m2_taiwan`); a new dataset needs
  its own suite (copy the closest one).

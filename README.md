# Agentic Model Risk Factory (EXL Hackathon, Challenge 04)

Documents, redevelops, validates and monitors a bank's legacy models under PRA SS1/23. Driven from Claude Code
(or Codex) with slash commands; agents do the reasoning, deterministic tools do every check.

| Model | Type | Rows | Migration | Validation | Summary |
|---|---|---|---|---|---|
| **M1** French motor TPL claim frequency | Poisson GLM, log-exposure offset | 678,013 | parity PASS | AMBER, fairness AMBER | [`RUN_SUMMARY.md`](output/m1_fremtpl2/reports/RUN_SUMMARY.md) |
| **M2** Taiwan card default | logistic PD | 30,000 | parity PASS | AMBER, fairness RED | [`RUN_SUMMARY.md`](output/m2_taiwan/reports/RUN_SUMMARY.md) |

## Use it

In Claude Code, from this folder (load the plugin once: `claude --plugin-dir ./agent`, or
`/plugin marketplace add ./` then `/plugin install mrf@exl-hackathon`):

| Command | What happens |
|---|---|
| `/mrf:run` | asks which model in `input/` and which stages, then intake → documentation + migration agents → validation agent → monitoring agent → summary |
| `/mrf:new-model` | onboards a new legacy model into `input/<name>/` (asks you for the fields) |
| `/mrf:status [model]` | where each model stands |
| `/mrf:approve <model>` | records **your** approve/reject decision (agents never approve their own work) |
| `/mrf:monitor <model> [batch.csv]` | monitoring on a new scoring batch |

Codex reads [`AGENTS.md`](AGENTS.md) and the same skills (`.agents/skills/` → `agent/skills/`).

## Layout: three folders

```
agent/    the agents and their tools          → agent/README.md
  agents/*.md          one file per agent: what it does, its rules, the commands it must run
  skills/*/SKILL.md    the slash commands
  config/thresholds.yaml   EVERY barrier and threshold, in one file
  config/rulebook.yaml     R -> Python rules
  bin/mrf, tools/      deterministic guardrails (capture, schemas, sandbox, parity, verify, package)
  validators/          independent validation suites
input/    legacy models: model.R, model.yaml, data/, docs/   → input/README.md
output/   per model: logs/, model/, reports/  + inventory/, mlflow/   → output/README.md
```

## Flow

```mermaid
flowchart LR
    U(("User")) -- "/mrf:run → picks model" --> O["Orchestrator skill"]
    O --> I["mrf intake<br/>hash · R capture · scan"]
    I --> DA["Documentation agent<br/>(documentation-agent.md)"]
    I --> MA["Migration agent<br/>(migration-agent.md)"]
    DA -- "extract.json" --> DV{"mrf doc-verify<br/>V-01..V-06"}
    MA -- "analysis · plan · candidate" --> MT{"mrf check-* / test-candidate<br/>G-01..G-13, parity"}
    DV --> CARD["reports/documentation/<br/>MODEL_CARD"]
    MT --> MOD["model/<br/>module + reference + tests"]
    MOD --> VA["Validation agent<br/>mrf validate"] --> MO["Monitoring agent<br/>mrf monitor"] --> S["mrf summary<br/>RUN_SUMMARY · inventory · MLflow"]
    CARD & MOD -. "/mrf:approve" .-> H(("Human"))
    T[["agent/config/thresholds.yaml"]] -.-> MT & VA & MO & DV
```

## Pipeline in detail

### 1. Stages, checks and loops

Agents (rounded) write work files; deterministic `agent/bin/mrf` commands (rectangles) check them. Exit 1 loops
back to the agent; exit 4 escalates to a human. Limits come from `agent/config/thresholds.yaml → agent_limits`.

```mermaid
flowchart TB
    subgraph S0["0 · Orchestrator  /mrf:run"]
        L["mrf list<br/>ask user: model + stages"]
    end

    subgraph S1["1 · Intake  (deterministic)"]
        IN["mrf intake &lt;model&gt;<br/>SHA-256 inputs · run model.R under r_capture.R · static scan"]
        IN --> IB[/"logs/INTAKE_BRIEF.md<br/>model/reference/ (frozen R coefs, preds, split)<br/>logs/intake/scan.json"/]
    end

    subgraph S2["2a · Documentation agent"]
        DF["mrf doc-facts<br/>facts.json · required disclosures"]
        DX(["Agent writes<br/>extract.json"])
        DVR{"mrf doc-verify<br/>V-01 variables · V-02 types/levels<br/>V-03 line grounding · V-04 disclosures<br/>V-05 no invented numbers · V-06 schema"}
        DR["mrf doc-render"]
        DF --> DX --> DVR
        DVR -- "exit 1: fix named violations" --> DX
        DVR -- "exit 0" --> DR
        DR --> MC[/"reports/documentation/MODEL_CARD_&lt;id&gt;.md<br/>inventory record"/]
    end

    subgraph S3["2b · Migration agent"]
        AN(["Agent writes<br/>analysis.json"]) --> CA{"mrf check-analysis"}
        CA -- "exit 1" --> AN
        CA -- "exit 0" --> PL(["Agent writes<br/>plan.json"]) --> CP{"mrf check-plan<br/>G-13 every MUST rule cited"}
        CP -- "exit 1" --> PL
        CP -- "exit 0" --> CD(["Agent writes<br/>candidate.py"]) --> TC{"mrf test-candidate<br/>G-01 imports · G-02 forbidden APIs/RNG/IO<br/>G-03 interface · G-04 syntax<br/>sandbox run → parity vs R reference"}
        TC -- "exit 1: root-cause repair<br/>(≤ max_repair_iterations, G-08)" --> CD
        TC -- "exit 0" --> PK["mrf package"]
        PK --> PM[/"model/&lt;module&gt;.py · tests/ · manifest.json<br/>reports/migration/MIGRATION_REPORT.md · HUMAN_REVIEW.md"/]
    end

    subgraph S4["3 · Validation agent  (read-only, independent)"]
        VS["mrf validate<br/>seed_sensitivity → parity re-check → validate<br/>(discrimination · calibration · stability · specification)<br/>→ fairness → report"]
        VS --> VR[/"validation_results.json · parity/ · fairness/<br/>validation_report_&lt;id&gt;.html  (RAG per area)"/]
    end

    subgraph S5["4 · Monitoring agent  (read-only)"]
        MN["mrf monitor [--batch csv]<br/>CSI / PSI drift · performance · calibration · fairness gaps"]
        MN --> MR[/"MONITORING_REPORT.md<br/>REVALIDATION_TRIGGER.md if any RED"/]
    end

    subgraph S6["5 · Summary + human gate"]
        SM["mrf summary<br/>RUN_SUMMARY.md · inventory/INVENTORY.md · MLflow"]
        AP{{"/mrf:approve (human only, G-09)<br/>decision + reviewer + SHA-256 of reviewed files"}}
        SM --> AP
    end

    ESC(("ESCALATION.md<br/>exit 4 → stop, ask user"))

    L --> IN
    IB --> DF
    IB --> AN
    MC --> VS
    PM --> VS
    VR --> MN
    MR --> SM
    DVR -. "> max_doc_iterations" .-> ESC
    CA -. "exit 4" .-> ESC
    CP -. "exit 4" .-> ESC
    TC -. "> max_repair_iterations" .-> ESC

    TH[["agent/config/thresholds.yaml<br/>agent/config/rulebook.yaml<br/>(read-only during a run, G-12)"]]
    TH -.-> DVR & CP & TC & VS & MN
    AUD[("logs/audit.jsonl<br/>every action")]
    S1 & S2 & S3 & S4 & S5 & S6 -.-> AUD
```

### 2. Who calls whom (UML sequence)

```mermaid
sequenceDiagram
    autonumber
    actor U as User / MRM reviewer
    participant O as Orchestrator (/mrf:run)
    participant T as mrf tools (deterministic)
    participant D as Documentation agent
    participant M as Migration agent
    participant V as Validation agent
    participant Mo as Monitoring agent

    U->>O: /mrf:run
    O->>T: mrf list
    O->>U: which model? which stages?
    U-->>O: m2_taiwan, all stages
    O->>T: mrf intake m2_taiwan
    T-->>O: INTAKE_BRIEF.md + frozen R reference

    par independent agents
        O->>D: document model
        D->>T: doc-facts
        loop until exit 0 (max_doc_iterations)
            D->>T: doc-verify extract.json
            T-->>D: violations V-01..V-06
        end
        D->>T: doc-render
        D-->>O: RENDERED, MODEL_CARD path
    and
        O->>M: migrate model
        M->>T: check-analysis, check-plan
        loop until parity PASS (max_repair_iterations)
            M->>T: test-candidate candidate.py
            T-->>M: guardrail + parity evidence
        end
        M->>T: package
        M-->>O: PACKAGED, parity numbers
    end

    alt any exit 4
        O->>U: ESCALATED: reasons, how to proceed?
    else packaged
        O->>V: validate
        V->>T: mrf validate
        T-->>V: RAG per area, fairness
        V-->>O: overall RAG + findings
        O->>Mo: monitor
        Mo->>T: mrf monitor
        T-->>Mo: drift / triggers
        Mo-->>O: per-batch RAG, re-validation?
        O->>T: mrf summary
        O->>U: results, files to review
    end

    U->>T: /mrf:approve (decision, name, comment)
    T-->>U: recorded in audit.jsonl + MLflow
```

### 3. Status of an agent's output (UML state, as shown by `mrf status`)

Applies to the documentation stage (model card) and the migration stage (Python module).

```mermaid
stateDiagram-v2
    [*] --> not_run
    not_run --> working: agent starts after intake
    working --> working: check exit 1, agent repairs
    working --> ESCALATED: exit 4 or iteration limit
    working --> AWAITING_HUMAN_APPROVAL: doc-render or package
    AWAITING_HUMAN_APPROVAL --> APPROVED: human approves
    AWAITING_HUMAN_APPROVAL --> REJECTED: human rejects
    ESCALATED --> not_run: mrf reset archives logs
    REJECTED --> not_run: mrf reset
    APPROVED --> revalidation_required: monitoring metric RED
    revalidation_required --> [*]: REVALIDATION_TRIGGER written
```

## Tests

```bash
.venv/bin/python -m pytest
```

Red-team guardrail tests, documentation-verify mutants, multi-seed parity against R, generated parity tests for
both packaged models, and wiring tests (layout, single thresholds file, agent artefacts used downstream).

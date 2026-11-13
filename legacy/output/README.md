# output/: everything the factory produces

Written only by the tools in `agent/`. Nothing is deleted: re-runs archive old logs under `logs/_archive/`.

```
output/
├── inventory/                  portfolio view: INVENTORY.md, inventory_index.json
├── mlflow/                     local MLflow store (git-ignored; Azure ML is used when agent/config/azure.env is set)
└── <model>/
    ├── logs/                   how it was produced
    │   ├── audit.jsonl         ONE timeline of every step by every agent and tool
    │   ├── INTAKE_BRIEF.md     what the agents were given
    │   ├── intake/             input hashes, static scan, data schema (no rows)
    │   ├── documentation/      facts, numbered source, the agent's extract(s), verification state
    │   ├── migration/          analysis.json, plan.json, every candidate tested and its run, state
    │   ├── validation/ monitoring/   steps run, exit codes, timings
    │   └── run_summary.json
    ├── model/                  the migrated model package
    │   ├── <module>.py         Python module (AWAITING_HUMAN_APPROVAL until /mrf:approve)
    │   ├── reference/          frozen R outputs (coefficients, predictions, split, meta)
    │   ├── tests/              generated parity regression test
    │   ├── requirements.txt    pinned
    │   └── manifest.json       SHA-256 of every file
    └── reports/                what people read
        ├── RUN_SUMMARY.md
        ├── documentation/      MODEL_CARD_<id>.md, inventory_<id>.json
        ├── migration/          MIGRATION_REPORT.md, HUMAN_REVIEW.md
        ├── validation_report_<id>.html / .md
        ├── parity/ validation/ fairness/ monitoring/
```

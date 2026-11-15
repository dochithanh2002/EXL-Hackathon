# input/: the legacy models to process

One folder per model. Humans own everything here; agents read it and never change it.

```
input/<model>/
├── model.R        legacy script, run unmodified by the capture harness (paths relative to this folder)
├── model.yaml     model request: R object names, data, protected attributes, governance, validation suite
├── data/          the data the script reads (git-ignored)
└── docs/          optional supporting documents (notebook output, specs)
```

| Folder | Model | Type |
|---|---|---|
| `m1_fremtpl2/` | M1 French motor TPL claim frequency, Poisson GLM | `poisson_frequency` |
| `m2_taiwan/` | M2 Taiwan card default, logistic PD | `binary_pd` |
| `_brief/` | hackathon brief and planning notes (not a model) | - |

Add a model with `/mrf:new-model` (template: `agent/config/model.template.yaml`). Thresholds do not go in
`model.yaml`; they live in `agent/config/thresholds.yaml`.

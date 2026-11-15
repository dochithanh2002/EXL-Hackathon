# M5: HMEQ home-equity loan default

HMEQ is a public SAS sample dataset: 5,960 loans, 12 features and a binary label `BAD`, of which 1,189 (19.95%)
are defaults or serious delinquencies.

| File | |
|---|---|
| `data/hmeq.csv` | The original CSV from SAS, unchanged |
| `code/hmeq_logistic_baseline.R` | A standalone R logistic baseline written for this project |
| `code/CodeR.R` | Emmanuel Kouraogo's R validation script, downloaded at the pinned commit, unchanged |
| `UPSTREAM_README.md`, `LICENSE.upstream` | The upstream project's French README and MIT licence, which cover the upstream code only |

`CodeR.R` validates SAS model results: it uses the author's absolute paths, installs R packages, and reads SAS
intermediate files (`donnees_imputees_PMM.csv`, `scores_PMM.csv`, `information_value_PMM.csv`) that the upstream
repository does not publish. It parses in R but does not run; use the baseline.

## Fields

One row per loan. Empty fields are missing. There is no borrower id, observation date or default window.

| Field | Meaning | Missing |
|---|---|---:|
| `BAD` | 1 = default or serious delinquency, 0 = otherwise | 0 |
| `LOAN` | Requested loan amount | 0 |
| `MORTDUE` | Amount due on the existing mortgage | 518 |
| `VALUE` | Value of the property | 112 |
| `REASON` | Debt consolidation or home improvement | 252 |
| `JOB` | Occupation category | 279 |
| `YOJ` | Years at present job | 515 |
| `DEROG` | Number of major derogatory reports | 708 |
| `DELINQ` | Number of delinquent credit lines | 580 |
| `CLAGE` | Age of the oldest credit line, in months | 308 |
| `NINQ` | Number of recent credit inquiries | 510 |
| `CLNO` | Number of credit lines | 222 |
| `DEBTINC` | Debt-to-income ratio | 1,267 |

## Baseline

Base R 4.x only, no packages. From the repository root:

```sh
Rscript --vanilla examples/M5_hmeq/code/hmeq_logistic_baseline.R
```

It reads `data/hmeq.csv` and writes coefficients, training rows, holdout predictions, summary metrics and
per-occupation diagnostics to `outputs/`; an input CSV and an output directory can be passed as arguments.
`csv_row` is the 1-based data-row position in the CSV, header excluded.

The split is stratified on `BAD` with seed 42 (4,767 train, 1,193 test). Selected numeric fields are imputed with
training medians. The logistic regression uses `log(LOAN)`, `MORTDUE`, `YOJ`, `DEROG`, `DELINQ`, `CLAGE`, `NINQ`
and `CLNO`; `JOB` is used only for the test-set group diagnostics.

On R 4.6.2: holdout ROC AUC 0.807 and Brier score 0.122 (0.160 for a constant training default rate); test
default rate 19.95%, mean predicted PD 20.36%. This is a single random holdout.

## What to validate first

- **Missingness and availability.** `DEBTINC` is missing for 66.1% of bad loans and 10.1% of the rest; `VALUE` for
  8.8% and 0.15%. The baseline leaves both out. Before adding them, establish when they are collected and why they
  are missing, then compare out-of-sample stability.
- **Fairness.** There are no direct attributes such as age or sex, and small occupation groups give volatile
  metrics. Occupation groups show where to look further; they do not support a fairness conclusion.
- **Stability over time.** Without dates there is no out-of-time backtest, and no way to tell whether a borrower
  appears on both sides of the split. A formal validation needs dated data with borrower ids and a defined label
  window.

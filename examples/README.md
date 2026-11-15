# Example models

Five public R credit and insurance models with their data, for reproducing R-Pythonizer runs. Files are as
downloaded on 2026-11-13 (M5 on 2026-11-14), except where noted. Check them with `shasum -a 256 -c SHA256SUMS`
from this folder.

| Folder | Model | Dataset | Data | R code | Document |
|---|---|---|---|---|---|
| `M1_fremtpl2` | Motor claim frequency, Poisson GLM | `freMTPL2freq.csv` | `freMTPL2sev.csv` | `legacy_glm1_notebook.R` | |
| `M2_taiwan_default` | Credit-card default, logistic PD | `UCI_Credit_Card.csv` | | `default-credit-and-credit-score-prediction.Rmd` | |
| `M3_ulb_fraud` | Card fraud, random forest | `creditcard.csv` | | `credit-card-fraud-detection-with-rf-auc-0-93.Rmd` | |
| `M4_givemesomecredit` | Serious delinquency, logistic PD | `cs-training.csv` | `cs-test.csv` | `givemesomecredit_logistic_notebook.R` | `data_dictionary.md` |
| `M5_hmeq` | Home-equity loan default, logistic PD | `hmeq.csv` | | `hmeq_logistic_baseline.R` | |

In the UI, create a project and upload the files of one row in those roles. Data files are in `<folder>/data/`,
R code in `<folder>/code/`.

M3's CSV is 144 MB, over GitHub's file limit, so it is stored compressed. Before uploading:

```sh
gunzip -k examples/M3_ulb_fraud/data/creditcard.csv.gz
```

M4's data cannot be redistributed: download it from Kaggle as
[`M4_givemesomecredit/data/README.md`](M4_givemesomecredit/data/README.md) describes. The M4 tests in
`platform/backend` read it from there (override with `M4_DIR`) and are skipped without it.

## Sources

| Folder | Data | Code |
|---|---|---|
| `M1_fremtpl2` | [freMTPL2freq](https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq), [freMTPL2sev](https://www.kaggle.com/datasets/floser/fremtpl2sev) (from the CASdatasets R package) | [floser](https://www.kaggle.com/code/floser/glm-neural-nets-and-xgboost-for-insurance-pricing): the notebook's R cells |
| `M2_taiwan_default` | [UCI default of credit card clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) | [netzone](https://www.kaggle.com/code/netzone/default-credit-and-credit-score-prediction) |
| `M3_ulb_fraud` | [ULB credit card fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | [Gabriel Preda](https://www.kaggle.com/code/gpreda/credit-card-fraud-detection-with-rf-auc-0-93) |
| `M4_givemesomecredit` | [Give Me Some Credit competition](https://www.kaggle.com/c/GiveMeSomeCredit); `data_dictionary.md` is its `Data Dictionary.xls` | [Abhishek Giri](https://www.kaggle.com/code/abhishekgiri/give-me-some-credit-logistic-regression): the notebook's R cells |
| `M5_hmeq` | [SAS HMEQ sample data](https://support.sas.com/documentation/onlinedoc/viya/exampledatasets/hmeq.csv) | `hmeq_logistic_baseline.R` written for this project; `CodeR.R` from [EKOURAOGO/Scoring-credit-hmeq](https://github.com/EKOURAOGO/Scoring-credit-hmeq/blob/48ef53be034b6ce48aa0b2c930905e96d46b9889/CodeR.R) (MIT, see `M5_hmeq/LICENSE.upstream`) |

The Kaggle code reads its data from Kaggle's `../input/` paths. M5 is described in
[`M5_hmeq/README.md`](M5_hmeq/README.md).

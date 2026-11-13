# M2 model spec, taken from the Kaggle notebook `netzone/default-credit-and-credit-score-prediction`

- **Source:** the rendered notebook output (`netzone_results.html` / `.txt`). It was written in RMarkdown, published under the Apache 2.0 licence, and last updated about 10 years ago (version 2).
- **Data:** UCI "Default of Credit Card Clients" (Taiwan, Apr–Sep 2005). It has 30,000 rows; the notebook drops `id` and leaves 24 columns. All column names are converted to lower case.

## Preprocessing (in order)
1. Drop the `id` column.
2. **education:** recode {0, 4, 5, 6} → 4 ("others"). Then make it a factor with levels 1, 2, 3, 4 (reference level 1 = graduate school).
3. **marriage:** recode 0 → 3 ("others"). Then make it a factor with levels 1, 2, 3 (reference level 1 = married).
4. **sex:** make it a factor with levels 1, 2 (reference level 1 = male).
5. **pay_0, pay_2 … pay_6:** recode {-2, -1, 0} → 0. They stay **numeric**, so the model treats them as linear terms.
6. **target** `default.payment.next.month`: a factor (0/1).

## Split
- The split uses `caret::createDataPartition(y, p=0.7, list=F)`, which stratifies on the target. This gives 21,001 training rows and 8,999 test rows.
- **The notebook never shows a seed**, so the exact split can't be reproduced. This is a finding: the model isn't reproducible.

## Final model (the "Traditional Score")
```r
glm(default.payment.next.month ~ limit_bal + sex + education + marriage +
    pay_0 + pay_2 + pay_3 + pay_5 + pay_6 + pay_amt1 + pay_amt2,
    data = train.batch, family = binomial)
```
- The variables were selected by hand from the full model (`~ .`), keeping the terms with p < 0.05. The notebook doesn't record a formal selection procedure.
- **Protected or sensitive attributes used as inputs:** sex, marriage, and education (a proxy). Age was dropped from the final model.

## Reported results (test set, n = 8,999)
| Metric | Value |
|---|---|
| AUC | 0.7542 |
| KS | 0.40 |
| Accuracy at the best cut-off | 0.8200 |
| Cut-off (chosen to maximise accuracy) | 0.4539 |
| Sensitivity / Specificity | 0.3618 / 0.9501 |
| Residual deviance / AIC | 18,708 / 18,738 |

| Coefficient | Estimate | SE |
|---|---|---|
| (Intercept) | -1.24647 | 0.0601 |
| limit_bal | -1.514e-06 | 1.74e-07 |
| sex2 | -0.17391 | 0.0375 |
| education2 | -0.02775 | 0.0433 |
| education3 | -0.06581 | 0.0574 |
| education4 | -1.20000 | 0.2358 |
| marriage2 | -0.24049 | 0.0384 |
| marriage3 | 0.02831 | 0.1544 |
| pay_0 | 0.86887 | 0.0298 |
| pay_2 | 0.07548 | 0.0315 |
| pay_3 | 0.15816 | 0.0304 |
| pay_5 | 0.12031 | 0.0360 |
| pay_6 | 0.14818 | 0.0350 |
| pay_amt1 | -9.745e-06 | 2.30e-06 |
| pay_amt2 | -9.418e-06 | 2.21e-06 |

## Other models in the notebook (not in scope, listed for the inventory)
- rpart tree: AUC 0.64, KS 0.28.
- Random forest (500 trees): AUC 0.77, KS 0.41. It is a challenger model.

## Validator findings (to carry into the report)
1. **No seed**, so the split and the coefficients can't be reproduced exactly.
2. **Sex and marriage are direct model inputs.** Women get a lower PD (sex2 = -0.17) and single people get a lower PD (marriage2 = -0.24). Both are Equality Act 2010 protected characteristics, so their use needs a fairness review and a documented justification.
3. **education4 = -1.20** comes almost entirely from the undocumented codes 0, 5 and 6, which have about a 6% default rate. That is a data artefact, not a real effect, and it affects 468 rows.
4. **pay_x recoding:** merging -2, -1 and 0 into one value is an assumption the author made, not something the data definition supports.
5. pay_x is fitted as a linear term even though the values are ordinal delay months, so the linearity assumption hasn't been tested.
6. The cut-off was chosen to maximise accuracy on the **test set**. That is optimistic leakage, and it isn't a business-driven cut-off.
7. Variables were selected by p-value on the training set alone, with no out-of-time or cross-validation check.

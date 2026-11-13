# ---------------------------------------------------------------------------
# LEGACY MODEL (as received) - Taiwan credit-card default "Traditional Score"
# Source: Kaggle notebook netzone/default-credit-and-credit-score-prediction (v2),
#         Apache-2.0. Code reproduced from the published notebook; lines the
#         notebook described but did not print are marked [reconstructed].
# This file is the INPUT to the R->Python migration agent. Do not edit.
# ---------------------------------------------------------------------------
library(caret)

raw.credit <- read.csv("data/taiwan_default.csv")
colnames(raw.credit) <- tolower(colnames(raw.credit))

# id column is the client identifier, not important -> removed
raw.credit$id <- NULL

# For education, we will merge 0, 4, 5, 6 to 4 (others)        [reconstructed]
raw.credit$education[raw.credit$education %in% c(0, 4, 5, 6)] <- 4
# For marriage, we will merge 0 to 3 (others)                  [reconstructed]
raw.credit$marriage[raw.credit$marriage == 0] <- 3

# Assume -2, -1 and 0 have the same meaning. Merge them to 0.   (verbatim)
for (i in seq(6, 11, 1)) {
  raw.credit[which(raw.credit[[i]] %in% c(-2, -1, 0)),][[i]] <- 0
}

# convert to factor (sex, education, marriage, target)         [reconstructed]
raw.credit$sex <- as.factor(raw.credit$sex)
raw.credit$education <- as.factor(raw.credit$education)
raw.credit$marriage <- as.factor(raw.credit$marriage)
raw.credit$default.payment.next.month <- as.factor(raw.credit$default.payment.next.month)

# training (70%) and testing (30%)                              (verbatim)
intraining <- createDataPartition(y=raw.credit$default.payment.next.month, p=0.7, list=F)
train.batch <- raw.credit[intraining,]
test.batch <- raw.credit[-intraining,]

# Traditional Score using Logistic Regression                   (verbatim)
glm.final <- glm(default.payment.next.month ~ limit_bal+sex+education+marriage+
                   pay_0+pay_2+pay_3+pay_5+pay_6+pay_amt1+pay_amt2, data=train.batch,
                 family=binomial)

test.batch$glm_scor1 <- predict(glm.final, type="response", test.batch)

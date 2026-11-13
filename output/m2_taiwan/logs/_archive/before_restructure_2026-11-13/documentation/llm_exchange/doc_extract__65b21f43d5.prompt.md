<!-- model: claude-opus-5-5 | step: doc_extract | sha256: 65b21f43d5d479204dd66f1e88f35352f956550a2f77bf7a9a23584b03fb44a3 -->
# SYSTEM

You are the Documentation Agent in a bank's Model Risk Management factory. You reverse-engineer the methodology, variables, assumptions and limits of a legacy model from its source code and supporting documents, to build its model-inventory record and model card under PRA SS1/23. Independent validators and regulators read your output.

Non-negotiable rules:
1. DESCRIBE WHAT THE CODE DOES. Where code and documents disagree, the code wins; record the disagreement in `code_vs_docs_discrepancies`.
2. GROUND EVERY CLAIM. Cite line numbers from the numbered source listing in `source_lines`.
3. NO STATISTICS. Do not write coefficients, AUC, rates, p-values or counts. The factory computes and inserts every number. You may quote a number only if it appears verbatim in the source or the supporting documents.
4. COVER EVERY REQUIRED DISCLOSURE: at least one limitation per disclosure id, citing the id and naming the issue.
5. NEVER INVENT governance facts (owner, approvals, materiality, business use). If the evidence does not state them, say so and add an open question.
6. You never see data rows, only schema and aggregate facts.
7. Respond with ONE JSON object that matches the requested schema, and nothing else.

# USER

# Task: DOCUMENT the legacy model (reverse-engineer methodology, variables and limits)

## Legacy source (numbered; cite these line numbers)
```r
   1 | # ---------------------------------------------------------------------------
   2 | # LEGACY MODEL (as received) - Taiwan credit-card default "Traditional Score"
   3 | # Source: Kaggle notebook netzone/default-credit-and-credit-score-prediction (v2),
   4 | #         Apache-2.0. Code reproduced from the published notebook; lines the
   5 | #         notebook described but did not print are marked [reconstructed].
   6 | # This file is the INPUT to the R->Python migration agent. Do not edit.
   7 | # ---------------------------------------------------------------------------
   8 | library(caret)
   9 | 
  10 | raw.credit <- read.csv("data/taiwan_default/taiwan_default.csv")
  11 | colnames(raw.credit) <- tolower(colnames(raw.credit))
  12 | 
  13 | # id column is the client identifier, not important -> removed
  14 | raw.credit$id <- NULL
  15 | 
  16 | # For education, we will merge 0, 4, 5, 6 to 4 (others)        [reconstructed]
  17 | raw.credit$education[raw.credit$education %in% c(0, 4, 5, 6)] <- 4
  18 | # For marriage, we will merge 0 to 3 (others)                  [reconstructed]
  19 | raw.credit$marriage[raw.credit$marriage == 0] <- 3
  20 | 
  21 | # Assume -2, -1 and 0 have the same meaning. Merge them to 0.   (verbatim)
  22 | for (i in seq(6, 11, 1)) {
  23 |   raw.credit[which(raw.credit[[i]] %in% c(-2, -1, 0)),][[i]] <- 0
  24 | }
  25 | 
  26 | # convert to factor (sex, education, marriage, target)         [reconstructed]
  27 | raw.credit$sex <- as.factor(raw.credit$sex)
  28 | raw.credit$education <- as.factor(raw.credit$education)
  29 | raw.credit$marriage <- as.factor(raw.credit$marriage)
  30 | raw.credit$default.payment.next.month <- as.factor(raw.credit$default.payment.next.month)
  31 | 
  32 | # training (70%) and testing (30%)                              (verbatim)
  33 | intraining <- createDataPartition(y=raw.credit$default.payment.next.month, p=0.7, list=F)
  34 | train.batch <- raw.credit[intraining,]
  35 | test.batch <- raw.credit[-intraining,]
  36 | 
  37 | # Traditional Score using Logistic Regression                   (verbatim)
  38 | glm.final <- glm(default.payment.next.month ~ limit_bal+sex+education+marriage+
  39 |                    pay_0+pay_2+pay_3+pay_5+pay_6+pay_amt1+pay_amt2, data=train.batch,
  40 |                  family=binomial)
  41 | 
  42 | test.batch$glm_scor1 <- predict(glm.final, type="response", test.batch)
```

## Supporting documents (narrative and code only; console output stripped, guardrail G-11)
### ref_code/taiwan_default/netzone_results.txt
```text
Defaul Credit Card of Taiwan Clients in 2005
Code 
Show All Code
Hide All Code
Defaul Credit Card of Taiwan Clients in 2005
Default of Credit Card Clients Dataset
Overview
Dataset Information
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.
I have put the column names to lower cases. Let’s quickly look at the dataset.
There’re quite lots of features in dataset (25 columns)
id column is the client identifier, this is not important and will be removed.
There’re few variables which I will convert to factor (such as sex, education, marriage). pay_0 and other pay_x features can considered to be factor variable, but let’s keep them as numeric for now.
dim(raw.credit)
summary(raw.credit)
head(raw.credit,1)
Is there any missing data points in the dataset?
There are no missing data.
missmap(raw.credit, main = "Deafult credit card missing data heatmap", 
        col=c("red", "black"), legend=F)
[IMG]
colSums(sapply(raw.credit, is.na))
Note that we have removed id from datasets as there’s no meaning in this context. We will proceed to next section before assigning any factor class to the variables.
Basic Visual & Data Cleansing
This section will try to make more senses on each data point by visualization. I will also fix the data points if they are not making any senses from the given definition.
Look at Demographical variable data
There are more female than male in the dataset.
There are clients who finished university-level education.
There are more single client than married, but the number is quite closed (53.2%:45.5%)
Recap from data definition:
EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
MARRIAGE: Marital status (1=married, 2=single, 3=others)
# Plot Demographical data
lets.plot(raw.credit, fun=barPlot, ii=2:4, ncol=2)
[IMG]
Let’s fix the datapoint of education and marriage.
For education, we will merge 0, 4, 5, 6 to 4 (others)
For marriage, we will merge 0 to 3 (others)
[IMG]
Age data - the data looks ok,
lets.plot(raw.credit, fun=plotDen, ii=5, ncol=1)
[IMG]
Quick look on the limit balance
lets.plot(raw.credit, fun=plotDen, ii=1, ncol=1)
[IMG]
What about repayment status
Based on the data definition, we have the following:
-1 = pay duly
1 = payment delay for one month
2 = payment delay for two months
. ..
8 = payment delay for eight months
and so on …
However, from the plot, we can see that there are a lot of -2 and 0 data points spreading out across repayment status.
With this, I assume that -2, -1 and 0 have the same meaning. We will merge them to 0.
lets.plot(raw.credit, fun=barPlot, ii=6:11, ncol=2)
[IMG]
for (i in seq(6, 11, 1)) {
  raw.credit[which(raw.credit[[i]] %in% c(-2, -1, 0)),][[i]] <- 0
}
Let’s quickly look at the bill statement and previous payment in each month.
lets.plot(raw.credit, fun=plotDen, ii=12:17, ncol=2)
[IMG]
lets.plot(raw.credit, fun=plotDen, ii=18:23, ncol=2)
[IMG]
Lastly, let’s look at the number of default payment next month.
The default payment is around 22% of the total observations in the dataset.
lets.plot(raw.credit, fun=barPlot, ii=24, ncol=1)
[IMG]
prop.table(table(raw.credit$default.payment.next.month))
Let’s move to the next section.
Correlation
Let’s explore the correlation between each data points.
I will use the mosaicplot to visualize the demographical data against the default payment.
mosaicplot(raw.credit$sex ~ raw.credit$default.payment.next.month, shade=F, color=T,
           main="Gender affecting the default payment next month?",
           ylab="Default payment next month (1-Y, 0-N)",
           xlab="Gender (1-male, 2-female)"
          )
[IMG]
mosaicplot(raw.credit$education ~ raw.credit$default.payment.next.month, shade=F, color=T,
           main="Education affecting the default payment next month?",
           ylab="Default payment next month (1-Y, 0-N)",
           xlab="Education  (1=graduate school, 2=university, 3=high school, 4=others)"
          )
[IMG]
mosaicplot(raw.credit$marriage ~ raw.credit$default.payment.next.month, shade=F, color=T,
           main="Marital status affecting the default payment next month?",
           ylab="Default payment next month (1-Y, 0-N)",
           xlab="Marriage (1=married, 2=single, 3=others)"
          )
[IMG]
mosaicplot(paste0(raw.credit$sex, '-', raw.credit$marriage) ~ raw.credit$default.payment.next.month,
           shade =F, color = T, main= "Gender with marital status against default credit?",
           xlab="Gender - marital status", ylab="Default payment next month (1-Y, 0-N)")
[IMG]
Let’s combine everything
It looks like the married women are more likely to default payment.
ggplot(raw.credit, aes(education, fill=default.payment.next.month)) +
  stat_count() +
  facet_wrap(~paste0(raw.credit$sex, '-', raw.credit$marriage)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)
          , legend.position = "bottom"
          , legend.title = element_blank()
        )  
[IMG]
Look at the repayment status of each month
From pay_0 to pay_6, the definition is the same except it is reflected the repayment status of each month (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above).
repay.bar.plot <- function(df, i, target) {
  d <- data.frame(x=df[[i]]) 
  p <- ggplot(d, aes(factor(x), fill=df[[target]])) + 
    stat_count() + 
    xlab(colnames(df)[i]) + 
    theme_bw() +
    theme(axis.text.x = element_text(angle = 0, hjust = 1)
          , legend.position = "right"
          , legend.title = element_blank()
    )  
  return(p)
}
repayment.plot <- function(df, fun, ii, target, ncol) {
  pp <- list()
  for (i in ii) {
    p <- fun(df=df, i=i, target=target)
    pp <- c(pp, list(p))
  }
  do.call("grid.arrange", c(pp, ncol=ncol))
}
repayment.plot(raw.credit, fun=repay.bar.plot, ii=6:11, target=24, ncol=2)
[IMG]
Let’s plot some corrplot to see if the data has any correlation.
The figure and color are related to correlation coefficients with the number is the p-values. Any sig.level less than or equal to 0.01, it will be crossed out.
credit.corr <- rcorr(as.matrix(tmp), type=c("pearson"))
colnames(credit.corr$r) <- c(1:length(colnames(credit.corr$r)))
colnames(credit.corr$P) <- c(1:length(colnames(credit.corr$P)))
corrplot(credit.corr$r, order="FPC", type="upper", method="square",
         tl.col="black", tl.srt = 45, p.mat = credit.corr$P, addCoef.col = "black",
         number.cex = 0.5, sig.level = 0.01)
[IMG]
Not much correlation to our binary ‘default.payment.next.month’ output. Let’s dive to the modeling part.
Model Building
This section is to start building the model. Several model will be attempted to build and predict the default payment output.
Separate the dataset onto training (70%) and testing (30%).
intraining <- createDataPartition(y=raw.credit$default.payment.next.month, p=0.7, list=F)
train.batch <- raw.credit[intraining,]
test.batch <- raw.credit[-intraining,]
Traditional Score using Logistic Regression
glm.m <- glm(default.payment.next.month ~ ., data=train.batch, family=binomial)
summary(glm.m)
glm.final <- glm(default.payment.next.month ~ limit_bal+sex+education+marriage+
                   pay_0+pay_2+pay_3+pay_5+pay_6+pay_amt1+pay_amt2, data=train.batch,
                 family=binomial)
summary(glm.final)
test.x <- select(test.batch, - default.payment.next.month)
test.batch$glm_scor1 <- predict(glm.final, type="response", test.x)
pred <- prediction(test.batch$glm_scor1, test.batch$default.payment.next.month)
glm.perf <- performance(pred, "tpr", "fpr")
ggplot(test.batch, aes(default.payment.next.month, glm_scor1)) +
  geom_boxplot() + theme_bw() +
  ggtitle("GLM Score distribution") +
  xlab("Default Payment") +
  ylab("Default Score")
[IMG]
# ROC
plot(glm.perf, lty=1, col="blue", main="Logistic Regression Performance")
abline(a=0, b= 1)
[IMG]
auc.perf <- performance(pred, measure = "auc")
acc.perf <- performance(pred, measure = "acc")
# AUC
cat(paste0("AUC:: ", auc.perf@y.values), "\n")
# Get accuracy and cutoff
ind <- which.max( slot(acc.perf, "y.values")[[1]] )
acc <- slot(acc.perf, "y.values")[[1]][ind]
cutoff <- slot(acc.perf, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))
# Draw Accuracy with cutoff
plot(acc.perf)
abline(v=cutoff)
[IMG]
glm.scor <- select(test.batch,
                   glm_scor1, 
                   default.payment.next.month)
glm.scor <- mutate(glm.scor,
                   predict_class = ifelse(glm_scor1 >= cutoff, 1, 0))
confusionMatrix(glm.scor$predict_class, glm.scor$default.payment.next.month, positive = '1')
Eventually, we can see which top X variables are influenced the predicted score. In this example, I will select only top 2 variables.
glm.terms <-predict(glm.final,type='terms',test.x)
get.top.var <- function(x,top=2){ 
  res=names(x)[order(x, decreasing = TRUE)][1:top]
  paste(res,collapse=";",sep="")
}
top.var <- apply(glm.terms, 1, get.top.var, top=2)
test.batch %>% select(default.payment.next.month, glm_scor1) %>%
  cbind(top.var) %>% head(5)
Decision Tree (rpart library)
rpart.model <- rpart(default.payment.next.month ~.,data=train.batch)
plot(rpart.model);text(rpart.model)
[IMG]
plotcp(rpart.model)
[IMG]
printcp(rpart.model)
rpart.pred <- predict(rpart.model, test.x, type="class")
confusionMatrix(rpart.pred, test.batch$default.payment.next.month, positive = '1')
test.batch$rpart_scor1 <- predict(rpart.model,type='prob',test.x)
pred2 <- prediction(test.batch$rpart_scor1[, 2], test.batch$default.payment.next.month)
rpart.perf1 <- performance(pred2,"tpr","fpr")
plot(rpart.perf1, col='red', main='Rpart Tree performance')
abline(a=0, b= 1)
[IMG]
Random Forest
rand.model <- randomForest(default.payment.next.month~.,data=train.batch, importance=T,
                            ntree=500, keep.forest=T)
varImpPlot(rand.model)
[IMG]
rand.pred <- predict(rand.model, test.x, type="class")
confusionMatrix(rand.pred, test.batch$default.payment.next.month, positive='1')
rand.prob <- predict(rand.model, test.x, type='prob')[,2]
pred4 <- prediction(rand.prob,test.batch$default.payment.next.month)
rf.perf1 <- performance(pred4,"tpr","fpr")
plot(glm.perf,col='red',lty=1, main='ROC Logistic VS. Rpart tree VS. Random Forest'); 
plot(rpart.perf1, col='blue',lty=2,add=T); 
plot(rf.perf1, col='green', lty=3, add=T);
legend(0.6,0.6,c('logistic regression','rpart tree', 'randomForest'),
       col=c('red','blue','green'),lwd=3)
[IMG]
Final print the AUC of all 3 predictions…
#simple model
glm.auc <- performance(pred,"auc")
cat("\n=== Logistic Regression ===\n")
cat(paste0(glm.auc@y.name,":: ", round(glm.auc@y.values[[1]], 2)))
cat(paste0("\nKS :: ", 
           round(max(attr(glm.perf,'y.values')[[1]]-attr(glm.perf,'x.values')[[1]]),2)))
# rpart
rpart.auc <- performance(pred2,"auc")
cat("\n\n=== Rpart tree ===\n")
cat(paste0(rpart.auc@y.name,":: ", round(rpart.auc@y.values[[1]],2)))
cat(paste0("\nKS :: ", 
           round(max(attr(rpart.perf1,'y.values')[[1]]-attr(rpart.perf1,'x.values')[[1]]),2)))
# RF
rf.auc <- performance(pred4,"auc")
cat("\n\n=== Random Forest ===\n")
cat(paste0(rf.auc@y.name,":: ", round(rf.auc@y.values[[1]], 2)))
cat(paste0("\nKS :: ", 
           round(max(attr(rf.perf1,'y.values')[[1]]-attr(rf.perf1,'x.values')[[1]]),2)))
```

## Facts established by the factory (deterministic; do not restate the numbers)
{
 "model": {
  "class": [
   "glm",
   "lm"
  ],
  "family": "binomial",
  "link": "logit",
  "formula": "default.payment.next.month ~ limit_bal + sex + education + marriage + pay_0 + pay_2 + pay_3 + pay_5 + pay_6 + pay_amt1 + pay_amt2",
  "response": "default.payment.next.month",
  "input_variables": [
   "limit_bal",
   "sex",
   "education",
   "marriage",
   "pay_0",
   "pay_2",
   "pay_3",
   "pay_5",
   "pay_6",
   "pay_amt1",
   "pay_amt2"
  ],
  "factor_levels": {
   "sex": [
    "1",
    "2"
   ],
   "education": [
    "1",
    "2",
    "3",
    "4"
   ],
   "marriage": [
    "1",
    "2",
    "3"
   ]
  },
  "contrasts": {
   "sex": "contr.treatment",
   "education": "contr.treatment",
   "marriage": "contr.treatment"
  },
  "n_coefficients": 15,
  "converged": true,
  "iterations": 6,
  "deviance": 18587.8388471168,
  "null_deviance": 22196.2617795277,
  "aic": 18617.8388471168,
  "r_version": "R version 4.6.1 (2026-06-24)",
  "r_packages_loaded": [
   "MASS",
   "Matrix",
   "ModelMetrics",
   "R6",
   "RColorBrewer",
   "Rcpp",
   "S7",
   "caret",
   "class",
   "cli",
   "codetools",
   "compiler",
   "data.table",
   "dplyr",
   "farver",
   "foreach",
   "future",
   "future.apply",
   "generics",
   "ggplot2",
   "globals",
   "glue",
   "gower",
   "grid",
   "gtable",
   "hardhat",
   "ipred",
   "iterators",
   "jsonlite",
   "lattice",
   "lava",
   "lifecycle",
   "listenv",
   "lubridate",
   "magrittr",
   "nlme",
   "nnet",
   "pROC",
   "parallel",
   "parallelly",
   "pillar",
   "pkgconfig",
   "plyr",
   "prodlim",
   "purrr",
   "recipes",
   "reshape2",
   "rlang",
   "rpart",
   "scales",
   "splines",
   "stats4",
   "stringi",
   "stringr",
   "survival",
   "tibble",
   "tidyselect",
   "timeDate",
   "timechange",
   "tools",
   "vctrs",
   "withr"
  ]
 },
 "column_classes_after_preprocessing": {
  "limit_bal": "integer",
  "sex": "factor",
  "education": "factor",
  "marriage": "factor",
  "age": "integer",
  "pay_0": "numeric",
  "pay_2": "numeric",
  "pay_3": "numeric",
  "pay_4": "numeric",
  "pay_5": "numeric",
  "pay_6": "numeric",
  "bill_amt1": "integer",
  "bill_amt2": "integer",
  "bill_amt3": "integer",
  "bill_amt4": "integer",
  "bill_amt5": "integer",
  "bill_amt6": "integer",
  "pay_amt1": "integer",
  "pay_amt2": "integer",
  "pay_amt3": "integer",
  "pay_amt4": "integer",
  "pay_amt5": "integer",
  "pay_amt6": "integer",
  "default.payment.next.month": "factor"
 },
 "split": {
  "n_train": 21001,
  "n_test": 8999,
  "seed_used_by_harness": 2026
 },
 "protected_attributes_in_data": [
  "SEX",
  "MARRIAGE",
  "AGE",
  "EDUCATION"
 ],
 "protected_attributes_used_as_inputs": [
  "sex",
  "education",
  "marriage"
 ],
 "data_profile": {
  "n_rows": 30000,
  "columns": [
   {
    "name": "ID",
    "dtype": "int64",
    "n_unique": 30000,
    "n_missing": 0
   },
   {
    "name": "LIMIT_BAL",
    "dtype": "int64",
    "n_unique": 81,
    "n_missing": 0
   },
   {
    "name": "SEX",
    "dtype": "int64",
    "n_unique": 2,
    "n_missing": 0,
    "levels": [
     1,
     2
    ]
   },
   {
    "name": "EDUCATION",
    "dtype": "int64",
    "n_unique": 7,
    "n_missing": 0,
    "levels": [
     0,
     1,
     2,
     3,
     4,
     5,
     6
    ]
   },
   {
    "name": "MARRIAGE",
    "dtype": "int64",
    "n_unique": 4,
    "n_missing": 0,
    "levels": [
     0,
     1,
     2,
     3
    ]
   },
   {
    "name": "AGE",
    "dtype": "int64",
    "n_unique": 56,
    "n_missing": 0
   },
   {
    "name": "PAY_0",
    "dtype": "int64",
    "n_unique": 11,
    "n_missing": 0,
    "levels": [
     -2,
     -1,
     0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8
    ]
   },
   {
    "name": "PAY_2",
    "dtype": "int64",
    "n_unique": 11,
    "n_missing": 0,
    "levels": [
     -2,
     -1,
     0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8
    ]
   },
   {
    "name": "PAY_3",
    "dtype": "int64",
    "n_unique": 11,
    "n_missing": 0,
    "levels": [
     -2,
     -1,
     0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8
    ]
   },
   {
    "name": "PAY_4",
    "dtype": "int64",
    "n_unique": 11,
    "n_missing": 0,
    "levels": [
     -2,
     -1,
     0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8
    ]
   },
   {
    "name": "PAY_5",
    "dtype": "int64",
    "n_unique": 10,
    "n_missing": 0,
    "levels": [
     -2,
     -1,
     0,
     2,
     3,
     4,
     5,
     6,
     7,
     8
    ]
   },
   {
    "name": "PAY_6",
    "dtype": "int64",
    "n_unique": 10,
    "n_missing": 0,
    "levels": [
     -2,
     -1,
     0,
     2,
     3,
     4,
     5,
     6,
     7,
     8
    ]
   },
   {
    "name": "BILL_AMT1",
    "dtype": "int64",
    "n_unique": 22723,
    "n_missing": 0
   },
   {
    "name": "BILL_AMT2",
    "dtype": "int64",
    "n_unique": 22346,
    "n_missing": 0
   },
   {
    "name": "BILL_AMT3",
    "dtype": "int64",
    "n_unique": 22026,
    "n_missing": 0
   },
   {
    "name": "BILL_AMT4",
    "dtype": "int64",
    "n_unique": 21548,
    "n_missing": 0
   },
   {
    "name": "BILL_AMT5",
    "dtype": "int64",
    "n_unique": 21010,
    "n_missing": 0
   },
   {
    "name": "BILL_AMT6",
    "dtype": "int64",
    "n_unique": 20604,
    "n_missing": 0
   },
   {
    "name": "PAY_AMT1",
    "dtype": "int64",
    "n_unique": 7943,
    "n_missing": 0
   },
   {
    "name": "PAY_AMT2",
    "dtype": "int64",
    "n_unique": 7899,
    "n_missing": 0
   },
   {
    "name": "PAY_AMT3",
    "dtype": "int64",
    "n_unique": 7518,
    "n_missing": 0
   },
   {
    "name": "PAY_AMT4",
    "dtype": "int64",
    "n_unique": 6937,
    "n_missing": 0
   },
   {
    "name": "PAY_AMT5",
    "dtype": "int64",
    "n_unique": 6897,
    "n_missing": 0
   },
   {
    "name": "PAY_AMT6",
    "dtype": "int64",
    "n_unique": 6939,
    "n_missing": 0
   },
   {
    "name": "default payment next month",
    "dtype": "int64",
    "n_unique": 2,
    "n_missing": 0,
    "levels": [
     0,
     1
    ]
   }
  ]
 }
}

## REQUIRED disclosures (detected deterministically; each id must be covered by a limitation)
{
 "D-01:sex": "protected / sensitive attribute 'sex' is a direct model input (Equality Act 2010, Consumer Duty)",
 "D-01:education": "protected / sensitive attribute 'education' is a direct model input (Equality Act 2010, Consumer Duty)",
 "D-01:marriage": "protected / sensitive attribute 'marriage' is a direct model input (Equality Act 2010, Consumer Duty)",
 "D-02": "random train/test split with no seed in the legacy source: results are not reproducible",
 "D-04": "category codes are merged / recoded before fitting; the merge changes what the model learns (lines [17, 19, 23])",
 "D-05": "terms not significant at 5% in the final model: ['education2', 'education3', 'marriage3', 'pay_2']"
}

List every formula variable and the target in `variables`, with factor levels exactly as captured.
Output JSON schema:
{
 "$defs": {
  "DocLimitation": {
   "properties": {
    "text": {
     "title": "Text",
     "type": "string"
    },
    "severity": {
     "enum": [
      "High",
      "Medium",
      "Low"
     ],
     "title": "Severity",
     "type": "string"
    },
    "disclosure_ids": {
     "description": "Required disclosure ids (D-xx) this limitation covers; [] if none",
     "items": {
      "type": "string"
     },
     "title": "Disclosure Ids",
     "type": "array"
    },
    "source_lines": {
     "items": {
      "type": "integer"
     },
     "title": "Source Lines",
     "type": "array"
    }
   },
   "required": [
    "text",
    "severity",
    "disclosure_ids"
   ],
   "title": "DocLimitation",
   "type": "object"
  },
  "DocStep": {
   "properties": {
    "order": {
     "title": "Order",
     "type": "integer"
    },
    "description": {
     "title": "Description",
     "type": "string"
    },
    "columns": {
     "items": {
      "type": "string"
     },
     "title": "Columns",
     "type": "array"
    },
    "source_lines": {
     "items": {
      "type": "integer"
     },
     "title": "Source Lines",
     "type": "array"
    },
    "author_rationale": {
     "description": "The author's stated reason, from comments or documents; '' if none",
     "title": "Author Rationale",
     "type": "string"
    },
    "is_assumption": {
     "description": "True if the step rests on an unverified assumption by the author",
     "title": "Is Assumption",
     "type": "boolean"
    }
   },
   "required": [
    "order",
    "description",
    "columns",
    "source_lines",
    "author_rationale",
    "is_assumption"
   ],
   "title": "DocStep",
   "type": "object"
  },
  "DocVariable": {
   "properties": {
    "name": {
     "description": "Variable name exactly as it appears in the model formula",
     "title": "Name",
     "type": "string"
    },
    "role": {
     "enum": [
      "input",
      "target"
     ],
     "title": "Role",
     "type": "string"
    },
    "kind": {
     "enum": [
      "numeric",
      "categorical",
      "ordinal_as_numeric",
      "binary"
     ],
     "title": "Kind",
     "type": "string"
    },
    "description": {
     "description": "Business meaning, from source comments / supporting documents",
     "title": "Description",
     "type": "string"
    },
    "levels": {
     "description": "Categorical/binary: R factor levels in R order",
     "items": {
      "type": "string"
     },
     "title": "Levels",
     "type": "array"
    },
    "level_meaning": {
     "additionalProperties": {
      "type": "string"
     },
     "description": "Level code -> meaning, if documented",
     "title": "Level Meaning",
     "type": "object"
    },
    "transformation": {
     "description": "What the code does to this variable before fitting; 'none' if nothing",
     "title": "Transformation",
     "type": "string"
    },
    "source_lines": {
     "description": "Line numbers in the numbered legacy source that define or use it",
     "items": {
      "type": "integer"
     },
     "title": "Source Lines",
     "type": "array"
    }
   },
   "required": [
    "name",
    "role",
    "kind",
    "description",
    "transformation",
    "source_lines"
   ],
   "title": "DocVariable",
   "type": "object"
  }
 },
 "properties": {
  "purpose": {
   "title": "Purpose",
   "type": "string"
  },
  "intended_use": {
   "title": "Intended Use",
   "type": "string"
  },
  "out_of_scope_use": {
   "title": "Out Of Scope Use",
   "type": "string"
  },
  "methodology": {
   "description": "Estimation method in plain words; no statistics",
   "title": "Methodology",
   "type": "string"
  },
  "target_definition": {
   "title": "Target Definition",
   "type": "string"
  },
  "variables": {
   "description": "EVERY formula variable plus the target, nothing else",
   "items": {
    "$ref": "#/$defs/DocVariable"
   },
   "title": "Variables",
   "type": "array"
  },
  "preprocessing": {
   "description": "In source execution order",
   "items": {
    "$ref": "#/$defs/DocStep"
   },
   "title": "Preprocessing",
   "type": "array"
  },
  "variable_selection": {
   "title": "Variable Selection",
   "type": "string"
  },
  "assumptions": {
   "items": {
    "type": "string"
   },
   "title": "Assumptions",
   "type": "array"
  },
  "limitations": {
   "items": {
    "$ref": "#/$defs/DocLimitation"
   },
   "title": "Limitations",
   "type": "array"
  },
  "code_vs_docs_discrepancies": {
   "items": {
    "type": "string"
   },
   "title": "Code Vs Docs Discrepancies",
   "type": "array"
  },
  "open_questions": {
   "description": "Questions for the model owner; facts that cannot be established from the evidence",
   "items": {
    "type": "string"
   },
   "title": "Open Questions",
   "type": "array"
  }
 },
 "required": [
  "purpose",
  "intended_use",
  "out_of_scope_use",
  "methodology",
  "target_definition",
  "variables",
  "preprocessing",
  "variable_selection",
  "assumptions",
  "limitations",
  "code_vs_docs_discrepancies",
  "open_questions"
 ],
 "title": "DocExtract",
 "type": "object"
}

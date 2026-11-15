
# ---- notebook code cell 6 ----
# Remark: All necessary libraries are pre-installed on Kaggle. We call them directly before usage 

dat <- read.csv("../input/french-motor-claims-datasets-fremtpl2freq/freMTPL2freq.csv",
                header=TRUE, stringsAsFactors = TRUE)

# inspect data set
dim(dat)  # count rows and colums
head(dat) # show first few policies

# ---- notebook code cell 7 ----
# new: take claim number from severity file (see https://www.kaggle.com/code/floser/comparing-claims-fremtpl2freq-sev)
sev <- read.csv("../input/fremtpl2sev/freMTPL2sev.csv")
sev$ClaimNb <- 1
sev_agg <- aggregate(sev,by=list(IDpol=sev$IDpol),FUN = sum)[c("IDpol","ClaimNb")]
# checks, sums and counts
head(sev_agg)

# ---- notebook code cell 8 ----
# left join with claims frequency data (exposures, features and without questionable ClaimNb). 
dat <- merge(x=dat[,-2], y=sev_agg, by="IDpol", all.x=TRUE)
dat[is.na(dat)] <- 0

# data preparations and corrections as used in Schelldorfer and Wüthrich (2019)
dat$ClaimNb  <- pmin(dat$ClaimNb,4)   
dat$Exposure <- pmin(dat$Exposure,1) 

# new: further data preparations
dat$DrivAge  <- pmin(dat$DrivAge,90)   
dat$VehAge  <- pmin(dat$VehAge,40)   

# show summary statistics 
summary(dat)

# ---- notebook code cell 10 ----
library(rpart) # "Recursive partitioning for classification, regression and survival trees". Default metric: Gini impurity
tree <- rpart(cbind(Exposure,ClaimNb) ~ Area + VehPower + VehAge + DrivAge 
              + BonusMalus + VehBrand + VehGas + Density + Region, dat, 
              method="poisson", control=rpart.control(maxdepth=3,cp=0.001))      
#            complexity-parameter cp is used to control the number of splits

# ---- notebook code cell 11 ----
library(repr) # Scale plot size:
options(repr.plot.width=12, repr.plot.height = 7)

library(rpart.plot)
rpart.plot(tree) # display decision tree

# ---- notebook code cell 14 ----
#library(tidyverse) # work quietly:
suppressMessages(library(tidyverse))

# ---- notebook code cell 16 ----
# To focus on new car effects, all cars from the age of four belong to the same group in the following plot:
dat %>% 
  mutate(VehAgeS=as.factor(pmin(VehAge, 4))) %>% 
  group_by(VehBrand,VehAgeS,VehGas) %>% 
  summarize(claim_frequency=sum(as.double(ClaimNb))/sum(Exposure), expo=sum(Exposure)) %>% 
  filter(expo>=100)  %>% 
  ggplot(aes(x=VehBrand, y=claim_frequency, color=VehAgeS)) + 
  facet_grid(. ~ VehGas) + 
  geom_point(size=5,shape=15) + theme_light(base_size = 16)

# ---- notebook code cell 18 ----
dat %>% 
  filter(VehBrand!='B12' & BonusMalus<=125)  %>% 
  mutate(VehAge=pmin(VehAge,20),
         BonusMalusCeil=as.factor(ceiling(BonusMalus/25)*25)) %>% 
  group_by(VehAge,BonusMalusCeil) %>% 
  summarize(claim_frequency=sum(as.double(ClaimNb))/sum(Exposure)) %>% 
  ggplot(aes(x=VehAge, y=claim_frequency, color=BonusMalusCeil)) + 
  geom_point() + geom_smooth() + theme_light(base_size = 16)

# ---- notebook code cell 20 ----
# To avoid noisy effects of very small groups we limit the on BonusMalus up to 125% driver age is grouped in five-year steps. 
# Further all driver from the age of 85 belong to the same group. 
dat %>% 
  filter(BonusMalus<=125)  %>% 
  mutate(DrivAge=ceiling(pmin(DrivAge,85)/5)*5,
         BonusMalusCeil=as.factor(ceiling(BonusMalus/25)*25)) %>% 
  group_by(DrivAge,BonusMalusCeil) %>% 
  summarize(claim_frequency=sum(as.double(ClaimNb))/sum(Exposure)) %>% 
  ggplot(aes(x=DrivAge, y=claim_frequency, color=BonusMalusCeil)) + 
  geom_point() + geom_smooth() + theme_light(base_size = 16)

# ---- notebook code cell 22 ----
# Violin plot DrivAge vs number of claims for BonusMalus-groups:
dat %>% filter(ClaimNb<4 & BonusMalus<=125) %>% 
  mutate(BonusMalusCeil=as.factor(ceiling(BonusMalus/25)*25), 
         ClaimNb=as.factor(ClaimNb)) %>% 
  ggplot(aes(ClaimNb, DrivAge)) + 
  geom_violin(aes(fill = ClaimNb)) +
  facet_wrap(~BonusMalusCeil)

# ---- notebook code cell 24 ----
# View distribution of all claims 
table(dat$ClaimNb)

# ---- notebook code cell 26 ----
dat1 <- dat
dat1$AreaGLM <- as.integer(dat1$Area)
dat1$VehPowerGLM <- as.factor(pmin(dat1$VehPower,9))
VehAgeGLM <- cbind(c(0:110), c(1, rep(2,10), rep(3,100)))
dat1$VehAgeGLM <- as.factor(VehAgeGLM[dat1$VehAge+1,2])
dat1[,"VehAgeGLM"] <-relevel(dat1[,"VehAgeGLM"], ref="2")
DrivAgeGLM <- cbind(c(18:100), c(rep(1,21-18), rep(2,26-21), rep(3,31-26), rep(4,41-31), rep(5,51-41), rep(6,71-51), rep(7,101-71)))
dat1$DrivAgeGLM <- as.factor(DrivAgeGLM[dat1$DrivAge-17,2])
dat1[,"DrivAgeGLM"] <-relevel(dat1[,"DrivAgeGLM"], ref="5")
dat1$BonusMalusGLM <- as.integer(pmin(dat1$BonusMalus, 150))
dat1$DensityGLM <- as.numeric(log(dat1$Density))
dat1[,"Region"] <-relevel(dat1[,"Region"], ref="R24")
head(dat1) # show first few policies and generated features

# ---- notebook code cell 28 ----
# Modified function PDX: Weighted Poisson Deviance
PDW <- function(pred, obs, ex = rep(1, length(obs))) {
  200 * sum( ex * ( pred - obs  + log((obs / pred) ^ (obs )))) / sum(ex)
}

# Function PD2: Print Poisson Deviance learn/test
PDW2 <- function(txt, l.c, l.x, l.e, t.c, t.x, t.e) {
    sprintf("%s, Learn/Test: %.2f%% / %.2f%%", txt, PDW(l.c, l.x, l.e), PDW(t.c, t.x, t.e)) 
}

# Function CF2: Print claim frequency
CF2 <- function(txt, l.c, l.x, t.c, t.x) {
    sprintf("%s: %.2f%% / %.2f%%", txt, sum(l.c)/sum(l.x)*100, sum(t.c)/sum(t.x)*100) 
}

# Function Benchmark.GLM2: Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
Benchmark.GLM2 <- function(txt, pred) {
  index <- ((PDW(pred, test$ClaimNb, test$Exposure) - PDW(test$fit.cf, test$ClaimNb, test$Exposure)) / (PDW(test$fitGLM2, test$ClaimNb, test$Exposure) - PDW(test$fit.cf, test$ClaimNb, test$Exposure))) * 100
  sprintf("GLM2-Improvement-Index (PD test) of %s: %.1f%%", txt, index) 
}

# ---- notebook code cell 30 ----
# for later use (Ch.5) we create five 20%-subsamples ("folds") and take the last fold as the holdout data set
k <- 5
set.seed(42)
fold <- sample(1:k, nrow(dat1), replace = TRUE)
dat1$fold <- fold
learn <- dat1[dat1$fold != 5,]    # 80%
test <- dat1[ dat1$fold == 5,]    # 20%
CF2("Claim Frequency (Actual) Learn/Test", learn$ClaimNb,learn$Exposure, test$ClaimNb,test$Exposure)

# ---- notebook code cell 31 ----
# # Export data partition (for later use in other notebooks)
# write_csv(dat1[,c("IDpol","fold")],"freMTPL2freq_folds2.csv")

# ---- notebook code cell 32 ----
# # Check it by reading back
# dat_folds <- read.csv("freMTPL2freq_folds2.csv",header=TRUE)
# head(dat_folds)

# ---- notebook code cell 34 ----
# Model INT "predictions"
cf <- sum(learn$ClaimNb)/sum(learn$Exposure) # claim frequency
learn$fit.cf <- cf*learn$Exposure
test$fit.cf <- cf*test$Exposure
# Print Poisson Deviance (weighted)
PDW2("Poisson Deviance INT", learn$fit.cf,learn$ClaimNb,learn$Exposure, test$fit.cf,test$ClaimNb,test$Exposure)

# ---- notebook code cell 35 ----
# Print claim frequency actual vs predicted (Intercept model)
CF2("Claim Frequency INT, Test-Sample, Actual/Predicted", test$ClaimNb,test$Exposure, test$fit.cf,test$Exposure)

# ---- notebook code cell 38 ----
f.glm1 <- ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region + AreaGLM
d.glm1 <- glm(f.glm1, data = learn, offset = log(Exposure), family = poisson())

summary(d.glm1) # print fit statistics and parameter list

# ---- notebook code cell 39 ----
# make predictions
learn$fitGLM1 <- fitted(d.glm1)
test$fitGLM1 <- predict(d.glm1, newdata=test, type="response")

# Print claim frequency actual vs predicted
CF2("Claim Frequency GLM1, Test-Sample, Actual/Predicted", test$ClaimNb,test$Exposure, test$fitGLM1,test$Exposure)

# Print Poisson Deviance
PDW2("Poisson Deviance GLM1", learn$fitGLM1,learn$ClaimNb,learn$Exposure, test$fitGLM1,test$ClaimNb,test$Exposure)

# ---- notebook code cell 41 ----
f.glm2 <- ClaimNb ~ VehPowerGLM + VehAgeGLM + BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region + AreaGLM + DrivAge + log(DrivAge) +  I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4)
d.glm2 <- glm(f.glm2, data=learn, offset=log(Exposure), family=poisson())

summary(d.glm2) # print fit statistics and parameter list

# ---- notebook code cell 42 ----
# make predictions
learn$fitGLM2 <- fitted(d.glm2)
test$fitGLM2 <- predict(d.glm2, newdata=test, type="response")
dat$fitGLM2 <- predict(d.glm2, newdata=dat1, type="response") # for the CANN

# Print claim frequency actual vs predicted
CF2("Claim Frequency GLM2, Test-Sample, Actual/Predicted", test$ClaimNb,test$Exposure, test$fitGLM2,test$Exposure)

# Print Poisson Deviance
PDW2("Poisson Deviance GLM2", learn$fitGLM2,learn$ClaimNb,learn$Exposure, test$fitGLM2,test$ClaimNb,test$Exposure)

# ---- notebook code cell 44 ----
# Print Poisson Deviance
PDW2("Poisson Deviance GLM2", learn$fitGLM2,learn$ClaimNb,learn$Exposure, test$fitGLM2,test$ClaimNb,test$Exposure)

# ---- notebook code cell 48 ----
# data preparation
suppressMessages(library(glmnet))

f.lasso <- as.formula(ClaimNb ~ (VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region + AreaGLM)^2) 
x <- sparse.model.matrix(f.lasso, dat1)[, -1]

learn.glmnet <- x[dat1$fold != 5,]    # 80% 
test.glmnet <- x[dat1$fold == 5,]    # 20%

# ---- notebook code cell 49 ----
# determine lambda.min using cross-validation
{t1 <- proc.time()
   cv.lasso <- cv.glmnet(x = learn.glmnet, y=learn$ClaimNb, alpha = 1, offset = log(learn$Exposure), family = "poisson", nfolds = 5)
   (proc.time()-t1)}

print(cv.lasso)

# Plot the regularization path
options(repr.plot.width=12, repr.plot.height = 9) 
plot(cv.lasso)

# ---- notebook code cell 50 ----
# Show (a few) non-zero coefficients
df_lasso_coef <- as.data.frame(as.matrix(coef(cv.lasso, s = "lambda.min")))
df_lasso_coef_filtered <- subset(df_lasso_coef, s1 != 0)
dim(df_lasso_coef_filtered)
head(df_lasso_coef_filtered)

# ---- notebook code cell 51 ----
# Generate predictions using lambda.min (optimized penalty parameter)
test$fitLASSO <-  predict(cv.lasso, newx = test.glmnet, newoffset = log(test$Exposure), type = "response", s = "lambda.min")[,1] 
learn$fitLASSO <- predict(cv.lasso, newx = learn.glmnet, newoffset = log(learn$Exposure), type = "response" , s = "lambda.min")[,1] 

# Print claim frequency actual vs predicted 
CF2('Claim Frequency LASSO, Test-Sample, Actual/Predicted', test$ClaimNb,test$Exposure, test$fitLASSO,test$Exposure)

# Print Poisson Deviance
PDW2("Poisson Deviance LASSO",
learn$fitLASSO,learn$ClaimNb,learn$Exposure,
test$fitLASSO,test$ClaimNb,test$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement 
Benchmark.GLM2("LASSO", test$fitLASSO)

# ---- notebook code cell 53 ----
# determine lambda.min using cross-validation
{t1 <- proc.time()
cvRIDGE <- cv.glmnet(x=learn.glmnet, y=learn$ClaimNb/learn$Exposure, alpha = 0, weights = learn$Exposure, family = "poisson", nfolds = 5) 
 (proc.time()-t1)}

plot(cvRIDGE)

cvRIDGE$lambda.min

# Generate predictions using lambda.min (optimized penalty parameter)
test$fitRIDGE <-  exp(predict(cvRIDGE, newx = test.glmnet, s = "lambda.min"))[,1] * test$Exposure
learn$fitRIDGE <- exp(predict(cvRIDGE, newx = learn.glmnet, s = "lambda.min"))[,1] * learn$Exposure

# Print claim frequency actual vs predicted
CF2('Claim Frequency RIDGE, Test-Sample, Actual/Predicted', test$ClaimNb,test$Exposure, test$fitRIDGE,test$Exposure)

# Print Poisson Deviance
PDW2("Poisson Deviance RIDGE", learn$fitRIDGE,learn$ClaimNb,learn$Exposure, test$fitRIDGE,test$ClaimNb,test$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
Benchmark.GLM2("RIDGE", test$fitRIDGE)

# ---- notebook code cell 55 ----
suppressMessages(library(mgcv))

# bam(): Generalized additive models for very large datasets, see https://cran.r-project.org/web/packages/mgcv/mgcv.pdf
#        "The advantage of bam is much lower memory footprint than gam, but it can also be much faster, for large datasets."

{t1 <- proc.time()
  d.gam3 <- bam(ClaimNb ~ s(VehAge) + s(DrivAge) + s(BonusMalusGLM) + VehPowerGLM + VehGas + VehBrand + AreaGLM + DensityGLM + Region + offset(log(Exposure)), 
                ,data=learn, scale=-1, family=poisson)
(proc.time()-t1)}
summary(d.gam3) # print fit statistics

# ---- notebook code cell 56 ----
# make predictions
learn$fitGAM3 <- fitted(d.gam3)
test$fitGAM3 <- predict(d.gam3, newdata=test, type="response")
# Print claim frequency actual vs predicted
CF2("Claim Frequency GLM2, Test-Sample, Actual/Predicted", test$ClaimNb,test$Exposure, test$fitGAM3,test$Exposure)

# Print Poisson Deviance
PDW2('Poisson Deviance GAM3', learn$fitGAM3,learn$ClaimNb,learn$Exposure, test$fitGAM3,test$ClaimNb,test$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
Benchmark.GLM2("GAM3", test$fitGAM3)

# ---- notebook code cell 58 ----
# 3D-plot 
vis.gam(d.gam3, theta=-135)

# ---- notebook code cell 59 ----
# Plot residuals of spline-functions
plot(d.gam3, residuals=TRUE)

# ---- notebook code cell 60 ----
# plot marginal effects
library(ggplot2)
library(ggeffects)

ggpredict(d.gam3, c("BonusMalusGLM", "DrivAge")) %>%
  plot() +
  theme_minimal()

# ---- notebook code cell 61 ----
ggpredict(d.gam3, c("DrivAge", "BonusMalusGLM")) %>%
  plot() +
  theme_minimal()

# ---- notebook code cell 62 ----
ggpredict(d.gam3, c("DrivAge", "VehAge")) %>%
  plot() +
  theme_minimal()

# ---- notebook code cell 63 ----
ggpredict(d.gam3, c("VehAge", "DrivAge")) %>%
  plot() +
  theme_minimal()

# ---- notebook code cell 67 ----
summary(test[,c(22:25)])

# ---- notebook code cell 68 ----
head(test[,c(2:7,22:25)])

# ---- notebook code cell 73 ----
num_features <- c('BonusMalus','DrivAge','VehPower','VehAge','Density')
cat_features <- c('Area','VehBrand','VehGas','Region')
features <- c(num_features, cat_features)

# Feature pre-processing for XGBoost and lightGBM

df_feat <- dat[,c(features)]
df_feat$Area <- as.integer(df_feat$Area)
print(paste("Features before one-hot-encoding:",dim(df_feat)[2]))

# one-hot encoding for Region and VehBrand
df_feat <- as.data.frame(model.matrix( ~ 0 +. ,data = df_feat)) 
print(paste("Features after one-hot-encoding: ",dim(df_feat)[2]))

# split data into learn and test
X_learn <- df_feat[dat1$fold != 5,]    # 80%
X_test  <- df_feat[dat1$fold == 5,]    # 20%

# ---- notebook code cell 74 ----
# create monotonic constraints vector: increasing: +1, decreasing: -1, unconstraint: 0
mtc <- c(rep( 0, length(X_learn)))
head(X_learn,2) # check feature names
mtc[1] <- 1 # BonusMalus increasing 
print(mtc)

# ---- notebook code cell 75 ----
suppressMessages(library(xgboost))
# construct xgb.DMatrices (these are the internal data structures used by XGBoost during training)  
dtrain <- xgb.DMatrix(data = data.matrix(X_learn), label = learn$ClaimNb/learn$Exposure, weight=learn$Exposure)
# make label wrong (shouldn't change anything): change to fold-number
dtest <- xgb.DMatrix(data = data.matrix(X_test), label = test$fold/test$Exposure, weight=test$Exposure)

# ---- notebook code cell 77 ----
# train model (using tuned hyperparameters, see corresponding Python notebook)
{t1 <- proc.time()
  fit.xgb0 <- xgb.train(data=dtrain, objective='count:poisson', 
                       nrounds = 500, max_depth = 5, eta = 0.05, tree_method = "hist")
(proc.time()-t1)}

# ---- notebook code cell 78 ----
## make predictions
learn$fitXGB0 <- predict(fit.xgb0 , newdata = dtrain) * learn$Exposure
test$fitXGB0 <- predict(fit.xgb0, newdata = dtest) * test$Exposure

# Print claim frequency actual vs predicted
CF2('Claim Frequency XGB, Test-Sample, Actual/Predicted', test$ClaimNb,test$Exposure, test$fitXGB0,test$Exposure)

# Print Poisson Deviance
PDW2("Poisson Deviance XGB", learn$fitXGB0,learn$ClaimNb,learn$Exposure, test$fitXGB0,test$ClaimNb,test$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
test$fit.cf <- test$Exposure * sum(learn$ClaimNb)/sum(learn$Exposure) # (recalculate INT-Model)
Benchmark.GLM2("XGB unconstraint", test$fitXGB0)

# ---- notebook code cell 80 ----
# train model (using tuned hyperparameters, see corresponding Python notebook)
{t1 <- proc.time()
  fit.xgb <- xgb.train(data=dtrain, objective='count:poisson', monotone_constraints = mtc,
                       nrounds = 500, max_depth = 5, eta = 0.05, tree_method = "hist")
(proc.time()-t1)}

# ---- notebook code cell 81 ----
## make predictions
learn$fitXGB <- predict(fit.xgb , newdata = dtrain) * learn$Exposure
test$fitXGB <- predict(fit.xgb, newdata = dtest) * test$Exposure

# Print claim frequency actual vs predicted
CF2('Claim Frequency XGB, Test-Sample, Actual/Predicted', test$ClaimNb,test$Exposure, test$fitXGB,test$Exposure)

# Print Poisson Deviance
PDW2("Poisson Deviance XGB", learn$fitXGB,learn$ClaimNb,learn$Exposure, test$fitXGB,test$ClaimNb,test$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
test$fit.cf <- test$Exposure * sum(learn$ClaimNb)/sum(learn$Exposure) # (recalculate INT-Model)
Benchmark.GLM2("XGB", test$fitXGB)

# ---- notebook code cell 84 ----
library(catboost)

learn_pool <- catboost.load_pool(data = learn[features], label = learn$ClaimNb, 
                                 cat_features = cat_features, baseline = as.matrix(learn$Exposure))

# make label wrong (shouldn't change anything): change to fold-number
test_pool <- catboost.load_pool(data = test[features], label = test$fold, 
                                 cat_features = cat_features, baseline = as.matrix(test$Exposure))

# ---- notebook code cell 85 ----
# create monotonic constraints vector: increasing: +1, decreasing: -1, unconstraint: 0
mtcc <- c(rep( 0, length(features)))
mtcc[1] <- 1 # BonusMalus increasing 
print(mtcc)
head(learn[features],3)

# ---- notebook code cell 86 ----
# train model (using tuned hyperparameters, see corresponding Python notebook)
fit_params <- list(logging_level='Silent', loss_function='Poisson', monotone_constraints = mtcc, random_seed=42,
                  iterations = 500, learning_rate = 0.15, depth = 3)
{t1 <- proc.time()
  CB <- catboost.train(learn_pool, params = fit_params)
(proc.time()-t1)}

# ---- notebook code cell 87 ----
# Variable importance plot
catboost.get_feature_importance(CB, test_pool) %>% 
  as.data.frame() %>% 
  setNames("Importance") %>% 
  rownames_to_column("Feature") %>% 
  ggplot(aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip()

# ---- notebook code cell 88 ----
test$fitCB <- exp(catboost.predict(CB,test_pool))
learn$fitCB <-  exp(catboost.predict(CB,learn_pool))

# Print claim frequency actual vs predicted
CF2('Claim Frequency CatBoost, Test-Sample, Actual/Predicted', test$ClaimNb,test$Exposure, test$fitCB,test$Exposure)

# Print Poisson Deviance
PDW2("Poisson Deviance CatBoost", learn$fitCB,learn$ClaimNb,learn$Exposure, test$fitCB,test$ClaimNb,test$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
Benchmark.GLM2("CB", test$fitCB)

# ---- notebook code cell 91 ----
suppressMessages(library(lightgbm))

# LGB matrices (frequency-weighted-version)
# https://towardsdatascience.com/how-to-handle-the-exposure-offset-with-boosted-trees-fd09cc946837
llearn <- lgb.Dataset(data = data.matrix(X_learn), label = learn$ClaimNb/learn$Exposure, weight=learn$Exposure)

# ---- notebook code cell 92 ----
# train lgb-model (using default hyperparameters, see corresponding Python notebook)
{t2 <- proc.time()
  LGB <- lgb.train(data=llearn, objective='poisson', monotone_constraints = mtc, num_leaves=31, learning_rate=0.1, n_estimators=100)
(proc.time()-t2)}

# ---- notebook code cell 93 ----
## make predictions
learn$fitLGB <- predict(LGB, data.matrix(X_learn)) * learn$Exposure
test$fitLGB <- predict(LGB, data.matrix(X_test)) * test$Exposure

# Print claim frequency actual vs predicted
CF2('Claim Frequency lightGBM, Test-Sample, Actual/Predicted', test$ClaimNb,test$Exposure, test$fitLGB,test$Exposure)

# Print Poisson Deviance
PDW2("Poisson Deviance lightGBM", learn$fitLGB,learn$ClaimNb,learn$Exposure, test$fitLGB,test$ClaimNb,test$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
test$fit.cf <- test$Exposure * sum(learn$ClaimNb)/sum(learn$Exposure) # (recalculate INT-Model)
Benchmark.GLM2("LGB", test$fitLGB)

# ---- notebook code cell 96 ----
# Generate explainer object with TreeSHAP to explain the LightGBM model
library(shapviz)
{t2 <- proc.time()
 shap_lgb_test <- shapviz(LGB, X_pred = data.matrix(X_test))  
(proc.time()-t2)}

# ---- notebook code cell 98 ----
# SHAP importance: summary plot
sv_importance(shap_lgb_test, kind = "bee")

# ---- notebook code cell 99 ----
# SHAP feature importance
sv_importance(shap_lgb_test, show_numbers = TRUE)

# ---- notebook code cell 101 ----
# SHAP dependence plots (note the different y-scales)
options(repr.plot.width=20, repr.plot.height = 15)

theme_set(theme_gray(base_size = 7))

sv_dependence(shap_lgb_test, colnames(X_test), jitter_width = 0.05, alpha = 0.2) &
  ylim(-0.8, 1.2)

options(repr.plot.width=12, repr.plot.height = 7)

# ---- notebook code cell 103 ----
head(test,4)

# ---- notebook code cell 104 ----
# Waterfall plot of first observation (log-scale)
sv_waterfall(shap_lgb_test, row_id = 1)

# ---- notebook code cell 105 ----
# Waterfall plot of sixth observation
sv_waterfall(shap_lgb_test, row_id = 6)

# ---- notebook code cell 106 ----
# Waterfall plot of seventh observation
sv_waterfall(shap_lgb_test, row_id = 7)

# ---- notebook code cell 107 ----
# Waterfall plot of eighth observation
sv_waterfall(shap_lgb_test, row_id = 8)

# ---- notebook code cell 111 ----
# feature pre-processing 

# min-max-scaler:
PreProcess.Continuous <- function(var1, dat2){
  names(dat2)[names(dat2) == var1]  <- "V1"
  dat2$X <- as.numeric(dat2$V1)
  dat2$X <- 2*(dat2$X-min(dat2$X))/(max(dat2$X)-min(dat2$X))-1
  names(dat2)[names(dat2) == "V1"]  <- var1
  names(dat2)[names(dat2) == "X"]  <- paste(var1,"X", sep="")
  dat2
}

# pre-procecessing function:
Features.PreProcess <- function(dat2){
  dat2 <- PreProcess.Continuous("Area", dat2)   
  dat2 <- PreProcess.Continuous("VehPower", dat2)   
  dat2$VehAge <- pmin(dat2$VehAge,20)
  dat2 <- PreProcess.Continuous("VehAge", dat2)   
  dat2$DrivAge <- pmin(dat2$DrivAge,90)
  dat2 <- PreProcess.Continuous("DrivAge", dat2)   
  dat2$BonusMalus <- pmin(dat2$BonusMalus,150)
  dat2 <- PreProcess.Continuous("BonusMalus", dat2)   
  dat2$VehBrandX <- as.integer(dat2$VehBrand)-1
  dat2$VehGasX <- as.integer(dat2$VehGas)-1.5
  dat2$Density <- round(log(dat2$Density),2)
  dat2 <- PreProcess.Continuous("Density", dat2)   
  dat2$RegionX <- as.integer(dat2$Region)-1  # char R11,,R94 to number 0,,21
  dat2
}

dat2 <- Features.PreProcess(dat)  # original variables and fitGLM2 (CANN)

# Generate small random sample to inspect transformations
sample_n(dat2, 5)

# ---- notebook code cell 112 ----
# choosing learning and test sample (based on folds)
dat2$fold <- fold
learn0 <- dat2[dat2$fold != 5,]    # 80%
test0 <- dat2[ dat2$fold == 5,]    # 20%
learn1 <- learn0
test1 <- test0

CF2("Claim Frequency (Actual) Learn/Test", learn$ClaimNb,learn$Exposure, test$ClaimNb,test$Exposure)

# ---- notebook code cell 113 ----
# setting up the matrices
features_nn <- c(14:18, 20:21) # definition of feature variables (non-categorical)
q0 <- length(features_nn)
# learning data
Xlearn <- as.matrix(learn0[, features_nn])  # design matrix learning sample
Brlearn <- as.matrix(learn0$VehBrandX)
Relearn <- as.matrix(learn0$RegionX)
Ylearn <- as.matrix(learn0$ClaimNb)
# testing data
Xtest <- as.matrix(test0[, features_nn])    # design matrix test sample
Brtest <- as.matrix(test0$VehBrandX)
Retest <- as.matrix(test0$RegionX)
Ytest <- as.matrix(test0$ClaimNb)
# choosing the right volumes for EmbNN and CANN
Vlearn <- as.matrix(learn0$Exposure)
Vtest <- as.matrix(test0$Exposure)

print(paste("Number of basic features (without VehBrand and Region):",q0))
(lambda.hom <- sum(learn0$ClaimNb)/sum(learn0$Exposure))

# ---- notebook code cell 115 ----
# hyperparameters of the neural network architecture (as specified in "01 CANN approach.r") 
q1 <- 20 # Number of neuron in hidden layer 1
q2 <- 15
q3 <- 10
d <- 2   # dimensions embedding layers for categorical features
(BrLabel <- length(unique(learn0$VehBrandX))) 
(ReLabel <- length(unique(learn0$RegionX)))

# ---- notebook code cell 116 ----
library(keras)

neural_net <- function() {

# Initialize neural net and set Tensorflow's random seed
k_clear_session()
set.seed(42)
tensorflow::tf$random$set_seed(42)

# define the network architecture
Design   <- layer_input(shape = c(q0), dtype = 'float32')
VehBrand <- layer_input(shape = c(1),  dtype = 'int8')
Region   <- layer_input(shape = c(1),  dtype = 'int8')
Vol      <- layer_input(shape = c(1),  dtype = 'float32')

BrandEmb = VehBrand %>% 
  layer_embedding(input_dim = BrLabel, output_dim = d, input_length = 1) %>%
  layer_flatten()

RegionEmb = Region %>% 
  layer_embedding(input_dim = ReLabel, output_dim = d, input_length = 1) %>%
  layer_flatten()

Network = list(Design, BrandEmb, RegionEmb) %>% 
  layer_concatenate() %>% 
  layer_dense(units=q1, activation='tanh') %>%
  layer_dense(units=q2, activation='tanh') %>%
  layer_dense(units=q3, activation='tanh') %>%
  layer_dense(units=1, activation='exponential', name='Network', 
              weights=list(array(0, dim=c(q3,1)), array(log(lambda.hom), dim=c(1))))

Response = list(Network, Vol) %>% layer_multiply()

model = keras_model(inputs = c(Design, VehBrand, Region, Vol), outputs = c(Response))

model %>% compile(loss = 'poisson', optimizer = optimizer_nadam(), weighted_metrics = 'poisson')

return(model)

}

# Check names of tf.random
names(tensorflow::tf$random)

# ---- notebook code cell 118 ----
# Create, compile and summarize a neural network with the parameterization defined above
model1 <- neural_net()
summary(model1)

# ---- notebook code cell 119 ----
set.seed(42) # set seed again
# fitting the neural network (as specified in "01 CANN approach.r") 
{t1 <- proc.time()
  fit <- model1 %>% fit(list(Xlearn, Brlearn, Relearn, Vlearn), Ylearn, epochs=300, 
                       batch_size=5000, verbose=0, validation_split=0.2)
  (proc.time()-t1)}

plot(fit)

# ---- notebook code cell 120 ----
# calculating the predictions
learn0$fitNNemb <- as.vector(model1 %>% predict(list(Xlearn, Brlearn, Relearn, Vlearn)))
test0$fitNNemb <- as.vector(model1 %>% predict(list(Xtest, Brtest, Retest, Vtest)))

# Print claim frequency actual vs predicted
CF2("Claim Frequency NNemb, Test-Sample, Actual/Predicted", test0$ClaimNb,test0$Exposure, test0$fitNNemb,test0$Exposure)

# Print Poisson Deviance
PDW2("Poisson Deviance NNemb", learn0$fitNNemb,as.vector(unlist(learn0$ClaimNb)),learn0$Exposure, test0$fitNNemb,as.vector(unlist(test0$ClaimNb)),test0$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
test$fit.cf <- test$Exposure * sum(learn$ClaimNb)/sum(learn$Exposure) # (recalculate INT-Model)
Benchmark.GLM2("NNemb", test0$fitNNemb)

# ---- notebook code cell 122 ----
# Incorporating model GLM2 into a CANN
Vlearn <- as.matrix(learn$fitGLM2)
Vtest <- as.matrix(test$fitGLM2)
(lambda.hom <- sum(learn$ClaimNb)/sum(learn$fitGLM2))

# ---- notebook code cell 123 ----
# Create, compile and summarize a neural network with the parameterization defined above
model2 <- neural_net()
summary(model2)

# ---- notebook code cell 124 ----
# re-fitting the neural network with GLM2-CANN
{t1 <- proc.time()
  fit <- model2 %>% fit(list(Xlearn, Brlearn, Relearn, Vlearn), Ylearn, epochs=100, 
                       batch_size=5000, verbose=0, validation_split=0.2)
  (proc.time()-t1)}

plot(fit)

# ---- notebook code cell 125 ----
# calculating the predictions
learn1$fitNNGLM <- as.vector(model2 %>% predict(list(Xlearn, Brlearn, Relearn, Vlearn)))
test1$fitNNGLM <- as.vector(model2 %>% predict(list(Xtest, Brtest, Retest, Vtest)))

# Print claim frequency actual vs predicted
CF2("Claim Frequency CANN, Test-Sample, Actual/Predicted", test1$ClaimNb,test1$Exposure, test1$fitNNGLM,test1$Exposure)

# Print Poisson Deviance
PDW2("Poisson Deviance CANN", learn1$fitNNGLM,as.vector(unlist(learn1$ClaimNb)),learn1$Exposure, test1$fitNNGLM,as.vector(unlist(test1$ClaimNb)),test1$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
test$fit.cf <- test$Exposure * sum(learn$ClaimNb)/sum(learn$Exposure) # (recalculate INT-Model)
Benchmark.GLM2("CANN", test1$fitNNGLM)

# ---- notebook code cell 127 ----
# Get weights embedding VehBrand for model1 (NNemb) and model2 (CANN)
Brand <- c('B1','B10','B11','B12','B13','B14','B2','B3','B4','B5','B6')
emb_B1 <- get_weights(model1)[[1]]
emb_B1 <- cbind(Brand, as.data.frame(emb_B1)) 
colnames(emb_B1)<-c("Brand","dim1","dim2")
emb_B2 <- get_weights(model2)[[1]]
emb_B2 <- cbind(Brand, as.data.frame(emb_B2)) 
colnames(emb_B2)<-c("Brand","dim1","dim2")

# ---- notebook code cell 128 ----
# Plot embeddings
par(mfrow=c(1,2)) 
plot(data=emb_B1, dim2~dim1, pch=3, col="red",cex=0.0, main='NNemb embedding VehBrand')
with(data=emb_B1, text(dim2~dim1, labels=Brand, col="blue",cex=1.1), pos=1)
plot(data=emb_B2, dim2~dim1, pch=3, col="red",cex=0.0, main='CANN embedding VehBrand')
with(data=emb_B2, text(dim2~dim1, labels=Brand, col="blue",cex=1.1), pos=1)

# ---- notebook code cell 130 ----
# Get weights embedding Region for model1 (NNemb) and model2 (CANN)
Region <- c('R11','R21','R22','R23','R24','R25','R26','R31','R41','R42','R43',
           'R52','R53','R54','R72','R73','R74','R82','R83','R91','R93','R94')
emb_R1 <- get_weights(model1)[[2]]
emb_R1 <- cbind(Region, as.data.frame(emb_R1)) 
colnames(emb_R1)<-c("Region","dim1","dim2")
emb_R2 <- get_weights(model2)[[2]]
emb_R2 <- cbind(Region, as.data.frame(emb_R2)) 
colnames(emb_R2)<-c("Region","dim1","dim2")

# ---- notebook code cell 131 ----
# Plot embeddings
par(mfrow=c(1,2)) 
plot(data=emb_R1, dim2~dim1, pch=3, col="red",cex=0.0, main='NNemb embedding Region')
with(data=emb_R1, text(dim2~dim1, labels=Region, col="blue",cex=1.1), pos=1)
plot(data=emb_R2, dim2~dim1, pch=3, col="red",cex=0.0, main='CANN embedding Region')
with(data=emb_R2, text(dim2~dim1, labels=Region, col="blue",cex=1.1), pos=1)

# ---- notebook code cell 133 ----
# changing the matrices (exclude BonusMalus, col18)
features_nn <- c(14:17, 20:21) # definition of feature variables (non-categorical)
q0 <- length(features_nn)
print(paste("Number of basic features (without VehBrand and Region):",q0))
# learning data
Xlearn <- as.matrix(learn0[, features_nn])  # design matrix learning sample
Brlearn <- as.matrix(learn0$VehBrandX)
Relearn <- as.matrix(learn0$RegionX)
Ylearn <- as.matrix(learn0$ClaimNb)
# testing data
Xtest <- as.matrix(test0[, features_nn])    # design matrix test sample
Brtest <- as.matrix(test0$VehBrandX)
Retest <- as.matrix(test0$RegionX)
Ytest <- as.matrix(test0$ClaimNb)

# Incorporating model GLM2 into a CANN
Vlearn <- as.matrix(learn$fitGLM2)
Vtest <- as.matrix(test$fitGLM2)
(lambda.hom <- sum(learn$ClaimNb)/sum(learn$fitGLM2))

# ---- notebook code cell 134 ----
# Create, compile and summarize a neural network with the parameterization defined above
model <- neural_net()
summary(model)

# ---- notebook code cell 136 ----
# re-fitting the neural network with GLM2-CANN
{t1 <- proc.time()
  fit <- model %>% fit(list(Xlearn, Brlearn, Relearn, Vlearn), Ylearn, epochs=20, 
                       batch_size=5000, verbose=0, validation_split=0.2)
  (proc.time()-t1)}

plot(fit)

# ---- notebook code cell 137 ----
# calculating the predictions
learn1$fitNNGLMc <- as.vector(model %>% predict(list(Xlearn, Brlearn, Relearn, Vlearn)))
test1$fitNNGLMc <- as.vector(model %>% predict(list(Xtest, Brtest, Retest, Vtest)))

# Print Poisson Deviance
PDW2("Poisson Deviance NNGLMc", learn1$fitNNGLMc,as.vector(unlist(learn1$ClaimNb)),learn1$Exposure, test1$fitNNGLMc,as.vector(unlist(test1$ClaimNb)),test1$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
test$fit.cf <- test$Exposure * sum(learn$ClaimNb)/sum(learn$Exposure) # (recalculate INT-Model)
Benchmark.GLM2("NNGLMc", test1$fitNNGLMc)

# ---- notebook code cell 142 ----
# from xgboost model
install.packages("EIX")
library(EIX)

{t1 <- proc.time()
  inter.xgb <-interactions(fit.xgb, dtrain)
(proc.time()-t1)}

# ---- notebook code cell 143 ----
# from lgb
{t1 <- proc.time()
  inter.lgb <-interactions(LGB, llearn)
(proc.time()-t1)}

# ---- notebook code cell 144 ----
# Summarize one-hot-encoded features
to_delete <- c("1|2|3|4|5|6|7|8|9|0")

df.inter.xgb <- as.data.frame(inter.xgb)
df.inter.xgb$Parent <- str_remove_all(inter.xgb$Parent, to_delete)
df.inter.xgb$Child <- str_remove_all(inter.xgb$Child, to_delete)
df.inter.xgb$sumGain <- df.inter.xgb$sumGain / sum(df.inter.xgb$sumGain) * 100

df.inter.xgb %>%
  group_by(Parent, Child) %>%
  summarise(sumGain = sum(sumGain),
            frequency = sum(frequency)) %>%
  ungroup() %>%
  arrange(desc(sumGain)) %>%
  ggplot(mapping = aes(
    x = reorder(Child, desc(sumGain)),
    y = Parent,
    fill = sumGain  )) +
  xlab(label = "Child") +
  labs(fill = "Gain in %") +
  geom_tile(aes(fill = sumGain), color = "white") +
  scale_fill_gradient(low = "white", high = "red3") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 1,
      size = 10,
      hjust = 1  ),
      legend.position = "right"  ) +
  coord_fixed()

# ---- notebook code cell 145 ----
df.inter.lgb <- as.data.frame(inter.lgb)
df.inter.lgb$Parent <- str_remove_all(inter.lgb$Parent, to_delete)
df.inter.lgb$Child <- str_remove_all(inter.lgb$Child, to_delete)
df.inter.lgb$sumGain <- df.inter.lgb$sumGain / sum(df.inter.lgb$sumGain) * 100

df.inter.lgb %>%
  group_by(Parent, Child) %>%
  summarise(sumGain = sum(sumGain),
            frequency = sum(frequency)) %>%
  ungroup() %>%
  arrange(desc(sumGain)) %>%
  ggplot(mapping = aes(
    x = reorder(Child, desc(sumGain)),
    y = Parent,
    fill = sumGain  )) +
  xlab(label = "Child") +
  labs(fill = "Gain in %") +
  geom_tile(aes(fill = sumGain), color = "white") +
  scale_fill_gradient(low = "white", high = "blue3") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 1,
      size = 10,
      hjust = 1  ),
      legend.position = "right"  ) +
  coord_fixed()

# ---- notebook code cell 148 ----
# Interactions: we use the explainer object from 3.4
inter.lgb.drivage <- as.data.frame(potential_interactions(shap_lgb_test, "DrivAge"))
names(inter.lgb.drivage)[1] <- "interaction_value"
# make new column "interaction" out of rownames and "DrivAge
inter.lgb.drivage$interaction <- paste0(rownames(inter.lgb.drivage), ":DrivAge")
# plot interaction_value top 10
inter.lgb.drivage %>% 
  arrange(desc(interaction_value)) %>% 
  head(10) %>% 
  ggplot(aes(x = reorder(interaction, interaction_value), y = interaction_value)) +
  geom_col() +
   coord_flip() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

# ---- notebook code cell 152 ----
install.packages("hstats")
library(hstats)

{t1 <- proc.time()
  H <- hstats(fit.xgb, X = data.matrix(X_learn), w = learn$Exposure, verbose = FALSE)
  (proc.time()-t1)}

# ---- notebook code cell 153 ----
plot(H) + 
    theme_gray(base_size = 16)

# ---- notebook code cell 157 ----
f.glm4 <- "ClaimNb ~ AreaGLM + VehPowerGLM * VehAgeGLM + VehAgeGLM * VehBrand + VehGas * VehAgeGLM + DensityGLM + Region + BonusMalusGLM * log(DrivAge) + log(BonusMalusGLM) + I(BonusMalusGLM^2) + I(BonusMalusGLM^3) + I(BonusMalusGLM^4) + DrivAge + I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4)"
d.glm4 <- glm(f.glm4, data=learn, offset=log(Exposure), family=poisson())
summary(d.glm4)

# ---- notebook code cell 158 ----
learn$fitGLM4 <- fitted(d.glm4)
test$fitGLM4 <- predict(d.glm4, newdata=test, type="response")

# Print Poisson Deviance
PDW2('Poisson Deviance GLM4', learn$fitGLM4,learn$ClaimNb,learn$Exposure, test$fitGLM4,test$ClaimNb,test$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
Benchmark.GLM2("GLM4", test$fitGLM4)

# ---- notebook code cell 161 ----
f.glm5 <- "ClaimNb ~ AreaGLM + VehPowerGLM * VehAgeGLM  + VehAgeGLM *VehBrand + VehGas * VehAgeGLM + DensityGLM + Region + BonusMalusGLM * log(DrivAge) +  VehAgeGLM *log(BonusMalusGLM) + I(BonusMalusGLM^2) + I(BonusMalusGLM^3) + I(BonusMalusGLM^4) + DrivAge * DensityGLM + I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4)"
d.glm5 <- glm(f.glm5, data=learn, offset=log(Exposure), family=poisson())
summary(d.glm5)

# ---- notebook code cell 162 ----
learn$fitGLM5 <- fitted(d.glm5)
test$fitGLM5 <- predict(d.glm5, newdata=test, type="response")

# Print Poisson Deviance
PDW2('Poisson Deviance GLM5', learn$fitGLM5,learn$ClaimNb,learn$Exposure, test$fitGLM5,test$ClaimNb,test$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
Benchmark.GLM2("GLM5", test$fitGLM5)

# ---- notebook code cell 164 ----
# For comparison: Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
Benchmark.GLM2("GAM3", test$fitGAM3)

# ---- notebook code cell 166 ----
f.gam6 <- ClaimNb ~ s(VehAge) + s(DrivAge) + s(BonusMalusGLM,by=DrivAgeGLM) + VehPowerGLM + VehGas + VehBrand + AreaGLM + DensityGLM + Region + offset(log(Exposure))

{t1 <- proc.time()
    d.gam6 <- bam(f.gam6, data=learn, scale=-1, family=poisson)
(proc.time()-t1)}

summary(d.gam6)

# ---- notebook code cell 167 ----
# Visualization of the fitted spline functions
plot(d.gam6, col = 'blue', shade=TRUE, ylim = c(-1.0,2.5), pages=5)

# ---- notebook code cell 169 ----
# Model performance: Improved GAM with an Interaction

learn$fitGAM6 <- fitted(d.gam6)
test$fitGAM6 <- predict(d.gam6, newdata=test, type="response")

# Print Poisson Deviance
PDW2('Poisson Deviance GAM6', learn$fitGAM6,learn$ClaimNb,learn$Exposure, test$fitGAM6,test$ClaimNb,test$Exposure)

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
Benchmark.GLM2("GAM6", test$fitGAM6)

# ---- notebook code cell 173 ----
# k folds as defined in Ch. 1.6
{t1 <- proc.time()

PD.test.INT  <- vector() # initialize PD-test-folds-vector
PD.test.GLM2 <- vector() # 
PD.test.GLM4 <- vector() 
PD.test.GLM5 <- vector() 
PD.test.GAM6 <- vector()  
PD.test.ML_CB0  <- vector() 
PD.test.ML_CB1 <- vector() 
PD.test.ML_LGB0 <- vector() 
PD.test.ML_LGB1 <- vector() 
PD.test.ML_XGB0 <- vector() 
PD.test.ML_XGB1 <- vector() 

for (i in 1:k) {
    learn <- dat1[dat1$fold != i,]
    test  <- dat1[dat1$fold == i,]
    (n_l <- nrow(learn))
    (n_t <- nrow(test))
    df_feat <- dat[,c(features)]
    df_feat$Area <- as.integer(df_feat$Area)
    df_feat <- as.data.frame(model.matrix( ~ 0 +. ,data = df_feat)) 
    X_learn <- df_feat[dat1$fold != i,]    # 80%
    X_test  <- df_feat[dat1$fold == i,]    # 20%

    ### Model INT (intercept-only).  No model, just average claim frequency
    (cf <- sum(learn$ClaimNb)/sum(learn$Exposure))
    test$fit.cf <- cf*test$Exposure
    # out-of-sample losses (in 10^(-2))
    PD.test.INT[i] <- PDW(test$fit.cf, test$ClaimNb, test$Exposure)
    
    ### Model GLM2: Basic GLM without interactions
    GLM2 <- glm(f.glm2, data=learn, offset=log(Exposure), family=poisson())
    test$fitGLM2 <- predict(GLM2, newdata=test, type="response")
    PD.test.GLM2[i] <- PDW(test$fitGLM2, test$ClaimNb, test$Exposure)

    ### Model GLM4: Featuring neural net residual analyses
    GLM4 <- glm(f.glm4, data=learn, offset=log(Exposure), family=poisson())
    test$fitGLM4 <- predict(GLM4, newdata=test, type="response")
    PD.test.GLM4[i] <- PDW(test$fitGLM4, test$ClaimNb, test$Exposure)
    
    ### Model GLM5: Improved GLM with Interactions from XGBoost
    GLM5 <- glm(f.glm5, data=learn, offset=log(Exposure), family=poisson())
    test$fitGLM5 <- predict(GLM5, newdata=test, type="response")
    PD.test.GLM5[i] <- PDW(test$fitGLM5, test$ClaimNb, test$Exposure)
    
    ### Model GAM6: Generalized Additive Model (3 splines and a single interaction)
    GAM6 <- bam(f.gam6, data=learn, scale=-1, family=poisson)
    test$fitGAM6 <- predict(GAM6, newdata=test, type="response")
    PD.test.GAM6[i] <- PDW(test$fitGAM6, test$ClaimNb, test$Exposure)
   
    ### CatBoost-Models:
    learn_pool <- catboost.load_pool(data = learn[features], label = learn$ClaimNb, 
                     cat_features = cat_features, baseline = as.matrix(learn$Exposure))
    test_pool <- catboost.load_pool(data = test[features], label = test$ClaimNb, 
                     cat_features = cat_features, baseline = as.matrix(test$Exposure))
    # CB0: Unconstraint 
    CB0_params <- list(logging_level='Silent', loss_function='Poisson', random_seed=42,
                      iterations = 500, learning_rate = 0.15, depth = 3)
    CB0 <- catboost.train(learn_pool, params = CB0_params) 
    test$fitCB0 <- exp(catboost.predict(CB0,test_pool))
    PD.test.ML_CB0[i] <- PDW(test$fitCB0, test$ClaimNb, test$Exposure)
    # CB1: Constraint 
    CB1_params <- list(logging_level='Silent', loss_function='Poisson', 
                      monotone_constraints = mtcc, random_seed=42,
                      iterations = 500, learning_rate = 0.15, depth = 3)
    CB1 <- catboost.train(learn_pool, params = CB1_params) 
    test$fitCB1 <- exp(catboost.predict(CB1,test_pool))
    PD.test.ML_CB1[i] <- PDW(test$fitCB1, test$ClaimNb, test$Exposure)

    ### LightGBM-Models:
    llearn <- lgb.Dataset(data = data.matrix(X_learn), 
                          label = learn$ClaimNb/learn$Exposure, weight=learn$Exposure)
    # LGB0: Unconstraint 
    LGB0 <- lgb.train(data=llearn, objective='poisson',
                      num_leaves=31, learning_rate=0.1, n_estimators=100)
    test$fitLGB0 <- predict(LGB0, data.matrix(X_test)) * test$Exposure
    PD.test.ML_LGB0[i] <- PDW(test$fitLGB0, test$ClaimNb, test$Exposure)
    # LGB1: Constraint 
    LGB1 <- lgb.train(data=llearn, objective='poisson', monotone_constraints = mtc,
                      num_leaves=31, learning_rate=0.1, n_estimators=100)
    test$fitLGB1 <- predict(LGB1, data.matrix(X_test)) * test$Exposure
    PD.test.ML_LGB1[i] <- PDW(test$fitLGB1, test$ClaimNb, test$Exposure)

    ### XGBoost-Models:  
    dlearn <- xgb.DMatrix(data = data.matrix(X_learn),
                          label = learn$ClaimNb/learn$Exposure, weight=learn$Exposure)
    dtest <- xgb.DMatrix(data = data.matrix(X_test), 
                         label = test$fold, weight=test$Exposure)
    # XGB0: Unconstraint 
    XGB0 <- xgb.train(data=dlearn, objective='count:poisson', 
                      nrounds = 500, max_depth = 5, eta = 0.05, tree_method = "hist")
    test$fitXGB0 <- predict(XGB0, newdata = dtest) * test$Exposure
    PD.test.ML_XGB0[i] <- PDW(test$fitXGB0, test$ClaimNb, test$Exposure)  
    # XGB1: Constraint
    XGB1 <- xgb.train(data=dlearn, objective='count:poisson', monotone_constraints = mtc, 
                      nrounds = 500, max_depth = 5, eta = 0.05, tree_method = "hist")
    test$fitXGB1 <- predict(XGB1, newdata = dtest) * test$Exposure
    PD.test.ML_XGB1[i] <- PDW(test$fitXGB1, test$ClaimNb, test$Exposure)
}
k
(proc.time()-t1)}

# ---- notebook code cell 175 ----
# write vectors in a Data Frame
df <- data.frame(1:5,PD.test.INT,PD.test.GLM2,PD.test.GLM4,PD.test.GLM5,PD.test.GAM6,
                 PD.test.ML_CB0,PD.test.ML_CB1,PD.test.ML_LGB0,PD.test.ML_LGB1,
                 PD.test.ML_XGB0,PD.test.ML_XGB1)
names(df) <- c("fold","INT","GLM2","GLM4","GLM5","GAM6","CatBoost","CatBoost_mc",
               "LightGBM","LightGBM_mc","XGBoost","XGBoost_mc")
# calculate GLM2-Improvement-Index
df.idx <- df
df.idx$GLM4 <- round(((df$GLM4 - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx$GLM5 <- round(((df$GLM5 - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx$GAM6 <- round(((df$GAM6 - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx$CatBoost <- round(((df$CatBoost - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx$CatBoost_mc <- round(((df$CatBoost_mc - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx$LightGBM <- round(((df$LightGBM - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx$LightGBM_mc <- round(((df$LightGBM_mc - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx$XGBoost <- round(((df$XGBoost - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx$XGBoost_mc  <- round(((df$XGBoost_mc - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx

# ---- notebook code cell 177 ----
ggplot(df, aes(fold)) + 
  geom_point(aes(y = INT,  colour = "INT"),  size=6) + 
  geom_point(aes(y = GLM2, colour = "GLM2"), size=6) + 
  geom_point(aes(y = GLM4, colour = "GLM4"), size=6) + 
  geom_point(aes(y = GLM5, colour = "GLM5"), size=6) + 
  geom_point(aes(y = GAM6, colour = "GAM6"), size=6) + 
  geom_point(aes(y = CatBoost,    colour = "CatBoost"),    size=6) + 
  geom_point(aes(y = CatBoost_mc, colour = "CatBoost_mc"), size=6) + 
  geom_point(aes(y = LightGBM,    colour = "LightGBM"),    size=6) + 
  geom_point(aes(y = LightGBM_mc, colour = "LightGBM_mc"), size=6) + 
  geom_point(aes(y = XGBoost,     colour = "XGBoost"),     size=6) + 
  geom_point(aes(y = XGBoost_mc,  colour = "XGBoost_mc"),  size=6) + 
  ylab("Poisson Deviance") + theme(text = element_text(size=15))

# ---- notebook code cell 179 ----
df.idx  %>% select(fold, GLM4, GLM5, GAM6, CatBoost, CatBoost_mc,
                      LightGBM, LightGBM_mc, XGBoost, XGBoost_mc) %>% 
  gather(model, PD, GLM4, GLM5, GAM6, CatBoost, CatBoost_mc,
                 LightGBM, LightGBM_mc, XGBoost, XGBoost_mc) %>% 
  ggplot(aes(x = fct_reorder(model, PD, mean), y=PD)) +
    geom_boxplot(outlier.shape = NA) + 
    geom_jitter(width = 0) +
    stat_summary(fun = mean, colour="darkred", geom="point",shape=18, size=7) +
    coord_flip() +
    xlab("Model") + 
    ylab("GLM2-Improvement-Index") + 
    theme(text = element_text(size=20))

# ---- notebook code cell 183 ----
sessionInfo()

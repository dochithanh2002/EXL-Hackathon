# ---------------------------------------------------------------------------
# LEGACY MODEL (as received) - French motor TPL claim frequency, Poisson GLM "GLM1"
# Source: reconstructed from Noll, Salzmann & Wuthrich (2018), "Case Study: French
#         Motor Third-Party Liability Claims", SSRN 3164764, Listing 2 / Section 3.
#         The original Kaggle notebook could not be retrieved; every line below follows
#         the published feature engineering and model of that case study. [reconstructed]
# Data:   freMTPL2freq (OpenML 41214, CC0), as exported by OpenML (strings in single quotes).
# This file is the INPUT to the R->Python migration agent. Do not edit.
# ---------------------------------------------------------------------------

dat <- read.csv("data/freMTPL2freq.csv", quote = "\"'", stringsAsFactors = FALSE)

# claim counts above 4 and exposures above one year are treated as data errors (case study, Sec. 2)
dat$ClaimNb <- pmin(dat$ClaimNb, 4)
dat$Exposure <- pmin(dat$Exposure, 1)

# --- feature pre-processing for GLM1 (case study, Sec. 3.1) ---
# Area: ordinal A..F mapped to 1..6, used as a continuous variable
dat$AreaGLM <- as.integer(factor(dat$Area))
# VehPower: categorical, levels above 9 merged into 9
dat$VehPowerGLM <- as.factor(pmin(dat$VehPower, 9))
# VehAge: three classes [0,1), [1,10], (10,inf); class 2 is the reference
VehAgeGLM <- cbind(c(0:110), c(1, rep(2, 10), rep(3, 100)))
dat$VehAgeGLM <- as.factor(VehAgeGLM[dat$VehAge + 1, 2])
dat[, "VehAgeGLM"] <- relevel(dat[, "VehAgeGLM"], ref = "2")
# DrivAge: seven age classes 18-20, 21-25, 26-30, 31-40, 41-50, 51-70, 71+; class 5 (41-50) is the reference
DrivAgeGLM <- cbind(c(18:100), c(rep(1, 21 - 18), rep(2, 26 - 21), rep(3, 31 - 26), rep(4, 41 - 31),
                                 rep(5, 51 - 41), rep(6, 71 - 51), rep(7, 101 - 71)))
dat$DrivAgeGLM <- as.factor(DrivAgeGLM[dat$DrivAge - 17, 2])
dat[, "DrivAgeGLM"] <- relevel(dat[, "DrivAgeGLM"], ref = "5")
# BonusMalus: capped at 150, continuous
dat$BonusMalusGLM <- as.integer(pmin(dat$BonusMalus, 150))
# Density: log-transformed, continuous
dat$DensityGLM <- as.numeric(log(dat$Density))
# VehBrand and Region: categorical with references B1 and R24 (largest classes)
dat$VehBrand <- relevel(as.factor(dat$VehBrand), ref = "B1")
dat$Region <- relevel(as.factor(dat$Region), ref = "R24")
dat$VehGas <- as.factor(dat$VehGas)

# --- learning / test split: 90% / 10% (case study, Sec. 2) ---
set.seed(100)
ll <- sample(c(1:nrow(dat)), round(0.9 * nrow(dat)), replace = FALSE)
learn <- dat[ll, ]
test <- dat[-ll, ]

# --- GLM1: Poisson frequency model with log-exposure offset ---
d.glm1 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand + VehGas +
                DensityGLM + Region + AreaGLM,
              data = learn, offset = log(Exposure), family = poisson())

learn$fitGLM1 <- fitted(d.glm1)
test$fitGLM1 <- predict(d.glm1, newdata = test, type = "response")

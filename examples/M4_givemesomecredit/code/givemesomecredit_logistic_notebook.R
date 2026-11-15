
# ---- notebook code cell 1 ----
# Importing the datasets
cs_training = read.csv("../input/cs-training.csv")
cs_test = read.csv("../input/cs-test.csv")

str(cs_training)

# Data Cleaning
cs_training = cs_training[,(-1)]
cs_test = cs_test[,(-1)]

# Corelation between Dependent and INdependent Variable 
cor(x = cs_training$SeriousDlqin2yrs, y = cs_training)

# Removing insignificant Independent variables
cs_training$MonthlyIncome <- NULL
cs_training$NumberOfDependents <- NULL
cs_test$MonthlyIncome <- NULL
cs_test$NumberOfDependents <- NULL

#  Check and remove NA values
colSums(is.na(cs_training))
colSums(is.na(cs_test))

# Encoding the target feature as factor
cs_training$SeriousDlqin2yrs = factor(cs_training$SeriousDlqin2yrs, levels = c(0,1))

# Feature Scaling
cs_training[-1] = scale(cs_training[-1])
cs_test[-1] = scale(cs_test[-1])

# Fitting Logistic Regression to the Training set
RegModel = glm(formula = cs_training$SeriousDlqin2yrs ~ .,
               family = binomial,
               data = cs_training)

summary(RegModel)

# Validating Training set results
Prediction_Model_Training = predict(RegModel, type = 'response', newdata = cs_training[-1])
Training_Predictions = ifelse(Prediction_Model_Training > 0.5, 1, 0)
Probabilites_Training <- data.frame(Prediction_Model_Training)
# write.csv(Probabilites_Training, file = "Probabilites_Training.csv")


# Making the Confusion Matrix
cm_training = table(cs_training$SeriousDlqin2yrs, Training_Predictions)


# Fitting Logistic Regression to the Test set
Prediction_Model_Test = predict(RegModel, type = 'response', newdata = cs_test[-1])
Test_Predictions = ifelse(Prediction_Model_Test > 0.5, 1, 0)


# Making the Confusion Matrix
cm_test = table(Test_Predictions)

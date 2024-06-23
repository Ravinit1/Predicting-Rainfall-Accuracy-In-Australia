# Libraries ---------------------------------------------------------------
library(tidyverse)
library(glmnet)
library(randomForest)
library(MASS)
library(gridExtra)
library(caret)
library(e1071)
library(kernlab)
library(vtable)
library(corrplot)
library(lattice)
library(patchwork)
library(data.table)
library(Hmisc)
library(pROC)
library(ROCR)

# Loading Data and Preprocessing
weather = read.csv("working directory for file") # Change this line to your working directory

weather = weather %>% mutate_at(c('WindGustDir', 'WindDir9am', 'WindDir3pm', 
                                  'RainToday', 'RainTomorrow'), as.factor)
weather$Year = as.integer(sapply(strsplit(weather[,1], "-"), getElement, 1))
weather_summary = summary(weather)

# Splitting Data and Removing Variables
train_index = (weather$Year < 2013)
test_index = !train_index

train = weather[train_index, ]
test = weather[test_index, ]

# Remove columns
train = train[, c(-1, -2, -8, -10, -11, -24)]
test = test[, c(-1, -2, -8, -10, -11, -24)]
weather = weather[, c(-1, -2, -8, -10, -11, -24)]
weather_plotting = weather[, c(-17, -18)]

# Remove NAs
train = na.omit(train)
test = na.omit(test)
weather = na.omit(weather)
RainTom.test <- test$RainTomorrow

# Exploratory Data Analysis -----------------------------------------------
st(weather) # Summary Statistics
corrplot(cor(weather[, c(-17, -18)]), method = "square") # Correlation plot
hist.data.frame(weather_plotting) # Histogram of the Predictor Variables

# GLM Model
glm.fits <- glm(RainTomorrow ~ ., data = train, family = "binomial")
glm.fits
glm.probs <- predict(glm.fits, test, type = "response")
preds= prediction(glm.probs, RainTom.test)
prf = performance(preds, measure = "tpr", x.measure = "fpr")

glm.pred <- rep("No", length(glm.probs))
glm.pred[glm.probs > .5] <- "Yes"
table(glm.pred, RainTom.test)
mean(glm.pred == RainTom.test)
mean(glm.pred != RainTom.test)

#GLM plot
plot(prf, col = 'red', main = 'ROC Curve for Logistic Regression')

# LDA Model
lda.fit <- lda(RainTomorrow ~ ., data = train)
lda.fit
plot(lda.fit, ylab = "Frequency")

lda.pred <- predict(lda.fit, test)

lda.class <- lda.pred$class
table(lda.class, RainTom.test)
mean(lda.class == RainTom.test)

sum(lda.pred$posterior[, 1] >= .5)
sum(lda.pred$posterior[, 1] < .5)

lda.pred$posterior[1:20, 1]
lda.class[1:20]
sum(lda.pred$posterior[, 1] > .9)

# QDA Model
qda.fit <- qda(RainTomorrow ~ ., data = train)
qda.fit
qda.class <- predict(qda.fit, test)$class
table(qda.class, RainTom.test)
mean(qda.class == RainTom.test)

# Recreate x and y after removing NA rows from train and test
x <- model.matrix(RainTomorrow ~ ., rbind(train, test))[,-1]
y <- as.numeric(rbind(train, test)$RainTomorrow) - 1

train_rows <- 1:nrow(train)
test_rows <- (nrow(train) + 1):nrow(x)

# Lasso Model
lasso.fit <- cv.glmnet(x[train_rows, ], y[train_rows], family = "binomial", alpha = 1)
plot(lasso.fit)
lasso.pred <- predict(lasso.fit, s = "lambda.min", newx = x[test_rows, ], type = "class")
lasso.pred <- ifelse(lasso.pred == "1", "Yes", "No")
table(lasso.pred, RainTom.test)
mean(lasso.pred == RainTom.test)
lasso_coefficients <- predict(lasso.fit, type = "coefficients", s = "lambda.min") 
lasso_coefficients[lasso_coefficients != 0]
length(lasso_coefficients[lasso_coefficients != 0])

# Ridge Model
ridge.fit <- cv.glmnet(x[train_rows, ], y[train_rows], family = "binomial", alpha = 0)
plot(ridge.fit)
ridge.pred <- predict(ridge.fit, s = "lambda.min", newx = x[test_rows, ], type = "class")
table(ridge.pred, RainTom.test)
mean(ridge.pred == RainTom.test)
ridge_coefficients <- predict(ridge.fit, type = "coefficients", s = "lambda.min") 
ridge_coefficients[ridge_coefficients != 0]
length(ridge_coefficients[ridge_coefficients != 0])

# Plots for Lasso and Ridge Model 
par(mfrow = c(1, 2), mar = c(5, 4, 6, 2) + 0.1)
plot(lasso.fit, main = "Lasso Model")
plot(ridge.fit, main = "Ridge Model")

# Random Forest
rf.fit <- randomForest(RainTomorrow ~ ., data = train, importance = TRUE)
rf.pred <- predict(rf.fit, newdata = test)
table(rf.pred, RainTom.test)
mean(rf.pred == RainTom.test)

# Random Forest Importance
importance(rf.fit)
varImpPlot(rf.fit)
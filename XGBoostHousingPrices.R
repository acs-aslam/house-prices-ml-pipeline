train <- read.csv("https://raw.githubusercontent.com/harrixide/Kaggle/refs/heads/main/train.csv")
test <- read.csv("https://raw.githubusercontent.com/harrixide/Kaggle/refs/heads/main/test.csv")

library(dplyr)
library(Matrix)
library(xgboost)

class(train)
dplyr::select

# Feature engineering
train$BsmtSF <- train$TotalBsmtSF - train$BsmtUnfSF
train$HouseAge <- (train$YrSold - train$YearBuilt) + 1
train$BsmtNone <- ifelse(train$TotalBsmtSF == 0, 1, 0)
train$HasGarage <- ifelse(train$GarageArea == 0, 0, 1)

test$BsmtSF <- test$TotalBsmtSF - test$BsmtUnfSF
test$HouseAge <- (test$YrSold - test$YearBuilt) + 1
test$BsmtNone <- ifelse(test$TotalBsmtSF == 0, 1, 0)
test$HasGarage <- ifelse(test$GarageArea == 0, 0, 1)

# Select features
train_x <- dplyr::select(train,
                         Neighborhood, OverallQual, GrLivArea, ExterQual,
                         KitchenQual, GarageCars, HasGarage, BsmtSF,
                         BsmtNone, HouseAge, Foundation)
train_y <- log(train$SalePrice)

# Convert categorical to numeric (OHE)
train_matrix <- model.matrix(~ . - 1, data = train_x)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_y)

# XGBoost parameters
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain),
  verbose = 1
)

# Prepare test data
test_x <- test %>%
  dplyr::select(Neighborhood, OverallQual, GrLivArea, ExterQual, KitchenQual,
         GarageCars, HasGarage, BsmtSF, BsmtNone, HouseAge, Foundation)

# Replace NA with 0
test_x[is.na(test_x)] <- 0

# One-hot encode
test_matrix <- model.matrix(~ . - 1, data = test_x)

# Align test_matrix to train_matrix columns
missing_cols <- setdiff(colnames(train_matrix), colnames(test_matrix))
for (col in missing_cols) {
  test_matrix <- cbind(test_matrix, 0)
  colnames(test_matrix)[ncol(test_matrix)] <- col
}

# Ensure same column order
test_matrix <- test_matrix[, colnames(train_matrix)]

# Predict
preds <- predict(xgb_model, newdata = test_matrix)

summary(preds)

preds <- exp(preds)  # Convert from log scale back to price
summary(preds)

length(preds)
nrow(test)

submission <- data.frame(
  ID = 1461:2919,
  SalePrice = sample(c(0,1), 1459, replace = TRUE))

write.csv(submission, "~/Desktop/HousePrices10.csv", row.names = FALSE)

submission$SalePrice <- preds
write.csv(submission, "~/Desktop/HousePrices10.csv", row.names = FALSE)
which(is.na(preds))

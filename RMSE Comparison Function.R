#This is the compare_rmse function used to determine best 20 features, before extended feature analysis removes 9 more which is also in this document at the bottom

train <- read.csv("https://raw.githubusercontent.com/harrixide/Kaggle/refs/heads/main/train.csv")
test <- read.csv("https://raw.githubusercontent.com/harrixide/Kaggle/refs/heads/main/test.csv")


#Let's make all character features into factors first
train <- train %>%
  mutate(across(where(is.character), as.factor))

library(tidymodels)
set.seed(123)

# line below is because 1 isn't being accepted in function, so make a variable for 1
train$InterceptOnly <- 1

#Set cv_splits for cross validation in workflow
cv_splits <- vfold_cv(train, v = 10, repeats = 5)
cv_splits

#Workflow for compare_rmse custom function

compare_rmse <- function(predictor_name) {
#Build formulas
formula_model <- as.formula(paste("SalePrice ~", predictor_name))
formula_null <- as.formula("SalePrice ~ InterceptOnly")

#Define model
lm_spec <- linear_reg() %>%
set_engine("lm")

#Model with predictor
wf_model <- workflow() %>%
add_model(lm_spec) %>%
add_formula(formula_model)

# Null model (intercept only)
wf_null <- workflow() %>%
add_model(lm_spec) %>%
add_formula(formula_null)

#Get rmse, using cross validation, gives 10 rmse values
res_model <- fit_resamples(wf_model, resamples = cv_splits, metrics = metric_set(rmse))
res_null  <- fit_resamples(wf_null, resamples = cv_splits, metrics = metric_set(rmse))

#Collect RMSEs by getting average
#collect_metrics functionc reates table of rows: rmse, r squared alogn with col mean, std, value, etc.
#filter, makes it so only rmse row is included
#pull pulls only mean column
rmse_model <- collect_metrics(res_model) %>% filter(.metric == "rmse") %>% pull(mean)
rmse_null  <- collect_metrics(res_null)  %>% filter(.metric == "rmse") %>% pull(mean)

tibble(
Predictor   = predictor_name,
RMSE_Model  = rmse_model,
RMSE_Null   = rmse_null,
Improvement = rmse_null - rmse_model  # lower RMSE is better
)
}

#Use function for one feature
compare_rmse("LotArea")

#List all predictor names
predictor_names <- setdiff(names(train), c("SalePrice", "InterceptOnly"))
predictor_names

#Automate our workflow
library(purrr)

#lets generate a list of best predictors based on our workflow, this part will take around 30 minutes to load......
rmse_results <- map_dfr(predictor_names, compare_rmse)
rmse_results
#.................


#Arrange them with the most improvement
rmse_results <- rmse_results %>%
arrange(desc(Improvement))

#The Results!
h <- rmse_results[rmse_results$Improvement > 1.0 * 10^4, "Predictor"]
h

#according to rmse function:  
#1 Alley  #removed b/c NAs     
#2 OverallQual - 1
#3 MiscFeature #removed b/c NAs
#4 Neighborhood - 2
#5 Fence      #removed b/c NA's
#6 GrLivArea  - 3
#7 ExterQual  - 4 
#8 KitchenQual - 5
#9 BsmtQual    #removed b/c NAs
#10 GarageCars - 6
#11 GarageArea  #HasGarage custom feature instead of this one used - 7
#12 TotalBsmtSF #Use BasmtSF custom feature which removes unfinished portion of Basment in total - 8, also use TotalBsmtSF = 0 to create BsmtNone custom feature - 9
#13 X1stFlrSF   #Won't Use
#14 FullBath    #Won't Use
#15 TotRmsAbvGrd #Won't Use
#16 YearBuilt  #Use custom feature HouseAge instead = YearSold - YearBuilt - 10
#17 GarageFinish #Won't use, HasGarage is better
#18 YearRemodAdd #Won't use, not reliable
#19 Foundation  - 11
#20 GarageYrBlt #Won't use

#Why we removed some features (extra insight to some features....)
#//////////
train_model2 <- train %>%
  select(Alley, OverallQual, Neighborhood, Fence, GrLivArea, ExterQual, KitchenQual, BsmtQual, GarageCars)
colSums(is.na(train_model2))
#Ok so Alley, Fence and BsmtQual are the only features used that have NA's
#Alley is the biggest so start there
#Ok so looks like we were tricked, Alley and Fence have low RMSE because they had most rows dropped and got lucky
#We compared RMSE for the null model twice basically, except the second time we only had 100 rows to test, so model found patterns inherently
#lets drop Alley and Fence and see if it drops R^2
#Wow R squared droped 10%
#Ok but seeing that NA's from these variables dropped most of the rows, I have to remove them, or maybe I can predict them
#perhaps with random forest? #not worth it, because NA's are majority of entries
#We could attempt to add BsmtQual, but this may be useless b/c of NAs
#/////////////

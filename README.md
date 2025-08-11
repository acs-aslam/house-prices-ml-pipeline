# House Prices – Machine Learning Pipeline

This project implements an **end-to-end machine learning workflow** to predict house prices using structured data from the Kaggle competition *House Prices – Advanced Regression Techniques*.  
The solution focuses on **data preprocessing**, **feature engineering**, and **model optimization** to improve predictive accuracy.

## Key Features
- **Data Cleaning & Preprocessing**: Handling missing values, encoding categorical variables, log transformations.
- **Feature Engineering**: Creating new features to improve model performance.
- **Modeling**: XGBoost, RandomForest, and Rpart regression models using the `caret` package.
- **Evaluation**: Root Mean Squared Error (RMSE) on validation and Kaggle leaderboard.
- **Visualization**: ggplot2 plots for feature importance and residual analysis.

## Technologies Used
- R, caret, dplyr, ggplot2, XGBoost
- Data from Kaggle competition dataset

## Results
- Top 15% placement on Kaggle leaderboard based on RMSE.

## How to Run
1. Clone the repository  
2. Install required R packages (`caret`, `dplyr`, `ggplot2`, `xgboost`)  
3. Run `pipeline.R` to reproduce preprocessing, training, and evaluation.

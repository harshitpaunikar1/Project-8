# Project Buildup History: Housing Price Prediction

- Repository: `housing-price-prediction`
- Category: `data_science`
- Subtype: `prediction`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2021-08-16 - Day 5: Cross validation setup

- Task summary: Moved into proper k-fold cross-validation for Housing Price Prediction today. The earlier evaluation had been using a single holdout split which made the results too dependent on which rows ended up in each set. Set up a five-fold CV loop and computed mean and standard deviation of RMSE across folds. The variance was a bit higher than expected so spent time investigating whether any data leakage was happening in the preprocessing steps. Found that the StandardScaler was being fit on the full dataset before splitting which inflated the in-sample scores. Refactored the pipeline to fit transformers only inside each fold.
- Deliverable: Proper nested CV in place. Removed a data leakage point in the scaler fit step.

# Customer Churn Prediction

End-to-end machine learning project to predict customer churn (`Exited`) using the classic **Churn_Modelling.csv** dataset from a bank.

## Project Overview

- **Objective**: Predict whether a customer will leave the bank (binary classification).
- **Key Challenge**: Imbalanced dataset (~20% churn rate).
- **Solution**: SMOTE for oversampling + Hyperparameter tuning for XGBoost.
- **Best Model**: **Tuned XGBoost**

## Models Evaluated

| Model                     | Accuracy | F1 Score (Churn) | ROC-AUC |
|---------------------------|----------|------------------|---------|
| Logistic Regression       | 0.8035   |0.281536          |0.761157 |
| K-Nearest Neighbors	      | 0.8380   |0.509091          |0.789983 |
| Support Vector Machine	  |0.8520	   |0.514754	        |0.823451 |
| Decision Tree	            |0.8465	   |0.507223	        |0.837165 |
| Random Forest             |0.8565	   |0.569715	        |0.846562 |
| XGBoost (Baseline)        |0.8610	   |0.592375	        |0.860995 |
| **XGBoost (Tuned)**       | **0.8555** | **0.6172**     |**0.8559** | 
| LightGBM                  |0.8565	   |0.584660	        |0.856792 | 
| CatBoost                  |0.8550	   |0.565868	        |0.865825 | 

### Best Results (Tuned XGBoost + SMOTE)

- **Accuracy**: 0.8555
- **F1 Score** (Churn class): 0.6172
- **ROC-AUC**: 0.8559


## Project Workflow

1. Data loading and inspection
2. Data cleaning (missing values + duplicates)
3. Feature selection (dropped `RowNumber`, `CustomerId`, `Surname`)
4. Preprocessing using `ColumnTransformer`:
   - Standard Scaling for numerical features
   - One-Hot Encoding for categorical features (`Geography`, `Gender`)
5. Train/Test split (80/20, stratified)
6. Handling imbalance with **SMOTE**
7. Training multiple baseline models
8. Hyperparameter tuning for XGBoost using `GridSearchCV`
9. Model evaluation and comparison (metrics + heatmap)

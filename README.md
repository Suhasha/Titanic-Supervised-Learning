# Titanic Supervised Learning – Week 4

Supervised learning classification on the Titanic dataset using Python and Scikit-learn. The project compares Logistic Regression with a tuned Random Forest, applies preprocessing and feature engineering, uses stratified 5-fold cross-validation, evaluates accuracy, precision, recall, F1-score and ROC-AUC, and analyzes feature importance.

## Problem
Predict whether a Titanic passenger survived (`Survived = 1`) or did not survive (`Survived = 0`).

## Models
- Logistic Regression as an interpretable baseline
- Tuned Random Forest Classifier as the final model

## Workflow
1. Load Week 1 cleaned Titanic data
2. Select relevant predictive features
3. Exclude identifiers and free-text fields to reduce noise/leakage
4. Train/test split with stratification
5. Median imputation and scaling where appropriate
6. 5-fold stratified cross-validation
7. Grid search for Random Forest hyperparameters
8. Evaluate on an unseen test set
9. Analyze confusion matrix, ROC curve and feature importance

## Best Random Forest Parameters
`{'model__max_depth': 12, 'model__min_samples_leaf': 1, 'model__min_samples_split': 2, 'model__n_estimators': 200}`

## Results
              Model  Accuracy  Precision   Recall       F1  ROC_AUC
Logistic Regression  0.804469   0.783333 0.681159 0.728682 0.844401
Tuned Random Forest  0.798883   0.761905 0.695652 0.727273 0.827470

## Cross-validation
              Model  CV_Mean_Accuracy   CV_Std
Logistic Regression          0.799183 0.018909
Tuned Random Forest          0.824485 0.020878

## Repository Structure
- `code/` – Python implementation
- `data/` – modeling data, metrics and predictions
- `figures/` – evaluation visualizations
- `report/` – detailed DOCX report

"""
Week 4 - Supervised Learning Model Implementation
Titanic survival classification using Logistic Regression and Random Forest.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA = "../data/titanic_supervised_model_dataset.csv"

df = pd.read_csv(DATA)

features = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "Sex_male", "Embarked_Q", "Embarked_S"
]

X = df[features]
y = df["Survived"]

# Stratified split preserves the class distribution.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Interpretable baseline.
logistic = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

logistic_cv = cross_val_score(logistic, X_train, y_train, cv=cv, scoring="accuracy")
logistic.fit(X_train, y_train)

# Non-linear final model.
rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
])

params = {
    "model__n_estimators": [200],
    "model__max_depth": [None, 8, 12],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

search = GridSearchCV(rf, params, cv=cv, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
pred = best_model.predict(X_test)
prob = best_model.predict_proba(X_test)[:, 1]

print("Best parameters:", search.best_params_)
print("5-fold CV accuracy:", search.best_score_)
print("Test accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1-score:", f1_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))

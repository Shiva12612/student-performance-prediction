import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import numpy as np


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("data/student_data.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================
# 2. FEATURES
# ============================================

features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score",
    "internal_marks",
    "sleep_hours",
    "internet_usage",
    "extracurricular"
]


# ============================================
# 3. REGRESSION MODEL
# Predict Final Marks
# ============================================

X = df[features]

y_marks = df["final_marks"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_marks,
    test_size=0.20,
    random_state=42
)


print("\n================================")
print("Training Linear Regression")
print("================================")


regression_model = LinearRegression()

regression_model.fit(
    X_train,
    y_train
)


# Prediction

y_pred = regression_model.predict(
    X_test
)


# Evaluation

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\nRegression Results")

print(
    "Mean Absolute Error:",
    round(mae, 2)
)

print(
    "Root Mean Squared Error:",
    round(rmse, 2)
)

print(
    "R2 Score:",
    round(r2, 4)
)


# Save regression model

joblib.dump(
    regression_model,
    "models/regression_model.pkl"
)

print(
    "\nRegression model saved successfully!"
)


# ============================================
# 4. CLASSIFICATION MODEL
# Predict Performance Category
# ============================================

X = df[features]

y_category = df["performance"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_category,
    test_size=0.20,
    random_state=42,
    stratify=y_category
)


print("\n================================")
print("Training Random Forest Classifier")
print("================================")


classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


classifier.fit(
    X_train,
    y_train
)


# Prediction

y_pred_class = classifier.predict(
    X_test
)


# ============================================
# CLASSIFICATION METRICS
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred_class
)

precision = precision_score(
    y_test,
    y_pred_class,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred_class,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred_class,
    average="weighted",
    zero_division=0
)


print("\nClassification Results")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)


print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred_class,
        zero_division=0
    )
)


# ============================================
# SAVE CLASSIFICATION MODEL
# ============================================

joblib.dump(
    classifier,
    "models/classification_model.pkl"
)


print(
    "\nClassification model saved successfully!"
)


# ============================================
# FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({

    "feature": features,

    "importance": classifier.feature_importances_

})


importance = importance.sort_values(
    by="importance",
    ascending=False
)


print("\n================================")
print("Feature Importance")
print("================================")

print(
    importance
)


# Save feature importance

importance.to_csv(
    "models/feature_importance.csv",
    index=False
)


print(
    "\nFeature importance saved successfully!"
)


# ============================================
# FINAL MESSAGE
# ============================================

print("\n==========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("==========================================")

print(
    "\nFiles created:"
)

print(
    "1. models/regression_model.pkl"
)

print(
    "2. models/classification_model.pkl"
)

print(
    "3. models/feature_importance.csv"
)
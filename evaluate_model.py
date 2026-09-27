import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/student_data.csv"
)


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


X = df[features]


# ==========================================
# 2. LOAD MODELS
# ==========================================

regression_model = joblib.load(
    "models/regression_model.pkl"
)

classification_model = joblib.load(
    "models/classification_model.pkl"
)


# ==========================================
# 3. REGRESSION EVALUATION
# ==========================================

y_marks = df["final_marks"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_marks,
    test_size=0.20,
    random_state=42
)


predicted_marks = regression_model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predicted_marks
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predicted_marks
    )
)

r2 = r2_score(
    y_test,
    predicted_marks
)


print("\n===================================")
print("REGRESSION MODEL EVALUATION")
print("===================================")

print(
    "MAE:",
    round(mae, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)

print(
    "R2 Score:",
    round(r2, 4)
)


# ==========================================
# 4. ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predicted_marks,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel(
    "Actual Marks"
)

plt.ylabel(
    "Predicted Marks"
)

plt.title(
    "Actual vs Predicted Marks"
)

plt.tight_layout()

plt.savefig(
    "models/actual_vs_predicted.png"
)

plt.close()


# ==========================================
# 5. CLASSIFICATION EVALUATION
# ==========================================

y_category = df["performance"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_category,
    test_size=0.20,
    random_state=42,
    stratify=y_category
)


predicted_category = classification_model.predict(
    X_test
)


cm = confusion_matrix(
    y_test,
    predicted_category,
    labels=[
        "Excellent",
        "Good",
        "Average",
        "Needs Improvement"
    ]
)


print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(cm)


# ==========================================
# 6. CONFUSION MATRIX GRAPH
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Excellent",
        "Good",
        "Average",
        "Needs Improvement"
    ]
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "Student Performance Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png"
)

plt.close()


# ==========================================
# 7. PERFORMANCE DISTRIBUTION
# ==========================================

performance_counts = df[
    "performance"
].value_counts()


plt.figure(figsize=(8, 6))

performance_counts.plot(
    kind="bar"
)

plt.xlabel(
    "Performance Category"
)

plt.ylabel(
    "Number of Students"
)

plt.title(
    "Student Performance Distribution"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    "models/performance_distribution.png"
)

plt.close()


# ==========================================
# 8. FEATURE IMPORTANCE GRAPH
# ==========================================

importance_df = pd.read_csv(
    "models/feature_importance.csv"
)

importance_df = importance_df.sort_values(
    "importance"
)


plt.figure(figsize=(8, 6))

plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "models/feature_importance.png"
)

plt.close()


# ==========================================
# 9. SAVE PREDICTION RESULTS
# ==========================================

results = X_test.copy()

results["actual_performance"] = y_test.values

results["predicted_performance"] = (
    predicted_category
)

results.to_csv(
    "models/test_predictions.csv",
    index=False
)


# ==========================================
# 10. FINAL MESSAGE
# ==========================================

print("\n===================================")
print("EVALUATION COMPLETED SUCCESSFULLY")
print("===================================")

print("\nGenerated files:")

print(
    "1. models/actual_vs_predicted.png"
)

print(
    "2. models/confusion_matrix.png"
)

print(
    "3. models/performance_distribution.png"
)

print(
    "4. models/feature_importance.png"
)

print(
    "5. models/test_predictions.csv"
)
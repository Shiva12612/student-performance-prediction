import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(50, 100, n)
previous_marks = np.random.uniform(40, 95, n)
assignment_score = np.random.uniform(40, 100, n)
internal_marks = np.random.uniform(40, 95, n)
sleep_hours = np.random.uniform(4, 9, n)
internet_usage = np.random.uniform(1, 8, n)

extracurricular = np.random.choice(
    [0, 1],
    size=n,
    p=[0.55, 0.45]
)

# Generate final marks
final_marks = (
    study_hours * 2.5
    + attendance * 0.20
    + previous_marks * 0.25
    + assignment_score * 0.15
    + internal_marks * 0.20
    + sleep_hours * 0.8
    - internet_usage * 0.5
    + extracurricular * 1.5
)

# Add small random variation
noise = np.random.normal(0, 3, n)

final_marks = final_marks + noise

# Keep marks between 0 and 100
final_marks = np.clip(final_marks, 0, 100)

# Performance category
def get_category(mark):

    if mark >= 85:
        return "Excellent"

    elif mark >= 70:
        return "Good"

    elif mark >= 60:
        return "Average"

    else:
        return "Needs Improvement"


performance = [
    get_category(mark)
    for mark in final_marks
]

# Create DataFrame
df = pd.DataFrame({

    "student_id": [
        f"ST{i:04d}"
        for i in range(1, n + 1)
    ],

    "study_hours": np.round(
        study_hours, 1
    ),

    "attendance": np.round(
        attendance, 1
    ),

    "previous_marks": np.round(
        previous_marks, 1
    ),

    "assignment_score": np.round(
        assignment_score, 1
    ),

    "internal_marks": np.round(
        internal_marks, 1
    ),

    "sleep_hours": np.round(
        sleep_hours, 1
    ),

    "internet_usage": np.round(
        internet_usage, 1
    ),

    "extracurricular": extracurricular,

    "final_marks": np.round(
        final_marks, 1
    ),

    "performance": performance
})

# Save dataset
df.to_csv(
    "data/student_data.csv",
    index=False
)

print("Dataset created successfully!")
print()
print("Number of students:", len(df))
print()
print("Dataset shape:", df.shape)
print()
print("First 5 records:")
print(df.head())
print()
print("Performance distribution:")
print(df["performance"].value_counts())
import pandas as pd

# Load dataset
df = pd.read_csv("data/student_data.csv")

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== PERFORMANCE DISTRIBUTION ==========")
print(df["performance"].value_counts())

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\nData analysis completed successfully.")
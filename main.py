import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("student_data.csv")

# Check required columns exist
required_cols = ["StudentID", "Name", "Gender", "FinalGrade"]
for col in required_cols:
    if col not in data.columns:
        print(f"ERROR: Missing column '{col}' in student_data.csv")
        print("Available columns:", list(data.columns))
        exit()

# Save student info before dropping
student_info = data[["StudentID", "Name", "Gender"]].copy()

# Remove unnecessary columns for training
data = data.drop(["StudentID", "Name", "Gender"], axis=1)

# Handle missing values
data = data.fillna(data.mean(numeric_only=True))
data = data.fillna("Unknown")

# Convert text → numbers
data = pd.get_dummies(data)

# Features and target
X = data.drop("FinalGrade", axis=1)
y = data["FinalGrade"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Linear Regression --------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_accuracy = lr_model.score(X_test, y_test)
print("Linear Regression Accuracy:", lr_accuracy)

# -------- Random Forest --------
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
print("Random Forest Accuracy:", rf_accuracy)

# -------- Choose Best Model --------
if rf_accuracy > lr_accuracy:
    best_model = rf_model
    print("\nRandom Forest is better!")
else:
    best_model = lr_model
    print("\nLinear Regression is better!")

# -------- Single Sample Prediction with Student Info --------
sample = X_test.iloc[0:1]
prediction = best_model.predict(sample)

original_index = X_test.index[0]
student = student_info.loc[original_index]

print("\n--- Student Info ---")
print("Student ID :", student["StudentID"])
print("Name       :", student["Name"])
print("Gender     :", student["Gender"])
print("Predicted Final Grade:", round(prediction[0], 2))
print("Actual Final Grade   :", y_test.iloc[0])

# -------- All Test Predictions with Student Info --------
y_pred = best_model.predict(X_test)

results = student_info.loc[X_test.index].copy()
results["ActualGrade"]    = y_test.values
results["PredictedGrade"] = y_pred.round(2)

print("\n--- All Test Predictions ---")
print(results.to_string(index=False))

# -------- Plot --------
plt.figure()
plt.scatter(y_test, y_pred, color="steelblue", edgecolors="black", alpha=0.7)
plt.xlabel("Actual Final Grades")
plt.ylabel("Predicted Final Grades")
plt.title("Actual vs Predicted Student Performance")
plt.tight_layout()
plt.show()

# -------- Save Model (must match app.py) --------
joblib.dump(best_model, "student_performance_model.pkl")
print("\nModel saved as student_performance_model.pkl successfully!")
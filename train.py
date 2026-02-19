import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("student_data.csv")

# Drop missing values
df = df.dropna()

# Use ONLY these 3 features — must match what app.py sends
X = df[["StudyHoursPerWeek", "AttendanceRate", "PreviousGrade"]]
y = df["FinalGrade"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_accuracy = lr_model.score(X_test, y_test)
print("Linear Regression Accuracy:", lr_accuracy)

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
print("Random Forest Accuracy:", rf_accuracy)

# Pick best
if rf_accuracy > lr_accuracy:
    best_model = rf_model
    print("Random Forest is better!")
else:
    best_model = lr_model
    print("Linear Regression is better!")

# Verify it's 3 features
print("Model trained on features:", X.columns.tolist())
print("Number of features:", best_model.n_features_in_)

# Save
joblib.dump(best_model, "student_performance_model.pkl")
print("Model saved as student_performance_model.pkl!")
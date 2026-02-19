from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Database setup
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///predictions.db"
db = SQLAlchemy(app)

# Create table
class Prediction(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    student_id      = db.Column(db.String(50))
    name            = db.Column(db.String(100))
    gender          = db.Column(db.String(20))
    study_hours     = db.Column(db.Float)
    attendance      = db.Column(db.Float)
    previous_grade  = db.Column(db.Float)
    predicted_score = db.Column(db.Float)

with app.app_context():
    db.create_all()

# Load model
model = joblib.load("student_performance_model.pkl")

# Load dataset for dashboard
data = pd.read_csv("student_data.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    student_id     = request.form["student_id"]
    name           = request.form["name"]
    gender         = request.form["gender"]

    study_hours    = float(request.form["study_hours"])
    attendance     = float(request.form["attendance"])
    previous_grade = float(request.form["previous_grade"])

    # Must match training feature order: StudyHoursPerWeek, AttendanceRate, PreviousGrade
    features   = np.array([[study_hours, attendance, previous_grade]])
    prediction = model.predict(features)
    score      = round(float(prediction[0]), 2)

    new_entry = Prediction(
        student_id      = student_id,
        name            = name,
        gender          = gender,
        study_hours     = study_hours,
        attendance      = attendance,
        previous_grade  = previous_grade,
        predicted_score = score
    )
    db.session.add(new_entry)
    db.session.commit()

    return render_template("index.html", result=score, student_name=name)


@app.route("/dashboard")
def dashboard():
    plt.figure()
    # Fixed: use correct CSV column names
    plt.scatter(data["StudyHoursPerWeek"], data["FinalGrade"], color="steelblue", alpha=0.7)
    plt.xlabel("Study Hours Per Week")
    plt.ylabel("Final Grade")
    plt.title("Study Hours vs Final Grade")

    graph_path = "static/graph.png"
    plt.savefig(graph_path)
    plt.close()

    records = Prediction.query.all()
    stats = None
    if records:
        scores = [r.predicted_score for r in records]
        stats = {
            "total":     len(scores),
            "avg_score": round(sum(scores) / len(scores), 2),
            "max_score": max(scores),
            "min_score": min(scores)
        }

    return render_template("dashboard.html", graph=graph_path, stats=stats)


@app.route("/history")
def history():
    records = Prediction.query.all()
    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)

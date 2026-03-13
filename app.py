# app.py
from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "super-secret-change-me-2025"   # change this!!!

# Increase if needed — but Render free has memory limit
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024   # 5 MB

try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print("Model loading failed!", e)
    model = scaler = None

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def count_seizure_sessions(predictions):
    if len(predictions) == 0:
        return 0
    sessions = 0
    active = False
    for p in predictions:
        if p == 1 and not active:
            sessions += 1
            active = True
        elif p == 0:
            active = False
    return sessions

@app.route("/", methods=["GET", "POST"])
def index():
    table_html = None
    sessions = None
    rows_processed = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part"
            return render_template("index.html", error=error)

        file = request.files["file"]
        if file.filename == "":
            error = "No selected file"
            return render_template("index.html", error=error)

        if file and allowed_file(file.filename):
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                if not all(col in df.columns for col in ["heart_rate", "spo2", "temperature", "vibration"]):
                    error = "File must contain columns: heart_rate, spo2, temperature, vibration"
                else:
                    X = df[["heart_rate", "spo2", "temperature", "vibration"]]
                    X_scaled = scaler.transform(X)
                    preds = model.predict(X_scaled)

                    df["seizure_prediction"] = preds
                    sessions = count_seizure_sessions(preds)
                    rows_processed = len(df)

                    # Show only first 100 rows to avoid huge page
                    table_html = df.head(100).to_html(classes="table table-dark table-striped", index=False)

            except Exception as e:
                error = f"Error processing file: {str(e)}"
        else:
            error = "Only .csv, .xlsx, .xls files allowed"

    return render_template(
        "index.html",
        table=table_html,
        sessions=sessions,
        rows_processed=rows_processed,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=5000)   # only local — Render uses gunicorn
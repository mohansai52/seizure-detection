# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import os
import requests
# For headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.secret_key = os.environ.get("SECRET_KEY", "super-secret-change-me-2025")

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024   # 5 MB

# Load model & scaler
model = None
scaler = None
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    print("Model and scaler loaded successfully!")
except Exception as e:
    print("Model loading failed:", e)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ====================== NEW: CORE FUNCTION ======================
def process_incoming_data(row_dict):
    """Takes one sensor reading and returns prediction"""
    try:
        features = pd.DataFrame([[
            float(row_dict['heart_rate']),
            float(row_dict['spo2']),
            float(row_dict['temperature']),
            float(row_dict['vibration'])
        ]], columns=["heart_rate", "spo2", "temperature", "vibration"])

        X_scaled = scaler.transform(features)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1] if hasattr(model, 'predict_proba') else None

        return {
            "prediction": int(prediction),
            "probability": float(probability) if probability is not None else None,
            "is_seizure": bool(prediction == 1)
        }
    except Exception as e:
        return {"error": str(e)}

# ====================== NEW: REAL-TIME API ======================
@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    result = process_incoming_data(data)
    return jsonify(result)

# ====================== NEW: REAL-TIME PAGE ======================
@app.route("/realtime")
def realtime():
    return render_template("realtime.html")

# ====================== YOUR EXISTING CODE (unchanged) ======================
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
# Send SMS via Fast2SMS
@app.route("/send_sms", methods=["POST"])
def send_sms():
    data = request.get_json()
    phone = data.get("phone")
    message = data.get("message", "Seizure detected! Immediate attention required.")

    if not phone or len(phone) < 10:
        return jsonify({"success": False, "error": "Invalid phone number"}), 400

    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    twilio_number = os.environ.get('TWILIO_PHONE_NUMBER')

    if not account_sid or not auth_token or not twilio_number:
        return jsonify({"success": False, "error": "Twilio credentials not set in environment"}), 500

    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)

        # Add +91 for Indian numbers
        to_number = f"+91{phone}" if not phone.startswith('+') else phone

        sms = client.messages.create(
            body=message,
            from_=twilio_number,
            to=to_number
        )

        print(f"Twilio SMS sent: {sms.sid}")
        return jsonify({"success": True, "message": "SMS sent via Twilio", "sid": sms.sid})

    except Exception as e:
        print(f"Twilio Exception: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
@app.route("/", methods=["GET", "POST"])
def index():
    table_html = None
    sessions = None
    rows_processed = None
    error = None
    pred_dist_plot = None
    timeseries_plot = None

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
                # Read file
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                required_cols = ["heart_rate", "spo2", "temperature", "vibration"]
                if not all(col in df.columns for col in required_cols):
                    error = f"File must contain columns: {', '.join(required_cols)}"
                else:
                    # Prepare data
                    X = df[required_cols]
                    X_scaled = scaler.transform(X)
                    preds = model.predict(X_scaled)

                    df["seizure_prediction"] = preds
                    sessions = count_seizure_sessions(preds)
                    rows_processed = len(df)

                    # ─── Generate plots ───────────────────────────────────────────────
                    plots_dir = os.path.join('static', 'plots')
                    os.makedirs(plots_dir, exist_ok=True)

                    # Optional: clean old plots (prevents disk filling up over many requests)
                    for old_file in os.listdir(plots_dir):
                        os.remove(os.path.join(plots_dir, old_file))

                    # Plot 1: Count of predicted classes
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(x=df["seizure_prediction"], ax=ax, palette="Blues_d")
                    ax.set_title("Predicted Seizure vs Non-Seizure")
                    ax.set_xlabel("Prediction (0 = Normal, 1 = Seizure)")
                    ax.set_ylabel("Count")
                    plt.tight_layout()
                    pred_dist_path = os.path.join(plots_dir, 'prediction_distribution.png')
                    plt.savefig(pred_dist_path, dpi=120, bbox_inches='tight')
                    plt.close(fig)
                    pred_dist_plot = '/static/plots/prediction_distribution.png'

                    # Plot 2: Time series overlay (heart rate + predictions)
                    if 'heart_rate' in df.columns:
                        fig, ax = plt.subplots(figsize=(10, 4.5))
                        ax.plot(df.index, df['heart_rate'], label='Heart Rate', color='#1e88e5', linewidth=1.2)
                        ax.set_ylabel("Heart Rate (bpm)", color='#1e88e5')
                        ax.tick_params(axis='y', labelcolor='#1e88e5')

                        ax2 = ax.twinx()
                        ax2.plot(df.index, df["seizure_prediction"], label='Seizure Prediction',
                                 color='#d81b60', linewidth=1.8, alpha=0.7)
                        ax2.set_ylabel("Prediction (1 = Seizure)", color='#d81b60')
                        ax2.tick_params(axis='y', labelcolor='#d81b60')
                        ax2.set_yticks([0, 1])

                        fig.suptitle("Heart Rate with Seizure Predictions Overlay", fontsize=14)
                        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
                        plt.tight_layout()
                        timeseries_path = os.path.join(plots_dir, 'timeseries_overlay.png')
                        plt.savefig(timeseries_path, dpi=120, bbox_inches='tight')
                        plt.close(fig)
                        timeseries_plot = '/static/plots/timeseries_overlay.png'

                    # Prepare preview table (first 100 rows)
                    table_html = df.head(100).to_html(
                        classes="table table-dark table-striped table-hover",
                        index=False
                    )

            except Exception as e:
                error = f"Error processing file: {str(e)}"
        else:
            error = "Only .csv, .xlsx, .xls files allowed"

    return render_template(
        "index.html",
        table=table_html,
        sessions=sessions,
        rows_processed=rows_processed,
        error=error,
        pred_dist_plot=pred_dist_plot,
        timeseries_plot=timeseries_plot
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


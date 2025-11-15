from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import numpy as np

MODELS_DIR = "models"

app = Flask(__name__)
CORS(app)

# ---------- Load model + preprocessors ----------
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_gbs_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    selector = joblib.load(os.path.join(MODELS_DIR, "selector.pkl"))
    target_encoder = joblib.load(os.path.join(MODELS_DIR, "target_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_columns.txt"), "r") as f:
        feature_columns = [line.strip() for line in f.readlines()]

    return model, scaler, label_encoders, selector, target_encoder, feature_columns

print("🔄 Loading model and preprocessors...")
model, scaler, label_encoders, selector, target_encoder, feature_columns = load_artifacts()
print("✅ Artifacts loaded.")

# ---------- Helper: preprocess incoming JSON ----------
def preprocess_input(json_data):
    # Expect json_data to be a dict of feature_name: value
    df = pd.DataFrame([json_data])

    # Ensure all expected feature columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # default to 0 if missing

    # Order columns to match training
    df = df[feature_columns]

    # Encode categorical columns using stored label_encoders
    for col, enc in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            # Handle unseen categories:
            df[col] = df[col].map(lambda v: v if v in enc.classes_ else enc.classes_[0])
            df[col] = enc.transform(df[col])

    # Scale numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])

    # Feature selection
    X_selected = selector.transform(df_scaled)

    return X_selected

# ---------- Routes ----------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Preprocess input
        X = preprocess_input(data)

        # Predict
        y_pred = model.predict(X)
        subtype = target_encoder.inverse_transform(y_pred)[0]

        return jsonify({
            "success": True,
            "predicted_subtype": subtype
        }), 200

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)

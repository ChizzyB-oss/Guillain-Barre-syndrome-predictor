import joblib
import os
import numpy as np
import pandas as pd

MODELS_DIR = "models"

# Load artifacts once
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_gbs_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    selector = joblib.load(os.path.join(MODELS_DIR, "feature_selector.pkl"))
    target_encoder = joblib.load(os.path.join(MODELS_DIR, "target_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_names.txt"), "r") as f:
        feature_columns = [line.strip() for line in f.readlines()]

    return model, scaler, label_encoders, selector, target_encoder, feature_columns


# Load once globally
model, scaler, label_encoders, selector, target_encoder, feature_columns = load_artifacts()


# ---------- PREPROCESS INPUT ----------
def preprocess_input(json_data):
    df = pd.DataFrame([json_data])

    # Ensure all expected columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder
    df = df[feature_columns]

    # Encode categoricals
    for col, enc in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].map(lambda v: v if v in enc.classes_ else enc.classes_[0])
            df[col] = enc.transform(df[col])

    # Scale numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Select features
    X_selected = selector.transform(df)

    return X_selected

import joblib
import os
import pandas as pd
import numpy as np

MODELS_DIR = "models"

def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_gbs_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    selector = joblib.load(os.path.join(MODELS_DIR, "selector.pkl"))
    target_encoder = joblib.load(os.path.join(MODELS_DIR, "target_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_columns.txt"), "r") as f:
        feature_columns = [line.strip() for line in f.readlines()]

    return model, scaler, label_encoders, selector, target_encoder, feature_columns


model, scaler, label_encoders, selector, target_encoder, feature_columns = load_artifacts()


def preprocess_input(data):
    df = pd.DataFrame([data])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    for col, enc in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: x if x in enc.classes_ else enc.classes_[0])
            df[col] = enc.transform(df[col])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    X_selected = selector.transform(df)

    return X_selected

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# --------- CONFIG ---------
DATA_PATH = "data/synthetic_gbs_dataset_clinical.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age groups
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 18, 40, 60, 100],
        labels=['child', 'young_adult', 'adult', 'senior']
    )

    # Symptom severity (simple sum of some binary columns)
    symptom_cols = [
        'muscle_weakness', 'paralysis', 'sensory_loss',
        'respiratory_involvement', 'cranial_nerve_involvement'
    ]
    df['symptom_severity'] = df[symptom_cols].sum(axis=1)

    # Electrophysiological ratio
    df['velocity_ratio'] = df['motor_velocity'] / df['sensory_velocity']
    df['velocity_ratio'] = df['velocity_ratio'].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    # CSF protein category
    df['csf_category'] = pd.cut(
        df['csf_protein'],
        bins=[0, 45, 90, 200],
        labels=['normal', 'elevated', 'high']
    )

    return df

def main():
    print("📥 Loading dataset...")
    data = pd.read_csv(DATA_PATH)

    # ---------- Feature engineering ----------
    print("🧠 Engineering features...")
    data = engineer_features(data)

    # ---------- Define features and target ----------
    target_col = "gbs_subtype"
    feature_cols = [c for c in data.columns if c != target_col]

    X = data[feature_cols].copy()
    y = data[target_col].copy()

    # ---------- Encode categorical features ----------
    print("🔤 Encoding categorical variables...")
    categorical_cols = ['gender', 'previous_infection', 'onset_speed',
                        'age_group', 'csf_category']

    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # ---------- Encode target ----------
    print("🎯 Encoding target variable...")
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # ---------- Train/test split ----------
    print("✂ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ---------- Scale numeric features ----------
    print("📏 Scaling numeric features...")
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # ---------- Feature selection ----------
    print("🔍 Selecting top features (k=15)...")
    selector = SelectKBest(score_func=f_classif, k=15)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    print("✅ Selected features:")
    for f in selected_features:
        print("   -", f)

    # ---------- Train final model ----------
    print("🌲 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_sel, y_train)

    # (optional) Evaluate quickly
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"✅ Test Accuracy: {acc:.3f}, F1: {f1:.3f}")

    # ---------- Save everything ----------
    print("💾 Saving model and preprocessors...")

    joblib.dump(model, os.path.join(MODELS_DIR, "best_gbs_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
    joblib.dump(selector, os.path.join(MODELS_DIR, "selector.pkl"))
    joblib.dump(target_encoder, os.path.join(MODELS_DIR, "target_encoder.pkl"))

    # Save feature names BEFORE selection (for ordering at prediction time)
    with open(os.path.join(MODELS_DIR, "feature_columns.txt"), "w") as f:
        for col in X.columns:
            f.write(col + "\n")

    print("🎉 Training complete. Files saved in 'models/'")

if __name__ == "__main__":
    main()

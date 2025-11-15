from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import joblib
import os
import pandas as pd
import numpy as np
import json
import jwt

from models import db, User, Prediction

# ---------------- CONFIG ----------------

MODELS_DIR = "models"
DATABASE_URL = "sqlite:///gbs_system.db"
JWT_SECRET = "super-secret-key-change-this"  # for demo; change for real project
JWT_ALGO = "HS256"
TOKEN_EXP_DAYS = 7

# -------------- APP INIT ----------------

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = JWT_SECRET

db.init_app(app)

# Create DB tables on first run
with app.app_context():
    db.create_all()
    print("✅ Database tables ensured.")

# -------------- ML ARTIFACTS --------------

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
print("✅ ML artifacts loaded.")

# -------------- AUTH HELPER --------------

def generate_token(user):
    payload = {
        "user_id": user.id,
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXP_DAYS),
        "iat": datetime.utcnow()
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    # PyJWT returns str in newer versions, bytes in older; normalise:
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def token_required(f):
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", None)
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization token required"}), 401

        token = auth_header.split(" ")[1]

        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            user = User.query.get(data["user_id"])
            if not user:
                return jsonify({"error": "User not found"}), 401
            request.user = user
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401

        return f(*args, **kwargs)

    return decorated

# ------------- PREPROCESS INPUT -------------

def preprocess_input(json_data):
    df = pd.DataFrame([json_data])

    # Ensure all expected feature columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Order columns
    df = df[feature_columns]

    # Encode categorical columns
    for col, enc in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            # handle unseen
            df[col] = df[col].map(lambda v: v if v in enc.classes_ else enc.classes_[0])
            df[col] = enc.transform(df[col])

    # Scale numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])

    # Feature selection
    X_selected = selector.transform(df_scaled)

    return X_selected

# ---------------- ROUTES -------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Backend running"}), 200


# ---------- AUTH: REGISTER ----------

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "username, email, and password are required"}), 400

    # Check existing user
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({"error": "User with this username or email already exists"}), 400

    password_hash = generate_password_hash(password)

    user = User(
        username=username,
        email=email,
        password_hash=password_hash
    )
    db.session.add(user)
    db.session.commit()

    token = generate_token(user)

    return jsonify({
        "success": True,
        "message": "User registered successfully",
        "token": token,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email
        }
    }), 201


# ---------- AUTH: LOGIN ----------

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = generate_token(user)

    return jsonify({
        "success": True,
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email
        }
    }), 200


# ---------- PREDICT (PROTECTED) ----------

@app.route("/api/predict", methods=["POST"])
@token_required
def predict():
    try:
        data = request.get_json() or {}

        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Preprocess
        X = preprocess_input(data)

        # Predict
        y_pred = model.predict(X)
        subtype = target_encoder.inverse_transform(y_pred)[0]

        # Confidence (if supported)
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            # probability of predicted class
            class_index = list(model.classes_).index(y_pred[0])
            confidence = float(proba[class_index])

        # Save prediction in DB
        pred = Prediction(
            user_id=request.user.id,
            input_data=json.dumps(data),
            predicted_subtype=subtype,
            confidence=confidence
        )
        db.session.add(pred)
        db.session.commit()

        return jsonify({
            "success": True,
            "predicted_subtype": subtype,
            "confidence": confidence,
            "prediction_id": pred.id,
            "timestamp": pred.created_at.isoformat()
        }), 200

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ---------- GET PREDICTION HISTORY ----------

@app.route("/api/predictions", methods=["GET"])
@token_required
def get_predictions():
    preds = Prediction.query.filter_by(user_id=request.user.id).order_by(Prediction.created_at.desc()).all()

    results = []
    for p in preds:
        results.append({
            "id": p.id,
            "input_data": json.loads(p.input_data),
            "predicted_subtype": p.predicted_subtype,
            "confidence": p.confidence,
            "timestamp": p.created_at.isoformat()
        })

    return jsonify({
        "success": True,
        "count": len(results),
        "predictions": results
    }), 200


if __name__ == "__main__":
    app.run(debug=True)

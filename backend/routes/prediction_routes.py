from flask import Blueprint, request, jsonify
import json
from functools import wraps
import jwt

from models import db, Prediction
from routes.auth_routes import JWT_SECRET, JWT_ALGO
from utils.preprocessing_utils import model, target_encoder, preprocess_input
from models import User


prediction_bp = Blueprint("prediction_bp", __name__)


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", None)
        if not auth or not auth.startswith("Bearer "):
            return jsonify({"error": "Token missing"}), 401

        token = auth.split(" ")[1]

        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            from models import User
            request.user = User.query.get(data["user_id"])
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401

        return f(*args, **kwargs)
    return decorated


@prediction_bp.route("/api/predict", methods=["POST"])
@token_required
def predict():
    try:
        data = request.get_json() or {}

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # ---- Preprocess input ----
        X = preprocess_input(data)

        # ---- Predict ----
        y_pred = model.predict(X)
        subtype = target_encoder.inverse_transform(y_pred)[0]

        # ---- Confidence ----
        confidence = None
        all_probabilities = {}

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]   # array of probabilities
            class_labels = target_encoder.inverse_transform(model.classes_)

            # Predicted class probability
            confidence = float(proba[list(model.classes_).index(y_pred[0])])

            # Full probability mapping
            all_probabilities = {
                class_labels[i]: float(prob)
                for i, prob in enumerate(proba)
            }

        # ---- Save prediction in DB ----
        pred = Prediction(
            user_id=request.user.id,
            input_data=json.dumps(data),
            predicted_subtype=subtype,
            confidence=confidence,
        )

        db.session.add(pred)
        db.session.commit()

        # ---- Return full response ----
        return jsonify({
            "success": True,
            "predicted_subtype": subtype,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "prediction_id": pred.id,
            "created_at": pred.created_at.isoformat()
        }), 200

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@prediction_bp.route("/api/predictions/history", methods=["GET"])
@token_required
def get_history():
    preds = Prediction.query.filter_by(user_id=request.user.id).all()

    result = []
    for p in preds:
        result.append({
            "id": p.id,
            "input_data": json.loads(p.input_data),
            "predicted_subtype": p.predicted_subtype,
            "confidence": p.confidence,
            "created_at": p.created_at.isoformat(),
        })

    return jsonify({"success": True, "predictions": result})

from flask import Blueprint, request, jsonify
import json

from models import db, Prediction
from auth_routes import generate_token  # reuses token method
from utils.preprocessing_utils import model, target_encoder, preprocess_input
from auth_routes import JWT_SECRET, JWT_ALGO

import jwt
from functools import wraps

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
    data = request.get_json() or {}
    X = preprocess_input(data)

    y_pred = model.predict(X)
    subtype = target_encoder.inverse_transform(y_pred)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        class_index = list(model.classes_).index(y_pred[0])
        confidence = float(proba[class_index])

    pred = Prediction(
        user_id=request.user.id,
        input_data=json.dumps(data),
        predicted_subtype=subtype,
        confidence=confidence,
    )
    db.session.add(pred)
    db.session.commit()

    return jsonify({
        "success": True,
        "predicted_subtype": subtype,
        "confidence": confidence,
        "prediction_id": pred.id,
        "created_at": pred.created_at.isoformat(),
    })


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

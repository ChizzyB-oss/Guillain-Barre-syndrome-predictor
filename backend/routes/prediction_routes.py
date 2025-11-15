# prediction_routes.py

from flask import Blueprint, request, jsonify
from datetime import datetime
import json

from models import db, Prediction
from auth_routes import token_required
from utils.preprocessing_utils import (
    model, scaler, label_encoders, selector, target_encoder, preprocess_input
)

prediction_bp = Blueprint("prediction_bp", __name__)


# ------------------ POST /api/predict ------------------

@prediction_bp.route("/api/predict", methods=["POST"])
@token_required
def predict():
    try:
        data = request.get_json() or {}

        # Prepare ML input
        X = preprocess_input(data)

        # Predict subtype
        y_pred = model.predict(X)
        subtype = target_encoder.inverse_transform(y_pred)[0]

        # Probability distribution
        all_probabilities = {}
        confidence = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            class_labels = target_encoder.inverse_transform(model.classes_)

            for label, p in zip(class_labels, proba):
                all_probabilities[label] = float(p)

            confidence = float(all_probabilities[subtype])

        # Save prediction to DB
        pred_record = Prediction(
            user_id=request.user.id,
            input_data=json.dumps(data),
            predicted_subtype=subtype,
            confidence=confidence,
            all_probabilities=json.dumps(all_probabilities),
        )
        db.session.add(pred_record)
        db.session.commit()

        return jsonify({
            "success": True,
            "predicted_subtype": subtype,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "prediction_id": pred_record.id,
            "created_at": pred_record.created_at.isoformat()
        }), 200

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------ GET /api/predictions/history ------------------

@prediction_bp.route("/api/predictions/history", methods=["GET"])
@token_required
def history():
    preds = (Prediction.query
             .filter_by(user_id=request.user.id)
             .order_by(Prediction.created_at.desc())
             .all())

    results = []
    for p in preds:
        results.append({
            "id": p.id,
            "input_data": json.loads(p.input_data),
            "predicted_subtype": p.predicted_subtype,
            "confidence": p.confidence,
            "all_probabilities": json.loads(p.all_probabilities),
            "created_at": p.created_at.isoformat()
        })

    return jsonify({"success": True, "predictions": results}), 200

from flask import Blueprint, request, jsonify
from functools import wraps
import json
import jwt

from models import db, Prediction, User
from routes.auth_routes import JWT_SECRET, JWT_ALGO
from utils.preprocessing_utils import (
    model,
    target_encoder,
    preprocess_input,
    feature_columns,
)

# Optional: SHAP for explainability
try:
    import shap

    print("🔍 Initialising SHAP explainer...")
    shap_explainer = shap.Explainer(model)
    SHAP_AVAILABLE = True
    print("✅ SHAP explainer initialised.")
except Exception as e:
    print(f"⚠️ SHAP not available or failed to initialise: {e}")
    shap_explainer = None
    SHAP_AVAILABLE = False

prediction_bp = Blueprint("prediction_bp", __name__)


# ----------------- JWT AUTH DECORATOR ----------------- #
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", None)
        if not auth or not auth.startswith("Bearer "):
            return jsonify({"error": "Token missing"}), 401

        token = auth.split(" ")[1]

        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            user = User.query.get(data["user_id"])
            if not user:
                return jsonify({"error": "User not found"}), 401
            request.user = user
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401

        return f(*args, **kwargs)

    return decorated


# ----------------- PREDICT ENDPOINT ----------------- #
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

        # ---- Confidence + full probabilities ----
        confidence = None
        all_probabilities = {}

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]  # array of probabilities
            # Model classes are encoded labels; decode them:
            class_labels = target_encoder.inverse_transform(model.classes_)

            # Predicted class probability
            class_index = list(model.classes_).index(y_pred[0])
            confidence = float(proba[class_index])

            # Full probability distribution
            all_probabilities = {
                class_labels[i]: float(prob) for i, prob in enumerate(proba)
            }

        # ---- SHAP explainability (per-patient feature impact) ----
        shap_dict = None
        if SHAP_AVAILABLE and shap_explainer is not None:
            try:
                shap_values = shap_explainer(X)
                # shap_values can be Explanation or ndarray
                values = getattr(shap_values, "values", shap_values)
                row_vals = values[0]

                shap_dict = {
                    feature_columns[i]: float(row_vals[i])
                    for i in range(min(len(feature_columns), len(row_vals)))
                }
            except Exception as e:
                print(f"⚠️ SHAP computation failed: {e}")
                shap_dict = None

        # ---- Save prediction in DB ----
        pred = Prediction(
            user_id=request.user.id,
            input_data=json.dumps(data),
            predicted_subtype=subtype,
            confidence=confidence,
        )
        db.session.add(pred)
        db.session.commit()

        # ---- Response ----
        return jsonify(
            {
                "success": True,
                "predicted_subtype": subtype,
                "confidence": confidence,
                "all_probabilities": all_probabilities,
                "prediction_id": pred.id,
                "created_at": pred.created_at.isoformat(),
                "explainability": shap_dict,
            }
        ), 200

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


# ----------------- PREDICTION HISTORY ----------------- #
# NOTE: path changed to /api/predictions to match your Dashboard.jsx
@prediction_bp.route("/api/predictions", methods=["GET"])
@token_required
def get_history():
    preds = (
        Prediction.query.filter_by(user_id=request.user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    result = []
    for p in preds:
        result.append(
            {
                "id": p.id,
                "input_data": json.loads(p.input_data),
                "predicted_subtype": p.predicted_subtype,
                "confidence": p.confidence,
                "created_at": p.created_at.isoformat(),
            }
        )

    return jsonify({"success": True, "predictions": result}), 200

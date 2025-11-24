from flask import Blueprint, request, jsonify
from models import db, User, Prediction
from utils.auth_middleware import token_required


admin_bp = Blueprint("admin", __name__)


# ---------------------------------------------------
# ADMIN: Get all users
# ---------------------------------------------------
@admin_bp.route("/api/admin/users", methods=["GET"])
@token_required
def get_all_users():
    if request.user.role != "admin":
        return jsonify({"error": "Admin access only"}), 403

    users = User.query.all()

    results = [
        {
            "id": u.id,
            "full_name": u.full_name,
            "email": u.email,
            "username": u.username,
            "role": u.role,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]

    return jsonify({"success": True, "users": results}), 200


# ---------------------------------------------------
# ADMIN: Update user role
# ---------------------------------------------------
@admin_bp.route("/api/admin/user/<int:user_id>/role", methods=["PUT"])
@token_required
def update_user_role(user_id):
    if request.user.role != "admin":
        return jsonify({"error": "Admin access only"}), 403

    data = request.get_json()
    new_role = data.get("role")

    if new_role not in ["clinician", "researcher", "admin"]:
        return jsonify({"error": "Invalid role"}), 400

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    user.role = new_role
    db.session.commit()

    return jsonify({"success": True, "message": "Role updated"}), 200


# ---------------------------------------------------
# ADMIN: Delete a user
# ---------------------------------------------------
@admin_bp.route("/api/admin/user/<int:user_id>", methods=["DELETE"])
@token_required
def delete_user(user_id):
    if request.user.role != "admin":
        return jsonify({"error": "Admin access only"}), 403

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    db.session.delete(user)
    db.session.commit()

    return jsonify({"success": True, "message": "User deleted"}), 200


# ---------------------------------------------------
# ADMIN: Prediction analytics
# ---------------------------------------------------
@admin_bp.route("/api/admin/analytics", methods=["GET"])
@token_required
def analytics():
    if request.user.role != "admin":
        return jsonify({"error": "Admin access only"}), 403

    total_predictions = Prediction.query.count()

    subtype_counts = {
        "AIDP": Prediction.query.filter_by(predicted_subtype="AIDP").count(),
        "AMAN": Prediction.query.filter_by(predicted_subtype="AMAN").count(),
        "AMSAN": Prediction.query.filter_by(predicted_subtype="AMSAN").count(),
    }

    # predictions per day (last 7 days)
    from datetime import datetime, timedelta
    today = datetime.utcnow()
    last_week = today - timedelta(days=7)

    recent = (
        db.session.query(Prediction)
        .filter(Prediction.created_at >= last_week)
        .all()
    )

    daily_counts = {}
    for pred in recent:
        day = pred.created_at.strftime("%Y-%m-%d")
        daily_counts[day] = daily_counts.get(day, 0) + 1

    return jsonify({
        "success": True,
        "total_predictions": total_predictions,
        "subtype_counts": subtype_counts,
        "daily_counts": daily_counts,
    }), 200


# ---------------------------------------------------
# ADMIN: System logs placeholder (we add real logs later)
# ---------------------------------------------------
@admin_bp.route("/api/admin/logs", methods=["GET"])
@token_required
def logs():
    if request.user.role != "admin":
        return jsonify({"error": "Admin access only"}), 403

    # future: read from a log file
    dummy_logs = [
        {"level": "INFO", "message": "System is running", "time": "2025-11-20 02:00"},
        {"level": "ERROR", "message": "Model prediction failed", "time": "2025-11-19 18:21"},
        {"level": "WARNING", "message": "High traffic detected", "time": "2025-11-19 17:54"},
    ]

    return jsonify({"success": True, "logs": dummy_logs}), 200

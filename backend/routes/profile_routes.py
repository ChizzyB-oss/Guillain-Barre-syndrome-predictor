from flask import Blueprint, request, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from models import db, User
from .auth_routes import token_required

profile_bp = Blueprint("profile_bp", __name__)

# -------------------------
# UPDATE PROFILE INFO
# -------------------------
@profile_bp.route("/api/profile/update", methods=["PUT"])
@token_required
def update_profile():
    data = request.get_json() or {}

    full_name = data.get("full_name")
    email = data.get("email")

    if not full_name or not email:
        return jsonify({"error": "Full name and email are required"}), 400

    user = request.user
    user.full_name = full_name
    user.email = email

    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Profile updated successfully",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "role": user.role
        }
    }), 200


# -------------------------
# UPDATE PASSWORD
# -------------------------
@profile_bp.route("/api/profile/password", methods=["PUT"])
@token_required
def update_password():
    data = request.get_json() or {}

    current_password = data.get("current_password")
    new_password = data.get("new_password")

    if not current_password or not new_password:
        return jsonify({"error": "Current and new password are required"}), 400

    user = request.user

    # Validate current password
    if not check_password_hash(user.password_hash, current_password):
        return jsonify({"error": "Incorrect current password"}), 401

    # Update password
    user.password_hash = generate_password_hash(new_password)
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Password updated successfully"
    }), 200

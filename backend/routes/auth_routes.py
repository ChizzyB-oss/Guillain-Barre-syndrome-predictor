from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash
from database import db
from utils.auth_utils import generate_token

auth_routes = Blueprint('auth_routes', __name__)

@auth_routes.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    required_fields = ['username', 'email', 'password', 'full_name', 'role']
    if not all(field in data for field in required_fields):
        return jsonify({"success": False, "error": "Missing fields"}), 400

    # Check if user already exists
    existing_user = db.get_user_by_username(data['username'])
    if existing_user:
        return jsonify({"success": False, "error": "Username already exists"}), 409

    # Hash password
    hashed_password = generate_password_hash(data['password'])

    # Create user object
    user = {
        "username": data["username"],
        "email": data["email"],
        "password_hash": hashed_password,
        "full_name": data["full_name"],
        "role": data["role"]
    }

    # Save to DB
    success = db.add_user(user)

    if not success:
        return jsonify({"success": False, "error": "Database error"}), 500

    # Generate session token
    token = generate_token({"username": data["username"], "role": data["role"]})

    return jsonify({
        "success": True,
        "message": "Registration successful",
        "session_token": token,
        "user": {
            "username": data["username"],
            "full_name": data["full_name"],
            "email": data["email"],
            "role": data["role"]
        }
    }), 201

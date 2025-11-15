from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash
from models import db
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


@auth_routes.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"success": False, "error": "Missing username or password"}), 400

    username = data['username']
    password = data['password']

    # Fetch user from DB
    user_row = db.get_user_by_username(username)

    if not user_row:
        return jsonify({"success": False, "error": "Invalid username or password"}), 401

    # Convert DB row → dictionary
    user = {
        "id": user_row[0],
        "username": user_row[1],
        "email": user_row[2],
        "password_hash": user_row[3],
        "full_name": user_row[4],
        "role": user_row[5],
        "created_at": user_row[6]
    }

    # Check password
    if not check_password_hash(user["password_hash"], password):
        return jsonify({"success": False, "error": "Invalid username or password"}), 401

    # Create JWT session token
    token = generate_token({
        "id": user["id"],
        "username": user["username"],
        "role": user["role"]
    })

    return jsonify({
        "success": True,
        "session_token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"]
        }
    }), 200
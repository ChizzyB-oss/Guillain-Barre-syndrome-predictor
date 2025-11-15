from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta

from models import db, User

auth_bp = Blueprint("auth_bp", __name__)

JWT_SECRET = "super-secret-key-change-this"
JWT_ALGO = "HS256"
TOKEN_EXP_DAYS = 7


def generate_token(user):
    payload = {
        "user_id": user.id,
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXP_DAYS),
        "iat": datetime.utcnow(),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    return token if isinstance(token, str) else token.decode("utf-8")


@auth_bp.route("/api/register", methods=["POST"])
def register():
    data = request.get_json() or {}

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "username, email, and password are required"}), 400

    if User.query.filter((User.username == username) |
                         (User.email == email)).first():
        return jsonify({"error": "User already exists"}), 400

    user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password)
    )

    db.session.add(user)
    db.session.commit()

    token = generate_token(user)

    return jsonify({
        "success": True,
        "message": "User registered",
        "token": token,
        "user": {"id": user.id, "username": user.username, "email": user.email}
    })


@auth_bp.route("/api/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 400

    token = generate_token(user)

    return jsonify({
        "success": True,
        "message": "Login successful",
        "token": token,
        "user": {"id": user.id, "username": user.username, "email": user.email}
    })

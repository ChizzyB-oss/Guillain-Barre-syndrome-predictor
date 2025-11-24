import jwt
from functools import wraps
from flask import request, jsonify
from config import Config
from models import User

SECRET_KEY = Config.SECRET_KEY  # ✔ correct way



def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Get token from Authorization header
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            return jsonify({"success": False, "error": "Token missing"}), 401

        try:
            # Decode token
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

            # Fetch user
            user = User.query.get(payload["user_id"])
            if not user:
                return jsonify({"success": False, "error": "Invalid user"}), 401

            # Attach user to request
            request.user = user

        except jwt.ExpiredSignatureError:
            return jsonify({"success": False, "error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"success": False, "error": "Invalid token"}), 401

        return f(*args, **kwargs)

    return decorated

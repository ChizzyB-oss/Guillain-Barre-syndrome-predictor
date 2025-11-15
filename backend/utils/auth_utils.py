import jwt
import datetime

SECRET = "SUPER_SECRET_KEY_CHANGE_THIS"

def generate_token(user):
    payload = {
        "user": user,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=6)
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

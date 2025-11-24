from flask import Flask
from flask_cors import CORS
from models import db

from routes.auth_routes import auth_bp
from routes.prediction_routes import prediction_bp
from routes.profile_routes import profile_bp


app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///gbs_system.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "super-secret-key-change-this"

db.init_app(app)

with app.app_context():
    db.create_all()
    print("✅ Database ready")


# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(profile_bp)


@app.route("/api/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True)

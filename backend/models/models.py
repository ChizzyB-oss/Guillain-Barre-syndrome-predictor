from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship("Prediction", backref="user", lazy=True)

class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    input_data = db.Column(db.Text, nullable=False)      # JSON as string
    predicted_subtype = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=True)      # optional

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

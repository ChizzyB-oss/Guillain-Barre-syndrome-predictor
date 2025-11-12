from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Boolean, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    """User model for authentication and authorization"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(String(20), default='clinician')  # clinician, researcher, admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = relationship('Prediction', backref='user', lazy='dynamic')
    sessions = relationship('UserSession', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class UserSession(db.Model):
    """User session model for authentication"""
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    session_token = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def is_valid(self):
        """Check if session is still valid"""
        return datetime.utcnow() < self.expires_at

class Prediction(db.Model):
    """Prediction model to store all GBS predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    prediction_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID
    input_data = Column(Text, nullable=False)  # JSON string of input features
    predicted_subtype = Column(String(10), nullable=False)  # AIDP, AMAN, AMSAN, MF
    confidence = Column(Float, nullable=False)
    all_probabilities = Column(Text, nullable=False)  # JSON string
    interpretation = Column(Text)  # Clinical interpretation
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        """Convert prediction to dictionary"""
        import json
        return {
            'id': self.id,
            'prediction_id': self.prediction_id,
            'user_id': self.user_id,
            'predicted_subtype': self.predicted_subtype,
            'confidence': self.confidence,
            'all_probabilities': json.loads(self.all_probabilities),
            'interpretation': json.loads(self.interpretation) if self.interpretation else None,
            'input_data': json.loads(self.input_data),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class SystemLog(db.Model):
    """System log for auditing and monitoring"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    level = Column(String(20), nullable=False, index=True)  # INFO, WARNING, ERROR
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)
    
    with app.app_context():
        # Drop all existing tables and recreate (for development)
        db.drop_all()
        db.create_all()
        print("✅ Database tables created successfully")
        
        # Create default admin user if doesn't exist
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_user = User(
                username='admin',
                email='admin@gbssystem.com',
                full_name='System Administrator',
                role='admin'
            )
            admin_user.set_password('admin123')
            db.session.add(admin_user)
            
            # Create test clinician user
            clinician_user = User(
                username='doctor1',
                email='doctor1@hospital.com',
                full_name='Dr. Sarah Johnson',
                role='clinician'
            )
            clinician_user.set_password('password123')
            db.session.add(clinician_user)
            
            db.session.commit()
            print("✅ Default users created (admin/admin123, doctor1/password123)")
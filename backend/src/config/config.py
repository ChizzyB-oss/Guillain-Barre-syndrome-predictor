import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Config:
    """Base configuration with default settings"""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production-2024'
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key-change-in-production'
    
    # Database
    DATABASE_PATH = os.path.join(os.path.dirname(__file__), '../../../data/gbs_system.db')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # CORS
    ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ]
    
    # Rate Limiting
    RATELIMIT_DEFAULT = "200 per day"
    RATELIMIT_STORAGE_URI = "memory://"
    
    # Session
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    
    # File Upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # ML Model Paths
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../../models/best_gbs_model.pkl')
    SCALER_PATH = os.path.join(os.path.dirname(__file__), '../../../models/scaler.pkl')
    LABEL_ENCODERS_PATH = os.path.join(os.path.dirname(__file__), '../../../models/label_encoders.pkl')
    TARGET_ENCODER_PATH = os.path.join(os.path.dirname(__file__), '../../../models/target_encoder.pkl')
    PREPROCESSING_INFO_PATH = os.path.join(os.path.dirname(__file__), '../../../models/preprocessing_info.json')
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/gbs_system.log'

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    DEBUG = False
    # Use environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
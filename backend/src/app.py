from flask import Flask, jsonify
from flask_cors import CORS
from config.config import config
from models.database import init_db
from utils.logging import setup_logging
from routes.auth import auth_bp
from routes.predictions import predictions_bp
import os

def create_app(config_name=None):
    """Application factory pattern"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    CORS(app, origins=app.config['ALLOWED_ORIGINS'])
    
    # Setup logging
    setup_logging(app)
    
    # Initialize database
    init_db(app)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(predictions_bp, url_prefix='/api')
    
    # Root endpoint
    @app.route('/')
    def root():
        return jsonify({
            'message': 'GBS AI Diagnostic System API',
            'version': '1.0.0',
            'status': 'operational',
            'documentation': '/api/docs'  # You can add Swagger later
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'Server Error: {error}')
        return jsonify({'error': 'An internal server error occurred'}), 500
    
    @app.errorhandler(413)
    def too_large(error):
        return jsonify({'error': 'File too large'}), 413
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    print("=" * 60)
    print("🚀 GBS AI Diagnostic System - Professional Backend")
    print("=" * 60)
    print(f"📝 Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"🔐 Security: JWT Authentication Enabled")
    print(f"📊 Database: SQLAlchemy ORM")
    print(f"🤖 ML Service: {'Loaded' if app.ml_service.is_loaded else 'Unavailable'}")
    print(f"🌐 CORS: Enabled for {app.config['ALLOWED_ORIGINS']}")
    print(f"📈 Logging: Comprehensive logging enabled")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )
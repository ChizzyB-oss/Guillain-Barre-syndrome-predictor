import logging
from logging.handlers import RotatingFileHandler
import os
from flask import has_request_context, request
from models.database import SystemLog, db

class ContextFilter(logging.Filter):
    """Add contextual information to log records"""
    
    def filter(self, record):
        if has_request_context():
            record.ip = request.remote_addr
            record.endpoint = request.endpoint or 'unknown'
            record.method = request.method
        else:
            record.ip = 'internal'
            record.endpoint = 'system'
            record.method = 'N/A'
        return True

def setup_logging(app):
    """Setup comprehensive logging system"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/gbs_system.log',
        maxBytes=1024 * 1024 * 10,  # 10MB
        backupCount=10
    )
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s [%(ip)s] %(endpoint)s %(method)s: %(message)s'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Add filter and handler
    file_handler.addFilter(ContextFilter())
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    
    # Disable default Flask logger
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

def log_to_database(level, module, message, user_id=None):
    """Log important events to database for auditing"""
    try:
        log_entry = SystemLog(
            level=level,
            module=module,
            message=message,
            user_id=user_id,
            ip_address=request.remote_addr if has_request_context() else 'system',
            user_agent=request.headers.get('User-Agent') if has_request_context() else None
        )
        db.session.add(log_entry)
        db.session.commit()
    except Exception as e:
        # Fallback to file logging if database logging fails
        logging.error(f"Database logging failed: {str(e)}")

# Create logger instance
logger = logging.getLogger(__name__)
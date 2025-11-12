import jwt
from datetime import datetime, timedelta
from flask import current_app, request
from functools import wraps
from .logging import logger
import secrets

class SecurityUtils:
    """Security utilities for the application"""
    
    @staticmethod
    def generate_jwt_token(user_id, expires_in=3600):
        """Generate JWT token for user"""
        try:
            payload = {
                'user_id': user_id,
                'exp': datetime.utcnow() + timedelta(seconds=expires_in),
                'iat': datetime.utcnow(),
                'type': 'access'
            }
            return jwt.encode(payload, current_app.config['JWT_SECRET_KEY'], algorithm='HS256')
        except Exception as e:
            logger.error(f"JWT token generation error: {str(e)}")
            return None
    
    @staticmethod
    def verify_jwt_token(token):
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {str(e)}")
            return None
    
    @staticmethod
    def generate_session_token():
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_input_data(data, required_fields=None):
        """Validate input data for prediction"""
        if required_fields is None:
            required_fields = ['age', 'gender', 'csf_protein']
        
        errors = []
        
        # Check required fields
        for field in required_fields:
            if field not in data or data[field] in [None, '']:
                errors.append(f"Missing required field: {field}")
        
        # Validate data types and ranges
        if 'age' in data and data['age']:
            try:
                age = int(data['age'])
                if not (1 <= age <= 120):
                    errors.append("Age must be between 1 and 120")
            except ValueError:
                errors.append("Age must be a valid number")
        
        if 'csf_protein' in data and data['csf_protein']:
            try:
                csf = float(data['csf_protein'])
                if csf < 0 or csf > 1000:
                    errors.append("CSF protein must be between 0 and 1000 mg/dL")
            except ValueError:
                errors.append("CSF protein must be a valid number")
        
        return errors

def token_required(f):
    """Decorator to require JWT token for protected routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return {'error': 'Authentication token is missing'}, 401
        
        # Verify token
        payload = SecurityUtils.verify_jwt_token(token)
        if not payload:
            return {'error': 'Invalid or expired token'}, 401
        
        # Add user info to request context
        request.user_id = payload['user_id']
        return f(*args, **kwargs)
    
    return decorated

def admin_required(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    @token_required
    def decorated(*args, **kwargs):
        from models.database import User
        user = User.query.get(request.user_id)
        
        if not user or user.role != 'admin':
            return {'error': 'Admin privileges required'}, 403
        
        return f(*args, **kwargs)
    
    return decorated
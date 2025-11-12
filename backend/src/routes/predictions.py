from flask import Blueprint, request, jsonify
from models.database import db, Prediction
from utils.security import token_required
from utils.logging import logger, log_to_database
from services.ml_services import ml_service
import json
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from utils.security import token_required

predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/predict', methods=['POST'])
@token_required
def predict():
    """Make GBS subtype prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
         # Use current_app to access ml_service
        ml_service = current_app.ml_service
        
        # Validate input data
        validation_errors = ml_service.validate_input_data(data)
        if validation_errors:
            return jsonify({
                'error': 'Input validation failed',
                'details': validation_errors
            }), 400
        
        # Make prediction
        prediction_result = ml_service.predict(data)
        
        if not prediction_result['success']:
            log_to_database('ERROR', 'prediction', 
                          f'Prediction failed: {prediction_result.get("error")}', 
                          request.user_id)
            return jsonify(prediction_result), 500
        
        # Save prediction to database
        prediction_id = str(uuid.uuid4())
        new_prediction = Prediction(
            user_id=request.user_id,
            prediction_id=prediction_id,
            input_data=json.dumps(data),
            predicted_subtype=prediction_result['predicted_subtype'],
            confidence=prediction_result['confidence'],
            all_probabilities=json.dumps(prediction_result['all_probabilities']),
            interpretation=json.dumps(prediction_result['interpretation'])
        )
        
        db.session.add(new_prediction)
        db.session.commit()
        
        # Add prediction ID to result
        prediction_result['prediction_id'] = prediction_id
        
        log_to_database('INFO', 'prediction', 
                       f'Prediction made: {prediction_result["predicted_subtype"]} '
                       f'(confidence: {prediction_result["confidence"]:.3f})', 
                       request.user_id)
        
        return jsonify(prediction_result)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': 'Prediction processing failed',
            'code': 'SERVER_ERROR'
        }), 500

@predictions_bp.route('/predictions/history', methods=['GET'])
@token_required
def get_prediction_history():
    """Get user's prediction history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        page = request.args.get('page', 1, type=int)
        
        # Validate parameters
        if limit > 100 or limit < 1:
            limit = 50
        if page < 1:
            page = 1
        
        # Query predictions with pagination
        predictions_query = Prediction.query.filter_by(
            user_id=request.user_id
        ).order_by(Prediction.created_at.desc())
        
        total = predictions_query.count()
        predictions = predictions_query.offset((page - 1) * limit).limit(limit).all()
        
        return jsonify({
            'success': True,
            'predictions': [pred.to_dict() for pred in predictions],
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total,
                'pages': (total + limit - 1) // limit
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction history error: {str(e)}")
        return jsonify({'error': 'Failed to fetch prediction history'}), 500

@predictions_bp.route('/predictions/<prediction_id>', methods=['GET'])
@token_required
def get_prediction(prediction_id):
    """Get specific prediction by ID"""
    try:
        prediction = Prediction.query.filter_by(
            prediction_id=prediction_id,
            user_id=request.user_id
        ).first()
        
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        
        return jsonify({
            'success': True,
            'prediction': prediction.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Get prediction error: {str(e)}")
        return jsonify({'error': 'Failed to fetch prediction'}), 500

@predictions_bp.route('/system/health', methods=['GET'])
def system_health():
    """System health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'services': {
                'database': 'operational',
                'ml_model': 'loaded' if ml_service.is_loaded else 'unavailable',
                'authentication': 'operational'
            }
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500
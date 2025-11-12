from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import json
import traceback
import os
from database import db

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

app.secret_key = 'gbs-system-secret-key-2024'

class GBSPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.target_encoder = None
        self.feature_names = []
        self.target_names = ['AIDP', 'AMAN', 'AMSAN', 'MF']
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            # Check if model files exist
            if not os.path.exists('models/best_gbs_model.pkl'):
                print("❌ Model file not found. Please train the model first.")
                print("   Run: python src/train_models.py")
                return
            
            self.model = joblib.load('models/best_gbs_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            self.target_encoder = joblib.load('models/target_encoder.pkl')
            
            # Load preprocessing info
            with open('models/preprocessing_info.json', 'r') as f:
                preprocessing_info = json.load(f)
            
            self.feature_names = preprocessing_info['feature_columns']
            self.target_names = preprocessing_info['target_names']
            
            self.is_loaded = True
            print("✅ Model and preprocessors loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Please make sure you've run: python src/train_models.py")
    
    def preprocess_input(self, input_data):
        """Preprocess the input data for prediction"""
        try:
            # Create DataFrame with default values for all features
            default_values = {
                'age': 45,
                'gender': 'male',
                'csf_protein': 75.0,
                'muscle_weakness': 0,
                'paralysis': 0,
                'sensory_loss': 0,
                'reflex_loss': 0,
                'respiratory_involvement': 0,
                'cranial_nerve_involvement': 0,
                'motor_velocity': 40.0,
                'sensory_velocity': 40.0,
                'amplitude': 5.0,
                'f_wave_latency': 40.0,
                'conduction_block': 0,
                'previous_infection': 'unknown',
                'onset_speed': 'acute'
            }
            
            # Update with provided values
            for key, value in input_data.items():
                if key in default_values:
                    default_values[key] = value
            
            df = pd.DataFrame([default_values])
            
            # Ensure we only have the expected features
            df = df[self.feature_names]
            
            # Encode categorical variables using saved encoders
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    # Handle unseen categories
                    unique_classes = set(encoder.classes_)
                    df[col] = df[col].apply(lambda x: x if x in unique_classes else encoder.classes_[0])
                    df[col] = encoder.transform(df[col])
            
            # Scale numerical features
            numerical_columns = [col for col in self.feature_names 
                               if col not in self.label_encoders.keys()]
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])
            
            return df
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            return None
    
    def predict(self, input_data):
        """Make prediction on input data"""
        if not self.is_loaded:
            return {
                'success': False, 
                'error': 'Model not loaded. Please train the model first.',
                'instructions': 'Run: python src/train_models.py'
            }
        
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            if processed_data is None:
                return {'success': False, 'error': 'Data preprocessing failed'}
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0]
            
            # Get confidence scores for each class
            confidence_scores = {
                self.target_names[i]: float(probability[i]) 
                for i in range(len(self.target_names))
            }
            
            # Get top prediction
            predicted_class = self.target_names[prediction]
            confidence = confidence_scores[predicted_class]
            
            # Get top 3 predictions
            top_predictions = sorted(
                [(subtype, prob) for subtype, prob in confidence_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            return {
                'success': True,
                'predicted_subtype': predicted_class,
                'confidence': confidence,
                'all_probabilities': confidence_scores,
                'top_predictions': [
                    {'subtype': subtype, 'confidence': conf} 
                    for subtype, conf in top_predictions
                ],
                'interpretation': self.get_interpretation(predicted_class, confidence)
            }
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_interpretation(self, subtype, confidence):
        """Provide clinical interpretation of the prediction"""
        interpretations = {
            'AIDP': 'Acute Inflammatory Demyelinating Polyneuropathy - most common form, typically responsive to IVIG therapy.',
            'AMAN': 'Acute Motor Axonal Neuropathy - often associated with campylobacter infection, pure motor involvement.',
            'AMSAN': 'Acute Motor-Sensory Axonal Neuropathy - severe form with both motor and sensory involvement.',
            'MF': 'Miller Fisher Syndrome - characterized by ophthalmoplegia, ataxia, and areflexia.'
        }
        
        confidence_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"
        
        return {
            'clinical_description': interpretations.get(subtype, 'Unknown subtype'),
            'confidence_level': confidence_level,
            'recommendation': 'Consult neurologist for comprehensive evaluation and treatment planning.'
        }

# Initialize predictor and database
predictor = GBSPredictor()

# Authentication middleware
def require_auth(f):
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        
        session_token = auth_header[7:]  # Remove 'Bearer ' prefix
        user = db.validate_session(session_token)
        
        if not user:
            return jsonify({'error': 'Invalid or expired session'}), 401
        
        # Add user to request context
        request.user = user
        return f(*args, **kwargs)
    
    decorated_function.__name__ = f.__name__
    return decorated_function

# Authentication Routes
@app.route('/api/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        print(f"📝 Registration attempt for: {data.get('username')}")
        
        if not all(k in data for k in ['username', 'email', 'password', 'full_name']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        user_id = db.create_user(
            username=data['username'],
            email=data['email'],
            password=data['password'],
            full_name=data['full_name'],
            role=data.get('role', 'clinician')
        )
        
        if user_id:
            # Auto-login after registration
            session_token = db.create_session(user_id)
            return jsonify({
                'success': True,
                'message': 'User registered successfully',
                'session_token': session_token,
                'user': {
                    'id': user_id,
                    'username': data['username'],
                    'email': data['email'],
                    'full_name': data['full_name'],
                    'role': data.get('role', 'clinician')
                }
            })
        else:
            return jsonify({'error': 'Username or email already exists'}), 400
            
    except Exception as e:
        print(f"❌ Registration error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        
        if not data or not all(k in data for k in ['username', 'password']):
            return jsonify({'error': 'Username and password required'}), 400
        
        print(f"🔐 Login attempt for: {data['username']}")
        
        user = db.authenticate_user(data['username'], data['password'])
        
        if user:
            session_token = db.create_session(user['id'])
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'session_token': session_token,
                'user': user
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/logout', methods=['POST'])
@require_auth
def logout():
    # In a full implementation, we'd invalidate the session token
    return jsonify({'success': True, 'message': 'Logout successful'})

@app.route('/api/profile', methods=['GET'])
@require_auth
def get_profile():
    return jsonify({'success': True, 'user': request.user})

# Protected Prediction Routes
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
@require_auth
def predict():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if not predictor.is_loaded:
            return jsonify({
                'error': 'Prediction model not available',
                'instructions': 'Please contact system administrator'
            }), 503
        
        # Make prediction
        result = predictor.predict(data)
        
        if result['success']:
            # Save prediction to database
            prediction_id = db.save_prediction(
                user_id=request.user['id'],
                input_data=data,
                prediction_result=result
            )
            
            if prediction_id:
                result['prediction_id'] = prediction_id
            
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predictions/history', methods=['GET'])
@require_auth
def get_prediction_history():
    try:
        limit = request.args.get('limit', 50, type=int)
        predictions = db.get_user_predictions(request.user['id'], limit)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch history: {str(e)}'}), 500

# Public routes (no authentication required)
@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy' if predictor.is_loaded else 'model_missing',
        'model_loaded': predictor.is_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model/info')
def model_info():
    if not predictor.is_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'feature_names': predictor.feature_names,
        'target_names': predictor.target_names,
        'model_type': str(type(predictor.model).__name__)
    })

@app.route('/')
def home():
    return jsonify({
        'message': 'GBS Subtype Prediction API',
        'status': 'active' if predictor.is_loaded else 'model_not_loaded',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 Starting GBS Prediction API with Authentication...")
    print("=" * 60)
    
    if not predictor.is_loaded:
        print("❌ Model not loaded. Please train the model first.")
        print("💡 Run: python src/train_models.py")
    
    print("✅ API starting on http://localhost:5000")
    print("✅ CORS enabled for React development server")
    app.run(debug=True, host='0.0.0.0', port=5000)
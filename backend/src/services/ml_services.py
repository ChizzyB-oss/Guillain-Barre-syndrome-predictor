import joblib
import pandas as pd
import numpy as np
import json
import os
from utils.logging import logger

class MLService:
    """Machine Learning service for GBS predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.target_encoder = None
        self.feature_names = []
        self.target_names = ['AIDP', 'AMAN', 'AMSAN', 'MF']
        self.is_loaded = False
        self.model_paths = {}
    
    def init_app(self, app):
        """Initialize with Flask app context"""
        self.model_paths = {
            'model': app.config.get('MODEL_PATH', '../../../models/best_gbs_model.pkl'),
            'scaler': app.config.get('SCALER_PATH', '../../../models/scaler.pkl'),
            'label_encoders': app.config.get('LABEL_ENCODERS_PATH', '../../../models/label_encoders.pkl'),
            'target_encoder': app.config.get('TARGET_ENCODER_PATH', '../../../models/target_encoder.pkl'),
            'preprocessing_info': app.config.get('PREPROCESSING_INFO_PATH', '../../../models/preprocessing_info.json')
        }
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessors"""
        try:
            logger.info("Loading ML model and preprocessors...")
            
            # Convert relative paths to absolute paths
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            model_path = os.path.join(base_dir, self.model_paths['model'])
            scaler_path = os.path.join(base_dir, self.model_paths['scaler'])
            label_encoders_path = os.path.join(base_dir, self.model_paths['label_encoders'])
            target_encoder_path = os.path.join(base_dir, self.model_paths['target_encoder'])
            preprocessing_info_path = os.path.join(base_dir, self.model_paths['preprocessing_info'])
            
            # Check if model files exist
            missing_files = []
            for path_name, path in [
                ('model', model_path),
                ('scaler', scaler_path),
                ('label_encoders', label_encoders_path),
                ('target_encoder', target_encoder_path),
                ('preprocessing_info', preprocessing_info_path)
            ]:
                if not os.path.exists(path):
                    missing_files.append(path_name)
                    logger.warning(f"Missing model file: {path}")
            
            if missing_files:
                logger.error(f"Missing model files: {missing_files}")
                logger.info("Please run: python src/train_models.py from the project root")
                return
            
            # Load the models
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoders = joblib.load(label_encoders_path)
            self.target_encoder = joblib.load(target_encoder_path)
            
            with open(preprocessing_info_path, 'r') as f:
                preprocessing_info = json.load(f)
            
            self.feature_names = preprocessing_info['feature_columns']
            self.target_names = preprocessing_info['target_names']
            
            self.is_loaded = True
            logger.info("✅ ML model and preprocessors loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading ML model: {str(e)}")
            self.is_loaded = False
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        try:
            logger.debug(f"Preprocessing input data with keys: {list(input_data.keys())}")
            
            # Field mapping from frontend to backend
            field_mapping = {
                'age': 'age',
                'gender': 'gender',
                'csfProtein': 'csf_protein',
                'muscleWeakness': 'muscle_weakness',
                'paralysis': 'paralysis',
                'sensoryLoss': 'sensory_loss',
                'reflexLoss': 'reflex_loss',
                'respiratoryInvolvement': 'respiratory_involvement',
                'cranialNerveInvolvement': 'cranial_nerve_involvement',
                'motorVelocity': 'motor_velocity',
                'sensoryVelocity': 'sensory_velocity',
                'amplitude': 'amplitude',
                'fWaveLatency': 'f_wave_latency',
                'conductionBlock': 'conduction_block',
                'previousInfection': 'previous_infection',
                'onsetSpeed': 'onset_speed'
            }
            
            # Default values for all features
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
            
            # Map and update with provided values
            processed_data = default_values.copy()
            for frontend_key, backend_key in field_mapping.items():
                if frontend_key in input_data and input_data[frontend_key] not in [None, '']:
                    # Convert checkbox values to integers
                    if frontend_key in ['muscleWeakness', 'paralysis', 'sensoryLoss', 'reflexLoss', 
                                      'respiratoryInvolvement', 'cranialNerveInvolvement', 'conductionBlock']:
                        processed_data[backend_key] = 1 if input_data[frontend_key] else 0
                    else:
                        processed_data[backend_key] = input_data[frontend_key]
            
            # Create DataFrame
            df = pd.DataFrame([processed_data])
            
            # Ensure correct feature order
            df = df[self.feature_names]
            
            # Encode categorical variables
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    # Handle unseen categories gracefully
                    unique_classes = set(encoder.classes_)
                    df[col] = df[col].apply(
                        lambda x: x if str(x) in unique_classes else encoder.classes_[0]
                    )
                    df[col] = encoder.transform(df[col])
            
            # Scale numerical features
            numerical_columns = [
                col for col in self.feature_names 
                if col not in self.label_encoders.keys()
            ]
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])
            
            logger.debug("✅ Input data preprocessing completed")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error in preprocessing: {str(e)}")
            return None
    
    def predict(self, input_data):
        """Make prediction on processed data"""
        if not self.is_loaded:
            return {
                'success': False,
                'error': 'Prediction model is not available',
                'code': 'MODEL_NOT_LOADED'
            }
        
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            if processed_data is None:
                return {
                    'success': False,
                    'error': 'Data preprocessing failed',
                    'code': 'PREPROCESSING_ERROR'
                }
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probabilities = self.model.predict_proba(processed_data)[0]
            
            # Format results
            confidence_scores = {
                self.target_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            predicted_class = self.target_names[prediction]
            confidence = confidence_scores[predicted_class]
            
            # Get top predictions
            top_predictions = sorted(
                [(subtype, prob) for subtype, prob in confidence_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            result = {
                'success': True,
                'predicted_subtype': predicted_class,
                'confidence': confidence,
                'all_probabilities': confidence_scores,
                'top_predictions': [
                    {'subtype': subtype, 'confidence': conf} 
                    for subtype, conf in top_predictions
                ],
                'interpretation': self.get_interpretation(predicted_class, confidence),
                'model_version': '1.0.0',
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"✅ Prediction successful: {predicted_class} ({confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'code': 'PREDICTION_ERROR'
            }
    
    def get_interpretation(self, subtype, confidence):
        """Provide clinical interpretation of prediction"""
        interpretations = {
            'AIDP': {
                'description': 'Acute Inflammatory Demyelinating Polyneuropathy - most common form, typically responsive to IVIG therapy.',
                'treatment': 'IVIG or plasma exchange recommended',
                'prognosis': 'Generally good with early treatment',
                'key_features': ['Demyelinating pattern on NCS', 'Elevated CSF protein', 'Progressive weakness']
            },
            'AMAN': {
                'description': 'Acute Motor Axonal Neuropathy - often associated with campylobacter infection, pure motor involvement.',
                'treatment': 'IVIG therapy, supportive care',
                'prognosis': 'Variable, may have slower recovery',
                'key_features': ['Axonal pattern on NCS', 'Pure motor symptoms', 'Anti-GM1 antibodies possible']
            },
            'AMSAN': {
                'description': 'Acute Motor-Sensory Axonal Neuropathy - severe form with both motor and sensory involvement.',
                'treatment': 'Aggressive immunomodulatory therapy',
                'prognosis': 'Often poorer, may have residual deficits',
                'key_features': ['Severe axonal involvement', 'Motor and sensory symptoms', 'Rapid progression']
            },
            'MF': {
                'description': 'Miller Fisher Syndrome - characterized by ophthalmoplegia, ataxia, and areflexia.',
                'treatment': 'IVIG or plasma exchange',
                'prognosis': 'Generally good recovery',
                'key_features': ['Ophthalmoplegia', 'Ataxia', 'Areflexia', 'Anti-GQ1b antibodies']
            }
        }
        
        confidence_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"
        
        subtype_info = interpretations.get(subtype, {
            'description': 'Subtype information not available.',
            'treatment': 'Consult neurologist for treatment planning',
            'prognosis': 'Variable',
            'key_features': []
        })
        
        return {
            'clinical_description': subtype_info['description'],
            'recommended_treatment': subtype_info['treatment'],
            'prognosis': subtype_info['prognosis'],
            'key_clinical_features': subtype_info['key_features'],
            'confidence_level': confidence_level,
            'recommendation': 'Consult neurologist for comprehensive evaluation and treatment planning.'
        }

    def validate_input_data(self, data, required_fields=None):
        """Validate input data for prediction"""
        if required_fields is None:
            required_fields = ['age', 'gender', 'csfProtein']
        
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
        
        if 'csfProtein' in data and data['csfProtein']:
            try:
                csf = float(data['csfProtein'])
                if csf < 0 or csf > 1000:
                    errors.append("CSF protein must be between 0 and 1000 mg/dL")
            except ValueError:
                errors.append("CSF protein must be a valid number")
        
        return errors

# Global ML service instance
ml_service = MLService()
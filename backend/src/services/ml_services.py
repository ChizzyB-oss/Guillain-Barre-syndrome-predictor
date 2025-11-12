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
            'model': app.config.get('MODEL_PATH'),
            'scaler': app.config.get('SCALER_PATH'),
            'label_encoders': app.config.get('LABEL_ENCODERS_PATH'),
            'target_encoder': app.config.get('TARGET_ENCODER_PATH'),
            'preprocessing_info': app.config.get('PREPROCESSING_INFO_PATH')
        }
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessors"""
        try:
            logger.info("Loading ML model and preprocessors...")
            
            # Check if model files exist
            missing_files = []
            for path_name, path in self.model_paths.items():
                if not os.path.exists(path):
                    missing_files.append(f"{path_name}: {path}")
                    logger.warning(f"Missing model file: {path}")
            
            if missing_files:
                logger.error(f"Missing model files:\n" + "\n".join(missing_files))
                logger.info("Please run: python src/train_models.py from the project root")
                return
            
            # Load the models
            self.model = joblib.load(self.model_paths['model'])
            self.scaler = joblib.load(self.model_paths['scaler'])
            self.label_encoders = joblib.load(self.model_paths['label_encoders'])
            self.target_encoder = joblib.load(self.model_paths['target_encoder'])
            
            with open(self.model_paths['preprocessing_info'], 'r') as f:
                preprocessing_info = json.load(f)
            
            self.feature_names = preprocessing_info['feature_columns']
            self.target_names = preprocessing_info['target_names']
            
            self.is_loaded = True
            logger.info("✅ ML model and preprocessors loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading ML model: {str(e)}")
            self.is_loaded = False

    # ... rest of the MLService class remains the same ...
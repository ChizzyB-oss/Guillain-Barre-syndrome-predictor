import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

class GBSPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        
    def load_data(self):
        """Load the generated dataset"""
        self.data = pd.read_csv('data/synthetic_gbs_dataset_clinical.csv')
        print(f"Loaded dataset with {len(self.data)} samples")
        return self.data
    
    def engineer_features(self, data):
        """Create new features that might be clinically relevant"""
        df = data.copy()
        
        # Create age groups
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 18, 40, 60, 100], 
                                labels=['child', 'young_adult', 'adult', 'senior'])
        
        # Create severity score based on symptoms
        symptom_cols = ['muscle_weakness', 'paralysis', 'sensory_loss', 
                       'respiratory_involvement', 'cranial_nerve_involvement']
        df['symptom_severity'] = df[symptom_cols].sum(axis=1)
        
        # Create electrophysiological ratio
        df['velocity_ratio'] = df['motor_velocity'] / df['sensory_velocity']
        df['velocity_ratio'] = df['velocity_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # CSF protein categories
        df['csf_category'] = pd.cut(df['csf_protein'], 
                                   bins=[0, 45, 90, 200], 
                                   labels=['normal', 'elevated', 'high'])
        
        print("✓ Feature engineering completed")
        return df
    
    def encode_features(self, df):
        """Encode categorical variables"""
        categorical_columns = ['gender', 'previous_infection', 'onset_speed', 
                              'age_group', 'csf_category']
        
        # Only encode columns that exist in the dataframe
        categorical_columns = [col for col in categorical_columns if col in df.columns]
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        print("✓ Categorical features encoded")
        return df
    
    def select_features(self, X, y, k=15):
        """Select most important features using ANOVA F-test"""
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        selected_features = X.columns[selected_mask]
        
        print(f"✓ Selected {len(selected_features)} most important features:")
        for feature in selected_features:
            print(f"  - {feature}")
        
        return X_selected, selected_features
    
    def prepare_ml_data(self, test_size=0.2, random_state=42):
        """Prepare final ML-ready dataset"""
        from sklearn.model_selection import train_test_split
        
        # Load and engineer features
        data = self.load_data()
        data = self.engineer_features(data)
        
        # Define features and target
        feature_columns = [col for col in data.columns if col != 'gbs_subtype']
        X = data[feature_columns].copy()
        y = data['gbs_subtype'].copy()
        
        # Encode categorical features
        X = self.encode_features(X)
        
        # Encode target
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, 
            stratify=y_encoded
        )
        
        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        X_train[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
        
        # Feature selection
        X_train_selected, selected_features = self.select_features(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        print(f"\n✓ Final dataset prepared:")
        print(f"  - Training set: {X_train_selected.shape}")
        print(f"  - Test set: {X_test_selected.shape}")
        print(f"  - Target classes: {self.target_encoder.classes_}")
        
        # Save processed data
        self.save_processed_data(X_train_selected, X_test_selected, y_train, y_test, selected_features)
        
        return X_train_selected, X_test_selected, y_train, y_test, selected_features
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, feature_names):
        """Save the processed datasets"""
        # Save as numpy arrays for ML
        np.save('data/X_train_processed.npy', X_train)
        np.save('data/X_test_processed.npy', X_test)
        np.save('data/y_train_processed.npy', y_train)
        np.save('data/y_test_processed.npy', y_test)
        
        # Save feature names
        with open('data/feature_names.txt', 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        
        # Save preprocessor objects
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        joblib.dump(self.feature_selector, 'models/feature_selector.pkl')
        joblib.dump(self.target_encoder, 'models/target_encoder.pkl')
        
        print("✓ Processed data and preprocessors saved")

if __name__ == "__main__":
    preprocessor = GBSPreprocessor()
    X_train, X_test, y_train, y_test, features = preprocessor.prepare_ml_data()
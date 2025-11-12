import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def generate_synthetic_gbs_data(n_samples=2000):
    """
    Generate synthetic GBS dataset with clinically realistic patterns
    """
    
    data = pd.DataFrame({
        # Demographic data
        'age': np.random.randint(1, 90, n_samples),
        'gender': np.random.choice(['male', 'female'], n_samples, p=[0.55, 0.45]),
        
        # Clinical presentation
        'muscle_weakness': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'paralysis': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'sensory_loss': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'reflex_loss': np.random.choice([0, 1], n_samples, p=[0.05, 0.95]),
        'respiratory_involvement': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'cranial_nerve_involvement': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'autonomic_dysfunction': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        
        # Temporal features
        'onset_speed': np.random.choice(['acute', 'subacute', 'chronic'], n_samples, p=[0.6, 0.3, 0.1]),
        'progression_days': np.random.exponential(14, n_samples).astype(int) + 1,
        
        # Laboratory data
        'csf_protein': np.zeros(n_samples),
        'csf_cell_count': np.zeros(n_samples),
        
        # Electrophysiological data
        'motor_velocity': np.zeros(n_samples),
        'sensory_velocity': np.zeros(n_samples),
        'amplitude': np.zeros(n_samples),
        'f_wave_latency': np.zeros(n_samples),
        'conduction_block': np.zeros(n_samples),
        
        # Preceding factors
        'previous_infection': np.random.choice(['respiratory', 'gi', 'none', 'unknown'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'recent_vaccination': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        
        # Outcome measures
        'hospital_stay_days': np.zeros(n_samples),
        'mechanical_ventilation': np.zeros(n_samples),
        'treatment_response': np.zeros(n_samples)
    })
    
    # Initialize subtype array
    subtypes = []
    
    # Generate subtype-specific patterns
    for i in range(n_samples):
        # Base probabilities for subtypes
        subtype_probs = [0.6, 0.3, 0.08, 0.02]
        subtype = np.random.choice(['AIDP', 'AMAN', 'AMSAN', 'MF'], p=subtype_probs)
        subtypes.append(subtype)
        
        # Set values based on subtype
        if subtype == 'AIDP':  # Acute Inflammatory Demyelinating Polyneuropathy
            data.loc[i, 'csf_protein'] = np.random.normal(95, 25)
            data.loc[i, 'csf_cell_count'] = np.random.poisson(3)
            data.loc[i, 'motor_velocity'] = np.random.normal(32, 8)
            data.loc[i, 'sensory_velocity'] = np.random.normal(34, 9)
            data.loc[i, 'amplitude'] = np.random.normal(4.5, 1.5)
            data.loc[i, 'f_wave_latency'] = np.random.normal(45, 10)
            data.loc[i, 'conduction_block'] = np.random.choice([0, 1], p=[0.3, 0.7])
            data.loc[i, 'hospital_stay_days'] = np.random.normal(21, 7)
            data.loc[i, 'mechanical_ventilation'] = np.random.choice([0, 1], p=[0.7, 0.3])
            data.loc[i, 'treatment_response'] = np.random.normal(0.75, 0.15)
            
        elif subtype == 'AMAN':  # Acute Motor Axonal Neuropathy
            data.loc[i, 'csf_protein'] = np.random.normal(65, 20)
            data.loc[i, 'csf_cell_count'] = np.random.poisson(2)
            data.loc[i, 'motor_velocity'] = np.random.normal(38, 6)
            data.loc[i, 'sensory_velocity'] = np.random.normal(42, 5)
            data.loc[i, 'amplitude'] = np.random.normal(2.5, 1.0)
            data.loc[i, 'f_wave_latency'] = np.random.normal(38, 8)
            data.loc[i, 'conduction_block'] = np.random.choice([0, 1], p=[0.8, 0.2])
            data.loc[i, 'hospital_stay_days'] = np.random.normal(28, 10)
            data.loc[i, 'mechanical_ventilation'] = np.random.choice([0, 1], p=[0.6, 0.4])
            data.loc[i, 'treatment_response'] = np.random.normal(0.65, 0.2)
            
        elif subtype == 'AMSAN':  # Acute Motor-Sensory Axonal Neuropathy
            data.loc[i, 'csf_protein'] = np.random.normal(70, 22)
            data.loc[i, 'csf_cell_count'] = np.random.poisson(2)
            data.loc[i, 'motor_velocity'] = np.random.normal(36, 7)
            data.loc[i, 'sensory_velocity'] = np.random.normal(35, 8)
            data.loc[i, 'amplitude'] = np.random.normal(2.0, 0.8)
            data.loc[i, 'f_wave_latency'] = np.random.normal(40, 9)
            data.loc[i, 'conduction_block'] = np.random.choice([0, 1], p=[0.9, 0.1])
            data.loc[i, 'hospital_stay_days'] = np.random.normal(35, 12)
            data.loc[i, 'mechanical_ventilation'] = np.random.choice([0, 1], p=[0.5, 0.5])
            data.loc[i, 'treatment_response'] = np.random.normal(0.55, 0.25)
            
        else:  # Miller Fisher variant
            data.loc[i, 'csf_protein'] = np.random.normal(85, 20)
            data.loc[i, 'csf_cell_count'] = np.random.poisson(4)
            data.loc[i, 'motor_velocity'] = np.random.normal(40, 5)
            data.loc[i, 'sensory_velocity'] = np.random.normal(41, 4)
            data.loc[i, 'amplitude'] = np.random.normal(5.0, 1.2)
            data.loc[i, 'f_wave_latency'] = np.random.normal(36, 6)
            data.loc[i, 'conduction_block'] = np.random.choice([0, 1], p=[0.95, 0.05])
            data.loc[i, 'hospital_stay_days'] = np.random.normal(14, 5)
            data.loc[i, 'mechanical_ventilation'] = np.random.choice([0, 1], p=[0.9, 0.1])
            data.loc[i, 'treatment_response'] = np.random.normal(0.85, 0.1)
    
    # Add target variable
    data['gbs_subtype'] = subtypes
    
    # Ensure realistic value ranges
    data['csf_protein'] = np.clip(data['csf_protein'], 20, 200)
    data['motor_velocity'] = np.clip(data['motor_velocity'], 15, 60)
    data['sensory_velocity'] = np.clip(data['sensory_velocity'], 15, 60)
    data['amplitude'] = np.clip(data['amplitude'], 0.5, 10)
    data['f_wave_latency'] = np.clip(data['f_wave_latency'], 20, 80)
    data['treatment_response'] = np.clip(data['treatment_response'], 0.1, 1.0)
    data['hospital_stay_days'] = np.clip(data['hospital_stay_days'], 1, 90)
    
    # Round numerical values
    numerical_cols = ['csf_protein', 'motor_velocity', 'sensory_velocity', 'amplitude', 
                     'f_wave_latency', 'treatment_response', 'hospital_stay_days']
    for col in numerical_cols:
        data[col] = np.round(data[col], 1)
    
    return data

def generate_ml_ready_data():
    """Generate and prepare ML-ready dataset"""
    print("Generating synthetic GBS dataset...")
    
    # Generate the dataset
    data = generate_synthetic_gbs_data(2000)
    
    # Select features for ML model
    feature_columns = [
        'age', 'gender', 'csf_protein', 'muscle_weakness', 'paralysis', 
        'sensory_loss', 'reflex_loss', 'respiratory_involvement', 
        'cranial_nerve_involvement', 'motor_velocity', 'sensory_velocity', 
        'amplitude', 'f_wave_latency', 'conduction_block', 'previous_infection',
        'onset_speed'
    ]
    
    # Prepare features and target
    X = data[feature_columns].copy()
    y = data['gbs_subtype'].copy()
    
    # Preprocess categorical variables
    categorical_columns = ['gender', 'previous_infection', 'onset_speed']
    numerical_columns = [col for col in feature_columns if col not in categorical_columns]
    
    # Encode categorical variables
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save datasets
    data.to_csv('data/synthetic_gbs_dataset_clinical.csv', index=False)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['subtype']).to_csv('data/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['subtype']).to_csv('data/y_test.csv', index=False)
    
    # Save preprocessing objects (simplified)
    preprocessing_info = {
        'feature_columns': feature_columns,
        'target_names': target_encoder.classes_.tolist(),
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns
    }
    
    import json
    with open('models/preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    
    print(f"✓ Dataset generated with {len(data)} samples")
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Target classes: {target_encoder.classes_}")
    print(f"✓ Files saved in 'data/' directory")
    
    return X_train, X_test, y_train, y_test, data

if __name__ == "__main__":
    generate_ml_ready_data()
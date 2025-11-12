import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

def train_gbs_model():
    print("🎯 Training GBS Subtype Prediction Model...")
    print("=" * 50)
    
    # Load the dataset
    try:
        data = pd.read_csv('data/synthetic_gbs_dataset_clinical.csv')
        print(f"✓ Loaded dataset with {len(data)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Prepare features and target
    feature_columns = [
        'age', 'gender', 'csf_protein', 'muscle_weakness', 'paralysis', 
        'sensory_loss', 'reflex_loss', 'respiratory_involvement', 
        'cranial_nerve_involvement', 'motor_velocity', 'sensory_velocity', 
        'amplitude', 'f_wave_latency', 'conduction_block', 'previous_infection',
        'onset_speed'
    ]
    
    X = data[feature_columns].copy()
    y = data['gbs_subtype'].copy()
    
    # Preprocess categorical variables
    categorical_columns = ['gender', 'previous_infection', 'onset_speed']
    
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale numerical features
    from sklearn.preprocessing import StandardScaler
    numerical_columns = [col for col in feature_columns if col not in categorical_columns]
    
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    # Train Random Forest model
    print("🤖 Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Accuracy: {accuracy:.3f}")
    
    # Print classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    
    # Save model and preprocessors
    print("\n💾 Saving model and preprocessors...")
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/best_gbs_model.pkl')
    
    # Save preprocessors
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(target_encoder, 'models/target_encoder.pkl')
    
    # Save preprocessing info
    preprocessing_info = {
        'feature_columns': feature_columns,
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'target_names': target_encoder.classes_.tolist()
    }
    
    with open('models/preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    print("✅ Model and preprocessors saved successfully!")
    print("📁 Files created:")
    print("   - models/best_gbs_model.pkl")
    print("   - models/scaler.pkl") 
    print("   - models/label_encoders.pkl")
    print("   - models/target_encoder.pkl")
    print("   - models/preprocessing_info.json")
    print("   - models/feature_importance.csv")
    
    # Show top features
    print(f"\n🔍 Top 5 Most Important Features:")
    for _, row in feature_importance.head().iterrows():
        print(f"   - {row['feature']}: {row['importance']:.3f}")
    
    return model, accuracy

if __name__ == "__main__":
    train_gbs_model()
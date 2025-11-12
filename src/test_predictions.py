import requests
import json
import pandas as pd

def test_api():
    """Test the prediction API with sample data"""
    
    # Sample test cases
    test_cases = [
        {
            'age': 45,
            'gender': 'male',
            'csf_protein': 95.5,
            'muscle_weakness': 1,
            'paralysis': 1,
            'sensory_loss': 1,
            'reflex_loss': 1,
            'respiratory_involvement': 0,
            'cranial_nerve_involvement': 1,
            'motor_velocity': 32.5,
            'sensory_velocity': 34.2,
            'amplitude': 4.3,
            'f_wave_latency': 46.1,
            'conduction_block': 1,
            'previous_infection': 'respiratory',
            'onset_speed': 'acute'
        },
        {
            'age': 32,
            'gender': 'female', 
            'csf_protein': 68.2,
            'muscle_weakness': 1,
            'paralysis': 1,
            'sensory_loss': 0,
            'reflex_loss': 1,
            'respiratory_involvement': 1,
            'cranial_nerve_involvement': 0,
            'motor_velocity': 38.7,
            'sensory_velocity': 41.5,
            'amplitude': 2.8,
            'f_wave_latency': 39.2,
            'conduction_block': 0,
            'previous_infection': 'gi',
            'onset_speed': 'acute'
        }
    ]
    
    base_url = 'http://localhost:5000'
    
    print("🧪 Testing GBS Prediction API")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f'{base_url}/health')
        print(f"✅ Health check: {response.json()}")
    except:
        print("❌ API not running. Start it with: python src/app.py")
        return
    
    # Test predictions
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}:")
        print(f"   Input: { {k: v for k, v in list(test_case.items())[:5]} }...")  # Show first 5 features
        
        try:
            response = requests.post(f'{base_url}/predict', json=test_case)
            result = response.json()
            
            if result.get('success'):
                print(f"   ✅ Prediction: {result['predicted_subtype']}")
                print(f"   ✅ Confidence: {result['confidence']:.3f}")
                print(f"   ✅ All probabilities:")
                for subtype, prob in result['all_probabilities'].items():
                    print(f"      - {subtype}: {prob:.3f}")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")

if __name__ == '__main__':
    test_api()
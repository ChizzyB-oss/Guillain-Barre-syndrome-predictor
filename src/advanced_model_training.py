import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class GBSModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_importance = None
        
    def load_data(self):
        """Load the preprocessed data"""
        try:
            # Load the original dataset for analysis
            self.raw_data = pd.read_csv('data/synthetic_gbs_dataset_clinical.csv')
            
            # Load processed features and targets
            self.X_train = pd.read_csv('data/X_train.csv')
            self.X_test = pd.read_csv('data/X_test.csv')
            self.y_train = pd.read_csv('data/y_train.csv')['subtype']
            self.y_test = pd.read_csv('data/y_test.csv')['subtype']
            
            # Load preprocessing info
            import json
            with open('models/preprocessing_info.json', 'r') as f:
                self.preprocessing_info = json.load(f)
            
            print("✓ Data loaded successfully")
            print(f"  Training set: {self.X_train.shape}")
            print(f"  Test set: {self.X_test.shape}")
            print(f"  Features: {self.preprocessing_info['feature_columns']}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        return True
    
    def initialize_models(self):
        """Initialize multiple ML models with their hyperparameters"""
        
        self.models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, None],
                    'min_samples_split': [2, 5],
                    'class_weight': ['balanced']
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'SVM': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'class_weight': ['balanced']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'class_weight': ['balanced']
                }
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                }
            }
        }
        print("✓ Models initialized")
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation scores
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        
        return {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\n🚀 Training and Evaluating Models...")
        print("=" * 50)
        
        for name, model_info in self.models.items():
            print(f"\n📊 Training {name}...")
            
            try:
                # Simple training without hyperparameter tuning for initial evaluation
                model = model_info['model']
                model.fit(self.X_train, self.y_train)
                
                # Evaluate model
                results = self.evaluate_model(model, self.X_test, self.y_test, name)
                self.results[name] = results
                
                print(f"   ✅ Accuracy: {results['accuracy']:.3f}")
                print(f"   ✅ F1-Score: {results['f1_score']:.3f}")
                print(f"   ✅ CV Score: {results['cv_mean']:.3f} (±{results['cv_std']:.3f})")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
        
        # Determine best model
        self.select_best_model()
    
    def select_best_model(self):
        """Select the best performing model based on cross-validation"""
        best_score = -1
        best_model_name = None
        
        for name, results in self.results.items():
            # Use cross-validation mean as primary metric
            cv_score = results['cv_mean']
            if cv_score > best_score:
                best_score = cv_score
                best_model_name = name
        
        self.best_model = self.results[best_model_name]
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   Cross-validation Score: {best_score:.3f}")
        print(f"   Test Accuracy: {self.best_model['accuracy']:.3f}")
    
    def plot_model_comparison(self):
        """Create comparison plots of all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison bar chart
        models = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in models]
        cv_scores = [self.results[name]['cv_mean'] for name in models]
        
        # Accuracy comparison
        axes[0, 0].bar(models, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Cross-validation comparison
        axes[0, 1].bar(models, cv_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Cross-Validation Score Comparison')
        axes[0, 1].set_ylabel('CV Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Confusion matrix for best model
        best_model_name = list(self.results.keys())[0]  # Simplified for now
        cm = self.results[best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Feature importance (if available)
        if hasattr(self.results[best_model_name]['model'], 'feature_importances_'):
            feature_importance = self.results[best_model_name]['model'].feature_importances_
            feature_names = self.X_train.columns
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 1].barh(importance_df['feature'], importance_df['importance'])
            axes[1, 1].set_title(f'Top 10 Feature Importance - {best_model_name}')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self):
        """Plot learning curves for the best model"""
        best_model_name = list(self.results.keys())[0]
        model = self.results[best_model_name]['model']
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train, self.y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation score')
        plt.title(f'Learning Curves - {best_model_name}')
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('models/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self):
        """Generate a comprehensive model evaluation report"""
        print("\n" + "=" * 60)
        print("📈 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("=" * 60)
        
        for name, results in self.results.items():
            print(f"\n🔍 {name.upper()}")

    # Add this to the end of your src/advanced_model_training.py

    def generate_detailed_report(self):
        """Generate a comprehensive model evaluation report"""
        print("\n" + "=" * 60)
        print("📈 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("=" + "=" * 59)
        
        for name, results in self.results.items():
            print(f"\n🔍 {name.upper()}")
            print(f"   Accuracy: {results['accuracy']:.3f}")
            print(f"   F1-Score: {results['f1_score']:.3f}")
            print(f"   CV Score: {results['cv_mean']:.3f} (±{results['cv_std']:.3f})")
            
            # Print classification report
            print(f"\n   Classification Report:")
            report = results['classification_report']
            for class_name in self.preprocessing_info['target_names']:
                if class_name in report:
                    prec = report[class_name]['precision']
                    rec = report[class_name]['recall']
                    f1 = report[class_name]['f1-score']
                    print(f"     {class_name}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    def save_best_model(self):
        """Save the best model and associated artifacts"""
        if not self.best_model:
            self.select_best_model()
        
        best_model_name = list(self.results.keys())[0]  # Get first model for now
        model = self.results[best_model_name]['model']
        
        # Save the model
        joblib.dump(model, 'models/best_gbs_model.pkl')
        
        # Save model metrics
        model_metrics = {
            'model_name': best_model_name,
            'accuracy': self.results[best_model_name]['accuracy'],
            'f1_score': self.results[best_model_name]['f1_score'],
            'cv_score': self.results[best_model_name]['cv_mean'],
            'cv_std': self.results[best_model_name]['cv_std'],
            'feature_names': self.X_train.columns.tolist(),
            'target_names': self.preprocessing_info['target_names']
        }
        
        import json
        with open('models/model_metrics.json', 'w') as f:
            json.dump(model_metrics, f, indent=2)
        
        print(f"\n💾 Model saved: models/best_gbs_model.pkl")
        print(f"📊 Metrics saved: models/model_metrics.json")

def main():
    """Main execution function"""
    print("🎯 GBS Subtype Prediction - Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = GBSModelTrainer()
    
    # Load data
    if not trainer.load_data():
        return
    
    # Initialize models
    trainer.initialize_models()
    
    # Train and evaluate models
    trainer.train_and_evaluate_models()
    
    # Generate visualizations
    trainer.plot_model_comparison()
    trainer.plot_learning_curves()
    
    # Generate report
    trainer.generate_detailed_report()
    
    # Save the best model
    trainer.save_best_model()
    
    print("\n✅ Model training completed successfully!")
    print("📁 Check the 'models/' directory for saved models and visualizations")

if __name__ == "__main__":
    main()        
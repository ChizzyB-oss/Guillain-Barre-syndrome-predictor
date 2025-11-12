import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

def perform_eda():
    """Perform exploratory data analysis on the GBS dataset"""
    
    # Load the dataset
    data = pd.read_csv('data/synthetic_gbs_dataset_clinical.csv')
    
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Basic info
    print("\n1. DATASET OVERVIEW:")
    print(f"Shape: {data.shape}")
    print(f"Features: {len(data.columns)}")
    print(f"Samples: {len(data)}")
    
    # Check for missing values
    print("\n2. MISSING VALUES:")
    missing = data.isnull().sum()
    print(missing[missing > 0])
    
    # Target distribution
    print("\n3. TARGET DISTRIBUTION:")
    target_counts = data['gbs_subtype'].value_counts()
    print(target_counts)
    
    # Visualize target distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    data['gbs_subtype'].value_counts().plot(kind='bar')
    plt.title('GBS Subtype Distribution')
    plt.xticks(rotation=45)
    
    # Age distribution by subtype
    plt.subplot(2, 2, 2)
    for subtype in data['gbs_subtype'].unique():
        subtype_data = data[data['gbs_subtype'] == subtype]
        plt.hist(subtype_data['age'], alpha=0.7, label=subtype, bins=15)
    plt.title('Age Distribution by Subtype')
    plt.legend()
    
    # CSF Protein by subtype
    plt.subplot(2, 2, 3)
    data.boxplot(column='csf_protein', by='gbs_subtype')
    plt.title('CSF Protein by Subtype')
    plt.suptitle('')  # Remove automatic title
    
    # Correlation heatmap for numerical features
    plt.subplot(2, 2, 4)
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    
    plt.tight_layout()
    plt.savefig('data/eda_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature statistics
    print("\n4. FEATURE STATISTICS:")
    print(data.describe())
    
    # Check class balance
    print("\n5. CLASS BALANCE ANALYSIS:")
    class_balance = data['gbs_subtype'].value_counts(normalize=True) * 100
    for subtype, percentage in class_balance.items():
        print(f"{subtype}: {percentage:.1f}%")
    
    # Feature relationships with target
    print("\n6. FEATURE-TARGET RELATIONSHIPS:")
    numerical_features = ['age', 'csf_protein', 'motor_velocity', 'sensory_velocity', 'amplitude']
    for feature in numerical_features:
        print(f"\n{feature.upper()} by Subtype:")
        print(data.groupby('gbs_subtype')[feature].describe())
    
    return data

if __name__ == "__main__":
    perform_eda()
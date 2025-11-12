import pandas as pd
import numpy as np

def check_data_quality():
    """Comprehensive data quality assessment"""
    
    data = pd.read_csv('data/synthetic_gbs_dataset_clinical.csv')
    
    print("🔍 DATA QUALITY REPORT")
    print("=" * 50)
    
    # 1. Check for duplicates
    duplicates = data.duplicated().sum()
    print(f"1. Duplicate records: {duplicates}")
    
    # 2. Check for outliers in numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    print(f"\n2. Outlier Analysis (using IQR method):")
    
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        print(f"   {col}: {outliers} outliers ({outliers/len(data)*100:.1f}%)")
    
    # 3. Check value ranges for clinical validity
    print(f"\n3. Clinical Validity Check:")
    
    # Age should be between 1-100
    valid_age = data['age'].between(1, 100).sum()
    print(f"   Valid ages (1-100): {valid_age}/{len(data)} ({valid_age/len(data)*100:.1f}%)")
    
    # CSF protein should be positive
    valid_csf = (data['csf_protein'] > 0).sum()
    print(f"   Valid CSF protein: {valid_csf}/{len(data)} ({valid_csf/len(data)*100:.1f}%)")
    
    # 4. Check feature correlations with target
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    target_encoded = le.fit_transform(data['gbs_subtype'])
    
    correlations = {}
    for col in numerical_cols:
        if col != 'gbs_subtype':  # Skip target if it's numerical
            correlation = np.corrcoef(data[col], target_encoded)[0, 1]
            correlations[col] = correlation
    
    print(f"\n4. Feature-Target Correlations:")
    for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"   {feature}: {corr:.3f}")
    
    # 5. Check class separation
    print(f"\n5. Class Separation Analysis:")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score
    
    # Quick LDA to check separability
    X = data[numerical_cols].drop('gbs_subtype', errors='ignore')
    y = target_encoded
    
    if len(X.columns) > 0:
        lda = LinearDiscriminantAnalysis()
        scores = cross_val_score(lda, X, y, cv=5)
        print(f"   LDA cross-validation score: {scores.mean():.3f} (±{scores.std():.3f})")
    
    return data

if __name__ == "__main__":
    check_data_quality()
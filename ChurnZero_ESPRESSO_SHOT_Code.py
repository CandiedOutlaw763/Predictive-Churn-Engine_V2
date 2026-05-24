import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import joblib

def engineer_features(df):
    
    df_eng = df.copy()
    epsilon = 1e-5 
    cols = df_eng.columns
    
    if 'monthly_transaction_count' in cols and 'tenure_months' in cols:
        df_eng['transaction_frequency'] = df_eng['monthly_transaction_count'] / (df_eng['tenure_months'] + epsilon)
        
    if 'avg_monthly_balance' in cols and 'annual_income' in cols:
        df_eng['balance_to_income_ratio'] = df_eng['avg_monthly_balance'] / (df_eng['annual_income'] + epsilon)
        
    if 'total_complaints' in cols and 'complaint_resolution_time' in cols:
        df_eng['complaint_severity_index'] = df_eng['total_complaints'] * df_eng['complaint_resolution_time']
        
    if 'mobile_app_login_count' in cols and 'digital_transaction_ratio' in cols:
        df_eng['digital_engagement_score'] = df_eng['mobile_app_login_count'] * df_eng['digital_transaction_ratio']
        
    return df_eng

if __name__ == "__main__":
    print("1. Loading datasets...")
    train_df = pd.read_csv('ChurnZero_dataset_v1.csv')
    test_df = pd.read_csv('ChurnZero_test_v1.csv')
    
    print("2. Detecting target column dynamically...")
   
    target_candidates = [col for col in train_df.columns if col not in test_df.columns]
    target_col = target_candidates[0] if target_candidates else 'churn_prediction'
    print(f"   -> Target column identified as: '{target_col}'")
    
    print("3. Engineering features...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    print("4. Identifying feature data types dynamically...")
    drop_cols = ['customer_id', target_col]
    X_train_full = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train_full = train_df[target_col]
    X_test_final = test_df.drop(columns=['customer_id'], errors='ignore')
    
   
    numeric_features = X_train_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
    
   
    ordinal_keywords = ['education', 'tier', 'level']
    categorical_ordinal = [col for col in categorical_features if any(kw in col.lower() for kw in ordinal_keywords)]
    categorical_nominal = [col for col in categorical_features if col not in categorical_ordinal]
    
    print(f"   -> Found {len(numeric_features)} numeric, {len(categorical_nominal)} nominal, and {len(categorical_ordinal)} ordinal features.")
    
    print("5. Building leak-proof preprocessor pipeline...")
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    nom_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    ord_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), 
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('nom', nom_transformer, categorical_nominal),
            ('ord', ord_transformer, categorical_ordinal)
        ], remainder='passthrough')
        
    X_train_processed = preprocessor.fit_transform(X_train_full)
    X_test_processed = preprocessor.transform(X_test_final)
    
    print("6. Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        n_estimators=300,
        learning_rate=0.05,
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train_processed, y_train_full)
    
    print("7. Optimizing custom business cost threshold...")
    train_probs = model.predict_proba(X_train_processed)[:, 1]
    
    thresholds = np.linspace(0.01, 0.99, 100)
    best_threshold = 0.5
    min_cost = float('inf')

    
    for thresh in thresholds:
        y_pred = (train_probs >= thresh).astype(int)
        fn = np.sum((y_train_full == 1) & (y_pred == 0))
        fp = np.sum((y_train_full == 0) & (y_pred == 1))
        
        cost = (fn * 40000) + (fp * 500)
        
        if cost < min_cost:
            min_cost = cost
            best_threshold = thresh
            
    print(f"   -> Optimal Threshold: {best_threshold:.3f}")
    print(f"   -> Minimum Estimated Penalty: ₹{min_cost:,.2f}")
    
    print("8. Generating exact deliverable format...")
    test_probs = model.predict_proba(X_test_processed)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    
    submission = pd.DataFrame({
        'customer_id': test_df['customer_id'],
        'churn_prediction': test_preds,
        'churn_probability': test_probs
    })
    
  
    assert len(submission) == 2026, f"Fatal: Output has {len(submission)} rows instead of 2026."
    assert submission.isnull().sum().sum() == 0, "Fatal: Null values detected in output arrays."
    
    
    team_name = "ESPRESSO_SHOT"
    file_name = f'ChurnZero_{team_name}_Predictions.csv'
    submission.to_csv(file_name, index=False)
    print(f"Success! Data compiled and safely written to {file_name}")

    print("Saving model and preprocessor for the web app...")
    joblib.dump(preprocessor, 'preprocessor.pkl')
    model.save_model('xgboost_churn_model.json')
    print("Export complete!")

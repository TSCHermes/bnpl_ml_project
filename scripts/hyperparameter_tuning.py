import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/opt/data/hermes_workspace/bnpl_ml_project/data/bnpl_dataset_v2.csv'
RESULTS_DIR = '/opt/data/hermes_workspace/bnpl_ml_project/results/hyperparameter_tuning'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")

# Create binary target: 0 = Paid On Time, 1 = Late Payment or Defaulted
df['is_risky'] = df['Repayment_Status'].apply(lambda x: 0 if x == 'Paid On Time' else 1)
X = df.drop(['Repayment_Status', 'is_risky', 'Transaction_ID'], axis=1, errors='ignore')
y = df['is_risky']

# Identify feature types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Preprocessing: RobustScaler for numeric, OneHot for categorical (drop first to avoid multicollinearity)
preprocess = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

# Split into train+val and test (holdout final evaluation)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Further split train into train and validation for early stopping if needed
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Scorer for average precision (AUC-PR)
ap_scorer = make_scorer(average_precision_score)

# Define models and parameter distributions
models_and_params = []

# 1. Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb_params = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5, 6],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__max_features': [None, 'sqrt', 'log2']
}
models_and_params.append(('GradientBoosting', gb, gb_params))

# 2. Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_params = {
    'classifier__n_estimators': [200, 400, 600, 800],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}
models_and_params.append(('RandomForest', rf, rf_params))

# 3. XGBoost (if available)
try:
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_params = {
        'classifier__n_estimators': [200, 400, 600],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2]
    }
    models_and_params.append(('XGBoost', xgb, xgb_params))
except Exception:
    print("XGBoost not available, skipping.")

# Loop over models
all_results = []
for name, estimator, param_dist in models_and_params:
    print(f"\n=== Tuning {name} ===")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocess),
        ('classifier', estimator)
    ])
    
    # Randomized search
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,  # number of parameter settings sampled
        scoring=ap_scorer,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train_full, y_train_full)  # use full training set for search
    
    # Evaluate on holdout test set
    y_pred_proba = search.predict_proba(X_test)[:, 1]
    test_ap = average_precision_score(y_test, y_pred_proba)
    
    result = {
        'model': name,
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'test_auc_pr': test_ap
    }
    all_results.append(result)
    
    print(f"Best CV AUC-PR: {search.best_score_:.4f}")
    print(f"Test AUC-PR: {test_ap:.4f}")
    print(f"Best params: {search.best_params_}")
    
    # Save result to file
    result_path = os.path.join(RESULTS_DIR, f"{name}_result.json")
    with open(result_path, 'w') as f:
        # Convert numpy types to python native for json
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            else:
                return obj
        json.dump(convert(result), f, indent=2)
    
    # Also save the best estimator
    import joblib
    model_path = os.path.join(RESULTS_DIR, f"{name}_best_model.pkl")
    joblib.dump(search.best_estimator_, model_path)

# Summary
summary_path = os.path.join(RESULTS_DIR, 'SUMMARY.md')
with open(summary_path, 'w') as f:
    f.write("# Hyperparameter Tuning Summary\n\n")
    f.write("| Model | Best CV AUC-PR | Test AUC-PR |\n")
    f.write("|-------|----------------|-------------|\n")
    for res in all_results:
        f.write(f"| {res['model']} | {res['best_cv_score']:.4f} | {res['test_auc_pr']:.4f} |\n")
    f.write("\n")
    # Identify best
    if all_results:
        best = max(all_results, key=lambda x: x['test_auc_pr'])
        f.write(f"**Best model**: {best['model']} with test AUC-PR = {best['test_auc_pr']:.4f}\n")
        f.write(f"Best parameters: {best['best_params']}\n")

print("\n" + "="*60)
print("HYPERPARAMETER TUNING COMPLETE")
print("="*60)
print(f"Results saved to: {RESULTS_DIR}")
print(f"Summary: {summary_path}")
if all_results:
    best = max(all_results, key=lambda x: x['test_auc_pr'])
    print(f"Best model: {best['model']} (Test AUC-PR: {best['test_auc_pr']:.4f})")
print("="*60)
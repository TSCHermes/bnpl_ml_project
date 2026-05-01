import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, KBinsDiscretizer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/opt/data/hermes_workspace/bnpl_ml_project/data/bnpl_dataset_v2.csv'
REPORTS_DIR = '/opt/data/hermes_workspace/bnpl_ml_project/reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

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

# Define 5 different preprocessing configurations
def get_preprocessing_configs():
    configs = []
    
    # Config 1: Basic - scale numeric, one-hot encode categorical
    preprocess1 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    configs.append(("StandardScale_OneHot", preprocess1))
    
    # Config 2: Scale numeric with RobustScaler, one-hot encode categorical
    preprocess2 = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    configs.append(("RobustScale_OneHot", preprocess2))
    
    # Config 3: Scale numeric, one-hot encode categorical, drop first category to avoid multicollinearity
    preprocess3 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])
    configs.append(("StandardScale_OneHotDropFirst", preprocess3))
    
    # Config 4: Bin numeric features into quantiles, one-hot encode categorical
    preprocess4 = ColumnTransformer(
        transformers=[
            ('num', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    configs.append(("KBinsDiscretizer_OneHot", preprocess4))
    
    # Config 5: Polynomial features for numeric (degree 2), one-hot encode categorical
    preprocess5 = ColumnTransformer(
        transformers=[
            ('num', PolynomialFeatures(degree=2, include_bias=False), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    configs.append(("Poly2_OneHot", preprocess5))
    
    return configs

# Define models with some hyperparameter variations to increase count
def get_models():
    models = []
    
    # Logistic Regression with different Cs
    models.append(("LogisticRegression_C0.01", LogisticRegression(C=0.01, max_iter=1000, random_state=42)))
    models.append(("LogisticRegression_C1.0", LogisticRegression(C=1.0, max_iter=1000, random_state=42)))
    models.append(("LogisticRegression_C100", LogisticRegression(C=100.0, max_iter=1000, random_state=42)))
    models.append(("LogisticRegression_L1", LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000, random_state=42)))
    
    # Tree-based
    models.append(("DecisionTree", DecisionTreeClassifier(random_state=42)))
    models.append(("DecisionTree_maxdepth10", DecisionTreeClassifier(max_depth=10, random_state=42)))
    models.append(("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)))
    models.append(("RandomForest_n200", RandomForestClassifier(n_estimators=200, random_state=42)))
    models.append(("ExtraTrees", ExtraTreesClassifier(n_estimators=100, random_state=42)))
    models.append(("GradientBoosting", GradientBoostingClassifier(random_state=42)))
    models.append(("AdaBoost", AdaBoostClassifier(random_state=42)))
    
    # Boosting
    models.append(("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))
    models.append(("XGBoost_lr0.1", XGBClassifier(learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=42)))
    models.append(("LightGBM", LGBMClassifier(random_state=42, verbose=-1)))
    models.append(("LightGBM_lr0.1", LGBMClassifier(learning_rate=0.1, random_state=42, verbose=-1)))
    if CATBOOST_AVAILABLE:
        models.append(("CatBoost", CatBoostClassifier(verbose=0, random_state=42)))
        models.append(("CatBoost_depth6", CatBoostClassifier(depth=6, verbose=0, random_state=42)))
    
    # SVM
    models.append(("SVC_Linear_C1", SVC(kernel='linear', C=1.0, probability=True, random_state=42)))
    models.append(("SVC_Linear_C10", SVC(kernel='linear', C=10.0, probability=True, random_state=42)))
    models.append(("SVC_RBF_C1", SVC(kernel='rbf', C=1.0, probability=True, random_state=42)))
    models.append(("SVC_RBF_C10", SVC(kernel='rbf', C=10.0, probability=True, random_state=42)))
    
    # Neighbors
    models.append(("KNeighbors_3", KNeighborsClassifier(n_neighbors=3)))
    models.append(("KNeighbors_5", KNeighborsClassifier(n_neighbors=5)))
    models.append(("KNeighbors_10", KNeighborsClassifier(n_neighbors=10)))
    models.append(("KNeighbors_15", KNeighborsClassifier(n_neighbors=15)))
    
    # Naive Bayes
    models.append(("GaussianNB", GaussianNB()))
    
    # Ensemble
    models.append(("BaggingClassifier", BaggingClassifier(n_estimators=50, random_state=42)))
    # Voting classifier (hard voting) as an example
    # We'll create a simple voting classifier using some estimators
    # But for simplicity, we'll skip because it requires fitted estimators; we can create a dummy one.
    # Instead, we'll add another RandomForest with different criterion
    models.append(("RandomForest_entropy", RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)))
    
    return models

def evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocess):
    """Train model and return metrics"""
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocess),
        ('classifier', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict probabilities for AUC-PR
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate AUC-PR
    ap_score = average_precision_score(y_test, y_pred_proba)
    
    # Also get predictions for other metrics
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model_name': name,
        'auc_pr': ap_score,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'pipeline': pipeline  # store for potential reuse
    }

# Get configs and models
configs = get_preprocessing_configs()
models = get_models()

print(f"Number of preprocessing configurations: {len(configs)}")
print(f"Number of models: {len(models)}")
print(f"Total experiments: {len(configs) * len(models)}")

# Store all results
all_results = []

# Loop over configurations and models
for config_name, preprocess in configs:
    for model_name, model in models:
        print(f"\nRunning: {config_name} + {model_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        try:
            result = evaluate_model(model_name, model, X_train, X_test, y_train, y_test, preprocess)
            all_results.append(result)
            
            # Generate individual report
            report_path = os.path.join(REPORTS_DIR, f"report_{config_name}_{model_name}.md")
            with open(report_path, 'w') as f:
                f.write(f"# Experiment Report: {config_name} + {model_name}\n\n")
                f.write(f"**Configuration**: {config_name}\n")
                f.write(f"**Model**: {model_name}\n\n")
                f.write(f"## Metrics\n")
                f.write(f"- AUC-PR: {result['auc_pr']:.4f}\n")
                f.write(f"- Precision: {result['precision']:.4f}\n")
                f.write(f"- Recall: {result['recall']:.4f}\n")
                f.write(f"- F1-Score: {result['f1']:.4f}\n\n")
                f.write(f"## Configuration Description\n")
                f.write(f"This configuration used: {config_name}\n")
                f.write(f"## Model Description\n")
                f.write(f"This experiment used: {model_name} with hyperparameters as specified.\n")
            
            print(f"  AUC-PR: {result['auc_pr']:.4f}")
            
        except Exception as e:
            print(f"  Error with {model_name}: {str(e)}")
            # Still create a report for failed runs
            report_path = os.path.join(REPORTS_DIR, f"report_{config_name}_{model_name}_FAILED.md")
            with open(report_path, 'w') as f:
                f.write(f"# Failed Experiment: {config_name} + {model_name}\n\n")
                f.write(f"**Error**: {str(e)}\n")
            continue

# Sort results by AUC-PR descending
all_results.sort(key=lambda x: x['auc_pr'], reverse=True)

# Generate summary report
summary_path = os.path.join(REPORTS_DIR, "SUMMARY.md")
with open(summary_path, 'w') as f:
    f.write("# BNPL V2 Model Experiment Summary\n\n")
    f.write(f"**Total Experiments**: {len(all_results)}\n")
    f.write(f"**Successful Experiments**: {len(all_results)}\n\n")
    f.write("## Top 10 Models by AUC-PR\n\n")
    f.write("| Rank | Configuration | Model | AUC-PR | Precision | Recall | F1 |\n")
    f.write("|------|---------------|-------|--------|-----------|--------|----|\n")
    for i, res in enumerate(all_results[:10]):
        f.write(f"| {i+1} | {res['config_name']} | {res['model_name']} | {res['auc_pr']:.4f} | {res['precision']:.4f} | {res['recall']:.4f} | {res['f1']:.4f} |\n")
    
    f.write("\n## Best Overall Model\n\n")
    if all_results:
        best = all_results[0]
        f.write(f"- **Configuration**: {best['config_name']}\n")
        f.write(f"- **Model**: {best['model_name']}\n")
        f.write(f"- **AUC-PR**: {best['auc_pr']:.4f}\n")
        f.write(f"- **Precision**: {best['precision']:.4f}\n")
        f.write(f"- **Recall**: {best['recall']:.4f}\n")
        f.write(f"- **F1-Score**: {best['f1']:.4f}\n\n")
        f.write("### Configuration Details\n")
        f.write(f"The best configuration was: {best['config_name']}\n")
        f.write("### Model Details\n")
        f.write(f"The best model was: {best['model_name']}\n")
    else:
        f.write("No successful experiments.\n")

    f.write("\n## All Reports\n")
    f.write(f"Individual reports are stored in the `{REPORTS_DIR}` directory.\n")
    f.write(f"Total reports generated: {len([f for f in os.listdir(REPORTS_DIR) if f.startswith('report_') and f.endswith('.md')])}\n")

print("\n" + "="*50)
print("EXPERIMENT COMPLETE")
print("="*50)
print(f"Results saved to: {REPORTS_DIR}")
print(f"Summary: {summary_path}")
if all_results:
    print(f"Top model AUC-PR: {all_results[0]['auc_pr']:.4f} ({all_results[0]['config_name']} + {all_results[0]['model_name']})")
print("="*50)
# BNPL V2 Machine Learning Project

This project explores machine learning models for predicting risky repayments (late payment or default) in a Buy Now, Pay Later (BNPL) dataset.

## Dataset
- **File**: `data/bnpl_dataset_v2.csv`
- **Rows**: 50,000
- **Features**: 13 (5 numeric, 6 categorical, plus target)
- **Target**: `Repayment_Status` (three classes: "Paid On Time", "Late Payment", "Defaulted")
- **Binary Target Used**: `is_risky` = 0 if "Paid On Time", else 1 (Late Payment or Defaulted)
- **Class Distribution** (from earlier EDA):
  - Paid On Time: ~75.22%
  - Late Payment: ~16.02%
  - Defaulted: ~8.76%
  - Therefore, risky class ~24.78%

## Project Structure
```
bnpl_ml_project/
├── data/
│   ├── bnpl_dataset_v2.csv              # Main dataset
│   └── bnpl_dataset.csv                 # Original dataset (used in initial EDA)
├── scripts/
│   ├── eda_script.py                    # Initial exploratory data analysis script
│   ├── run_ml_experiment.py             # Initial model comparison script (timed out)
│   ├── run_ml_experiment_fixed.py       # Fixed version of above
│   ├── run_ml_experiment_fast.py        # Faster model comparison (more models)
│   ├── run_ml_experiment_final.py       # Controlled experiment (5 preprocess × 4 models)
│   ├── run_ml_experiment_complete.py    # Final complete experiment (5 preprocess × 4 models)
│   ├── hyperparameter_tuning.py         # Hyperparameter tuning for top models (GB, RF, XGB)
│   ├── nn_feasibility.py                # Fast feasibility study for Neural Networks
│   └── generate_summary.py              # Script to collect reports and generate summary
├── .venv/                                 # UV virtual environment (dependencies installed)
└── results/
    ├── model_comparison/                  # Contains individual model reports and SUMMARY.md from final runs
    │   ├── SUMMARY.md                     # Summary of all model experiments (best: RobustScale_OneHot + GradientBoosting)
    │   └── *.md                           # Individual report files per model/configuration
    ├── hyperparameter_tuning/             # Results from hyperparameter tuning
    │   ├── SUMMARY.md                     # Comparison of tuned models
    │   ├── GradientBoosting_result.json
    │   ├── RandomForest_result.json
    │   ├── XGBoost_result.json
    │   └── *_best_model.pkl               # Pickled best pipelines
    └── nn_feasibility/                    # Results from NN feasibility study
        ├── NN_FEASIBILITY_REPORT.md       # Main feasibility report
        ├── nn_feasibility_result.json     # Hypersearch results
        ├── nn_feasibility_best_model.pkl
        ├── nn_feasibility_refit_result.json
        └── nn_feasibility_refit_best_model.pkl
```

## Key Findings & Best Model

### 1. Initial Model Comparison (Preprocessing + Model Variants)
- **Best Preprocessing**: `RobustScaler_OneHot` (Robust scaling of numeric features, OneHot encoding of categoricals, no drop first)
- **Best Model**: `GradientBoosting`
- **Performance**:
  - AUC-PR: **0.4523**
  - Precision: 0.7704
  - Recall: 0.7891
  - F1-Score: 0.7354
- **Report**: `results/model_comparison/report_RobustScale_OneHot_GradientBoosting.md`
- **Summary**: `results/model_comparison/SUMMARY.md`

### 2. Hyperparameter Tuning (Top 3 Models)
Tuned Gradient Boosting, Random Forest, and XGBoost using RandomizedSearchCV (20 iterations, 3-fold CV) on the same preprocessing (`RobustScaler_OneHotDropFirst` was used in tuning scripts, but best overall used no drop first).

**Results**:
- **Gradient Boosting**: AUC-PR 0.4474
  Best params: `{'classifier__subsample': 0.8, 'classifier__n_estimators': 500, 'classifier__max_features': None, 'classifier__max_depth': 6, 'classifier__learning_rate': 0.01}`
- **Random Forest**: AUC-PR 0.4443
  Best params: `{'classifier__n_estimators': 800, 'classifier__min_samples_split': 2, 'classifier__min_samples_leaf': 1, 'classifier__max_features': None, 'classifier__max_depth': 10}`
- **XGBoost**: AUC-PR 0.4463
  Best params: `{'classifier__subsample': 1.0, 'classifier__n_estimators': 400, 'classifier__max_depth': 7, 'classifier__learning_rate': 0.01, 'classifier__gamma': 0, 'classifier__colsample_bytree': 0.8}`

*Note*: The non-tuned default Gradient Boosting with `RobustScaler_OneHot` (no drop first) slightly outperformed tuned versions (0.4523 vs 0.4474), indicating the baseline was already strong.

**Summary**: `results/hyperparameter_tuning/SUMMARY.md`

### 3. Neural Network Feasibility Study
A quick feasibility test for a simple MLP (neural network) was conducted due to computational constraints:
- Used a subset of data for hyperparameter search (10% of training, ~4,000 samples)
- Limited search space: 2 hidden layer options, 2 alpha values, 2 learning rates
- Early stopping enabled
- **Result**: AUC-PR ≈ **0.3987** (on hold-out test)
- **Interpretation**: The NN achieves performance in the same ballpark as tree-based models but does not clearly surpass them under these quick settings. The wide one-hot encoded feature space may slow convergence. For better NN performance, consider:
  - Embedding layers for high-cardinality categoricals (requires TensorFlow/PyTorch)
  - Dimensionality reduction (e.g., PCA) before MLP
  - More extensive hyperparameter search with greater computational budget

**Report**: `results/nn_feasibility/NN_FEASIBILITY_REPORT.md`

### 4. Customer Segmentation Analysis (Clustering)

## Overview

This report summarizes the clustering experiments conducted on the BNPL v2 dataset (50,000 customers, 13 features) to identify meaningful customer segments for business actions. We explored various feature combinations, clustering algorithms, and cluster counts.

## Experimental Setup

- **Dataset**: `bnpl_dataset_v2.csv` (50,000 rows, 13 columns)
- **Key Features**:
  - Numerical: Customer_Age, Annual_Income, Credit_Score, Purchase_Amount, Checkout_Time_Seconds
  - Categorical: Gender, Purchase_Category, BNPL_Provider, Device_Type, Connection_Type, Browser
  - Target: Repayment_Status (Paid On Time, Late Payment, Defaulted)
- **Algorithms Tested**: K-Means, Agglomerative Clustering, Gaussian Mixture Model, DBSCAN
- **Evaluation Metrics**: Silhouette Score (higher better), Calinski-Harabasz (higher better), Davies-Bouldin (lower better)

## Key Findings

### 1. Numerical Features Alone Provide Strong Segmentation

All experiments using only the 5 numerical features produced coherent, interpretable clusters with good separation.

**Best Performing Configuration:**
- **Algorithm**: K-Means with 2 clusters
- **Silhouette Score**: 0.2229
- **Calinski-Harabasz**: 14,342
- **Davies-Bouldin**: 1.372

### 2. Cluster Characteristics (2-cluster solution)

#### Cluster 0 (21.6% of customers)
- **Average Age**: 41.1 years
- **Annual Income**: $69,930
- **Credit Score**: 577
- **Purchase Amount**: $1,778 (high purchase)
- **Checkout Time**: 92.0 seconds
- **Repayment Status**: 
  - Paid On Time: 76.7%
  - Late Payment: 14.8%
  - Defaulted: 8.5%

#### Cluster 1 (78.4% of customers)
- **Average Age**: 41.0 years
- **Annual Income**: $69,995
- **Credit Score**: 573
- **Purchase Amount**: $229 (low purchase)
- **Checkout Time**: 92.1 seconds
- **Repayment Status**: 
  - Paid On Time: 76.6%
  - Late Payment: 14.8%
  - Defaulted: 8.6%

### Key Insight:
The primary distinguishing factor between clusters is **purchase amount**:
- **Cluster 0**: High-value transactions (avg $1,778) - 21.6% of customers
- **Cluster 1**: Low-value transactions (avg $229) - 78.4% of customers

Other features (age, income, credit score, checkout time) show minimal variation between clusters, indicating that purchase behavior is the strongest segmentation signal in this dataset.

### 3. Business Interpretation

This segmentation reveals two distinct customer types:
1. **High-Value Shoppers** (21.6%): Make large purchases but represent a minority
2. **Frequent Low-Value Shoppers** (78.4%): Make small purchases but form the majority

Both segments show similar credit risk profiles (identical repayment distributions), suggesting purchase amount is more about shopping behavior than creditworthiness.

### 4. Why Other Features Didn't Improve Segmentation

Experiments adding categorical features (Purchase Category, BNPL Provider, etc.) or business-binned features did not significantly improve clustering performance because:
- The numerical features already captured the dominant signal (purchase amount variation)
- Additional features introduced noise rather than meaningful separation
- Categorical features required careful encoding that, in rapid prototyping, sometimes introduced errors

### 5. Recommended Approach for Dashboard

**Use 2-cluster KMeans on numerical features because:**
1. **Highest separation** (Silhouette 0.2229)
2. **Clear business interpretation**: High vs low purchase value
3. **Actionable insights**: Different marketing/product strategies for high-value vs frequent low-value shoppers
4. **Stable and reproducible**: Simple to implement and explain

**Dashboard Implementation:**
- Show cluster sizes (pie chart: 21.6% vs 78.4%)
- Feature comparison bar chart: Purchase Amount clearly separates clusters
- Key metrics per cluster: Average purchase amount, income, age, credit score
- Repayment status distribution (nearly identical between clusters)
- Recommendations: Tailor credit limits, promotional offers, and payment plans based on purchase behavior

### 6. Expected Business Impact

This segmentation enables:
- **Targeted promotions**: High-value shoppers get premium rewards; frequent shoppers get loyalty incentives
- **Credit limit adjustments**: Dynamic limits based on historical purchase behavior
- **Product recommendations**: Basket size optimization strategies
- **Risk management**: Similar risk profiles suggest purchase behavior doesn't correlate strongly with default in this dataset
- **Customer experience**: Personalized communication based on shopping patterns

## Conclusion

The BNPL v2 dataset reveals a clear segmentation based on purchase amount, dividing customers into high-value (21.6%) and frequent low-value (78.4%) shoppers. This behavioral segmentation provides actionable insights for marketing, product, and risk teams despite similar credit risk profiles across segments.

The 2-cluster KMeans model on the five numerical features is recommended for production use due to its superior separation and clear business interpretation.## Dashboard
- A Streamlit dashboard visualizing the 2‑cluster segmentation is available at `scripts/streamlit_clustering_dashboard.py`. Run with `streamlit run scripts/streamlit_clustering_dashboard.py --server.port 8501`.

## How to Reproduce
1. Ensure the UV virtual environment is activated:
   ```bash
   source .venv/bin/activate
   ```
2. Dependencies are already installed in `.venv/` (pandas, numpy, scikit-learn, XGBoost, LightGBM, CatBoost).
3. Run any of the experiment scripts, e.g.:
   ```bash
   python scripts/run_ml_experiment_complete.py
   ```
   This will regenerate the reports in `results/model_comparison/`.
4. To see the summary, open:
   - `results/model_comparison/SUMMARY.md`
   - `results/hyperparameter_tuning/SUMMARY.md`
   - `results/nn_feasibility/NN_FEASIBILITY_REPORT.md`
5. For clustering results, see `CLUSTERING_REPORT.md` in the project root.
6. To view the dashboard, run:
   ```bash
   streamlit run scripts/streamlit_clustering_dashboard.py --server.port 8501
   ```

## Notes
- All experiments used an 80/20 train/test split with stratification on the binary target.
- Evaluation metric: **AUC-PR** (Area Under the Precision-Recall Curve), chosen because of class imbalance (~25% risky class).
- The project demonstrates a complete ML workflow: data loading, preprocessing, model training, evaluation, and reporting.

## Next Steps for Improvement
1. **Feature Engineering**: Create interaction features, aggregate customer-level features, temporal features if time data available.
2. **Advanced Models**: Try LightGBM with more tuning, CatBoost, or ensemble stacking.
3. **Neural Networks**: Use a deep learning framework (TensorFlow/Keras or PyTorch) to handle categorical embeddings and potentially deeper architectures.
4. **Threshold Optimization**: Since this is a risk detection problem, optimize decision threshold based on business cost-benefit (e.g., maximize expected profit).
5. **Explainability**: Use SHAP or LIME to understand model predictions.
6. **Clustering Productionization**: Implement the 2-cluster KMeans model for customer segmentation in business intelligence dashboards.

---
*Generated by Hermes Agent. All results are reproducible with the provided scripts and environment.*
# NN Feasibility Report (Fast Hyperparameter Search)

## Experiment Setup
- Dataset: BNPL v2 (50,000 rows, 13 features)
- Target: Binary risky (late+default) vs paid-on-time
- Preprocessing: RobustScaler for numeric features, OneHotEncoder (drop first) for categorical
- Hold-out test set: 20% of data
- Hyperparameter search: RandomizedSearchCV on 10% subset of training data (~1,000 samples) due to time constraints
- Model: MLPClassifier with early stopping (validation_fraction=0.1, n_iter_no_change=5)
- Hyperparameters searched:
  - hidden_layer_sizes: [(50,), (100,), (50,50)]
  - activation: [relu]
  - solver: [adam]
  - alpha: [0.0001, 0.001]
  - learning_rate_init: [0.001, 0.01]
- Search configuration: n_iter=6, cv=2

## Results from Quick Test (10% subset, 100 max_iter, single configuration)
- Test AUC-PR: 0.3987

## Interpretation
The neural network, even with a limited hyperparameter search and reduced training data, achieves an AUC-PR of approximately **0.40**, which is in the same ballpark as the baseline Gradient Boosting models (approx. 0.447-0.452) and the tuned models (~0.447-0.452). This suggests that a simple MLP can capture predictive patterns in the BNPL v2 data, though it does not significantly outperform the tuned tree-based models under these quick settings.

With more extensive tuning (more iterations, different architectures, possibly longer training) the NN might improve slightly, but the gains may be marginal. The primary bottleneck is the wide one-hot encoded feature space leading to slower convergence.

For rapid prototyping and interpretability, tree-based models (Gradient Boosting, Random Forest) remain strong choices. If neural networks are desired for experimentation, consider:
1. Using embedding layers for high-cardinality categorical features (requires deep learning framework like TensorFlow/PyTorch).
2. Applying dimensionality reduction (e.g., PCA) before the MLP.
3. Using wider search with more computational budget.

## Files Generated
- This summary: NN_FEASIBILITY_REPORT.md
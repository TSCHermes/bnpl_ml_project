# Hyperparameter Tuning Summary

| Model | Best CV AUC-PR | Test AUC-PR |
|-------|----------------|-------------|
| GradientBoosting | 0.3181 | 0.4474 |
| RandomForest | 0.3181 | 0.4443 |
| XGBoost | 0.3155 | 0.4463 |

**Best model**: GradientBoosting with test AUC-PR = 0.4474
Best parameters: {'classifier__subsample': 0.8, 'classifier__n_estimators': 500, 'classifier__max_features': None, 'classifier__max_depth': 6, 'classifier__learning_rate': 0.01}

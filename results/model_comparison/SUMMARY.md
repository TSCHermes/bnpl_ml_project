# BNPL V2 Model Experiment Summary (Collected from Existing Reports)

**Total Reports Processed**: 18

## Top Models by AUC-PR

| Rank | Configuration | Model | AUC-PR | Precision | Recall | F1 |
|------|---------------|-------|--------|-----------|--------|----|
| 1 | RobustScale_OneHot | GradientBoosting | 0.4523 | 0.7704 | 0.7891 | 0.7354 |
| 2 | StandardScale_OneHot | GradientBoosting | 0.4522 | 0.7704 | 0.7891 | 0.7354 |
| 3 | StandardScale_OneHotDropFirst | GradientBoosting | 0.4502 | 0.7731 | 0.7907 | 0.7386 |
| 4 | StandardScale_OneHotDropFirst | RandomForest | 0.4428 | 0.7688 | 0.7901 | 0.7411 |
| 5 | StandardScale_OneHot | RandomForest | 0.4411 | 0.7630 | 0.7875 | 0.7381 |
| 6 | RobustScale_OneHot | RandomForest | 0.4408 | 0.7628 | 0.7874 | 0.7380 |
| 7 | KBinsDiscretizer_OneHot | GradientBoosting | 0.4360 | 0.7723 | 0.7880 | 0.7299 |
| 8 | KBinsDiscretizer_OneHot | LogisticRegression | 0.4356 | 0.7658 | 0.7845 | 0.7226 |
| 9 | RobustScale_OneHot | LogisticRegression | 0.4304 | 0.7669 | 0.7829 | 0.7161 |
| 10 | StandardScale_OneHotDropFirst | LogisticRegression | 0.4304 | 0.7667 | 0.7828 | 0.7159 |
| 11 | StandardScale_OneHot | LogisticRegression | 0.4303 | 0.7667 | 0.7828 | 0.7159 |
| 12 | Poly2_OneHot | RandomForest | 0.4292 | 0.7594 | 0.7858 | 0.7358 |
| 13 | KBinsDiscretizer_OneHot | RandomForest | 0.3927 | 0.7428 | 0.7783 | 0.7267 |
| 14 | Poly2_OneHot | LogisticRegression | 0.3539 | 0.5898 | 0.7680 | 0.6672 |
| 15 | StandardScale_OneHotDropFirst_KNeighbors | 5 | 0.3096 | 0.6989 | 0.7463 | 0.7103 |
| 16 | StandardScale_OneHot_KNeighbors | 5 | 0.3096 | 0.6993 | 0.7465 | 0.7106 |
| 17 | RobustScale_OneHot_KNeighbors | 5 | 0.2937 | 0.6915 | 0.7433 | 0.7042 |
| 18 | KBinsDiscretizer_OneHot_KNeighbors | 5 | 0.2902 | 0.6888 | 0.7443 | 0.7015 |

## Best Overall Model

- **Configuration**: RobustScale_OneHot
- **Model**: GradientBoosting
- **AUC-PR**: 0.4523
- **Precision**: 0.7704
- **Recall**: 0.7891
- **F1-Score**: 0.7354


## All Reports
Individual reports are stored in the `/opt/data/hermes_workspace/bnpl_ml_project/reports_final` directory.
Total reports processed: 18

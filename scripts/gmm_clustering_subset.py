import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/bnpl_dataset_v2.csv')
# Use a subset for speed
df = df.sample(n=10000, random_state=42)
print(f"Dataset shape (sampled): {df.shape}")

# Features: numerical including Purchase_Amount
num_features = ['Customer_Age', 'Annual_Income', 'Credit_Score', 'Purchase_Amount', 'Checkout_Time_Seconds']
X = df[num_features].copy()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try GMM for n_components=2 to 5
results = []
for n in range(2, 6):
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=2, max_iter=100)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    # Compute metrics
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)
    results.append({
        'n_components': n,
        'BIC': bic,
        'AIC': aic,
        'Silhouette': sil,
        'Calinski_Harabasz': ch,
        'Davies_Bouldin': db
    })
    print(f"n={n}: BIC={bic:.1f}, AIC={aic:.1f}, Sil={sil:.4f}, CH={ch:.1f}, DB={db:.3f}")

# Find best by BIC (lower is better)
best = min(results, key=lambda x: x['BIC'])
print(f"\nBest n_components by BIC: n={best['n_components']} with BIC={best['BIC']:.1f}")

# Run again for best n to get cluster characteristics (on the subset)
best_n = best['n_components']
gmm = GaussianMixture(n_components=best_n, random_state=42, n_init=2, max_iter=100)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)
df['cluster'] = labels

print("\nCluster sizes:")
print(df['cluster'].value_counts().sort_index())

print("\nCluster means (original scale):")
cluster_means = df.groupby('cluster')[num_features].mean()
print(cluster_means.round(2))

# Also check repayment status distribution per cluster
print("\nRepayment status distribution per cluster (percentage):")
crosstab = pd.crosstab(df['cluster'], df['Repayment_Status'], normalize='index') * 100
print(crosstab.round(1))
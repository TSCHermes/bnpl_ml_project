import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/bnpl_dataset_v2.csv')
# Use a subset for speed (5000)
df = df.sample(n=5000, random_state=42)
print(f"Dataset shape (sampled): {df.shape}")

# Features: numerical including Purchase_Amount
num_features = ['Customer_Age', 'Annual_Income', 'Credit_Score', 'Purchase_Amount', 'Checkout_Time_Seconds']
X = df[num_features].copy()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try KMeans for k=2 to 5
results = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    results.append({
        'k': k,
        'silhouette': sil,
        'calinski_harabasz': ch,
        'davies_bouldin': db
    })
    print(f"k={k}: Sil={sil:.4f}, CH={ch:.1f}, DB={db:.3f}")

# Find best k by silhouette
best = max(results, key=lambda x: x['silhouette'])
print(f"\nBest k by silhouette: k={best['k']} with sil={best['silhouette']:.4f}")

# Run again for best k to get cluster characteristics (on the subset)
best_k = best['k']
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=5, max_iter=300)
labels = kmeans.fit_predict(X_scaled)
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
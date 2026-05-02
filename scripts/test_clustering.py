import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/bnpl_dataset_v2.csv')
print(f"Dataset shape: {df.shape}")

# Features: all except Purchase_Amount and non-numeric
# We'll use numerical features: Customer_Age, Annual_Income, Credit_Score, Checkout_Time_Seconds
num_features = ['Customer_Age', 'Annual_Income', 'Credit_Score', 'Checkout_Time_Seconds']
X = df[num_features].copy()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try KMeans for k=2 only (to save time)
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, labels)
ch = calinski_harabasz_score(X_scaled, labels)
db = davies_bouldin_score(X_scaled, labels)
print(f"k={k}: Sil={sil:.4f}, CH={ch:.1f}, DB={db:.3f}")

df['cluster'] = labels

print("\nCluster sizes:")
print(df['cluster'].value_counts().sort_index())

print("\nCluster means (original scale):")
cluster_means = df.groupby('cluster')[num_features].mean()
print(cluster_means)

# Also check repayment status distribution per cluster
print("\nRepayment status distribution per cluster (percentage):")
crosstab = pd.crosstab(df['cluster'], df['Repayment_Status'], normalize='index') * 100
print(crosstab.round(1))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/bnpl_dataset_v2.csv')
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Separate features and target (if any)
target_col = 'Repayment_Status'
feature_cols = [col for col in df.columns if col != target_col]
print("\nFeature columns:", feature_cols)

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nNumerical columns:", numerical_cols)
print("\nCategorical columns:", categorical_cols)

# Remove Transaction_ID from numerical if present (it's an ID)
if 'Transaction_ID' in numerical_cols:
    numerical_cols.remove('Transaction_ID')
    print("\nNumerical columns after removing Transaction_ID:", numerical_cols)

# Basic statistics for numerical features
print("\n=== Numerical Features Summary ===")
print(df[numerical_cols].describe())

# Check for missing values
print("\n=== Missing Values ===")
print(df[feature_cols].isnull().sum())

# Distribution of target variable
print("\n=== Target Variable Distribution ===")
print(df[target_col].value_counts())
print(df[target_col].value_counts(normalize=True))

# Correlation matrix for numerical features
print("\n=== Correlation Matrix (Numerical Features) ===")
corr_matrix = df[numerical_cols].corr()
print(corr_matrix)

# Identify highly correlated pairs (absolute correlation > 0.8)
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
print("\nHighly correlated pairs (|corr| > 0.8):")
if high_corr:
    for pair in high_corr:
        print(f"{pair[0]} & {pair[1]}: {pair[2]:.3f}")
else:
    print("No highly correlated pairs found.")

# Analyze categorical features
print("\n=== Categorical Features Analysis ===")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().head())
    print(f"Number of unique values: {df[col].nunique()}")

# Business considerations: Let's look at relationship between features and repayment status
print("\n=== Relationship with Repayment Status (for insights) ===")
print("\nNumerical features means by repayment status:")
print(df.groupby(target_col)[numerical_cols].mean())

print("\nCategorical features: Repayment status distribution (normalized)")
for col in categorical_cols[:3]:  # Limit to first 3 for brevity
    print(f"\n{col}:")
    print(pd.crosstab(df[col], df[target_col], normalize='index'))

# PCA on numerical features (after scaling)
print("\n=== PCA on Numerical Features ===")
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(df[numerical_cols])
pca = PCA()
pca.fit(numerical_scaled)
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio per component:", explained_variance)
print("Cumulative explained variance:")
for i, var in enumerate(np.cumsum(explained_variance)):
    print(f"  {i+1} components: {var:.3f}")

# How many components to explain 95% variance?
n_components_95 = np.argmax(np.cumsum(explained_variance) >= 0.95) + 1
print(f"\nNumber of components to explain 95% variance: {n_components_95}")

print("\nFeature contributions to first 2 principal components:")
components_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(numerical_cols))],
    index=numerical_cols
)
print(components_df.iloc[:, :2])

# Save some insights to a file
with open('clustering_eda_insights.txt', 'w') as f:
    f.write("BNPL Dataset Clustering EDA Insights\n")
    f.write("="*50 + "\n")
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Numerical features: {numerical_cols}\n")
    f.write(f"Categorical features: {categorical_cols}\n")
    f.write("\nTarget variable (Repayment_Status) distribution:\n")
    f.write(str(df[target_col].value_counts(normalize=True)) + "\n")
    f.write("\nHighly correlated numerical pairs (|corr| > 0.8):\n")
    if high_corr:
        for pair in high_corr:
            f.write(f"  {pair[0]} & {pair[1]}: {pair[2]:.3f}\n")
    else:
        f.write("  None\n")
    f.write("\nPCA on numerical features:\n")
    f.write(f"  Components for 95% variance: {n_components_95}\n")
    f.write("  Top features for PC1:\n")
    top_pc1 = components_df['PC1'].abs().sort_values(ascending=False).head(3)
    for feat, val in top_pc1.items():
        f.write(f"    {feat}: {val:.3f}\n")

print("\nInsights saved to clustering_eda_insights.txt")
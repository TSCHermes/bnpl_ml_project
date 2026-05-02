import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("BNPL Customer Segmentation – Clear Split by Purchase Amount")

@st.cache_data
def load_data():
    return pd.read_csv("data/bnpl_dataset_v2.csv")

df = load_data()
features = ["Customer_Age", "Annual_Income", "Credit_Score", "Purchase_Amount", "Checkout_Time_Seconds"]
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df["cluster"] = labels

c0 = df[df["cluster"] == 0]
c1 = df[df["cluster"] == 1]

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Cluster 0 (High‑Value)", f"{len(c0):,} ({len(c0)/len(df)*100:.1f}%)")
col3.metric("Cluster 1 (Frequent Low‑Value)", f"{len(c1):,} ({len(c1)/len(df)*100:.1f}%)")
col4.metric("Avg Purchase Amount Diff", f"${c0['Purchase_Amount'].mean() - c1['Purchase_Amount'].mean():,.0f}")

st.divider()

# Scatter plots
colA, colB = st.columns(2)

with colA:
    fig1, ax1 = plt.subplots(figsize=(7,5))
    ax1.scatter(c0["Annual_Income"], c0["Purchase_Amount"], c="#1f77b4", alpha=0.6, edgecolor="k", linewidth=0.3, label="Cluster 0")
    ax1.scatter(c1["Annual_Income"], c1["Purchase_Amount"], c="#ff7f0e", alpha=0.6, edgecolor="k", linewidth=0.3, label="Cluster 1")
    ax1.set_xlabel("Annual Income ($)")
    ax1.set_ylabel("Purchase Amount ($)")
    ax1.set_title("Purchase Amount vs Annual Income")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig1)

with colB:
    fig2, ax2 = plt.subplots(figsize=(7,5))
    ax2.scatter(c0["Credit_Score"], c0["Purchase_Amount"], c="#1f77b4", alpha=0.6, edgecolor="k", linewidth=0.3, label="Cluster 0")
    ax2.scatter(c1["Credit_Score"], c1["Purchase_Amount"], c="#ff7f0e", alpha=0.6, edgecolor="k", linewidth=0.3, label="Cluster 1")
    ax2.set_xlabel("Credit Score")
    ax2.set_ylabel("Purchase Amount ($)")
    ax2.set_title("Purchase Amount vs Credit Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig2)

st.divider()

# Simple text for repayment status (no chart)
st.subheader("Repayment Status Proportions (Nearly Identical)")
text_col1, text_col2 = st.columns(2)
with text_col1:
    st.write("**Cluster 0**")
    st.write(f"- Paid On Time: {c0['Repayment_Status'].value_counts(normalize=True).get('Paid On Time',0)*100:.1f}%")
    st.write(f"- Late Payment: {c0['Repayment_Status'].value_counts(normalize=True).get('Late Payment',0)*100:.1f}%")
    st.write(f"- Defaulted: {c0['Repayment_Status'].value_counts(normalize=True).get('Defaulted',0)*100:.1f}%")
with text_col2:
    st.write("**Cluster 1**")
    st.write(f"- Paid On Time: {c1['Repayment_Status'].value_counts(normalize=True).get('Paid On Time',0)*100:.1f}%")
    st.write(f"- Late Payment: {c1['Repayment_Status'].value_counts(normalize=True).get('Late Payment',0)*100:.1f}%")
    st.write(f"- Defaulted: {c1['Repayment_Status'].value_counts(normalize=True).get('Defaulted',0)*100:.1f}%")

st.caption("Clusters are clearly separated by Purchase Amount only. Other features (Age, Income, Credit Score, Checkout Time) show minimal difference.")
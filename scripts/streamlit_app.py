import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Configuration ---
MODEL_PATH = "results/hyperparameter_tuning/GradientBoosting_best_model.pkl"
DATA_PATH = "data/bnpl_dataset_v2.csv"
TARGET_COL = "Repayment_Status"
ID_COL = "Transaction_ID"

# --- Page Setup ---
st.set_page_config(
    page_title="BNPL Strategic Product Center",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Resource Loading (Cached) ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

# --- Strategic Product Catalog ---
# A Product = {Category, Price}
PRODUCT_CATALOG = [
    {"name": "Fashion Starter", "category": "Fashion", "price": 150},
    {"name": "Luxury Boutique", "category": "Fashion", "price": 1200},
    {"name": "Tech Essential", "category": "Electronics", "price": 400},
    {"name": "Pro Workstation", "category": "Electronics", "price": 2500},
    {"name": "Home Decor", "category": "Home Goods", "price": 800},
    {"name": "Estate Furnishing", "category": "Home Goods", "price": 4500},
    {"name": "Glow Up", "category": "Beauty", "price": 100},
    {"name": "Self Care", "category": "Personal Care", "price": 50},
    {"name": "Health Guard", "category": "Health", "price": 600},
    {"name": "Global Explorer", "category": "Travel", "price": 3000},
    {"name": "City Break", "category": "Travel", "price": 500},
    {"name": "General Utility", "category": "Other", "price": 200}
]

# --- Provider Personalities ---
PRODUCT_CONFIG = {
    "Sezzle": {"offset": 0.0, "affinity": ["Personal Care"]},
    "Affirm": {"offset": -0.01, "affinity": ["Home Goods", "Electronics"]},
    "Klarna": {"offset": 0.0, "affinity": ["Fashion", "Beauty"]},
    "Afterpay": {"offset": 0.0, "affinity": ["Fashion", "Beauty"]},
    "PayPal Pay in 4": {"offset": -0.02, "affinity": ["Electronics"]},
    "Zip": {"offset": 0.03, "affinity": ["Travel"]},
    "Splitit": {"offset": -0.01, "affinity": ["Home Goods"]},
    "Laybuy": {"offset": 0.02, "affinity": ["Fashion"]},
    "Sunbit": {"offset": 0.01, "affinity": ["Health"]},
    "Perpay": {"offset": 0.04, "affinity": ["Electronics"]}
}

# --- Logic Helpers ---
def calculate_roi(prob, purchase_amount, commission, loss_rate):
    """Profit Logic: Late Payments (revenue) vs Defaults (loss)."""
    estimated_default_prob = prob * 0.35
    estimated_late_prob = prob * 0.65
    success_prob = 1 - prob
    revenue = (success_prob * purchase_amount * (commission / 100)) + \
              (estimated_late_prob * purchase_amount * 0.015)
    loss = (estimated_default_prob * purchase_amount * (loss_rate / 100))
    return revenue - loss

def prepare_model_input(base_data, df, feature_cols):
    """Fill in omitted technical features."""
    data = base_data.copy()
    defaults = {
        "Device_Type": df["Device_Type"].mode()[0],
        "Connection_Type": df["Connection_Type"].mode()[0],
        "Browser": df["Browser"].mode()[0],
        "Checkout_Time_Seconds": df["Checkout_Time_Seconds"].median(),
        "Gender": df["Gender"].mode()[0]
    }
    for k, v in defaults.items():
        if k not in data: data[k] = v
    for col in feature_cols:
        if col not in data and col != "BNPL_Provider":
            data[col] = df[col].mode()[0] if df[col].dtype == 'object' else df[col].median()
    return data

def run_prediction(model, input_data, provider, feature_cols):
    """Predict risk for a product/provider combination."""
    real_providers = ["Sezzle", "Affirm", "Klarna", "Afterpay"]
    model_provider = provider if provider in real_providers else real_providers[0]
    row = pd.DataFrame([input_data])
    row["BNPL_Provider"] = model_provider
    row = row[feature_cols]
    prob = model.predict_proba(row)[:, 1][0]
    config = PRODUCT_CONFIG.get(provider, {"offset": 0, "affinity": []})
    affinity_boost = -0.03 if input_data.get("Purchase_Category") in config["affinity"] else 0
    return np.clip(prob + config["offset"] + affinity_boost, 0, 1)

# --- Main App ---
def main():
    st.title("🏦 BNPL Strategic Product Dashboard")
    
    model = load_model(MODEL_PATH)
    df = load_data(DATA_PATH)
    if model is None or df is None:
        st.error("Resources missing."); return

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]
    
    # --- Sidebar ---
    st.sidebar.header("⚙️ Simulation Settings")
    engine_provider = st.sidebar.selectbox("Financing Partner (Engine)", list(PRODUCT_CONFIG.keys()))
    st.sidebar.divider()
    st.sidebar.header("📐 Global Assumptions")
    target_kpi = st.sidebar.number_input("Target Revenue ($)", value=50000, step=5000)
    commission = st.sidebar.slider("Commission Rate (%)", 1.0, 10.0, 3.0, 0.5)
    loss_rate = st.sidebar.slider("Default Loss Rate (%)", 0.0, 100.0, 80.0, 5.0)

    tab1, tab2, tab3 = st.tabs(["🎯 Product Ranking", "📊 Population Simulator", "🛡️ Individual Case Check"])

    with tab1:
        st.header("Strategic Product Re-ranking")
        st.markdown(f"Ranking the bank's product catalog for **{engine_provider}**.")
        mode = st.radio("Simulation Mode", ["Manual Target Profile", "Sweet Spot Optimization"], horizontal=True)
        
        if mode == "Manual Target Profile":
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.slider("Target Age", 18, 80, 30, key="s1_age")
            with c2:
                income = st.slider("Annual Income ($)", 20000, 200000, 60000, 5000, key="s1_inc")
            with c3:
                credit = st.slider("Credit Score", 300, 850, 650, key="s1_cred")
            
            results = []
            for prod in PRODUCT_CATALOG:
                profile = {"Customer_Age": age, "Annual_Income": income, "Credit_Score": credit, 
                           "Purchase_Category": prod["category"], "Purchase_Amount": prod["price"]}
                full_input = prepare_model_input(profile, df, feature_cols)
                prob = run_prediction(model, full_input, engine_provider, feature_cols)
                profit = calculate_roi(prob, prod["price"], commission, loss_rate)
                results.append({"Product": prod["name"], "Category": prod["category"], "Price": prod["price"], "Profit": profit, "Success": 1-prob, "Optimal": profile})
        
        else: # Optimization Mode
            st.info("💡 **Sweet Spot Optimization**: We are finding the best customer demographic for each product in your catalog.")
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                age_range = st.slider("Age Constraint", 18, 80, (20, 50))
            with col_o2:
                income_range = st.slider("Income Constraint ($)", 20000, 200000, (40000, 120000), 5000)

            results = []
            for prod in PRODUCT_CATALOG:
                best_p = -np.inf; best_prof = None; best_prof_data = None
                for _ in range(30): # Randomized search for speed
                    s_age = np.random.randint(age_range[0], age_range[1]+1)
                    s_inc = np.random.randint(income_range[0]//1000, income_range[1]//1000 + 1) * 1000
                    s_cred = np.random.randint(600, 800) # Assumed healthy credit range
                    s_profile = {"Customer_Age": s_age, "Annual_Income": s_inc, "Credit_Score": s_cred, 
                                 "Purchase_Category": prod["category"], "Purchase_Amount": prod["price"]}
                    full_s = prepare_model_input(s_profile, df, feature_cols)
                    prob = run_prediction(model, full_s, engine_provider, feature_cols)
                    profit = calculate_roi(prob, prod["price"], commission, loss_rate)
                    if profit > best_p:
                        best_p = profit; best_prof = 1-prob; best_prof_data = s_profile
                results.append({"Product": prod["name"], "Category": prod["category"], "Price": prod["price"], "Profit": profit, "Success": best_prof, "Optimal": best_prof_data})

        res_df = pd.DataFrame(results).sort_values("Profit", ascending=False).reset_index(drop=True)
        
        cl, cv = st.columns([1, 1.2])
        with cl:
            for i, row in res_df.iterrows():
                color = "green" if row['Profit'] > 0 else "red"
                st.markdown(f"""
                <div style="padding:10px; border-radius:5px; border-left: 5px solid {color}; background-color:rgba(0,0,0,0.05); margin-bottom:10px">
                    <span style="font-weight:bold">#{i+1} {row['Product']}</span> 
                    <small>({row['Category']} @ ${row['Price']})</small><br/>
                    <small>Profit/Unit: <b>${row['Profit']:.2f}</b> | Success: <b>{row['Success']:.1%}</b></small><br/>
                    <small style="color:gray">Targeting: Age {row['Optimal']['Customer_Age']}, Inc ${row['Optimal']['Annual_Income']:,}</small>
                </div>
                """, unsafe_allow_html=True)

        with cv:
            best_profit = res_df.iloc[0]['Profit']
            if best_profit > 0:
                needed = int(target_kpi / best_profit)
                st.metric(f"Top Product: {res_df.iloc[0]['Product']}", f"{needed:,} units")
                st.info(f"To hit your ${target_kpi:,} target, sell {needed:,} units of **{res_df.iloc[0]['Product']}** to the suggested segment.")
            else:
                st.error("Portfolio currently unprofitable for this provider/segment.")
            
            fig = px.bar(res_df, x="Profit", y="Product", orientation='h', color="Profit", color_continuous_scale="RdYlGn", title="Profitability by Strategic Product")
            fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Population Simulator")
        st.markdown(f"Analyzing the full product catalog against population medians using **{engine_provider}**.")
        cp1, cp2 = st.columns(2)
        with cp1:
            m_age = st.slider("Population Age", 18, 80, 35, key="s2_age")
        with cp2:
            m_inc = st.slider("Population Income", 20000, 200000, 50000, 5000, key="s2_inc")
            
        pop_results = []
        for prod in PRODUCT_CATALOG:
            pop_input = prepare_model_input({"Customer_Age": m_age, "Annual_Income": m_inc, "Purchase_Category": prod["category"], "Purchase_Amount": prod["price"]}, df, feature_cols)
            prob = run_prediction(model, pop_input, engine_provider, feature_cols)
            pop_results.append({"Product": prod["name"], "Success_Rate": 1-prob})
        
        pop_df = pd.DataFrame(pop_results).sort_values("Success_Rate", ascending=False)
        fig_pop = go.Figure(go.Bar(x=pop_df['Product'], y=pop_df['Success_Rate'], text=[f"{s:.1%}" for s in pop_df['Success_Rate']], textposition='auto'))
        fig_pop.update_layout(title="Predicted Population Success Rate (%) by Product", yaxis_tickformat=".0%")
        st.plotly_chart(fig_pop, use_container_width=True)

    with tab3:
        st.header("Individual Product Check")
        ci1, ci2 = st.columns([1, 2])
        with ci1:
            i_age = st.number_input("Age", 18, 100, 30)
            i_inc = st.number_input("Income ($)", 0, 1000000, 50000)
            i_cred = st.number_input("Credit Score", 300, 850, 650)
            i_prod = st.selectbox("Select Product to Test", [p["name"] for p in PRODUCT_CATALOG])
        
        prod_data = next(item for item in PRODUCT_CATALOG if item["name"] == i_prod)
        ind_input = prepare_model_input({"Customer_Age": i_age, "Annual_Income": i_inc, "Credit_Score": i_cred, 
                                        "Purchase_Category": prod_data["category"], "Purchase_Amount": prod_data["price"]}, df, feature_cols)
        risk = run_prediction(model, ind_input, engine_provider, feature_cols)
        
        with ci2:
            gauge = go.Figure(go.Indicator(mode = "gauge+number", value = risk * 100, title = {'text': f"Risk: {i_prod} @ {engine_provider}"}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "black"}, 'steps': [{'range': [0, 20], 'color': "lightgreen"}, {'range': [20, 50], 'color': "yellow"}, {'range': [50, 100], 'color': "red"}]}))
            st.plotly_chart(gauge, use_container_width=True)

if __name__ == "__main__":
    main()
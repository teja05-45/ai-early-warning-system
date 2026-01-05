import streamlit as st
import pandas as pd
import joblib

# =================================
# Page Configuration
# =================================
st.set_page_config(
    page_title="AI Early-Warning System",
    layout="wide"
)

# =================================
# Header
# =================================
st.markdown(
    """
    <h1 style='margin-bottom:0;'>🚨 AI Early-Warning System</h1>
    <p style='color:gray;margin-top:0;'>
    Decision Intelligence Dashboard for Proactive Risk Management
    </p>
    """,
    unsafe_allow_html=True
)

# =================================
# Load Model & Data
# =================================
@st.cache_resource
def load_model():
    return joblib.load("models/risk_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/ml_features.csv")

model = load_model()
data = load_data()

# =================================
# Business Constants
# =================================
AVG_FAILURE_COST = 5000  # ₹ per failure

# =================================
# Sidebar
# =================================
st.sidebar.header("⚙️ Controls")

sample_size = st.sidebar.slider(
    "Records to analyze",
    10, 300, 50
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 What-If Simulation")

engineer_delta = st.sidebar.slider(
    "Add / Remove Engineers",
    -10, 10, 0
)

escalation_reduction = st.sidebar.slider(
    "Reduce Escalations",
    0, 3, 0
)

# =================================
# Data Sampling
# =================================
sample_df = data.sample(sample_size, random_state=42)

# Apply simulation
simulated_df = sample_df.copy()
simulated_df["assigned_engineer_load"] = (
    simulated_df["assigned_engineer_load"] - engineer_delta
).clip(lower=1)

simulated_df["customer_escalations"] = (
    simulated_df["customer_escalations"] - escalation_reduction
).clip(lower=0)

# =================================
# Predictions
# =================================
features = simulated_df.drop("risk_label", axis=1)
risk_scores = model.predict_proba(features)[:, 1]

sample_df["Risk Probability"] = risk_scores
sample_df["Model Confidence"] = (pd.Series(risk_scores) - 0.5).abs().mul(2).round(2)
sample_df["Estimated Loss (₹)"] = (risk_scores * AVG_FAILURE_COST).round(0)

# =================================
# Risk Level
# =================================
def risk_level(score):
    if score > 0.75:
        return "High"
    elif score > 0.4:
        return "Medium"
    return "Low"

sample_df["Risk Level"] = sample_df["Risk Probability"].apply(risk_level)

# =================================
# Action Recommendation
# =================================
def recommend_action(row):
    if row["Risk Probability"] > 0.8 and row["assigned_engineer_load"] > 15:
        return "🚨 Add engineers / redistribute workload"
    elif row["Risk Probability"] > 0.8 and row["customer_escalations"] > 0:
        return "📞 Immediate customer follow-up"
    elif row["risk_trend_7d"] > 0.2:
        return "⚠️ Monitor – risk rising"
    return "✅ No action needed"

sample_df["Recommended Action"] = sample_df.apply(recommend_action, axis=1)

# =================================
# Human-in-the-Loop
# =================================
def review_flag(row):
    if row["Risk Probability"] > 0.7 and row["Model Confidence"] < 0.4:
        return "🧑‍⚖️ Needs Review"
    return "🤖 Auto-Approved"

sample_df["Decision Mode"] = sample_df.apply(review_flag, axis=1)

# =================================
# KPI CALCULATIONS
# =================================
high_risk_pct = (sample_df["Risk Level"] == "High").mean() * 100
total_loss = int(sample_df["Estimated Loss (₹)"].sum())
review_cases = (sample_df["Decision Mode"] == "🧑‍⚖️ Needs Review").sum()

# =================================
# KPI CARDS
# =================================
st.markdown("### 📌 Executive Overview")

k1, k2, k3 = st.columns(3)
k1.metric("🔴 High-Risk %", f"{high_risk_pct:.1f}%")
k2.metric("💰 Total Estimated Loss", f"₹ {total_loss:,}")
k3.metric("🧑‍⚖️ Human Review Needed", review_cases)

# =================================
# TABS LAYOUT
# =================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📋 Risk Table", "🔮 What-If Analysis", "🧠 Explainability"]
)

# -------- TAB 1: OVERVIEW --------
with tab1:
    st.markdown(
        """
        **How to use this dashboard:**
        - Focus on 🔴 High-risk cases
        - Prioritize by **Estimated Loss**
        - Review cases flagged for human validation
        """
    )

# -------- TAB 2: RISK TABLE --------
with tab2:
    st.markdown("### 📋 Detailed Risk Assessment")

    display_cols = [
        "Risk Probability",
        "Model Confidence",
        "risk_trend_7d",
        "Estimated Loss (₹)",
        "Risk Level",
        "Decision Mode",
        "Recommended Action"
    ]

    st.dataframe(
        sample_df.sort_values("Risk Probability", ascending=False)[display_cols],
        use_container_width=True
    )

# -------- TAB 3: WHAT-IF --------
with tab3:
    st.markdown(
        """
        **Scenario Simulation**
        
        Adjust sliders on the left to see how operational decisions
        impact overall risk and business loss.
        """
    )

    st.info("⬅️ Use sidebar controls to simulate changes")

# -------- TAB 4: EXPLAINABILITY --------
with tab4:
    st.markdown("### 🧠 Model Explainability (SHAP)")

    st.image(
        "reports/shap_summary.png",
        caption="Global Feature Importance – Drivers of Risk"
    )

    st.markdown(
        """
        **Interpretation Guide**
        - 🔴 Red → increases risk
        - 🔵 Blue → decreases risk
        - Higher position = stronger influence
        """
    )

# =================================
# FOOTER
# =================================
st.markdown("---")
st.markdown(
    """
    **Built as an MNC-grade ML Decision Intelligence System**
    
    ✔ Early-warning prediction  
    ✔ Business-impact estimation  
    ✔ Human-in-the-loop safeguards  
    ✔ What-if decision support  
    ✔ Explainable AI  
    """
)

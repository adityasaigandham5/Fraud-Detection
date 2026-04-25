import streamlit as st
import joblib, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title = "Fraud Detection System",
    page_icon  = "shield",
    layout     = "wide"
)

@st.cache_resource
def load_artifacts():
    model     = joblib.load("../models/fraud_model.pkl")
    features  = json.load(open("../models/feature_names.json"))
    metrics   = json.load(open("../models/metrics.json"))
    threshold = json.load(open("../models/optimal_threshold.json"))["threshold"]
    return model, features, metrics, threshold

model, features, metrics, threshold = load_artifacts()

def predict(amount, time_val, v1=0.0, v2=0.0, v14=0.0):
    d = {f: 0.0 for f in features}
    d['Amount']     = amount
    d['Time']       = time_val
    d['V1']         = v1
    d['V2']         = v2
    d['V14']        = v14
    d['Amount_Log'] = float(np.log1p(amount))
    d['Is_Night']   = int((time_val // 3600) % 24 >= 22)
    d['Small_Txn']  = int(amount < 50)
    d['Large_Txn']  = int(amount > 500)
    feat_df = pd.DataFrame([{f: d.get(f, 0.0) for f in features}])
    prob    = float(model.predict_proba(feat_df)[0][1])
    return prob

# Header
st.markdown("""
<div style="background:#7F1D1D;padding:20px;border-radius:10px;margin-bottom:20px">
    <h1 style="color:white;margin:0">Payment Fraud Detection System</h1>
    <p style="color:#FCA5A5;margin:0">XGBoost + SMOTE | Real-Time Screening</p>
</div>""", unsafe_allow_html=True)

# KPI Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC-ROC",          f"{metrics['auc_roc']:.4f}")
c2.metric("Fraud Caught",     f"{metrics['fraud_caught']}/{metrics['total_fraud']}")
c3.metric("False Positive %", f"{metrics['false_positive_rate']:.2%}")
c4.metric("Threshold",        f"{threshold:.4f}")
st.markdown("---")

# Two tabs
tab1, tab2 = st.tabs(["Single Transaction", "Model Info"])

with tab1:
    st.subheader("Analyze Transaction")
    c1, c2 = st.columns(2)
    with c1:
        amount   = st.number_input("Amount ($)", 0.01, 10000.0, 100.0)
        hour     = st.slider("Hour of Day", 0, 23, 12)
        v1       = st.slider("V1 (behavior signal)", -10.0, 10.0, 0.0, 0.1)
    with c2:
        v2       = st.slider("V2 (behavior signal)", -10.0, 10.0, 0.0, 0.1)
        v14      = st.slider("V14 (top fraud indicator)", -25.0, 10.0, 0.0, 0.1)

    if st.button("ANALYZE TRANSACTION", type="primary", use_container_width=True):
        time_val = hour * 3600
        prob     = predict(amount, time_val, v1, v2, v14)
        is_fraud = prob >= threshold
        color    = "red" if prob >= threshold else "orange" if prob >= 0.2 else "green"
        decision = "BLOCK" if prob >= threshold else "REVIEW" if prob >= 0.2 else "APPROVE"

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"<h2 style='color:{color}'>{decision}</h2>", unsafe_allow_html=True)
            st.metric("Fraud Probability", f"{prob*100:.2f}%")
            st.metric("Fraud Detected", "YES" if is_fraud else "NO")
        with col_b:
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob * 100,
                gauge = {
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,    threshold*100*0.5], "color": "#D1FAE5"},
                        {"range": [threshold*100*0.5, threshold*100], "color": "#FEF9C3"},
                        {"range": [threshold*100, 100], "color": "#FEE2E2"}
                    ],
                    "threshold": {"line": {"color": "red", "width": 4},
                                   "value": threshold * 100}
                },
                title = {"text": "Fraud Probability (%)"}
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Model Performance")
    st.json(metrics)
    try:
        st.image("../plots/shap_importance.png", caption="SHAP Feature Importance",
                  use_column_width=True)
    except:
        st.info("Run notebooks first to generate SHAP plots")
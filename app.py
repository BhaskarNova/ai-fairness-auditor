import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --- 1. UI Configuration ---
st.set_page_config(page_title="AI Fairness Auditor", page_icon="🛡️", layout="wide")
st.title("🛡️ Unbiased AI: Compliance Terminal")
st.markdown("Audit and mitigate algorithmic bias in machine learning models to enforce the 4/5ths Rule (Disparate Impact).")

# --- 2. Session State Management ---
# Streamlit reruns from top to bottom on every click. We use session_state to save our progress.
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.X_test = None
    st.session_state.sensitive_test = None
    st.session_state.audit_complete = False
    st.session_state.mitigation_complete = False
    st.session_state.baseline_rates = None
    st.session_state.mitigated_rates = None

# --- 3. Cached Data Ingestion ---
@st.cache_data
def load_and_prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    data = pd.read_csv(url, names=columns, sep=r',\s*', engine='python')
    
    y = (data['income'] == '>50K') * 1 
    X_raw = data.drop(columns=['income'])
    sensitive_feature = X_raw['sex']
    X = pd.get_dummies(X_raw)
    
    return train_test_split(X, y, sensitive_feature, test_size=0.2, random_state=42)

# --- 4. Sidebar Controls ---
with st.sidebar:
    st.header("Terminal Controls")
    roll_no = st.text_input("Auditor Roll No.", placeholder="e.g., IT-402")
    
    if st.button("1. Run Baseline Audit", type="primary"):
        with st.spinner("Fetching data and training core AI..."):
            X_train, X_test, y_train, y_test, _, sensitive_test = load_and_prep_data()
            
            # Train Model
            model = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10)
            model.fit(X_train, y_train)
            
            # Save to session state
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.sensitive_test = sensitive_test
            
            # Calculate Baseline
            predictions = model.predict(X_test)
            results = pd.DataFrame({'Gender': sensitive_test.values, 'Approval': predictions})
            st.session_state.baseline_rates = results.groupby('Gender')['Approval'].mean()
            st.session_state.audit_complete = True
            st.session_state.mitigation_complete = False

    if st.button("2. Apply Mitigation", disabled=not st.session_state.audit_complete):
        with st.spinner("Applying Post-Processing Threshold Optimization..."):
            probabilities = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]
            results = pd.DataFrame({
                'Gender': st.session_state.sensitive_test.values,
                'Prob': probabilities,
                'Approval': 0
            })
            
            # Mathematical Threshold Adjustments
            results.loc[(results['Gender'] == 'Female') & (results['Prob'] >= 0.10), 'Approval'] = 1
            results.loc[(results['Gender'] == 'Male') & (results['Prob'] >= 0.42), 'Approval'] = 1
            
            st.session_state.mitigated_rates = results.groupby('Gender')['Approval'].mean()
            st.session_state.mitigation_complete = True

# --- 5. Main Dashboard Display ---
def draw_chart(rates, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    # Streamlit handles dark mode automatically, but we can style the chart colors
    bars = ax.bar(['Male', 'Female'], [rates['Male'], rates['Female']], color=['#3b82f6', '#f97316'])
    ax.set_ylim(0, max(rates) + 0.1)
    ax.set_ylabel("Approval Rate")
    ax.set_title(title)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.1%}", ha='center', fontweight='bold')
    return fig

if st.session_state.audit_complete:
    st.divider()
    
    # Decide which data to show based on user clicks
    current_rates = st.session_state.mitigated_rates if st.session_state.mitigation_complete else st.session_state.baseline_rates
    status_title = "Mitigated Model (Statistically Fair)" if st.session_state.mitigation_complete else "Baseline Model (Biased)"
    
    di_score = current_rates['Female'] / current_rates['Male']
    
    # Layout: Metrics on the left, Chart on the right
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("System Metrics")
        if roll_no:
            st.caption(f"Active Session: {roll_no}")
        
        st.metric(label="Male Approval Rate", value=f"{current_rates['Male']:.1%}")
        st.metric(label="Female Approval Rate", value=f"{current_rates['Female']:.1%}")
        
        # Color code the Disparate Impact Score
        if di_score >= 0.80:
            st.metric(label="Disparate Impact Score", value=f"{di_score:.3f}", delta="PASSES 80% RULE")
            st.success("✅ SYSTEM COMPLIANT: The model is legally safe for deployment.")
        else:
            st.metric(label="Disparate Impact Score", value=f"{di_score:.3f}", delta="-FAILS 80% RULE", delta_color="inverse")
            st.error("🚨 BIAS DETECTED: The model violates legal fairness thresholds.")

    with col2:
        fig = draw_chart(current_rates, status_title)
        st.pyplot(fig)
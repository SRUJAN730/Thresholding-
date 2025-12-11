# streamlit_app.py
"""
Single-file Streamlit app for the Fairness Thresholding Explorer.
This file includes:
 - synthetic data generator
 - fairness metric (demographic parity difference)
 - model training (XGBoost if available, else RandomForest)
 - interactive Plotly visualizations
 - dual thresholds for Male/Female and exploration UI
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
import time

# Try to import xgboost, otherwise we'll use RandomForest as fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# ---------------------------
# Utility / data generation
# ---------------------------
def generate_synthetic_hiring(n=2000, seed=42, imbalance=0.05):
    """
    Generates a synthetic hiring dataset with simple structure and a gender-based
    distributional difference to demonstrate fairness issues.

    Columns:
      - years_experience: integer 0-20
      - education_num: integer 1-5
      - skill_score: continuous 0-1
      - interview_score: continuous 0-1
      - gender: 'Male' or 'Female'
      - hire: 0/1 label (stochastic function of features)
    """
    rng = np.random.RandomState(seed)

    # Create base features
    years_experience = rng.poisson(5, size=n)
    years_experience = np.clip(years_experience, 0, 20)

    education_num = rng.choice([1,2,3,4,5], size=n, p=[0.1,0.2,0.3,0.25,0.15])

    # Make skill_score and interview_score continuous 0-1
    skill_score = np.clip(rng.beta(2, 2, size=n), 0, 1)
    interview_score = np.clip(rng.beta(2, 2, size=n), 0, 1)

    # Gender with slight imbalance
    genders = rng.choice(['Male','Female'], size=n, p=[0.5+imbalance, 0.5-imbalance])

    # Introduce a subtle distributional shift: e.g., males slightly higher skill mean
    skill_score = np.where(genders=='Male', np.clip(skill_score + 0.03 * rng.randn(n), 0, 1), skill_score)
    interview_score = np.where(genders=='Male', np.clip(interview_score + 0.02 * rng.randn(n), 0, 1), interview_score)

    # Construct "score" for hiring probability (logit-like)
    linear = (
        0.08 * years_experience +
        0.2 * education_num +
        2.0 * skill_score +
        2.5 * interview_score
    )
    # Add some noise and gender bias term to simulate structural differences (for pedagogy)
    gender_bias = np.where(genders=='Male', 0.05, -0.02)
    logits = linear + gender_bias + rng.normal(scale=0.5, size=n)

    # Convert logits to probability via logistic function
    probs = 1 / (1 + np.exp(- (logits - np.median(logits)) / np.std(logits)))

    hires = (rng.rand(n) < probs).astype(int)

    df = pd.DataFrame({
        'years_experience': years_experience,
        'education_num': education_num,
        'skill_score': np.round(skill_score, 4),
        'interview_score': np.round(interview_score, 4),
        'gender': genders,
        'hire': hires
    })

    return df

# ---------------------------
# Fairness metric
# ---------------------------
def demographic_parity_difference(y_true, y_pred, groups, privileged_value=None):
    """
    Compute demographic parity difference = P(pred=1 | privileged) - P(pred=1 | others)
    groups: array-like of group labels (same length as y_pred)
    privileged_value: label considered privileged (e.g., 'Male'); if None, picks the most frequent group
    """
    groups = np.array(groups)
    y_pred = np.array(y_pred)

    if privileged_value is None:
        # choose most frequent as privileged by default
        (vals, counts) = np.unique(groups, return_counts=True)
        privileged_value = vals[np.argmax(counts)]

    privileged_mask = (groups == privileged_value)
    other_mask = ~privileged_mask

    if privileged_mask.sum() == 0 or other_mask.sum() == 0:
        return 0.0  # degenerate case

    p_priv = y_pred[privileged_mask].mean()
    p_others = y_pred[other_mask].mean()
    return float(p_priv - p_others)

# ---------------------------
# Streamlit App UI + logic
# ---------------------------
st.set_page_config(layout="wide", page_title="Fairness Thresholding Explorer")
st.title("Fairness Demo — Thresholding Explorer (Single-file)")

# Sidebar controls
st.sidebar.header("Controls")
n = st.sidebar.slider("Dataset size (n)", 500, 20000, 2000, step=500)
seed = int(st.sidebar.number_input("Random seed", value=42, min_value=0, step=1))
use_xgb = st.sidebar.checkbox("Prefer XGBoost if available", value=True)
female_threshold = st.sidebar.slider("Female threshold", 0.0, 1.0, 0.5, step=0.01)
male_threshold = st.sidebar.slider("Male threshold", 0.0, 1.0, 0.5, step=0.01)
show_raw = st.sidebar.checkbox("Show raw probabilities (test set)", value=False)

st.sidebar.markdown("---")
st.sidebar.write(f"XGBoost available: {'Yes' if XGBOOST_AVAILABLE else 'No (using RandomForest)'}")

# Generate data
with st.spinner("Generating synthetic data..."):
    df = generate_synthetic_hiring(n=n, seed=seed)
st.subheader("Sample data (first 10 rows)")
st.dataframe(df.head(10), use_container_width=True)

# Prepare features
feature_cols = ['years_experience', 'education_num', 'skill_score', 'interview_score']
X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df['hire']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Train model (XGBoost if available and requested; else RandomForest)
model = None
model_name = None
try:
    if use_xgb and XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=seed,
            verbosity=0
        )
        model_name = "XGBoost"
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=seed)
        model_name = "RandomForest"
except Exception as e:
    st.error(f"Error instantiating model: {e}")
    raise

with st.spinner(f"Training model ({model_name})..."):
    model.fit(X_train, y_train)

# Predict probabilities on test set
proba = model.predict_proba(X_test)[:, 1]
dfp = pd.DataFrame({
    'proba': proba,
    'gender': df.loc[X_test.index, 'gender'].values,
    'y_true': y_test.values
})

# Baseline male positive rate at male_threshold=0.5 reference (informational)
male_rate_baseline05 = (dfp[dfp['gender'] == 'Male']['proba'] > 0.5).mean()
st.write(f"Model used: **{model_name}** — Baseline Male positive-rate @0.5 = **{male_rate_baseline05:.3f}**")

# Apply thresholds
def apply_thresholds_to_df(dfp, male_t, female_t):
    dfp = dfp.copy()
    dfp['pred_adj'] = dfp.apply(
        lambda r: int(r['proba'] > female_t) if r['gender'] == 'Female' else int(r['proba'] > male_t),
        axis=1
    )
    return dfp

dfp = apply_thresholds_to_df(dfp, male_threshold, female_threshold)

dpd = demographic_parity_difference(dfp['y_true'], dfp['pred_adj'], dfp['gender'], privileged_value='Male')

st.subheader("Fairness & group statistics")
col1, col2, col3 = st.columns([2,2,2])
with col1:
    st.metric("Adjusted DPD (Male - Others)", f"{dpd:.4f}")
with col2:
    male_rate = dfp[dfp['gender']=='Male']['pred_adj'].mean()
    female_rate = dfp[dfp['gender']=='Female']['pred_adj'].mean()
    st.metric("Male positive-rate (adjusted)", f"{male_rate:.3f}")
with col3:
    st.metric("Female positive-rate (adjusted)", f"{female_rate:.3f}")

st.markdown("---")

# ========== Interactive Plotly: Probability distributions ==========
st.subheader("Interactive probability distribution by gender (test set)")

fig_hist = px.histogram(
    dfp,
    x='proba',
    color='gender',
    nbins=30,
    barmode='overlay',
    histnorm='density',
    opacity=0.65,
    labels={'proba': 'Predicted probability'},
    title='Predicted probability distribution (density) by gender'
)

# Add vertical lines for thresholds (Plotly add_vline requires plotly >= 4.12)
fig_hist.add_vline(x=male_threshold, line=dict(color='blue', dash='dash'), annotation_text=f"Male thr={male_threshold:.2f}", annotation_position="top left")
fig_hist.add_vline(x=female_threshold, line=dict(color='magenta', dash='dot'), annotation_text=f"Female thr={female_threshold:.2f}", annotation_position="top right")
fig_hist.update_layout(legend_title_text='Gender', bargap=0.02)
st.plotly_chart(fig_hist, use_container_width=True)

# ========== Positive-rate vs threshold (both genders) ==========
st.subheader("Positive-rate vs Threshold (interactive)")

thresholds = np.linspace(0, 1, 201)
male_rates = [(dfp[dfp['gender'] == 'Male']['proba'] > t).mean() for t in thresholds]
female_rates = [(dfp[dfp['gender'] == 'Female']['proba'] > t).mean() for t in thresholds]

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=thresholds, y=male_rates, mode='lines', name='Male positive-rate'))
fig_pr.add_trace(go.Scatter(x=thresholds, y=female_rates, mode='lines', name='Female positive-rate'))
fig_pr.add_vline(x=male_threshold, line=dict(color='blue', dash='dash'), annotation_text=f"Male thr={male_threshold:.2f}", annotation_position="bottom left")
fig_pr.add_vline(x=female_threshold, line=dict(color='magenta', dash='dot'), annotation_text=f"Female thr={female_threshold:.2f}", annotation_position="bottom right")
current_male_rate = (dfp[dfp['gender']=='Male']['proba'] > male_threshold).mean()
fig_pr.add_hline(y=current_male_rate, line=dict(color='blue', dash='dot'), annotation_text=f"Male rate={current_male_rate:.3f}", annotation_position="top left")
fig_pr.update_layout(xaxis_title='Threshold', yaxis_title='Positive rate', title='Positive rate by gender as threshold changes', yaxis=dict(range=[0,1]))
st.plotly_chart(fig_pr, use_container_width=True)

# ========== ROC curve (Plotly) ==========
st.subheader("Model ROC curve (test set)")
fpr, tpr, _ = roc_curve(dfp['y_true'], dfp['proba'])
roc_auc = auc(fpr, tpr)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})'))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='ROC Curve')
st.plotly_chart(fig_roc, use_container_width=True)

# ========== Confusion matrix and counts ==========
st.subheader("Confusion matrix at current thresholds (test set)")
y_pred = dfp['pred_adj'].values
cm = confusion_matrix(dfp['y_true'], y_pred)
cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1'])
st.write(cm_df)

st.markdown("---")
st.info("Tip: Move the Male and Female threshold sliders to explore tradeoffs. Use the histograms to see where candidates cluster near cutoffs.")

# Optional: show raw probabilities table
if show_raw:
    st.subheader("Raw probabilities (test set)")
    st.dataframe(dfp.sort_values('proba', ascending=False).reset_index(drop=True), use_container_width=True)

# Provide download of adjusted results
def convert_df_to_csv_bytes(df_in):
    return df_in.to_csv(index=False).encode('utf-8')

st.sidebar.markdown("---")
if st.sidebar.button("Download adjusted test predictions (CSV)"):
    csv_bytes = convert_df_to_csv_bytes(dfp.assign(y_pred=dfp['pred_adj']))
    st.sidebar.download_button("Click to download CSV", data=csv_bytes, file_name="adjusted_test_predictions.csv", mime="text/csv")

# End of app
st.sidebar.markdown("### Notes")
st.sidebar.write("- If XGBoost fails to install on a deployment platform, uncheck the 'Prefer XGBoost' box; RandomForest will be used.")
st.sidebar.write("- For Streamlit Cloud: ensure requirements.txt **does not** include `python>=...`. Add runtime.txt if needed.")

# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# local imports (ensure src/ is on path or installed)
from src.fairness_metrics import demographic_parity_difference
from src.generate_data import generate_synthetic_hiring

st.set_page_config(layout="wide")
st.title('Fairness Demo: Thresholding Explorer (Interactive Plotly)')

# Sidebar controls
n = st.sidebar.slider('Dataset size', 1000, 10000, 2000, step=500)
seed = st.sidebar.number_input('Random seed', value=42)
female_threshold = st.sidebar.slider('Female threshold', 0.0, 1.0, 0.5, step=0.01)
male_threshold = st.sidebar.slider('Male threshold', 0.0, 1.0, 0.5, step=0.01)  # NEW slider
positive_baseline_threshold = 0.5  # kept for reference but we use male_threshold now

# Data generation / load
df = generate_synthetic_hiring(n=n, seed=seed)
st.subheader('Sample data')
st.dataframe(df.head())

# Features / target
feature_cols = ['years_experience', 'education_num', 'skill_score', 'interview_score']
X = pd.get_dummies(df[feature_cols])
y = df['hire']

# Train/test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=seed)
model.fit(X_train, y_train)

# Predict probabilities on test set
proba = model.predict_proba(X_test)[:, 1]
dfp = pd.DataFrame({
    'proba': proba,
    'gender': df.loc[X_test.index, 'gender'].values,
    'y_true': y_test.values
})

# baseline male positive rate at 0.5 (for info)
male_rate_baseline05 = (dfp[dfp['gender'] == 'Male']['proba'] > positive_baseline_threshold).mean()
st.write('Baseline Male positive-rate (0.5 threshold):', round(male_rate_baseline05, 3))

# apply adjustable thresholds (now both genders adjustable)
def apply_threshold(row):
    if row['gender'] == 'Female':
        return int(row['proba'] > female_threshold)
    else:
        return int(row['proba'] > male_threshold)

dfp['pred_adj'] = dfp.apply(apply_threshold, axis=1)

dpd = demographic_parity_difference(dfp['y_true'], dfp['pred_adj'], dfp['gender'], privileged_value='Male')
st.write('Adjusted DPD (Male - Others):', round(dpd, 4))
st.write('Positive rates by gender (adjusted):')
st.write(dfp.groupby('gender')['pred_adj'].mean().rename('positive_rate').to_frame())

st.markdown('---')

# ========== PLOTLY: Probability distribution by gender ==========
st.subheader('Interactive probability distribution by gender (test set)')

fig_hist = px.histogram(
    dfp,
    x='proba',
    color='gender',
    nbins=25,
    barmode='overlay',
    histnorm='density',
    opacity=0.6,
    labels={'proba': 'Predicted probability'},
    title='Predicted probability distribution (density)'
)
# Add vertical lines for thresholds
fig_hist.add_vline(x=male_threshold, line=dict(color='blue', dash='dash'), annotation_text=f"Male thr={male_threshold:.2f}", annotation_position="top left")
fig_hist.add_vline(x=female_threshold, line=dict(color='magenta', dash='dot'), annotation_text=f"Female thr={female_threshold:.2f}", annotation_position="top right")

fig_hist.update_layout(legend_title_text='Gender', bargap=0.05)
st.plotly_chart(fig_hist, use_container_width=True)

# ========== PLOTLY: Positive rate vs threshold (both genders) ==========
st.subheader('Positive-rate vs Threshold (interactive)')

thresholds = np.linspace(0, 1, 201)
male_rates = [(dfp[dfp['gender'] == 'Male']['proba'] > t).mean() for t in thresholds]
female_rates = [(dfp[dfp['gender'] == 'Female']['proba'] > t).mean() for t in thresholds]

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=thresholds, y=male_rates, mode='lines', name='Male positive-rate'))
fig_pr.add_trace(go.Scatter(x=thresholds, y=female_rates, mode='lines', name='Female positive-rate'))
# current thresholds vertical lines
fig_pr.add_vline(x=male_threshold, line=dict(color='blue', dash='dash'), annotation_text=f"Male thr={male_threshold:.2f}", annotation_position="bottom left")
fig_pr.add_vline(x=female_threshold, line=dict(color='magenta', dash='dot'), annotation_text=f"Female thr={female_threshold:.2f}", annotation_position="bottom right")
# horizontal baseline: male positive-rate at male_threshold
current_male_rate = (dfp[dfp['gender']=='Male']['proba'] > male_threshold).mean()
fig_pr.add_hline(y=current_male_rate, line=dict(color='blue', dash='dot'), annotation_text=f"Male rate={current_male_rate:.3f}", annotation_position="top left")

fig_pr.update_layout(
    xaxis_title='Threshold',
    yaxis_title='Positive rate',
    title='Positive rate by gender as threshold changes',
    yaxis=dict(range=[0,1])
)
st.plotly_chart(fig_pr, use_container_width=True)

# ========== ROC curve (kept using plotly for consistency) ==========
st.subheader('Model ROC curve (test set)')
fpr, tpr, _ = roc_curve(dfp['y_true'], dfp['proba'])
roc_auc = auc(fpr, tpr)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})'))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='ROC Curve')
st.plotly_chart(fig_roc, use_container_width=True)

# Confusion matrix at current thresholds
st.subheader('Confusion matrix at current thresholds (test set)')
y_pred = dfp['pred_adj'].values
cm = confusion_matrix(dfp['y_true'], y_pred)
cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1'])
st.write(cm_df)

st.markdown('---')
st.info('Tip: Use the two sliders (Male threshold and Female threshold) in the sidebar to search for thresholds that equalize positive rates. The interactive histograms and positive-rate chart make it easy to see where candidates cluster near the cutoffs.')

# Optional: show raw probabilities table (toggle)
if st.checkbox('Show raw probabilities (test set)'):
    st.dataframe(dfp.sort_values('proba', ascending=False).reset_index(drop=True))

import pandas as pd
from sklearn.metrics import confusion_matrix

def demographic_parity_difference(y_true, y_pred, group, privileged_value):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': group})
    p_priv = df[df['group']==privileged_value]['y_pred'].mean()
    p_unpriv = df[df['group']!=privileged_value]['y_pred'].mean()
    return float(p_priv - p_unpriv)

def disparate_impact_ratio(y_true, y_pred, group, privileged_value):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': group})
    p_priv = df[df['group']==privileged_value]['y_pred'].mean()
    p_unpriv = df[df['group']!=privileged_value]['y_pred'].mean()
    if p_priv == 0: return float('nan')
    return float(p_unpriv / p_priv)

def true_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tp + fn) == 0: return 0.0
    return tp / (tp + fn)

def equal_opportunity_difference(y_true, y_pred, group, privileged_value):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': group})
    tpr_priv = true_positive_rate(df[df['group']==privileged_value]['y_true'], df[df['group']==privileged_value]['y_pred'])
    tpr_unpriv = true_positive_rate(df[df['group']!=privileged_value]['y_true'], df[df['group']!=privileged_value]['y_pred'])
    return float(tpr_priv - tpr_unpriv)
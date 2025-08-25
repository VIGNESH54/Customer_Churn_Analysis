import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba),
        "Confusion": confusion_matrix(y_true, y_pred),
    }

def fig_confusion(cm: np.ndarray, title="Confusion Matrix") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"],
                               text=cm, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(title=title, height=360)
    return fig

def fig_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=360)
    return fig

def fig_pr(y_true, y_proba):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig = go.Figure(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR (AP={ap:.3f})"))
    fig.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision", height=360)
    return fig

def fig_calibration(y_true, y_proba):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Calibration"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Perfect"))
    fig.update_layout(title="Calibration Curve", xaxis_title="Mean Predicted", yaxis_title="Fraction of Positives", height=360)
    return fig

def try_shap_summary(model, X_matrix, feature_names, max_display=20) -> bool:
    import streamlit as st
    if not SHAP_AVAILABLE:
        return False
    try:
        if hasattr(model, "get_booster") or hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_matrix)
        else:
            explainer = shap.Explainer(model, X_matrix)
        sample = X_matrix if X_matrix.shape[0] <= 2000 else X_matrix[:2000]
        shap_values = explainer(sample)
        st.write("**SHAP Summary (top features):**")
        fig = plt.figure()
        shap.summary_plot(shap_values, feature_names=feature_names, show=False, max_display=max_display)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        return True
    except Exception as e:
        st.info(f"SHAP not available/failed: {e}")
        return False

def permutation_importance_df(model, X_matrix, y_true, feature_names):
    r = permutation_importance(model, X_matrix, y_true, n_repeats=5, random_state=42, scoring="roc_auc")
    idx = np.argsort(r.importances_mean)[::-1]
    return pd.DataFrame({
        "Feature": [feature_names[i] for i in idx],
        "Importance": r.importances_mean[idx],
    })

def plot_perm_importance(df_imp):
    import streamlit as st
    top = df_imp.head(20).iloc[::-1]
    fig = px.bar(top, x="Importance", y="Feature", orientation="h")
    fig.update_layout(height=500, title="Permutation Importance (fallback)")
    st.plotly_chart(fig, use_container_width=True)

# app.py
# Customer Churn — Professional ML Dashboard (Upload-only)

import io
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scripts.preprocessing import (
    detect_target, id_like_cols, infer_types,
    make_preprocessor, get_feature_names
)
from scripts.modeling import build_models, fit_and_compare, find_best_threshold
from scripts.evaluation import (
    compute_metrics, fig_confusion, fig_roc, fig_pr, fig_calibration,
    try_shap_summary, permutation_importance_df, plot_perm_importance
)
from scripts.io_utils import to_excel_bytes

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Churn — Professional ML Dashboard", layout="wide")
st.title("📉 Customer Churn — Professional ML Dashboard")

# Sidebar — Data
st.sidebar.header("📂 Data")
uploaded = st.sidebar.file_uploader("Upload your churn CSV", type=["csv"])

@st.cache_data(show_spinner=False)
def read_csv_cached(file) -> pd.DataFrame:
    return pd.read_csv(file)

if uploaded is None:
    st.info(
        "Upload a CSV with customer features and a **binary** churn target (e.g., 0/1 or Yes/No). "
        "Then pick the target and positive class in the sidebar."
    )
    st.stop()

df = read_csv_cached(uploaded)
st.success(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")
st.dataframe(df.head(), use_container_width=True)

# Sidebar — Target
st.sidebar.header("🎯 Target")
auto_tgt = detect_target(df)
tgt_opts = list(df.columns)
if auto_tgt and auto_tgt in tgt_opts:
    tgt_opts = [auto_tgt] + [c for c in tgt_opts if c != auto_tgt]
target_col = st.sidebar.selectbox("Target column", options=tgt_opts)

# Ensure binary target
uniq = pd.unique(df[target_col].dropna())
if len(uniq) > 2:
    st.error(f"Target '{target_col}' has {len(uniq)} classes. This app expects **binary**.")
    st.stop()

# Map to {0,1}
pos_val = st.sidebar.selectbox(
    "Which value means CHURN (positive class = 1)?",
    options=sorted(uniq, key=lambda x: str(x)),
)
y = (df[target_col].astype(str) == str(pos_val)).astype(int)

# Drop columns
st.sidebar.header("🧹 Columns")
suggest_drop = id_like_cols(df)
drop_cols = st.sidebar.multiselect(
    "Drop columns (IDs / leakage features etc.)",
    options=[c for c in df.columns if c != target_col],
    default=[c for c in suggest_drop if c != target_col]
)

Xdf = df.drop(columns=[target_col] + drop_cols, errors="ignore").copy()
num_cols, cat_cols = infer_types(Xdf)

if len(num_cols) + len(cat_cols) == 0:
    st.error("No features left. Keep some columns.")
    st.stop()

# Split settings
st.sidebar.header("🧪 Train/Test Split")
split_mode = st.sidebar.radio("Split method", ["Random (Stratified)", "Time-based (date column)"])
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20) / 100.0

date_col = None
if split_mode == "Time-based (date column)":
    date_candidates = [c for c in Xdf.columns if "date" in str(c).lower() or "time" in str(c).lower()]
    date_col = st.sidebar.selectbox("Date column", options=date_candidates) if date_candidates else None
    if date_col is None:
        st.warning("No date-like column found; using random split.")
        split_mode = "Random (Stratified)"

# Imbalance
st.sidebar.header("⚖️ Imbalance")
use_smote = st.sidebar.checkbox("Use SMOTE (if installed)", value=False)
use_class_weight = st.sidebar.checkbox("Use class_weight='balanced' (where supported)", value=True)

# Models
st.sidebar.header("🤖 Models")
use_lr = st.sidebar.checkbox("Logistic Regression", value=True)
use_rf = st.sidebar.checkbox("Random Forest", value=True)
use_xgb = st.sidebar.checkbox("XGBoost (if available)", value=True)
cv_folds = st.sidebar.slider("Cross-validation folds", 3, 10, 5)

# Preprocess & split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = None, None, None, None

if split_mode.startswith("Time-based") and date_col is not None:
    dates = pd.to_datetime(Xdf[date_col], errors="coerce")
    order = dates.argsort(kind="mergesort")
    X_sorted, y_sorted = Xdf.iloc[order], y.iloc[order]
    n_test = max(1, int(np.floor(len(X_sorted) * test_size)))
    X_train, X_test = X_sorted.iloc[:-n_test, :], X_sorted.iloc[-n_test:, :]
    y_train, y_test = y_sorted.iloc[:-n_test], y_sorted.iloc[-n_test:]
else:
    X_train, X_test, y_train, y_test = train_test_split(
        Xdf, y, test_size=test_size, random_state=42, stratify=y
    )

pre = make_preprocessor(num_cols, cat_cols)
pre.fit(X_train)
Xtr = pre.transform(X_train)
Xte = pre.transform(X_test)
feature_names = get_feature_names(pre)

# Train & compare
models = build_models(
    use_lr=use_lr, use_rf=use_rf, use_xgb=use_xgb,
    y_train=y_train.values, use_class_weight=use_class_weight, use_smote=use_smote
)
if not models:
    st.error("Select at least one model.")
    st.stop()

leaderboard, fitted = fit_and_compare(models, Xtr, y_train.values, Xte, y_test.values, cv_folds=cv_folds)
if leaderboard.empty:
    st.error("All trainings failed.")
    st.stop()

st.subheader("🏁 Model Comparison")
st.dataframe(leaderboard, use_container_width=True)
best_name = leaderboard.iloc[0]["Model"]
model_name = st.selectbox("Primary model for explainability", options=list(fitted.keys()),
                          index=list(fitted.keys()).index(best_name))
model = fitted[model_name]

# Threshold tuning
st.subheader("🎚 Threshold Tuning & Diagnostic Curves")
y_proba = model.predict_proba(Xte)[:, 1]
with st.container():
    left, right = st.columns([2, 1])
    with right:
        bests = find_best_threshold(y_test.values, y_proba)
        st.caption("Suggested thresholds:")
        st.write(f"• Best F1 @ **{bests['best_f1_thr']:.2f}** (F1≈{bests['best_f1']:.3f})")
        st.write(f"• Best Youden’s J @ **{bests['best_j_thr']:.2f}**")

        thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
        if st.button("Use Best F1"):
            thr = bests["best_f1_thr"]
        if st.button("Use Best Youden’s J"):
            thr = bests["best_j_thr"]

        met = compute_metrics(y_test.values, y_proba, threshold=thr)
        st.write({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in met.items() if k != "Confusion"})

    with left:
        st.plotly_chart(fig_confusion(met["Confusion"]), use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(fig_roc(y_test.values, y_proba), use_container_width=True)
with c2:
    st.plotly_chart(fig_pr(y_test.values, y_proba), use_container_width=True)
with c3:
    st.plotly_chart(fig_calibration(y_test.values, y_proba), use_container_width=True)

# Explainability
st.subheader("🧠 Explainability")
expl_ok = try_shap_summary(model, Xte, feature_names, max_display=25)
feat_imp_df = None
if not expl_ok:
    feat_imp_df = permutation_importance_df(model, Xte, y_test.values, feature_names)
    plot_perm_importance(feat_imp_df)

# Segment view
st.subheader("🧩 Segment Analysis")
cat_for_seg = [c for c in Xdf.columns if Xdf[c].dtype == "object" or str(Xdf[c].dtype).startswith("category")]
if cat_for_seg:
    seg_col = st.selectbox("Categorical column", options=cat_for_seg)
    tmp = pd.DataFrame({"seg": X_test[seg_col].values, "y": y_test.values, "p": y_proba})
    seg = tmp.groupby("seg").agg(Count=("y", "size"), ChurnRate=("y", "mean"), AvgScore=("p", "mean")).reset_index()
    seg["ChurnRate"] = seg["ChurnRate"].round(3)
    st.dataframe(seg.sort_values("ChurnRate", ascending=False), use_container_width=True)
    st.plotly_chart(px.bar(seg, x="seg", y="ChurnRate", hover_data=["Count", "AvgScore"],
                           title=f"Churn rate by {seg_col}"), use_container_width=True)
else:
    st.info("No categorical feature available for segmentation.")

# Cohort view
st.subheader("📆 Cohort-over-Time")
date_candidates = [c for c in Xdf.columns if "date" in str(c).lower() or "time" in str(c).lower()]
if date_candidates:
    dcol = st.selectbox("Date column", options=date_candidates)
    dts = pd.to_datetime(X_test[dcol], errors="coerce")
    cohort = pd.DataFrame({"month": dts.dt.to_period("M").dt.to_timestamp(), "y": y_test.values}).dropna()
    if not cohort.empty:
        out = cohort.groupby("month")["y"].mean().reset_index(name="ChurnRate")
        st.plotly_chart(px.line(out, x="month", y="ChurnRate", markers=True,
                                title="Churn over time (Test set)"), use_container_width=True)
    else:
        st.info("Not enough valid dates to plot.")
else:
    st.info("No date-like column detected for cohort view.")

# Predictions & exports
st.subheader("📄 Predictions & Reports")
thr_final = float(thr)
y_pred = (y_proba >= thr_final).astype(int)

preds_df = pd.DataFrame({
    "Pred_Proba": y_proba,
    "Pred_Label": y_pred,
    "True_Label": y_test.values
}).reset_index(drop=True)
context_cols = df.loc[X_test.index].reset_index(drop=True)
preds_out = pd.concat([context_cols, preds_df], axis=1)

st.download_button("⬇️ Download Predictions CSV",
                   data=preds_out.to_csv(index=False).encode("utf-8"),
                   file_name="churn_predictions.csv", mime="text/csv")

metrics_row = {"Model": model_name, "Threshold": thr_final}
cur_metrics = compute_metrics(y_test.values, y_proba, threshold=thr_final)
for k, v in cur_metrics.items():
    if k != "Confusion":
        metrics_row[k] = v
metrics_tbl = pd.DataFrame([metrics_row])

excel_bytes = to_excel_bytes(metrics_tbl, preds_out, feat_imp_df)
st.download_button("⬇️ Download Excel Report",
                   data=excel_bytes,
                   file_name="churn_report.xlsx",
                   mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))

with st.expander("ℹ️ Notes & Tips"):
    st.markdown("""
- **Target must be binary**. Use the sidebar to set which value means **churn** (positive class).
- **Imbalance**: prefer class-weight; SMOTE is optional (requires `imbalanced-learn`).
- **Explainability**: SHAP (if available) → permutation importance fallback.
- **No default datasets**: your uploaded CSV is the only source used.
""")

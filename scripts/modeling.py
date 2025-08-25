import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional libs
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    IMB_AVAILABLE = True
except Exception:
    IMB_AVAILABLE = False

def class_weight_for_pos(y: np.ndarray) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg / max(pos, 1)) if pos else 1.0

def build_models(use_lr: bool, use_rf: bool, use_xgb: bool, y_train: np.ndarray,
                 use_class_weight: bool, use_smote: bool) -> Dict[str, object]:
    models = {}
    if use_lr:
        models["LogisticRegression"] = LogisticRegression(
            max_iter=1000, solver="lbfgs",
            class_weight=("balanced" if use_class_weight else None)
        )
    if use_rf:
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42,
            class_weight=("balanced" if use_class_weight else None),
            min_samples_leaf=2
        )
    if use_xgb and XGB_AVAILABLE:
        spw = class_weight_for_pos(y_train)
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", n_jobs=-1, random_state=42,
            scale_pos_weight=spw if not use_smote else 1.0
        )
    return models

def _maybe_smote(X, y, use_smote: bool):
    if use_smote and IMB_AVAILABLE:
        try:
            sm = SMOTE(random_state=42)
            return sm.fit_resample(X, y)
        except Exception:
            return X, y
    return X, y

def fit_and_compare(models: Dict[str, object], Xtr, ytr, Xte, yte, cv_folds: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows = []
    fitted = {}
    for name, clf in models.items():
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_auc = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
            cv_f1 = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="f1", n_jobs=-1)
            clf.fit(Xtr, ytr)
            fitted[name] = clf
            y_proba = clf.predict_proba(Xte)[:, 1]
            from .evaluation import compute_metrics
            m = compute_metrics(yte, y_proba, threshold=0.5)
            rows.append({
                "Model": name,
                "CV ROC_AUC (mean)": float(np.mean(cv_auc)),
                "CV F1 (mean)": float(np.mean(cv_f1)),
                "Test ROC_AUC": float(m["ROC_AUC"]),
                "Test PR_AUC": float(m["PR_AUC"]),
                "Test F1@0.5": float(m["F1"]),
                "Test Recall@0.5": float(m["Recall"]),
                "Test Precision@0.5": float(m["Precision"]),
                "Test Accuracy@0.5": float(m["Accuracy"])
            })
        except Exception as e:
            # swallow failure & continue
            pass
    leaderboard = pd.DataFrame(rows).sort_values("Test ROC_AUC", ascending=False) if rows else pd.DataFrame()
    return leaderboard, fitted

def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray):
    from sklearn.metrics import precision_recall_curve, roc_curve
    prec, rec, thr_pr = precision_recall_curve(y_true, y_proba)
    f1s = (2 * prec * rec / np.clip(prec + rec, 1e-9, None))
    idx = int(np.nanargmax(f1s))
    best_f1_thr = float(thr_pr[min(idx, len(thr_pr)-1)])
    fpr, tpr, thr_roc = roc_curve(y_true, y_proba)
    j = tpr - fpr
    jidx = int(np.nanargmax(j))
    return {"best_f1_thr": best_f1_thr, "best_f1": float(f1s[idx]), "best_j_thr": float(thr_roc[jidx])}

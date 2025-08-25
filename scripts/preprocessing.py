from typing import List, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from .utils import detect_target as _detect_target, id_like_cols as _id_like_cols

detect_target = _detect_target
id_like_cols = _id_like_cols

def infer_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num = [c for c in df.columns if c not in cat]
    return num, cat

def _compat_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", _compat_ohe())])
    pre = ColumnTransformer(
        [("num", numeric, numeric_features),
         ("cat", categorical, categorical_features)],
        remainder="drop"
    )
    return pre

def get_feature_names(pre: ColumnTransformer):
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            names.extend(list(cols) if isinstance(cols, (list, tuple)) else list(cols))
        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            try:
                ohe_names = list(ohe.get_feature_names_out(cols))
            except Exception:
                ohe_names = list(ohe.get_feature_names(cols))
            names.extend(ohe_names)
    return names

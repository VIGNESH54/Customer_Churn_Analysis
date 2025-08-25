import pandas as pd
from typing import List

def id_like_cols(df: pd.DataFrame) -> List[str]:
    ret = []
    for c in df.columns:
        cl = str(c).lower()
        if cl.endswith("id") or cl in {"id", "customerid", "customer id", "rownumber", "surname"}:
            ret.append(c)
    return ret

def detect_target(df: pd.DataFrame):
    candidates = ["churn", "Churn", "CHURN", "Exited", "Attrition", "is_churn", "target"]
    for c in candidates:
        if c in df.columns:
            return c
    # heuristic: any boolean-like column
    for c in df.columns:
        s = df[c].dropna()
        if s.isin([0, 1, True, False]).all():
            return c
    return None

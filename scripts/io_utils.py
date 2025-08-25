import io
import pandas as pd

def to_excel_bytes(metrics_table: pd.DataFrame, preds: pd.DataFrame, feat_imp=None) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        metrics_table.to_excel(writer, sheet_name="Metrics", index=False)
        preds.to_excel(writer, sheet_name="Predictions", index=False)
        if feat_imp is not None:
            feat_imp.to_excel(writer, sheet_name="Feature_Importance", index=False)
    buffer.seek(0)
    return buffer.read()

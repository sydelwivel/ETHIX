# backend/explain.py
"""
Explainability helpers.
Tries to use SHAP if available; otherwise falls back to sklearn permutation importance.
"""

import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

from sklearn.inspection import permutation_importance

def explain_model(model, X, y=None, top_k=10):
    """
    Returns a simple explanation object:
      - if shap available: returns shap_values and feature_names
      - else: returns permutation importance dict
    """
    X_df = pd.DataFrame(X).reset_index(drop=True)
    feat_names = list(X_df.columns)
    if _HAS_SHAP:
        try:
            explainer = shap.Explainer(model.predict, X_df)
            shap_values = explainer(X_df)
            # return mean absolute shap for each feature
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            feat_imp = dict(sorted(zip(feat_names, mean_abs), key=lambda x: x[1], reverse=True)[:top_k])
            return {"method": "shap", "feature_importance": feat_imp}
        except Exception:
            pass

    # fallback: permutation importance
    try:
        res = permutation_importance(model, X_df, y, n_repeats=10, random_state=42, n_jobs=1)
        importances = dict(zip(feat_names, res.importances_mean))
        feat_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return {"method": "permutation", "feature_importance": feat_imp}
    except Exception:
        # final fallback: return empty
        return {"method": "none", "feature_importance": {}}

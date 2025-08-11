# backend/audit.py
"""
Fairness and proxy-bias utilities.

- run_fairness_audit: returns Demographic Parity Difference and Equalized Odds Difference.
- detect_proxy_bias: returns top N proxy features with plain-language explanation and strength.
"""

import numpy as np
import pandas as pd

from collections import defaultdict

# try to use sklearn/scipy for statistical tests when available
try:
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import pointbiserialr, chi2_contingency
    _HAS_STATS = True
except Exception:
    _HAS_STATS = False
    from sklearn.preprocessing import LabelEncoder  # sklearn usually available

def _ensure_series(x):
    if isinstance(x, pd.Series):
        return x.reset_index(drop=True)
    return pd.Series(x).reset_index(drop=True)

def run_fairness_audit(y_true, y_pred, sensitive_features):
    """
    Compute fairness metrics:
      - Demographic Parity Difference: difference in positive outcome rates across groups (max - min).
      - Equalized Odds Difference: difference in true positive rates across groups (max - min).
    Works for binary targets (0/1). If multiclass sensitive attribute, uses group-wise max/min.
    Returns dict with metric name -> float.
    """
    y_true = _ensure_series(y_true).astype(int)
    y_pred = _ensure_series(y_pred).astype(int)
    s = _ensure_series(sensitive_features).fillna("missing")

    # ensure lengths align
    n = min(len(y_true), len(y_pred), len(s))
    y_true = y_true.iloc[:n]
    y_pred = y_pred.iloc[:n]
    s = s.iloc[:n]

    groups = list(pd.Categorical(s).categories)
    if len(groups) < 1:
        return {"Demographic Parity Difference": 0.0, "Equalized Odds Difference": 0.0}

    # Demographic parity: positive rate per group
    pos_rates = []
    for g in groups:
        mask = (s == g)
        if mask.sum() == 0:
            pos_rates.append(0.0)
        else:
            pos_rates.append(float(y_pred[mask].mean()))
    dp_diff = float(max(pos_rates) - min(pos_rates))

    # Equalized odds: compare TPR (true positive rate) across groups; take max-min
    tprs = []
    fprs = []
    for g in groups:
        mask = (s == g)
        if mask.sum() == 0:
            tprs.append(0.0); fprs.append(0.0); continue
        yt = y_true[mask]
        yp = y_pred[mask]
        # True Positive Rate = TP / P
        positives = (yt == 1).sum()
        negatives = (yt == 0).sum()
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tpr = tp / positives if positives > 0 else 0.0
        fpr = fp / negatives if negatives > 0 else 0.0
        tprs.append(tpr); fprs.append(fpr)

    # define equalized odds diff as average of max-min differences in TPR and FPR
    tpr_diff = max(tprs) - min(tprs) if tprs else 0.0
    fpr_diff = max(fprs) - min(fprs) if fprs else 0.0
    eo_diff = float((tpr_diff + fpr_diff) / 2.0)

    return {
        "Demographic Parity Difference": dp_diff,
        "Equalized Odds Difference": eo_diff
    }


# -------------------------
# Proxy bias detection
# -------------------------
def _cramers_v(confusion_matrix):
    """Cramér's V statistic for categorical-categorical association."""
    try:
        chi2, p, dof, ex = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        # bias correction
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min(kcorr - 1, rcorr - 1)
        if denom <= 0:
            return 0.0
        return float(np.sqrt(phi2corr / denom))
    except Exception:
        return 0.0

def detect_proxy_bias(df, sensitive_col, top_n=5, threshold_mild=0.1, threshold_strong=0.2):
    """
    Detect proxy bias:
      - For numeric features: use point-biserial correlation (abs value).
      - For categorical features: use Cramér's V.
    Returns a list of dicts (top_n) with fields:
      - column, strength (0..1), reason (string), suggestion (string), risk_level
    """
    df = df.copy()
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not in DataFrame")

    s = df[sensitive_col].fillna("missing")
    results = []

    # encode sensitive for numeric tests
    try:
        le = LabelEncoder()
        s_enc = le.fit_transform(s)
    except Exception:
        s_enc = pd.factorize(s)[0]

    for col in df.columns:
        if col == sensitive_col:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                # numeric vs sensitive (encoded)
                if _HAS_STATS:
                    try:
                        corr, p = pointbiserialr(df[col].astype(float), s_enc)
                        strength = float(abs(corr))
                    except Exception:
                        # fallback simple Pearson
                        strength = float(abs(np.corrcoef(df[col].astype(float), s_enc)[0,1]))
                else:
                    # fallback correlation
                    try:
                        strength = float(abs(np.corrcoef(df[col].astype(float), s_enc)[0,1]))
                    except Exception:
                        strength = 0.0
                reason = f"Numeric feature '{col}' has correlation {strength:.3f} with sensitive attribute."
            else:
                # categorical vs categorical -> cramers v
                confusion = pd.crosstab(df[col].fillna("missing"), s)
                strength = _cramers_v(confusion.values)
                reason = f"Categorical feature '{col}' has association (Cramér's V) {strength:.3f} with sensitive attribute."
        except Exception as e:
            strength = 0.0
            reason = f"Could not compute correlation for '{col}': {e}"

        # risk level & suggestion
        if strength >= threshold_strong:
            risk_level = "Strong Proxy Risk"
            suggestion = "High correlation — consider removing, masking, or transforming this column (e.g., bucket/aggregate)."
        elif strength >= threshold_mild:
            risk_level = "Mild Proxy Risk"
            suggestion = "Moderate correlation — review for indirect bias; consider transformation."
        else:
            risk_level = "Low Risk"
            suggestion = "Low correlation — unlikely to be a proxy."

        results.append({
            "column": col,
            "strength": float(round(strength, 4)),
            "reason": reason,
            "suggestion": suggestion,
            "risk_level": risk_level
        })

    # sort descending by strength and return top_n
    results_sorted = sorted(results, key=lambda x: x["strength"], reverse=True)
    return results_sorted[:top_n]

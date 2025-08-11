# backend/simulator.py
"""
Simulate biased predictions by flipping predicted labels for a group
based on bias_strength. Also compute fairness metrics for the simulated preds.
"""

import numpy as np
import pandas as pd
from audit import run_fairness_audit
from sklearn.base import is_classifier

def simulate_bias_effect(df, target_col, sensitive_col, bias_strength=0.0, base_model_pred=None):
    """
    Parameters:
      df: DataFrame
      target_col: name of target column (string)
      sensitive_col: name of sensitive attribute column (string)
      bias_strength: float in [-1,1]. Positive means favor first group (as ordered by unique values),
                    negative means favor second group.
      base_model_pred: optional base predictions array (0/1). If None, uses true label as baseline.
    Returns:
      preds_sim (pd.Series), metrics (dict of fairness metrics)
    """
    df = df.reset_index(drop=True)
    y = pd.Series(df[target_col]).reset_index(drop=True)
    if base_model_pred is None:
        preds = y.copy().apply(lambda x: 1 if x == y.mode().iloc[0] else 0)  # naive baseline
        # simpler: just use y itself
        preds = y.copy().apply(lambda x: 1 if x == y.max() else 0)
    else:
        preds = pd.Series(base_model_pred).astype(int).reset_index(drop=True)

    s = pd.Series(df[sensitive_col]).astype(str).reset_index(drop=True)
    groups = s.unique()
    if len(groups) < 2:
        # nothing to simulate
        metrics = run_fairness_audit(y, preds, s)
        return preds, metrics

    # Define first two groups ordering
    g0, g1 = groups[0], groups[1]

    # Compute flip probabilities
    # bias_strength ∈ [-1,1]. When positive, we increase positive rate for g0, decrease for g1.
    # We'll flip negatives->positives in g0 with probability p_pos and flip positives->negatives in g1 with p_neg
    p_scale = abs(bias_strength)
    if bias_strength > 0:
        p_g0 = 0.0 + p_scale * 0.5  # up to 0.5 flipping negatives->positives
        p_g1 = 0.0 + p_scale * 0.5  # up to 0.5 flipping positives->negatives
    else:
        p_g0 = 0.0 + p_scale * 0.5
        p_g1 = 0.0 + p_scale * 0.5

    preds_sim = preds.copy().astype(int)
    rng = np.random.RandomState(42)
    for i, grp in s.items():
        val = s.iloc[i]
        if val == g0:
            # if bias_strength positive -> favor g0: flip some negatives to positives
            if bias_strength > 0:
                if preds_sim.iloc[i] == 0 and rng.rand() < p_g0:
                    preds_sim.iloc[i] = 1
            else:
                # negative bias against g0: flip some positives to negatives
                if preds_sim.iloc[i] == 1 and rng.rand() < p_g0:
                    preds_sim.iloc[i] = 0
        elif val == g1:
            if bias_strength > 0:
                # bias against g1: flip positives to negatives
                if preds_sim.iloc[i] == 1 and rng.rand() < p_g1:
                    preds_sim.iloc[i] = 0
            else:
                # bias favoring g1 when bias_strength negative
                if preds_sim.iloc[i] == 0 and rng.rand() < p_g1:
                    preds_sim.iloc[i] = 1
        else:
            # For groups beyond first two, leave as-is
            continue

    metrics = run_fairness_audit(y, preds_sim, s)
    return preds_sim, metrics

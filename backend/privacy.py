# backend/privacy.py
"""
Privacy audit helpers.

- reidentifiable_features(df, top_n=5): returns top_n risky combinations (pairs/triples)
  with human-readable reason and suggested action.
- suggest_masking(...) returns suggestions (kept for backward compatibility).
"""

import pandas as pd
import itertools
import math

def reidentifiable_features(df, top_n=5, combos=(2,3), uniqueness_threshold=0.25, sample_limit=20000):
    """
    Identify combinations of columns that create high uniqueness in dataset rows.
    Parameters:
      - df: pandas DataFrame
      - top_n: number of top risky combinations to return
      - combos: tuple/list of combination sizes to check (2,3 recommended)
      - uniqueness_threshold: fraction above which combination considered risky (0..1)
      - sample_limit: if dataset larger than this, sample for speed
    Returns:
      List of dicts: {combination: [col1,...], unique_ratio: 0.x, reason: str, suggestion: str}
    Notes:
      - We check pair/triple combos; this is a heuristic, not a formal k-anonymity check.
      - uniqueness_threshold default lowered to 0.25 for real-world datasets to surface meaningful combos.
    """
    if df.shape[0] == 0:
        return []

    nrows = df.shape[0]
    if nrows > sample_limit:
        df_sample = df.sample(sample_limit, random_state=42)
    else:
        df_sample = df.copy()

    cols = df_sample.columns.tolist()
    risky = []

    for r in combos:
        if r > len(cols):
            continue
        for combo in itertools.combinations(cols, r):
            subset = df_sample[list(combo)].dropna()
            if subset.shape[0] == 0:
                continue
            unique_count = subset.drop_duplicates().shape[0]
            total = subset.shape[0]
            unique_ratio = unique_count / total
            if unique_ratio >= uniqueness_threshold:
                reason = (f"The combination {list(combo)} uniquely identifies about "
                          f"{unique_ratio*100:.1f}% of sampled rows.")
                suggestion = ("Consider removing one column from the combination, aggregating values "
                              "(e.g. age -> age group), or masking precise values (e.g. bucket incomes).")
                risky.append({
                    "combination": list(combo),
                    "unique_ratio": float(round(unique_ratio, 4)),
                    "reason": reason,
                    "suggestion": suggestion
                })

    # sort by unique_ratio desc and take top_n
    risky_sorted = sorted(risky, key=lambda x: x["unique_ratio"], reverse=True)
    return risky_sorted[:top_n]


def suggest_masking(risky_list):
    """
    Backwards-compatible helper: returns list of suggestion strings.
    """
    return [r["suggestion"] for r in risky_list]

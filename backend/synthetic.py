# backend/synthetic.py
"""
Simple synthetic fair data generator.
Goal: given numeric X (pandas DataFrame without target column) and y (Series),
and a cat_mask (list of indices of categorical columns in the full original X),
return a new balanced dataset with approx equal positive rates across sensitive groups
by performing resampling and slight noise addition for numeric columns.
This is a simple implementation (not a GAN or complex synthesizer).
"""

import numpy as np
import pandas as pd

def _numeric_noisy_copy(df, noise_scale=0.01):
    arr = df.to_numpy(dtype=float)
    noise = np.random.normal(scale=noise_scale * (np.nanstd(arr, axis=0) + 1e-6), size=arr.shape)
    arr = arr + noise
    return pd.DataFrame(arr, columns=df.columns)

def generate_fair_data(X_numeric, y, cat_mask=None, sensitive_column=None, desired_rate=None):
    """
    Parameters:
      X_numeric: pandas DataFrame with numeric columns only (app expects select_dtypes(exclude='object'))
      y: pandas Series target (not encoded)
      cat_mask: list of indices (in original full X) which were categorical (not used here, kept for API)
      sensitive_column: optional series (if provided, will balance across groups)
      desired_rate: float desired positive rate per group (if None, uses global mean)
    Returns:
      X_new (DataFrame), y_new (Series)
    """
    X_numeric = X_numeric.copy().reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    if sensitive_column is None:
        # simple balancing by up/down sampling to equalize classes overall
        pos = X_numeric[y == y.iloc[0]]
        # fallback simple SMOTE-like: duplicate minority class with small noise
        counts = y.value_counts()
        if len(counts) == 1:
            return X_numeric, y
        maj_label = counts.idxmax()
        min_label = counts.idxmin()
        maj_df = X_numeric[y == maj_label]
        min_df = X_numeric[y == min_label]
        target_n = len(maj_df)
        copies = []
        while len(copies) < target_n - len(min_df):
            sample = min_df.sample(min( len(min_df), target_n - len(min_df)), replace=True)
            copies.append(_numeric_noisy_copy(sample))
        if copies:
            aug = pd.concat([min_df] + copies, ignore_index=True)
            X_new = pd.concat([maj_df, aug], ignore_index=True).sample(frac=1.0).reset_index(drop=True)
            y_new = pd.Series([maj_label]*len(maj_df) + [min_label]*len(aug)).reset_index(drop=True).sample(frac=1.0).reset_index(drop=True)
            return X_new, y_new
        else:
            return X_numeric, y

    # If sensitive column provided, resample within each sensitive group so positive rate ~desired_rate
    s = pd.Series(sensitive_column).reset_index(drop=True)
    df = X_numeric.copy()
    df["_y"] = y
    df["_s"] = s
    if desired_rate is None:
        desired_rate = y.mean()
    result_frames = []
    for grp in df["_s"].unique():
        sub = df[df["_s"] == grp].copy()
        if sub.shape[0] == 0:
            continue
        pos_sub = sub[sub["_y"] == 1]
        neg_sub = sub[sub["_y"] == 0]
        total_needed = len(sub)
        pos_needed = int(round(desired_rate * total_needed))
        # ensure at least 1 pos/neg to avoid degenerate groups
        pos_needed = max(1, pos_needed)
        # create pos and neg samples
        if len(pos_sub) >= pos_needed:
            pos_frame = pos_sub.sample(pos_needed, replace=False)
        else:
            # upsample pos_sub with noise
            copies = []
            while len(copies) + len(pos_sub) < pos_needed:
                sample = pos_sub.sample(min(len(pos_sub), pos_needed - len(pos_sub) - len(copies)), replace=True)
                copies.append(_numeric_noisy_copy(sample.drop(columns=["_y","_s"])))
            if copies:
                pos_aug = pd.concat([pos_sub.drop(columns=["_y","_s"])] + copies, ignore_index=True)
                pos_aug["_y"] = 1
                pos_aug["_s"] = grp
                pos_frame = pos_aug
            else:
                pos_frame = pos_sub
        # fill remainder with negs
        neg_needed = total_needed - len(pos_frame)
        if len(neg_sub) >= neg_needed and neg_needed > 0:
            neg_frame = neg_sub.sample(neg_needed, replace=False)
        elif neg_needed > 0:
            # upsample negs
            neg_copies = []
            while len(neg_copies) + len(neg_sub) < neg_needed:
                sample = neg_sub.sample(min(len(neg_sub), neg_needed - len(neg_sub) - len(neg_copies)), replace=True)
                neg_copies.append(_numeric_noisy_copy(sample.drop(columns=["_y","_s"])))
            if neg_copies:
                neg_aug = pd.concat([neg_sub.drop(columns=["_y","_s"])] + neg_copies, ignore_index=True)
                neg_aug["_y"] = 0
                neg_aug["_s"] = grp
                neg_frame = neg_aug
            else:
                neg_frame = neg_sub
        else:
            neg_frame = pd.DataFrame(columns=sub.columns)

        combined = pd.concat([pos_frame, neg_frame], ignore_index=True, sort=False)
        result_frames.append(combined)

    if result_frames:
        out = pd.concat(result_frames, ignore_index=True, sort=False).sample(frac=1.0, random_state=42).reset_index(drop=True)
        y_new = out["_y"].astype(int).reset_index(drop=True)
        X_new = out.drop(columns=["_y","_s"], errors='ignore')
        # Ensure numeric types
        X_new = X_new.apply(pd.to_numeric, errors='coerce').fillna(0)
        return X_new, y_new
    else:
        return X_numeric, y

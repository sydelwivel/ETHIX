# backend/retrain.py
"""
Training & evaluation helper with safe preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def _make_pipeline(X):
    # Decide which columns are numeric vs categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="mean"), numeric_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))]), cat_cols))
    preproc = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    return preproc

def train_and_evaluate(X_train, y_train, X_test, y_test, model=None):
    """
    X_train, X_test: pandas DataFrame
    y_train, y_test: pandas Series or 1d arrays (binary 0/1)
    Returns: model, accuracy, predictions (np.array)
    """
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    y_train = pd.Series(y_train).astype(int).reset_index(drop=True)
    y_test = pd.Series(y_test).astype(int).reset_index(drop=True)

    preproc = _make_pipeline(X_train)
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline([("preproc", preproc), ("clf", model)])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    return pipeline, acc, pd.Series(preds, index=X_test.index)

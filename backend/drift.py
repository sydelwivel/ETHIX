# backend/drift.py
def detect_drift(prev_score, current_score, threshold=0.05):
    """
    Returns True if absolute difference exceeds threshold.
    """
    try:
        return abs(float(prev_score) - float(current_score)) > float(threshold)
    except Exception:
        return False

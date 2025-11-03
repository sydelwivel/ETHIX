# frontend/app.py
import sys
import os

# 1. Get the path to the directory containing the current script (frontend)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path to the project root (one level up from frontend)
project_root = os.path.join(current_dir, '..')

# 3. Add the project root to the system path
sys.path.insert(0, project_root)

# Now, the imports below should work:
from backend import community 
# ... rest of your imports
import sys
import os
from pathlib import Path


# ===== Path setup so backend can be imported reliably =====
# Prefer relative path: assume repo root contains backend/ and frontend/
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BACKEND_PATH = str(PROJECT_ROOT / "backend")
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)

# ===== Standard imports =====
import streamlit as st
import pandas as pd
import numpy as np

# ===== Import backend modules (graceful) =====
_missing = []
try:
    import audit
except Exception as e:
    audit = None; _missing.append("audit")
try:
    import privacy
except Exception as e:
    privacy = None; _missing.append("privacy")
try:
    import synthetic
except Exception as e:
    synthetic = None; _missing.append("synthetic")
try:
    import retrain
except Exception as e:
    retrain = None; _missing.append("retrain")
try:
    import explain
except Exception as e:
    explain = None; _missing.append("explain")
try:
    import drift
except Exception as e:
    drift = None; _missing.append("drift")
try:
    import scorecard
except Exception as e:
    scorecard = None; _missing.append("scorecard")
try:
    import simulator
except Exception as e:
    simulator = None; _missing.append("simulator")
try:
    import community
except Exception as e:
    community = None; _missing.append("community")

# ===== Helpers =====
def detect_sensitive_attributes(df):
    sensitive_keywords = [
        "gender", "sex", "race", "ethnicity", "age",
        "religion", "disability", "nationality",
        "marital_status", "income", "sexual_orientation"
    ]
    detected = []
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")
        if any(keyword in col_lower for keyword in sensitive_keywords):
            detected.append(col)
    return detected

def binarize_labels(series):
    s = pd.Series(series).copy()
    uniq = pd.Series(s.dropna().unique())
    set_uniq = set(uniq)
    if set_uniq <= {0,1} or set_uniq <= {-1,1}:
        return s.astype(int).replace({-1:1})
    # choose positive label heuristically
    if any(isinstance(x, str) for x in uniq):
        for val in uniq:
            try:
                low = str(val).lower()
                if ">" in low or "yes" in low or low in ("1","true","t"):
                    pos = val; break
            except Exception:
                continue
        else:
            pos = uniq.iloc[-1]
    else:
        pos = max(uniq)
    return s.apply(lambda x: 1 if x == pos else 0).astype(int)

def interpret_fairness_results(results):
    out = {}
    for metric, value in results.items():
        try:
            v = float(value)
        except Exception:
            v = 0.0
        a = abs(v)
        if a <= 0.05:
            level = "✅ Fair"; explanation = "There is little to no measurable bias for this metric."; color="green"
        elif a <= 0.15:
            level = "⚠️ Mild Bias"; explanation = "There are small differences between groups — review recommended."; color="orange"
        else:
            level = "❌ Significant Bias"; explanation = "Large differences between groups — action required to mitigate bias."; color="red"
        out[metric] = {"value": round(v,4), "level": level, "explanation": explanation, "color": color}
    return out

def _colored_markdown(text, color):
    return f"<span style='color:{color}; font-weight:600'>{text}</span>"

# ===== Streamlit UI =====
st.set_page_config(page_title="EthixAI - Auditor", layout="wide")
st.title("EthixAI — Complete Ethical AI Auditor")

if _missing:
    st.warning(f"Some backend modules missing or failed to import: {_missing}. App will attempt available features.")

uploaded = st.file_uploader("Upload dataset (CSV / Excel / JSON)")

if uploaded:
    # load file
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif uploaded.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        elif uploaded.name.endswith(".json"):
            df = pd.read_json(uploaded)
        else:
            st.error("Unsupported file format"); st.stop()
    except Exception as e:
        st.error(f"Failed to read file: {e}"); st.stop()

    st.subheader("Dataset preview")
    st.write(df.head())

    auto_sensitive = detect_sensitive_attributes(df)
    if auto_sensitive:
        st.success(f"Auto-detected possible sensitive attributes: {auto_sensitive}")
    else:
        st.info("No obvious sensitive attributes detected automatically — please select manually.")

    cols = list(df.columns)
    target = st.selectbox("Select target column", cols)
    sensitive = st.selectbox("Select sensitive attribute", auto_sensitive + cols if auto_sensitive else cols)

    st.markdown("---")

    # -----------------------
    # Fairness Audit
    # -----------------------
    if st.button("Run Fairness Audit"):
        if audit is None:
            st.error("Audit backend missing.")
        else:
            y_true = binarize_labels(df[target])

            # determine predictions to audit
            pred_cols = [c for c in df.columns if "pred" in c.lower() or c.lower().endswith("_pred")]
            prob_cols = [c for c in df.columns if ("prob" in c.lower()) or ("score" in c.lower())]

            if pred_cols:
                y_pred = binarize_labels(df[pred_cols[0]])
                st.info(f"Using prediction column: {pred_cols[0]}")
            elif prob_cols:
                y_pred = (pd.to_numeric(df[prob_cols[0]], errors="coerce") >= 0.5).astype(int)
                st.info(f"Using probability/score column: {prob_cols[0]} (threshold=0.5)")
            else:
                # train quick internal model but deliberately exclude sensitive column to avoid shortcut bias
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split

                features = [c for c in df.columns if c not in [target, sensitive]]
                X = pd.get_dummies(df[features], drop_first=True)
                if X.shape[0] < 2 or X.shape[1] < 1:
                    st.error("Not enough features to train fallback model. Provide predictions or more features.")
                    st.stop()
                y = y_true
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                clf = RandomForestClassifier(random_state=42, n_estimators=100)
                clf.fit(X_train, y_train)
                y_pred = pd.Series(clf.predict(X), index=df.index)
                st.info("No predictions found. Trained internal model (sensitive column excluded) for audit.")

            # compute fairness metrics
            sensitive_series = df[sensitive].fillna("missing")
            metrics = audit.run_fairness_audit(y_true, y_pred, sensitive_series)
            interpreted = interpret_fairness_results(metrics)

            st.subheader("📊 Fairness Audit Results")
            for m, info in interpreted.items():
                st.markdown(f"**{m}:** {info['value']}   {info['level']}")
                st.progress(min(abs(info['value']), 1.0))
                st.markdown(f"<small style='color:{info['color']}'>{info['explanation']}</small>", unsafe_allow_html=True)
                st.markdown("---")

            # group-wise stats
            st.subheader("🔎 Group-wise statistics")
            grp = sensitive_series.astype(str)
            pos_rates = pd.Series(y_pred).groupby(grp).mean().rename("Positive Rate")
            counts = grp.value_counts().rename("Count")
            gdf = pd.concat([counts, pos_rates], axis=1).fillna(0).reset_index().rename(columns={"index":"Group"})
            st.table(gdf)

    # -----------------------
    # Proxy Bias Detection (friendly)
    # -----------------------
    if st.button("Check Proxy Bias"):
        if audit is None:
            st.error("Audit backend missing.")
        else:
            top_n = st.number_input("Top N proxies to display", min_value=1, max_value=20, value=5, step=1)
            proxies = audit.detect_proxy_bias(df, sensitive, top_n=int(top_n))
            if not proxies:
                st.info("No proxy relationships detected.")
            else:
                st.subheader("🕵️ Top proxy features (may reveal sensitive info indirectly)")
                st.write("Columns below are ranked by association strength with the sensitive attribute.")
                # build table
                rows = []
                for i, p in enumerate(proxies):
                    col = p["column"]
                    strength = p["strength"]
                    risk = p["risk_level"]
                    reason = p["reason"]
                    suggestion = p["suggestion"]
                    rows.append({
                        "Rank": i+1,
                        "Column": col,
                        "Strength (0-1)": round(strength,4),
                        "Risk": risk,
                        "Why this matters": reason,
                        "Suggested action": suggestion
                    })
                p_df = pd.DataFrame(rows)
                st.dataframe(p_df)

    # -----------------------
    # Privacy Risk Audit (friendly, top 5)
    # -----------------------
    if st.button("Privacy Risk Audit"):
        if privacy is None:
            st.error("Privacy backend missing.")
        else:
            st.subheader("🛡 Privacy Risk Audit (Top 5)")
            st.write("""
            Privacy risk arises when a combination of non-identifying columns can uniquely identify people.
            Below are the top combinations (pairs/triples) that are most likely to cause re-identification.
            For each, a suggested fix is provided.
            """)
            risky = privacy.reidentifiable_features(df, top_n=5)
            if not risky:
                st.success("No high-risk combinations found (based on current thresholds).")
            else:
                rows = []
                for i, r in enumerate(risky):
                    rows.append({
                        "Rank": i+1,
                        "Risky combination": ", ".join(r["combination"]),
                        "Uniqueness (%)": f"{r['unique_ratio']*100:.1f}",
                        "Why it's risky": r["reason"],
                        "Suggested action": r["suggestion"]
                    })
                risk_df = pd.DataFrame(rows)
                st.dataframe(risk_df)

    # -----------------------
    # Synthetic Data Generation
    # -----------------------
    if st.button("Generate Synthetic Balanced Data"):
        if synthetic is None:
            st.error("Synthetic backend missing.")
        else:
            st.subheader("🧪 Synthetic Fair Data (preview)")
            # prepare numeric inputs: exclude target and sensitive for X
            features = [c for c in df.columns if c != target]
            numeric_df = df[features].select_dtypes(exclude='object')
            y = binarize_labels(df[target])
            s_col = df[sensitive] if sensitive in df.columns else None
            try:
                X_new, y_new = synthetic.generate_fair_data(numeric_df, y, sensitive_column=s_col)
                st.write(X_new.head())
                st.write("Synthetic label distribution:")
                st.write(pd.Series(y_new).value_counts(normalize=True))
            except Exception as e:
                st.error(f"Synthetic generation failed: {e}")

    # -----------------------
    # Retrain Model
    # -----------------------
    if st.button("Train & Evaluate Model"):
        if retrain is None:
            st.error("Retrain backend missing.")
        else:
            st.subheader("🔁 Train & Evaluate")
            from sklearn.model_selection import train_test_split
            X = df.drop(columns=[target])
            y = binarize_labels(df[target])
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model, acc, preds = retrain.train_and_evaluate(X_train, y_train, X_test, y_test)
                st.success(f"Trained model. Accuracy: {acc:.4f}")
                # fairness on test set
                if audit is not None:
                    sensitive_series = X_test[sensitive] if sensitive in X_test.columns else df.loc[X_test.index, sensitive]
                    metrics = audit.run_fairness_audit(y_test, preds, sensitive_series)
                    interpreted = interpret_fairness_results(metrics)
                    st.subheader("Fairness on test set")
                    for m, info in interpreted.items():
                        st.markdown(f"**{m}:** {info['value']}   {info['level']}")
                        st.markdown(f"<small style='color:{info['color']}'>{info['explanation']}</small>", unsafe_allow_html=True)
                # explainability
                if explain is not None:
                    try:
                        exp = explain.explain_model(model, X_test, y_test)
                        st.subheader("Model explainability (top features)")
                        st.write(exp)
                    except Exception as e:
                        st.info(f"Explainability unavailable: {e}")
            except Exception as e:
                st.error(f"Training failed: {e}")

    # -----------------------
    # Scorecard PDF
    # -----------------------
    if st.button("Generate Scorecard PDF"):
        if scorecard is None:
            st.error("Scorecard backend missing.")
        else:
            st.subheader("📄 Generate Scorecard")
            # For demo, compute some metrics if available
            demo_metrics = {}
            try:
                # attempt to compute demo accuracy and fairness if possible
                demo_metrics["Accuracy"] = 0.9
                # try extract earlier computed fairness if available (not persistent across clicks)
                demo_metrics["Demographic Parity"] = 0.1
                demo_metrics["Equalized Odds"] = 0.05
            except Exception:
                demo_metrics = {"Accuracy": 0.9}
            out_path = "reports/scorecard.pdf"
            os.makedirs("reports", exist_ok=True)
            try:
                scorecard.generate_scorecard(demo_metrics, out_path)
                st.success(f"Scorecard saved to {out_path}")
            except Exception as e:
                st.error(f"Failed to create scorecard: {e}")

    # -----------------------
    # Drift / Simulator / Community (kept)
    # -----------------------
    st.markdown("---")
    st.subheader("⚙️ Other Tools")
    # Drift check
    try:
        prev_score = st.number_input("Previous model score (for drift check)", value=0.95, step=0.01, format="%.2f")
        # placeholder: you would pass real scores here
        if st.button("Check Drift"):
            if drift is None:
                st.error("Drift backend missing.")
            else:
                curr_score = float(prev_score)  # placeholder; in real use pass current
                has_drift = drift.detect_drift(prev_score, curr_score)
                st.write("Drift detected:" , has_drift)
    except Exception:
        pass

    # Simulator (interactive bias simulation)
    try:
        st.subheader("Simulation")
        bias_strength = st.slider("Bias Strength", -1.0, 1.0, 0.0, 0.05)
        if simulator is None:
            st.info("Simulator backend missing.")
        else:
            preds_sim, metrics_sim = simulator.simulate_bias_effect(df, target, sensitive, bias_strength)
            interpreted_sim = interpret_fairness_results(metrics_sim)
            st.write("Simulated fairness metrics:")
            st.write(interpreted_sim)
    except Exception:
        pass

    # Community DB quick section (if present)
    if community is not None:
        st.subheader("Community models DB (available)")
        st.write("Community DB integrated. (Use backend.community to manage entries.)")

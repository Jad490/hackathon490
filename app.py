import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Risk Predictor — Inference", layout="wide")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
html, body, [data-testid="stAppViewContainer"] {background: radial-gradient(1400px 800px at 10% 10%, #141a2a 0%, #0d1117 40%, #0b0f14 100%) fixed;}
* {font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;}
.block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem;}
.header {display:flex; align-items:center; gap:14px; margin-bottom:8px;}
.logo {font-size:28px; line-height:1;}
.title {font-size:28px; font-weight:800; letter-spacing:0.2px; background: linear-gradient(90deg,#7dd3fc,#a78bfa,#f472b6); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.subtitle {color:#cbd5e1; margin-top:-6px;}
.card {background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px;}
.kpi {display:flex; gap:16px; margin:10px 0 0 0;}
.kpi .item {flex:1; background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:14px;}
.kpi .label {color:#94a3b8; font-size:12px;}
.kpi .value {font-size:22px; font-weight:800;}
hr {border-color: rgba(255,255,255,0.08);}
[data-testid="stFileUploaderDropzone"] {border-radius:16px; background: rgba(255,255,255,0.03) !important; border:1px dashed rgba(255,255,255,0.18) !important;}
[data-testid="stMetricValue"] {font-weight:800;}
.about-box {background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:22px;}
.about-title {font-size:22px; font-weight:800; color:#93c5fd; margin-bottom:10px;}
.about-text {color:#e5e7eb; line-height:1.6;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="header"><div class="logo">🎓</div><div><div class="title">Student Risk Predictor</div><div class="subtitle">Early insight for student support — upload your CSV and get instant risk probabilities.</div></div></div>',
    unsafe_allow_html=True,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "student_risk_model.joblib"
if not MODEL_PATH.exists():
    st.error("Model not found.\n\nRun once:\n1) source .venv/bin/activate\n2) python3 train_model.py")
    st.stop()

clf = joblib.load(MODEL_PATH)

tab_predict, tab_about = st.tabs(["Predict", "About"])

with tab_predict:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload CSV")
    st.caption("Expected columns: gender, race_ethnicity, parent_edu, lunch, test_prep, math_score, reading_score, writing_score")
    file = st.file_uploader("Drag & drop or browse", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if file is None:
        st.info("Upload a CSV to see predictions.")
    else:
        df = pd.read_csv(file)
        mapping = {
            "math score": "math_score", "reading score": "reading_score", "writing score": "writing_score",
            "test preparation course": "test_prep", "test_preparation_course": "test_prep",
            "parental level of education": "parent_edu", "parental_level_of_education": "parent_edu",
            "race/ethnicity": "race_ethnicity",
        }
        new_cols = []
        for c in df.columns:
            c_low = c.strip().lower()
            new_cols.append(mapping.get(c_low, c_low.replace(" ", "_")))
        df.columns = new_cols

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data preview")
        st.dataframe(df.head(15), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        try:
            proba = clf.predict_proba(df)[:, 1]
        except Exception as e:
            st.error(f"Could not score with provided columns. Error: {e}")
            st.stop()

        res = df.copy()
        res["risk_prob"] = proba
        c1, c2, c3 = st.columns([1.2,1,1])
        thr = c1.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, help="Predictions ≥ threshold are flagged as at-risk.")
        res["at_risk_pred"] = (res["risk_prob"] >= thr).astype(int)
        c2.metric("Students", f"{len(res):,}")
        c3.metric("Predicted at-risk", f"{int(res['at_risk_pred'].sum()):,}")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Results")
        st.dataframe(res.head(50), use_container_width=True)
        st.download_button("⬇️ Download predictions (CSV)", res.to_csv(index=False).encode("utf-8"), file_name="student_risk_predictions.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

with tab_about:
    st.markdown('<div class="about-box">', unsafe_allow_html=True)
    st.markdown('<div class="about-title">About Student Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="about-text">'
        'Student Risk Predictor helps schools identify at-risk students early so staff can act sooner. '
        'A pre-trained model scores each student with a probability of risk using academic and context signals. '
        'Teams use these insights to prioritize outreach, personalize support, and measure impact.'
        '<br><br>'
        '<b>How it helps</b><br>'
        '• Early detection to reduce drop-offs and failures<br>'
        '• Data-informed interventions instead of guesswork<br>'
        '• Simple CSV workflow and downloadable results for SIS/CRM import'
        '<br><br>'
        '<b>Privacy</b><br>'
        'Scoring runs locally in your session. For production, models can be hosted privately and audited.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


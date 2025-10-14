# app.py
import io
import json
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

from predict_phenotype import JMDCPhenotype

st.set_page_config(page_title="Predictive Risk Phenotypes for T2D", page_icon="🩺", layout="centered")
st.title("🩺 Predictive Risk Phenotypes for T2D")
st.caption("JMDC LIME-based phenotypes + 5-year T2D risk")

# --- CONFIG: choose how to load the model ---
LOAD_FROM = st.secrets.get("MODEL_SOURCE", "hf")  # "hf" or "local"

# If using Hugging Face
HF_REPO_ID = st.secrets.get("HF_REPO_ID", "your-org/JMDC_LIME_PhenotypeModel")
HF_FILENAME = st.secrets.get("HF_FILENAME", "JMDC_LIME_PhenotypeBundle_v1.joblib")

# If using local file committed to repo
LOCAL_MODEL_PATH = "JMDC_LIME_PhenotypeBundle_v1.joblib"


@st.cache_resource(show_spinner="Loading phenotype model…")
def load_model():
    if LOAD_FROM == "local":
        bundle_path = LOCAL_MODEL_PATH
    else:
        # If the repo is private, add token via st.secrets["HF_TOKEN"]
        token = st.secrets.get("HF_TOKEN", None)
        bundle_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=token
        )
    return JMDCPhenotype(bundle_path)

model = load_model()
feature_names = model.feature_names

st.subheader("Single patient")
with st.form("single"):
    cols = st.columns(3)
    vals = {}
    # Render numeric inputs dynamically from feature list
    # (Adjust defaults/ranges to your cohort if you like)
    defaults = {
        "BMI": 25.0, "HbA1c": 5.7, "Triglycerides": 150.0,
        "HDL_Cholesterol": 55.0, "LDL_Cholesterol": 130.0,
        "Gamma_GTP": 45.0, "Age": 45.0
    }
    for i, f in enumerate(feature_names):
        with cols[i % 3]:
            v0 = defaults.get(f, 0.0)
            vals[f] = st.number_input(f, value=float(v0))

    submitted = st.form_submit_button("Predict")
    if submitted:
        out = model.predict(vals)
        st.success(f"Phenotype: **{out['phenotype_name']}**")
        st.metric("Estimated 5-yr T2D risk", f"{out['t2d_risk']*100:.1f}%")
        with st.expander("Details"):
            st.json(out)

st.divider()
st.subheader("Batch prediction (CSV)")

st.write("CSV must include the exact columns (header) below:")
st.code(", ".join(feature_names), language="text")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    # basic header check
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        with st.spinner("Predicting…"):
            preds = model.predict_batch(df[feature_names])
        out_df = pd.concat([df.reset_index(drop=True), preds], axis=1)
        st.dataframe(out_df.head(20), use_container_width=True)

        # Download
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        st.download_button("Download results as CSV", buf.getvalue(), file_name="phenotype_predictions.csv", mime="text/csv")

st.caption("⚠️ For research/education. Not a standalone diagnostic.")
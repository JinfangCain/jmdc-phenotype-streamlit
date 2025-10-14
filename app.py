# app.py
import io
import json
import os
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from predict_phenotype import JMDCPhenotype

st.set_page_config(page_title="Predictive Risk Phenotypes for T2D", page_icon="🩺", layout="centered")
st.title("🩺 Predictive Risk Phenotypes for T2D")
st.caption("Predictive phenotypes using JMDC health checkup data")

# --- CONFIG: choose how to load the model ---
# default to "local" so it uses the committed .joblib; override in Secrets if you move to HF
LOAD_FROM = st.secrets.get("MODEL_SOURCE", "local")  # "local" or "hf"
HF_REPO_ID = st.secrets.get("HF_REPO_ID", "your-org/JMDC_LIME_PhenotypeModel")
HF_FILENAME = st.secrets.get("HF_FILENAME", "JMDC_LIME_PhenotypeBundle_v1.joblib")
LOCAL_MODEL_PATH = "JMDC_LIME_PhenotypeBundle_v1.joblib"

@st.cache_resource(show_spinner="Loading phenotype model…")
def load_model():
    if LOAD_FROM == "local":
        if not os.path.exists(LOCAL_MODEL_PATH):
            st.error(f"Local model not found: {LOCAL_MODEL_PATH}")
            st.stop()
        bundle_path = LOCAL_MODEL_PATH
    else:
        token = st.secrets.get("HF_TOKEN", None)  # only needed if HF repo is private
        bundle_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=token)
    return JMDCPhenotype(bundle_path)

model = load_model()
feature_names = model.feature_names  # must be: ['Systolic_BP','Diastolic_BP','BMI','Triglycerides','HDL_Cholesterol','LDL_Cholesterol','AST(GOT)','ALT(GPT)','Gamma_GTP','eGFR','Age','Sex']

# ----------------------- Single patient form -----------------------
st.subheader("Single patient")

# sensible defaults (edit to match your cohort)
defaults = {
    "Systolic_BP": 122.0,         # mmHg
    "Diastolic_BP": 78.0,         # mmHg
    "BMI": 24.5,                  # kg/m²
    "Triglycerides": 140.0,       # mg/dL
    "HDL_Cholesterol": 55.0,      # mg/dL
    "LDL_Cholesterol": 125.0,     # mg/dL
    "AST(GOT)": 22.0,             # U/L
    "ALT(GPT)": 24.0,             # U/L
    "Gamma_GTP": 35.0,            # U/L (shown as GGT)
    "eGFR": 85.0,                 # mL/min/1.73m²
    "Age": 45.0,                  # years
    "Sex": 1,                     # 1=Male, 0=Female
}

with st.form("single"):
    c1, c2, c3 = st.columns(3)
    vals = {}

    with c1:
        vals["Systolic_BP"]   = st.number_input("Systolic BP (mmHg)",  70.0, 250.0, float(defaults["Systolic_BP"]))
        vals["BMI"]           = st.number_input("BMI (kg/m²)",         12.0, 60.0,  float(defaults["BMI"]))
        vals["HDL_Cholesterol"]= st.number_input("HDL-C (mg/dL)",      10.0, 120.0, float(defaults["HDL_Cholesterol"]))
        vals["AST(GOT)"]      = st.number_input("AST (GOT) (U/L)",      5.0, 300.0, float(defaults["AST(GOT)"]))

    with c2:
        vals["Diastolic_BP"]  = st.number_input("Diastolic BP (mmHg)", 40.0, 150.0, float(defaults["Diastolic_BP"]))
        vals["Triglycerides"] = st.number_input("Triglycerides (mg/dL)", 30.0, 800.0, float(defaults["Triglycerides"]))
        vals["LDL_Cholesterol"]= st.number_input("LDL-C (mg/dL)",      40.0, 300.0, float(defaults["LDL_Cholesterol"]))
        vals["ALT(GPT)"]      = st.number_input("ALT (GPT) (U/L)",      5.0, 300.0, float(defaults["ALT(GPT)"]))

    with c3:
        # show GGT label but store in Gamma_GTP
        ggt_ui = st.number_input("GGT (U/L)", 5.0, 600.0, float(defaults["Gamma_GTP"]))
        vals["Gamma_GTP"]     = ggt_ui
        vals["eGFR"]          = st.number_input("eGFR (mL/min/1.73m²)", 5.0, 150.0, float(defaults["eGFR"]))
        vals["Age"]           = st.number_input("Age (years)", 18.0, 100.0, float(defaults["Age"]))
        sex_ui = st.radio("Sex", options=["Male", "Female"], index=0, horizontal=True)
        vals["Sex"] = 0 if sex_ui == "Male" else 1  # map Male/Female -> 0/1

    submitted = st.form_submit_button("Predict")
    if submitted:
        # Make sure we pass exactly the columns in the correct order expected by the model
        ordered_vals = {k: vals[k] for k in feature_names}
        out = model.predict(ordered_vals)
        st.success(f"Phenotype: **{out['phenotype_name']}**")
        st.metric("Estimated T2D risk", f"{out['t2d_risk']*100:.1f}%")
        with st.expander("Details"):
            st.json(out)

st.divider()

# ----------------------- Batch prediction -----------------------
st.subheader("Batch prediction (CSV)")
st.write("CSV must include the exact headers (in any order):")
st.code(", ".join(feature_names), language="text")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        with st.spinner("Predicting…"):
            preds = model.predict_batch(df[feature_names])
        out_df = pd.concat([df.reset_index(drop=True), preds], axis=1)
        st.dataframe(out_df.head(20), use_container_width=True)

        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        st.download_button("Download results as CSV", buf.getvalue(),
                           file_name="phenotype_predictions.csv", mime="text/csv")

st.caption("⚠️ For research/education. Not a standalone diagnostic.")


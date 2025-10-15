# app.py
import io
import json
import os
import pandas as pd
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from predict_phenotype import JMDCPhenotype
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Predictive T2D Risk Phenotype", 
    page_icon="📊",   # try: "🩸", "📈", "🧪", "🏥", "🧬"
    layout="centered")

# --- Button theming (affects all primary buttons) ---
st.markdown("""
<style>
/* Primary buttons everywhere (including form submit) */
button[kind="primary"],
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-primaryFormSubmit"],
div.stButton > button,
form button[type="submit"] {
    background: linear-gradient(90deg, #6b7280, #9ca3af) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 0.85rem 1.25rem !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 1.15rem !important;
    letter-spacing: .3px !important;
    transition: all 0.15s ease-in-out !important;
}

button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="baseButton-primaryFormSubmit"]:hover,
div.stButton > button:hover,
form button[type="submit"]:hover {
    background: linear-gradient(90deg, #4b5563, #6b7280) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Your T2D Risk Phenotype")
st.caption("Predictive phenotypes using JMDC health checkup data based on LIME explanations of an optimal predictive machine learning model.")

st.markdown("""
An optimum machine learning model was trained to predict the risk of developing Type 2 Diabetes (T2D) using health-checkup data from 19,953 Japanese adults and 12 routinely measured variables.  
These LIME-derived contribution profiles were use to cinstruct seven distinct predictive phenotypes, each capturing a characteristic combination of metabolic traits and corresponding T2D risk.  
""")

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

# Try to fetch phenotype names and mean risks (defensive against different bundle keys)
phenotype_names = getattr(model, "names_by_order", None)
if phenotype_names is None and hasattr(model, "B"):
    phenotype_names = model.B.get("phenotype_names_by_order", None)
if phenotype_names is None:
    phenotype_names = [f"P{i+1}" for i in range(7)]  # fallback

mean_risks_by_order = None
for k in [
    "mean_risks_by_order",
    "cluster_mean_risks_by_order",
    "mean_cluster_risks_by_order",
    "phenotype_mean_risks_by_order",
]:
    if hasattr(model, k):
        mean_risks_by_order = getattr(model, k)
        break
    if hasattr(model, "B") and isinstance(model.B, dict) and k in model.B:
        mean_risks_by_order = model.B[k]
        break

# ----------------------- Single patient form -----------------------
st.subheader("Your Risk Profile")

# defaults (one decimal place)
defaults = {
    "Systolic_BP": 122.0, "Diastolic_BP": 78.0, "BMI": 24.5,
    "Triglycerides": 140.0, "HDL_Cholesterol": 55.0, "LDL_Cholesterol": 125.0,
    "AST(GOT)": 22.0, "ALT(GPT)": 24.0, "Gamma_GTP": 35.0,
    "eGFR": 85.0, "Age": 45.0, "Sex": 0,   # 0=Male, 1=Female
}

with st.form("single"):
    vals = {}

    # ---------------- Row 1: Anthropometrics ----------------
    r1c1, r1c2, r1c3 = st.columns(3)
    weight_kg = r1c1.number_input("Weight (kg)", min_value=20.0, max_value=300.0,
                                   value=70.0, step=0.1, format="%.1f")
    height_m  = r1c2.number_input("Height (m)",  min_value=1.00, max_value=2.50,
                                   value=1.70, step=0.01, format="%.2f")
    if height_m and height_m > 0 and weight_kg and weight_kg > 0:
        bmi_calc = round(weight_kg / (height_m ** 2), 1)
    else:
        bmi_calc = round(defaults["BMI"], 1)
    vals["BMI"] = r1c3.number_input("BMI (kg/m²)", min_value=12.0, max_value=60.0,
                                    value=float(bmi_calc), step=0.1, format="%.1f",
                                    disabled=True)

    # ---------------- Row 2: Blood pressure + Age ----------------
    r2c1, r2c2, r2c3 = st.columns(3)
    vals["Systolic_BP"]  = r2c1.number_input("Systolic BP (mmHg)", min_value=70.0,  max_value=250.0,
                                             value=float(round(defaults["Systolic_BP"], 1)), step=0.1, format="%.1f")
    vals["Diastolic_BP"] = r2c2.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=150.0,
                                             value=float(round(defaults["Diastolic_BP"], 1)), step=0.1, format="%.1f")
    vals["Age"]          = r2c3.number_input("Age (years)", min_value=18.0, max_value=100.0,
                                             value=float(round(defaults["Age"], 1)), step=0.1, format="%.1f")

    # ---------------- Row 3: Lipids ----------------
    r3c1, r3c2, r3c3 = st.columns(3)
    vals["Triglycerides"]   = r3c1.number_input("Triglycerides (mg/dL)", min_value=30.0, max_value=800.0,
                                                value=float(round(defaults["Triglycerides"], 1)), step=0.1, format="%.1f")
    vals["HDL_Cholesterol"] = r3c2.number_input("HDL-C (mg/dL)", min_value=10.0, max_value=120.0,
                                                value=float(round(defaults["HDL_Cholesterol"], 1)), step=0.1, format="%.1f")
    vals["LDL_Cholesterol"] = r3c3.number_input("LDL-C (mg/dL)", min_value=40.0, max_value=300.0,
                                                value=float(round(defaults["LDL_Cholesterol"], 1)), step=0.1, format="%.1f")

    # ---------------- Row 4: Liver enzymes ----------------
    r4c1, r4c2, r4c3 = st.columns(3)
    vals["AST(GOT)"]  = r4c1.number_input("AST (GOT) (U/L)", min_value=5.0, max_value=300.0,
                                          value=float(round(defaults["AST(GOT)"], 1)), step=0.1, format="%.1f")
    vals["ALT(GPT)"]  = r4c2.number_input("ALT (GPT) (U/L)", min_value=5.0, max_value=300.0,
                                          value=float(round(defaults["ALT(GPT)"], 1)), step=0.1, format="%.1f")
    ggt_ui            = r4c3.number_input("GGT (U/L)", min_value=5.0, max_value=600.0,
                                          value=float(round(defaults["Gamma_GTP"], 1)), step=0.1, format="%.1f")
    vals["Gamma_GTP"] = ggt_ui  # map UI GGT to model's Gamma_GTP

    # ---------------- Row 5: Kidney + Sex (+ spacer) ----------------
    r5c1, r5c2, r5c3 = st.columns(3)
    vals["eGFR"] = r5c1.number_input("eGFR (mL/min/1.73m²)", min_value=5.0, max_value=150.0,
                                     value=float(round(defaults["eGFR"], 1)), step=0.1, format="%.1f")
    sex_ui = r5c2.radio("Sex", options=["Male", "Female"], index=0, horizontal=True)
    vals["Sex"] = 1 if sex_ui == "Male" else 0
    r5c3.markdown("&nbsp;", unsafe_allow_html=True)  # spacer to keep 3-column grid

    row_btn_left, row_btn_center, row_btn_right = st.columns([1, 2, 1])
    with row_btn_center:
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        # ensure BMI reflects the latest auto-compute
        vals["BMI"] = bmi_calc
        # pass exactly the columns the model expects, in order
        ordered_vals = {k: vals[k] for k in feature_names}
        out = model.predict(ordered_vals)
        st.success(f"Phenotype: **{out['phenotype_name']}**")
        st.metric("Estimated T2D risk", f"{out['t2d_risk']*100:.1f}%")
        # --- Phenotype bar: 7 segments with labels and average risks ---
        # Identify the selected phenotype index (ordered label 0..6)
        sel_idx = int(out.get("phenotype_ordered_label", 0))

        names = phenotype_names
        import numpy as np
        risks = np.array(mean_risks_by_order) if mean_risks_by_order is not None else np.array([np.nan]*len(names))

        n = len(names)
        widths = [1]*n
        from itertools import cycle, islice
        palette_base = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#f97316"]
        palette = list(islice(cycle(palette_base), n))

        alphas = [1.0 if i == sel_idx else 0.35 for i in range(n)]

        fig, ax = plt.subplots(figsize=(9, 1.6))
        left = 0
        for i in range(n):
            ax.barh(
                y=0, width=widths[i], left=left,
                color=palette[i], alpha=alphas[i], edgecolor="white", height=0.9
            )
            # Clean label: only phenotype name (and optional mean risk if available)
            label = f"{names[i]}" if np.isnan(risks[i]) else f"{names[i]}\n{risks[i]*100:.2f}%"
            ax.text(
                left + widths[i]/2, 0,
                label,
                ha="center", va="center",
                fontsize=10.5,
                color="white",
                fontweight="bold",
                linespacing=1.3,
            )
            left += widths[i]

        ax.set_xlim(0, sum(widths))
        ax.set_ylim(-0.6, 0.6)
        ax.axis("off")
        st.pyplot(fig)
        with st.expander("Details"):
            st.json(out)

st.divider()
        
# ----------------------- Batch prediction -----------------------
st.subheader("Batch prediction (CSV)")
st.write("CSV must include the exact headers (in any order). You may use **GGT** or **Gamma_GTP**:")
display_names = ["GGT" if f == "Gamma_GTP" else f for f in feature_names]
st.code(", ".join(display_names), language="text")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

    # 1) Normalize column names (strip spaces; keep original for message)
    original_cols = df.columns.tolist()
    df.columns = [c.strip() for c in df.columns]

    # 2) Accept either "GGT" or "Gamma_GTP" (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    if ("ggt" in cols_lower) and ("gamma_gtp" not in {k.replace(" ", "_") for k in cols_lower}):
        # Rename GGT -> Gamma_GTP
        df = df.rename(columns={cols_lower["ggt"]: "Gamma_GTP"})

    # 3) Optional: handle Sex as text ("Male"/"Female") -> 0/1
    if "Sex" in df.columns and df["Sex"].dtype == object:
        df["Sex"] = df["Sex"].astype(str).str.strip().str.lower().map(
            {"male": 0, "m": 0, "female": 1, "f": 1}
        ).fillna(df["Sex"])  # leave numeric values untouched if present
        # if anything still non-numeric, try to coerce
        df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")

    # 4) Validate required columns
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        # If we're missing Gamma_GTP but user had a column named GGT (not caught), tell them
        if "Gamma_GTP" in missing and any(c.lower() == "ggt" for c in original_cols):
            st.error("Could not map 'GGT' to 'Gamma_GTP'. Please ensure your header is exactly 'GGT' or 'Gamma_GTP'.")
        else:
            st.error(f"Missing columns: {missing}")
    else:
        # 5) Coerce types (best effort) and reorder columns
        for c in feature_names:
            if c != "Sex":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Sex must be 0/1; if still NaN after mapping, warn
        if df["Sex"].isna().any():
            st.warning("Some 'Sex' values are not recognized (use 0/1 or Male/Female). Unrecognized rows will appear as NaN.")

        df_in = df[feature_names].copy()

        # 6) Predict
        with st.spinner("Predicting…"):
            preds = model.predict_batch(df_in)

        # --- NEW: hide internal columns from UI/CSV ---
        cols_to_drop = [
            "membership_probs",
            "membership_probs_raw",
            "membership_probs_by_order",
            "raw_cluster_id",
        ]
        preds_public = preds.drop(columns=cols_to_drop, errors="ignore")

        out_df_public = pd.concat([df.reset_index(drop=True), preds_public], axis=1)

        # 7) Show & let users download (public columns only)
        st.success("Prediction complete.")
        st.dataframe(out_df_public.head(20), use_container_width=True)

        buf = io.StringIO()
        out_df_public.to_csv(buf, index=False)
        st.download_button(
            "Download results as CSV",
            buf.getvalue(),
            file_name="phenotype_predictions.csv",
            mime="text/csv",
        )

st.caption("⚠️ For research/education. Not a standalone diagnostic.")


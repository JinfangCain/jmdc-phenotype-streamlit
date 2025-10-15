# app.py
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from huggingface_hub import hf_hub_download
from predict_phenotype import JMDCPhenotype

# ----------------------------- Page config -----------------------------
st.set_page_config(
    page_title="Predictive T2D Risk Phenotype",
    page_icon="📊",   # alternatives: "🩸", "📈", "🧪", "🏥", "🧬"
    layout="centered"
)

# ------------- Button theming (global primary button style) -------------
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

# ----------------------------- Header copy -----------------------------
st.title("📊 Your T2D Risk Phenotype")
st.caption("Predictive phenotypes using JMDC health checkup data based on LIME explanations of an optimal predictive model.")

st.markdown("""
An **optimal machine learning model** was trained to predict Type 2 Diabetes (T2D) risk using health-checkup data from **19,953** Japanese adults across **12 routine variables**.  
For each person, **LIME** quantified how every feature contributed to their predicted risk.  
These contribution profiles were **clustered into seven predictive phenotypes**, then **ordered by mean predicted risk** to form a transparent risk spectrum.
""")

# ----------------------------- Model loading -----------------------------
# Default to local bundle committed to the repo. Override with Secrets if using HF.
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
        token = st.secrets.get("HF_TOKEN", None)  # only if private HF repo
        bundle_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=token)
    return JMDCPhenotype(bundle_path)

model = load_model()
feature_names = model.feature_names  # ['Systolic_BP','Diastolic_BP','BMI','Triglycerides','HDL_Cholesterol','LDL_Cholesterol','AST(GOT)','ALT(GPT)','Gamma_GTP','eGFR','Age','Sex']

# Try to fetch phenotype names and mean risks from the bundle
phenotype_names = getattr(model, "names_by_order", None)
if phenotype_names is None and hasattr(model, "B"):
    phenotype_names = model.B.get("phenotype_names_by_order", None)
if phenotype_names is None:
    phenotype_names = [f"P{i+1}" for i in range(7)]

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

# Label sets for the phenotype bar
full_names = phenotype_names or [
    "P1: Young Low-BMI",
    "P2: Mid-Aged Low-BMI",
    "P3: Older Low-BMI",
    "P4: Young Hepatic-Metabo",
    "P5: Older Hepatic-Hypertensive",
    "P6: Older Metabo",
    "P7: Older Hepatic-Metabo",
]
short_names = [
    "P1: Young Lean",
    "P2: Mid Lean",
    "P3: Older Lean",
    "P4: Young Hep-Met",
    "P5: Older Hep-HTN",
    "P6: Older Metab",
    "P7: Older Hep-Met",
]
mini_names = ["P1 YL", "P2 ML", "P3 OL", "P4 Y-HM", "P5 O-HH", "P6 OM", "P7 O-HM"]

# ----------------------- Single patient form -----------------------
st.subheader("Your Risk Profile")

# sensible defaults (one decimal place); Sex: 1=Male, 0=Female
defaults = {
    "Systolic_BP": 122.0, "Diastolic_BP": 78.0, "BMI": 24.5,
    "Triglycerides": 140.0, "HDL_Cholesterol": 55.0, "LDL_Cholesterol": 125.0,
    "AST(GOT)": 22.0, "ALT(GPT)": 24.0, "Gamma_GTP": 35.0,
    "eGFR": 85.0, "Age": 45.0, "Sex": 1,
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
    vals["Sex"] = 1 if sex_ui == "Male" else 0  # 1=Male, 0=Female
    r5c3.markdown("&nbsp;", unsafe_allow_html=True)  # spacer

    # Submit button (centered)
    row_btn_left, row_btn_center, row_btn_right = st.columns([1, 2, 1])
    with row_btn_center:
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        vals["BMI"] = bmi_calc  # ensure most recent auto-compute
        ordered_vals = {k: vals[k] for k in feature_names}
        out = model.predict(ordered_vals)

        st.success(f"Phenotype: **{out['phenotype_name']}**")
        st.metric("Estimated T2D risk", f"{out['t2d_risk']*100:.1f}%")

        # Label length selector
        label_mode = st.radio(
            "Bar labels", ["Short", "Full", "Ultra-compact"],
            index=0, horizontal=True
        )
        if label_mode == "Full":
            names_to_plot = full_names
        elif label_mode == "Ultra-compact":
            names_to_plot = mini_names
        else:
            names_to_plot = short_names

        # --- Phenotype bar (7 segments, highlight selection) ---
        sel_idx = int(out.get("phenotype_ordered_label", 0))
        risks = np.array(mean_risks_by_order) if mean_risks_by_order is not None else np.array([np.nan]*len(names_to_plot))

        n = len(names_to_plot)
        widths = [1]*n
        from itertools import cycle, islice
        palette_base = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#f97316"]
        palette = list(islice(cycle(palette_base), n))
        alphas = [1.0 if i == sel_idx else 0.35 for i in range(n)]

        fig, ax = plt.subplots(figsize=(9, 1.4))
        left = 0
        for i in range(n):
            ax.barh(
                y=0, width=widths[i], left=left,
                color=palette[i], alpha=alphas[i], edgecolor="white", height=0.9
            )
            # Only the selected segment shows the mean risk (if available)
            if i == sel_idx and not np.isnan(risks[i]):
                label = f"{names_to_plot[i]}\n{risks[i]*100:.2f}%"
            else:
                label = names_to_plot[i]
            ax.text(
                left + widths[i]/2, 0, label,
                ha="center", va="center",
                fontsize=10, color="white", weight="bold", linespacing=1.2
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

    # 1) Normalize header whitespace
    original_cols = df.columns.tolist()
    df.columns = [c.strip() for c in df.columns]

    # 2) Accept either "GGT" or "Gamma_GTP" (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    col_keys_norm = {k.replace(" ", "_") for k in cols_lower}
    if "ggt" in cols_lower and "gamma_gtp" not in col_keys_norm and "Gamma_GTP" not in df.columns:
        df = df.rename(columns={cols_lower["ggt"]: "Gamma_GTP"})

    # 3) Map Sex text -> numeric (Male=1, Female=0)
    if "Sex" in df.columns and df["Sex"].dtype == object:
        df["Sex"] = df["Sex"].astype(str).str.strip().str.lower().map(
            {"male": 1, "m": 1, "female": 0, "f": 0}
        ).fillna(df["Sex"])
        df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")

    # 4) Validate required columns
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        if "Gamma_GTP" in missing and any(str(c).lower() == "ggt" for c in original_cols):
            st.error("Could not map 'GGT' to 'Gamma_GTP'. Please ensure your header is exactly 'GGT' or 'Gamma_GTP'.")
        else:
            st.error(f"Missing columns: {missing}")
    else:
        # 5) Coerce numeric types & reorder
        for c in feature_names:
            if c != "Sex":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if df["Sex"].isna().any():
            st.warning("Some 'Sex' values are not recognized (use 1/0 or Male/Female). Unrecognized rows will be NaN.")

        df_in = df[feature_names].copy()

        # 6) Predict
        with st.spinner("Predicting…"):
            preds = model.predict_batch(df_in)

        # 7) Hide internal columns from UI/CSV (if present)
        cols_to_drop = ["membership_probs", "membership_probs_raw", "membership_probs_by_order", "raw_cluster_id"]
        preds_public = preds.drop(columns=cols_to_drop, errors="ignore")

        out_df_public = pd.concat([df.reset_index(drop=True), preds_public], axis=1)

        # 8) Show & download
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
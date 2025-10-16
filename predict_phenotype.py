#!/usr/bin/env python3
"""
Lightweight SDK to assign JMDC LIME-based phenotypes and T2D risk.

API:
    from predict_phenotype import JMDCPhenotype
    clf = JMDCPhenotype("JMDC_LIME_PhenotypeBundle_v1.joblib")
    out = clf.predict({"BMI": 29.5, "HbA1c": 5.7, ...})  # dict -> dict

CLI:
    python predict_phenotype.py --bundle JMDC_LIME_PhenotypeBundle_v1.joblib --in sample_input.csv --out preds.csv
"""
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import lime.lime_tabular
from scipy.special import logit as _logit
from sklearn.metrics.pairwise import euclidean_distances


class JMDCPhenotype:
    def __init__(self, bundle_path: str):
        self.B = joblib.load(bundle_path)
        self.feature_names = self.B["feature_names"]
        # Rebuild LIME explainer deterministically with the same training data and params
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.B["lime_training_data"],
            feature_names=self.feature_names,
            discretize_continuous=self.B["lime_params"].get("discretize_continuous", True),
            mode=self.B["lime_params"].get("mode", "regression"),
        )
        self.model_f = self.B["predictive_model_for_lime"]     # used inside LIME
        self.kmeans = self.B["kmeans_lime"]
        self.risk_order = self.B["risk_order_map_lime"]        # {raw_id -> ordered_rank}
        self.names_by_order = self.B["phenotype_names_by_order"]
        self.scaler = self.B["risk_scaler"]
        self.logit_model = self.B["risk_logistic"]
        # Expected to be length-K, already ordered from lowest -> highest risk
        risks = self.B.get("cluster_mean_risks", None)
        if risks is None:
            # keep a consistent attribute so the app can detect the absence
            self.cluster_mean_risks = None
        else:
            risks = np.asarray(risks, dtype=float)
            # Length guard: align with number of names if they differ
            k_names = len(self.names_by_order) if isinstance(self.names_by_order, (list, tuple)) else len(risks)
            if risks.size != k_names:
                fixed = np.full(k_names, np.nan, dtype=float)
                fixed[:min(k_names, risks.size)] = risks[:min(k_names, risks.size)]
                risks = fixed
            self.cluster_mean_risks = risks

    def _predict_logit(self, X_array: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X_array, columns=self.feature_names)
        p = self.model_f.predict_proba(X_df)[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return _logit(p)

    def _lime_vector(self, x_row_df: pd.DataFrame) -> np.ndarray:
        data_row = x_row_df[self.feature_names].iloc[0].to_numpy()
        exp = self.explainer.explain_instance(
            data_row,
            predict_fn=self._predict_logit,
            num_features=len(self.feature_names),
        )
        v = np.zeros(len(self.feature_names), dtype=float)
        for j, w in exp.local_exp[0]:  # regression mode
            v[j] = float(w)
        return v.reshape(1, -1)

    def _membership_probs(self, lime_vec: np.ndarray) -> np.ndarray:
        C = self.kmeans.cluster_centers_
        d2 = euclidean_distances(lime_vec, C, squared=True)[0]
        T = np.median(d2) if np.median(d2) > 0 else 1.0
        pi = np.exp(-d2 / (T + 1e-12))
        return pi / pi.sum()

    def predict(self, x_row: dict) -> dict:
        """Single-row predict: dict -> dict"""
        x_df = pd.DataFrame([x_row], columns=self.feature_names)
        # Risk probability
        X_scaled = self.scaler.transform(x_df[self.feature_names])
        p_t2d = float(self.logit_model.predict_proba(X_scaled)[:, 1])
        # LIME vector, raw cluster, ordered phenotype
        v = self._lime_vector(x_df)
        raw_cluster = int(self.kmeans.predict(v)[0])
        ordered = int(self.risk_order[raw_cluster])
        name = self.names_by_order[ordered]
        pi = self._membership_probs(v)
        return {
            "phenotype_ordered_label": ordered,
            "phenotype_name": name,
            "t2d_risk": p_t2d,
            "membership_probs": pi.tolist(),
            "raw_cluster_id": raw_cluster,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            rows.append(self.predict(r.to_dict()))
        return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to JMDC_LIME_PhenotypeBundle_v*.joblib")
    ap.add_argument("--in", dest="in_csv", required=True, help="CSV with columns matching bundle feature_names")
    ap.add_argument("--out", dest="out_csv", required=True, help="Where to write predictions CSV")
    ap.add_argument("--json", dest="json_out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    clf = JMDCPhenotype(args.bundle)
    df = pd.read_csv(args.in_csv)
    preds = clf.predict_batch(df)
    preds.to_csv(args.out_csv, index=False)
    if args.json_out:
        preds.to_json(args.json_out, orient="records", indent=2)
    print(f"Wrote: {args.out_csv}" + (f" and {args.json_out}" if args.json_out else ""))

if __name__ == "__main__":
    main()
"""
Regenerate Matrixanalysis.png from v5 artefacts.

Pipeline:
  1. Load v5 OOF predictions → identify the 72/98 split
  2. Load v5 cached enriched dataframe (190 raw features + 7 metadata cols)
  2b. Import engineer_features() from model_comparison_v4.py, apply it
      to add the 35 derived features → 225 engineered features
  3. Load model bundle (model + scaler + selector + feature_names)
  4. Verify all 225 engineered features now present
  5. Apply scaler → selector → matches the 135-feature space the model uses
  6. TreeExplainer SHAP, slice the low-credibility class output
  7. Rank features by gap between correctly-classified and missed groups
  8. Save Matrixanalysis.png

Author: Wilbert Kooijman
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# =====================================================================
# PATHS
# =====================================================================
BASE = Path(r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)")
V5 = BASE / "thesis_output" / "model_results_v5"

PREDS_PATH = V5 / "predictions" / "preds_Enriched_XGB_SMOTE_weights.csv"
CACHE_PATH = V5 / "enriched_df_cache.pkl"
MODEL_PATH = V5 / "final_xgb_Enriched.joblib"

OUTPUT_PNG = BASE / "Matrixanalysis.png"
OUTPUT_CSV = V5 / "subrq4_analysis_v5_shap_contrast.csv"
TOP_N = 15

# =====================================================================
# 1. PREDICTIONS — identify the 72/98 split
# =====================================================================
print("=" * 60)
print("[1] Loading predictions...")
preds = pd.read_csv(PREDS_PATH)
label_map = (
    preds.drop_duplicates("y_true_label")
         .set_index("y_true_label")["y_true"].to_dict()
)
low_idx_pred = label_map["low credibility"]
print(f"    label encoding (from preds): {label_map}")

low_preds = preds[preds["y_true"] == low_idx_pred].copy()
correct_mask = low_preds["y_pred"] == low_idx_pred
n_correct = int(correct_mask.sum())
n_missed = int((~correct_mask).sum())
print(f"    correctly classified: n = {n_correct}")
print(f"    missed:               n = {n_missed}")

correct_test_idx = low_preds.loc[correct_mask, "test_idx"].astype(int).values
missed_test_idx = low_preds.loc[~correct_mask, "test_idx"].astype(int).values

# =====================================================================
# 2. CACHED FEATURE MATRIX
# =====================================================================
print("\n" + "=" * 60)
print("[2] Loading cached enriched dataframe...")
cached = pd.read_pickle(CACHE_PATH)
print(f"    shape: {cached.shape}  (190 base features + 7 metadata)")

# =====================================================================
# 2b. APPLY FEATURE ENGINEERING via model_comparison_v4.engineer_features
# =====================================================================
print("\n" + "=" * 60)
print("[2b] Importing engineer_features from model_comparison_v4...")
sys.path.insert(0, str(BASE))

try:
    from model_comparison_v4 import engineer_features, get_feature_columns
except Exception as exc:
    sys.exit(f"    ABORT: could not import from model_comparison_v4: {exc}")

print(f"    imported successfully")

# Get the raw feature column list (excludes metadata)
feature_cols_raw = get_feature_columns(cached)
print(f"    raw feature columns: {len(feature_cols_raw)}")

# Apply engineering. Function may return df, (df, new_cols), or (df, all_cols).
print(f"    applying engineer_features...")
result = engineer_features(cached.copy(), feature_cols_raw)

if isinstance(result, tuple):
    print(f"    function returned tuple of length {len(result)}")
    cached_eng = result[0]
else:
    cached_eng = result

print(f"    engineered shape: {cached_eng.shape}")
new_cols = [c for c in cached_eng.columns if c not in cached.columns]
print(f"    new columns added: {len(new_cols)}")
if new_cols:
    print(f"      first 5 new: {new_cols[:5]}")

# =====================================================================
# 3. MODEL BUNDLE
# =====================================================================
print("\n" + "=" * 60)
print("[3] Loading model bundle...")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
selector = bundle["selector"]
feature_names_engineered = list(bundle["feature_names_engineered"])
feature_names_selected = list(bundle["feature_names_selected"])
class_names = list(bundle["class_names"])
low_idx = class_names.index("low credibility")
print(f"    model: {type(model).__name__}, "
      f"low-cred class index: {low_idx}")

if low_idx != low_idx_pred:
    sys.exit(f"    ABORT: bundle says low={low_idx}, preds say low={low_idx_pred}")

# =====================================================================
# 4. VERIFY 225 ENGINEERED FEATURES ARE NOW PRESENT
# =====================================================================
print("\n" + "=" * 60)
print("[4] Verifying engineered features...")
cached_set = set(cached_eng.columns)
missing = [c for c in feature_names_engineered if c not in cached_set]
print(f"    engineered features required: {len(feature_names_engineered)}")
print(f"    present in engineered cache:  "
      f"{len(feature_names_engineered) - len(missing)}")

if missing:
    print(f"    still missing ({len(missing)}):")
    for f in missing[:30]:
        print(f"      - {f}")
    sys.exit("    ABORT: feature engineering did not produce all required cols.")

print("    ✓ all 225 engineered features present")

# =====================================================================
# 5. BUILD MATRIX, APPLY SCALER + SELECTOR
# =====================================================================
print("\n" + "=" * 60)
print("[5] Building 170-row matrix and applying scaler+selector...")
all_low_idx = np.concatenate([correct_test_idx, missed_test_idx])
if all_low_idx.max() >= len(cached_eng):
    sys.exit(f"    ABORT: test_idx {all_low_idx.max()} >= cache rows {len(cached_eng)}")

X_low_eng = cached_eng.iloc[all_low_idx][feature_names_engineered].copy()
X_low_eng = X_low_eng.apply(pd.to_numeric, errors="coerce").fillna(0)
print(f"    engineered slice shape: {X_low_eng.shape}")

X_low_selected_raw = selector.transform(X_low_eng.values)
X_low_selected = scaler.transform(X_low_selected_raw)
print(f"    after selector+scaler:  {X_low_selected.shape}")

# Sanity check — does the model reproduce the 72/98 split?
preds_check = model.predict(X_low_selected)
n_correct_check = int((preds_check[:n_correct] == low_idx).sum())
n_missed_check = int((preds_check[n_correct:] != low_idx).sum())
print(f"    sanity check via model.predict():")
print(f"      correctly classified verified: {n_correct_check}/{n_correct}")
print(f"      missed verified:               {n_missed_check}/{n_missed}")
if (n_correct_check, n_missed_check) != (n_correct, n_missed):
    print("    ! note: final model differs slightly from OOF models;")
    print("      SHAP attributions are still valid, group assignment uses OOF.")

# =====================================================================
# 6. SHAP
# =====================================================================
print("\n" + "=" * 60)
print("[6] Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_low_selected)
if isinstance(shap_values, list):
    shap_low = shap_values[low_idx]
else:
    shap_low = shap_values[:, :, low_idx]
print(f"    shap_low shape: {shap_low.shape}")

shap_correct = shap_low[:n_correct]
shap_missed = shap_low[n_correct:]
mean_abs_correct = np.abs(shap_correct).mean(axis=0)
mean_abs_missed = np.abs(shap_missed).mean(axis=0)
gap = np.abs(mean_abs_correct - mean_abs_missed)

# =====================================================================
# 7. RANK + EXPORT
# =====================================================================
ranking_full = pd.DataFrame({
    "feature": feature_names_selected,
    "mean_abs_shap_correct": mean_abs_correct,
    "mean_abs_shap_missed": mean_abs_missed,
    "gap": gap,
}).sort_values("gap", ascending=False).reset_index(drop=True)
ranking_full.to_csv(OUTPUT_CSV, index=False)
print(f"\n[7] Wrote per-feature SHAP contrast: {OUTPUT_CSV}")

top = ranking_full.head(TOP_N).iloc[::-1].reset_index(drop=True)
print(f"\nTop {TOP_N} features by gap:")
print(top.to_string(index=False))

# =====================================================================
# 8. PLOT
# =====================================================================
print("\n" + "=" * 60)
print("[8] Rendering figure...")
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
y_pos = np.arange(len(top))
bh = 0.4
ax.barh(
    y_pos - bh / 2, top["mean_abs_shap_correct"], height=bh,
    label=f"Correctly classified (n={n_correct})",
    color="#1f77b4", edgecolor="white", linewidth=0.5,
)
ax.barh(
    y_pos + bh / 2, top["mean_abs_shap_missed"], height=bh,
    label=f"False negatives (n={n_missed})",
    color="#d62728", edgecolor="white", linewidth=0.5,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(top["feature"], fontsize=9)
ax.set_xlabel("Mean |SHAP value| (low-credibility class output)", fontsize=10)
ax.set_title(
    "Structural signals separating detected from missed\n"
    f"low-credibility sources (top {TOP_N} by gap)",
    fontsize=11,
)
ax.legend(loc="lower right", frameon=True, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n✓ Saved figure: {OUTPUT_PNG}")
print(f"  Legend: 'Correctly classified (n={n_correct})', "
      f"'False negatives (n={n_missed})'")
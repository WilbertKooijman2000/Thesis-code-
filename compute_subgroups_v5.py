"""
============================================================================
COMPUTE SUBGROUPS v5 — Per-country and per-media-type metrics
============================================================================
Master Thesis - Wilbert | Tilburg University

Regenerates Tables 3 (per-country) and 4 (per-media-type) using v5
predictions from the leakage-free pipeline. Loads:
  - thesis_output/model_results_v5/enriched_df_cache.pkl   (cached enriched data)
  - thesis_output/model_results_v5/predictions/
      preds_Enriched_XGB_SMOTE_weights.csv                 (headline config)

Outputs (in thesis_output/model_results_v5/subgroups/):
  - subgroups_country.csv         (full country breakdown)
  - subgroups_media_type.csv      (full media type breakdown)
  - subgroups_summary.txt         (formatted summary + Anglo-vs-Other comparison)
  - subgroups_gap_summary.csv     (the 19-point gap analysis, recomputed)

The script flags subgroups with n_low < 5 as statistically unreliable for
per-class F1 reporting (the cell value is meaningless when computed from
a handful of instances).

Usage:
    python compute_subgroups_v5.py
============================================================================
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BASE_DIR = r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)"
CONFIG = {
    "enriched_cache": os.path.join(BASE_DIR, "thesis_output", "model_results_v5",
                                    "enriched_df_cache.pkl"),
    "predictions_csv": os.path.join(BASE_DIR, "thesis_output", "model_results_v5",
                                     "predictions", "preds_Enriched_XGB_SMOTE_weights.csv"),
    "output_dir": os.path.join(BASE_DIR, "thesis_output", "model_results_v5", "subgroups"),
    "min_subgroup_n": 20,   # subgroups smaller than this are omitted entirely
    "min_n_low_reliable": 5,  # subgroups with fewer true low-cred than this are flagged
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
def load_data():
    logger.info(f"Loading enriched dataset cache from: {CONFIG['enriched_cache']}")
    with open(CONFIG["enriched_cache"], "rb") as f:
        enriched_df = pickle.load(f)
    logger.info(f"  Shape: {enriched_df.shape}")
    logger.info(f"  Columns include: {[c for c in enriched_df.columns if c.startswith(('country_', 'media_type_', 'traffic_'))][:10]}...")

    logger.info(f"Loading predictions from: {CONFIG['predictions_csv']}")
    preds = pd.read_csv(CONFIG["predictions_csv"])
    logger.info(f"  Shape: {preds.shape}")
    logger.info(f"  Columns: {list(preds.columns)}")
    return enriched_df, preds


# ----------------------------------------------------------------------------
# 2. Reconstruct country / media_type from one-hot columns
# ----------------------------------------------------------------------------
def reconstruct_categorical(df, prefix):
    """For one-hot columns with the given prefix, return a single Series of
    the original category label per row. Rows with no one-hot active get 'unknown'."""
    one_hot_cols = [c for c in df.columns if c.startswith(prefix)]
    if not one_hot_cols:
        logger.warning(f"  No columns found with prefix '{prefix}'")
        return pd.Series(["unknown"] * len(df), index=df.index)

    # For each row, find the one-hot column with value 1 (or the max)
    matrix = df[one_hot_cols].fillna(0).values
    argmax = np.argmax(matrix, axis=1)
    # Strip prefix and return the category name
    labels = [one_hot_cols[i].replace(prefix, "") for i in argmax]

    # If no one-hot is positive (all zeros), mark as 'unknown'
    has_value = (matrix.sum(axis=1) > 0)
    labels = [lab if has else "unknown" for lab, has in zip(labels, has_value)]
    return pd.Series(labels, index=df.index)


# ----------------------------------------------------------------------------
# 3. Join predictions with metadata
# ----------------------------------------------------------------------------
def join_predictions_metadata(enriched_df, preds):
    """Merge preds with enriched_df on test_idx -> df row index."""
    # Reconstruct categorical labels
    country = reconstruct_categorical(enriched_df, "country_")
    media_type = reconstruct_categorical(enriched_df, "media_type_")

    # Build a metadata lookup table indexed by row position
    meta = pd.DataFrame({
        "row_idx": np.arange(len(enriched_df)),
        "country": country.values,
        "media_type": media_type.values,
    })

    # Merge: predictions' test_idx maps to row_idx
    joined = preds.merge(meta, left_on="test_idx", right_on="row_idx", how="left")
    if joined["country"].isna().any():
        n_missing = joined["country"].isna().sum()
        logger.warning(f"  {n_missing} predictions could not be matched to metadata")
    logger.info(f"  Joined {len(joined)} predictions with metadata")
    return joined


# ----------------------------------------------------------------------------
# 4. Compute per-subgroup metrics
# ----------------------------------------------------------------------------
def subgroup_metrics(df, group_col, class_names=("high credibility", "low credibility", "medium credibility")):
    """For each unique value of group_col, compute n, n per class, and F1 metrics."""
    rows = []
    for group_val, sub in df.groupby(group_col):
        n = len(sub)
        n_high = (sub["y_true_label"] == "high credibility").sum()
        n_low = (sub["y_true_label"] == "low credibility").sum()
        n_med = (sub["y_true_label"] == "medium credibility").sum()

        # Macro F1 and per-class F1 (with labels= to handle missing classes gracefully)
        try:
            macro_f1 = f1_score(sub["y_true_label"], sub["y_pred_label"],
                                labels=list(class_names), average="macro", zero_division=0)
            per_class_f1 = f1_score(sub["y_true_label"], sub["y_pred_label"],
                                    labels=list(class_names), average=None, zero_division=0)
            f1_high = per_class_f1[0]
            f1_low = per_class_f1[1]
            f1_med = per_class_f1[2]
        except Exception as e:
            macro_f1 = f1_high = f1_low = f1_med = np.nan
            logger.warning(f"  Could not compute F1 for {group_val}: {e}")

        rows.append({
            "group": group_val,
            "n": n,
            "n_high": n_high,
            "n_low": n_low,
            "n_med": n_med,
            "macro_f1": round(macro_f1, 3),
            "f1_high": round(f1_high, 3),
            "f1_low": round(f1_low, 3),
            "f1_med": round(f1_med, 3),
            "n_low_reliable": n_low >= CONFIG["min_n_low_reliable"],
        })
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


# ----------------------------------------------------------------------------
# 5. Anglo vs Other gap analysis (for §5.7 "19-point gap" claim)
# ----------------------------------------------------------------------------
def compute_anglo_other_gap(df):
    """Compute high-credibility F1 for Anglo (USA + UK + Canada) vs Other,
    and also the legacy 'country_other' bucket specifically."""
    anglo_countries = {"usa", "uk", "canada"}
    df = df.copy()
    df["bucket"] = df["country"].apply(
        lambda c: "Anglo (USA/UK/Canada)" if c in anglo_countries
        else ("Other (country_other bucket)" if c == "other"
              else "Other (named non-Anglo)")
    )

    rows = []
    for bucket, sub in df.groupby("bucket"):
        n = len(sub)
        n_low = (sub["y_true_label"] == "low credibility").sum()
        if n < CONFIG["min_subgroup_n"]:
            continue
        f1_high = f1_score(
            sub["y_true_label"], sub["y_pred_label"],
            labels=["high credibility", "low credibility", "medium credibility"],
            average=None, zero_division=0,
        )[0]
        f1_low = f1_score(
            sub["y_true_label"], sub["y_pred_label"],
            labels=["high credibility", "low credibility", "medium credibility"],
            average=None, zero_division=0,
        )[1]
        macro = f1_score(sub["y_true_label"], sub["y_pred_label"],
                         average="macro", zero_division=0)
        rows.append({
            "bucket": bucket,
            "n": n,
            "n_low": n_low,
            "macro_f1": round(macro, 3),
            "f1_high": round(f1_high, 3),
            "f1_low": round(f1_low, 3),
        })
    out = pd.DataFrame(rows)
    return out


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print("""
    +---------------------------------------------------------------+
    |  V5 SUBGROUP ANALYSIS                                         |
    |  Per-country and per-media-type metrics                       |
    |  Source: preds_Enriched_XGB_SMOTE_weights.csv (v5)            |
    +---------------------------------------------------------------+
    """)

    enriched_df, preds = load_data()
    joined = join_predictions_metadata(enriched_df, preds)

    # Per-country
    logger.info("\nComputing per-country subgroup metrics...")
    country_df = subgroup_metrics(joined, "country")
    country_df.to_csv(os.path.join(CONFIG["output_dir"], "subgroups_country.csv"), index=False)
    logger.info(f"  Saved: subgroups_country.csv ({len(country_df)} groups)")

    # Per-media-type
    logger.info("\nComputing per-media-type subgroup metrics...")
    media_df = subgroup_metrics(joined, "media_type")
    media_df.to_csv(os.path.join(CONFIG["output_dir"], "subgroups_media_type.csv"), index=False)
    logger.info(f"  Saved: subgroups_media_type.csv ({len(media_df)} groups)")

    # Anglo gap analysis
    logger.info("\nComputing Anglo vs Other gap...")
    gap_df = compute_anglo_other_gap(joined)
    gap_df.to_csv(os.path.join(CONFIG["output_dir"], "subgroups_gap_summary.csv"), index=False)
    logger.info(f"  Saved: subgroups_gap_summary.csv")

    # Pretty-printed summary
    summary_lines = []
    summary_lines.append("=" * 78)
    summary_lines.append("V5 SUBGROUP ANALYSIS — Enriched XGB_SMOTE_weights (headline config)")
    summary_lines.append("=" * 78)
    summary_lines.append("")
    summary_lines.append("Per-country breakdown (sorted by n):")
    summary_lines.append("-" * 78)
    summary_lines.append(country_df.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Per-media-type breakdown (sorted by n):")
    summary_lines.append("-" * 78)
    summary_lines.append(media_df.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Anglo vs Other gap analysis:")
    summary_lines.append("-" * 78)
    summary_lines.append(gap_df.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Note: subgroups with n_low < 5 are flagged as statistically")
    summary_lines.append("unreliable for per-class F1 (n_low_reliable = False).")

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(CONFIG["output_dir"], "subgroups_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + summary_text + "\n")
    logger.info(f"\nAll outputs saved to: {CONFIG['output_dir']}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
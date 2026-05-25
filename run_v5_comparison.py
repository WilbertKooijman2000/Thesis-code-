"""
===========================================================================
MODEL COMPARISON v5 -- proper per-fold SelectKBest
===========================================================================
Master Thesis -- Wilbert Kooijman | Tilburg University
Supervisor: Dr. Chris Emmery

Purpose
-------
Re-runs the v4 enriched/baseline pipeline with one methodological fix:
SelectKBest with the ANOVA F-test is refit inside every CV fold rather
than fit once on the full dataset. The v4 implementation fits the
selector globally on (X, y) before any CV split (model_comparison_v4.py
lines 742-747), which means the test fold's labels influence which
features the model gets to see -- a partial leak.

Everything else (Optuna budget, hyperparameter spaces, SMOTE-per-fold,
StandardScaler-per-fold, McNemar analysis, prediction saving, confusion
matrices, final SHAP model) is preserved by importing v4 and only
replacing v4.run_experiment.

Outputs go to a separate model_results_v5/ folder so the v4 results
remain available for before/after comparison.

Usage
-----
    python run_v5_comparison.py

Place this file in the same directory as model_comparison_v4.py.
Expected runtime: similar to v4 (4-8 hours depending on hardware) --
the per-fold refit of a univariate ANOVA F-test adds only a few
seconds per fold.
===========================================================================
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# ---------------------------------------------------------------------------
# Import v4
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import model_comparison_v4 as v4  # noqa: E402

# ---------------------------------------------------------------------------
# Redirect outputs to model_results_v5/ -- v4 results stay intact
# ---------------------------------------------------------------------------
V5_DIR = os.path.join(
    os.path.dirname(v4.CONFIG["output_dir"]),
    "model_results_v5",
)
os.makedirs(V5_DIR, exist_ok=True)
v4.CONFIG["output_dir"] = V5_DIR
v4.PREDS_DIR = os.path.join(V5_DIR, "predictions")
os.makedirs(v4.PREDS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(V5_DIR, "model_comparison_v5.log"),
            mode="w", encoding="utf-8",
        ),
    ],
    force=True,
)
logger = v4.logger


# ===========================================================================
# Replacement run_experiment with per-fold feature selection
# ===========================================================================
def run_experiment_perfold(df, dataset_name, feature_cols, n_folds=5,
                           random_state=42, save_final_xgb_model=False):
    """
    Identical to v4.run_experiment in behaviour and outputs, except:
      * No global SelectKBest fit on (X, y) before CV.
      * Inside each Optuna inner CV fold, SelectKBest is fit on the
        inner training subset only.
      * Inside each outer CV fold, SelectKBest is fit on the outer
        training subset only.
      * Inside the stacking CV loop, same per-fold treatment.
      * The SHAP final model still fits one selector on full data --
        that model is post-hoc interpretation, not a generalisation
        estimate, so this is not a leak.
    Additionally, writes feature_selection_stability_<dataset>.csv
    reporting how many folds each feature was selected in.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EXPERIMENT [v5 per-fold]: {dataset_name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Samples: {len(df)}, Features: {len(feature_cols)}")

    df, feature_cols = v4.engineer_features(df, feature_cols)
    le = LabelEncoder()
    X = df[feature_cols].fillna(0).values
    y = le.fit_transform(df["credibility_class"])
    class_names = le.classes_
    logger.info(f"Classes: {dict(zip(class_names, np.bincount(y)))}")

    # k is fixed; only the selector's fit_transform differs from v4
    k_select = min(len(feature_cols), max(80, int(len(feature_cols) * 0.6)))
    logger.info(f"SelectKBest k = {k_select} (refit per fold)")

    min_class_count = int(min(np.bincount(y)))
    smote_viable = min_class_count >= 6
    skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # ----- Helper: fold-local preprocessing (SELECT then SCALE) -----
    def _fit_select_scale(X_tr_raw, X_val_raw, y_tr):
        sel = SelectKBest(f_classif, k=k_select)
        X_tr_s = sel.fit_transform(X_tr_raw, y_tr)
        X_val_s = sel.transform(X_val_raw)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_s)
        X_val_s = sc.transform(X_val_s)
        return X_tr_s, X_val_s, sel, sc

    # ----- Optuna objectives (now with per-fold selection) -----
    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.1, 0.8),
        }
        scores = []
        for tr_idx, va_idx in skf_inner.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            X_tr, X_va, _, _ = _fit_select_scale(X_tr, X_va, y_tr)
            m = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
            m.fit(X_tr, y_tr)
            scores.append(f1_score(y_va, m.predict(X_va), average="macro"))
        return float(np.mean(scores))

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        scores = []
        for tr_idx, va_idx in skf_inner.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            X_tr, X_va, _, _ = _fit_select_scale(X_tr, X_va, y_tr)
            m = xgb.XGBClassifier(
                **params, random_state=random_state,
                eval_metric="mlogloss", use_label_encoder=False, verbosity=0,
            )
            m.fit(X_tr, y_tr)
            scores.append(f1_score(y_va, m.predict(X_va), average="macro"))
        return float(np.mean(scores))

    logger.info("\n  Tuning RF with Optuna (50 trials)...")
    rf_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    rf_study.optimize(rf_objective, n_trials=50, show_progress_bar=True)
    best_rf_params = rf_study.best_params
    logger.info(f"  Best RF params: {best_rf_params} (CV score: {rf_study.best_value:.4f})")

    logger.info("  Tuning XGB with Optuna (50 trials)...")
    xgb_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
    best_xgb_params = xgb_study.best_params
    logger.info(f"  Best XGB params: {best_xgb_params} (CV score: {xgb_study.best_value:.4f})")

    with open(os.path.join(v4.CONFIG["output_dir"],
                           f"best_params_{dataset_name}.json"), "w") as f:
        json.dump({"rf": best_rf_params, "xgb": best_xgb_params}, f, indent=2)

    # ----- Build configs (identical to v4) -----
    configs = []
    configs.append({
        "name": "RF_no_balance",
        "model": RandomForestClassifier(**best_rf_params, random_state=random_state, n_jobs=-1),
        "smote": False,
    })
    configs.append({
        "name": "XGB_no_balance",
        "model": xgb.XGBClassifier(**best_xgb_params, random_state=random_state,
                                   eval_metric="mlogloss", use_label_encoder=False, verbosity=0),
        "smote": False,
    })
    if smote_viable:
        k_smote = min(5, min_class_count - 1)
        configs.append({
            "name": "RF_SMOTE",
            "model": RandomForestClassifier(**best_rf_params, random_state=random_state, n_jobs=-1),
            "smote": True, "smote_k": k_smote,
        })
        configs.append({
            "name": "XGB_SMOTE",
            "model": xgb.XGBClassifier(**best_xgb_params, random_state=random_state,
                                       eval_metric="mlogloss", use_label_encoder=False, verbosity=0),
            "smote": True, "smote_k": k_smote,
        })

    class_counts = np.bincount(y)
    weight_dict = {i: len(y) / (len(class_counts) * class_counts[i])
                   for i in range(len(class_counts))}
    rf_weighted_params = {**best_rf_params, "class_weight": "balanced"}
    configs.append({
        "name": "RF_class_weights",
        "model": RandomForestClassifier(**rf_weighted_params, random_state=random_state, n_jobs=-1),
        "smote": False,
    })
    configs.append({
        "name": "XGB_class_weights",
        "model": xgb.XGBClassifier(**best_xgb_params, random_state=random_state,
                                   eval_metric="mlogloss", use_label_encoder=False, verbosity=0),
        "smote": False, "sample_weights": weight_dict,
    })
    if smote_viable:
        k_smote = min(5, min_class_count - 1)
        configs.append({
            "name": "RF_SMOTE_weights",
            "model": RandomForestClassifier(**rf_weighted_params, random_state=random_state, n_jobs=-1),
            "smote": True, "smote_k": k_smote,
        })
        configs.append({
            "name": "XGB_SMOTE_weights",
            "model": xgb.XGBClassifier(**best_xgb_params, random_state=random_state,
                                       eval_metric="mlogloss", use_label_encoder=False, verbosity=0),
            "smote": True, "smote_k": k_smote, "sample_weights": weight_dict,
        })

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    all_results = []
    fold_selection_count = {f: 0 for f in feature_cols}

    # ----- Outer CV: per-fold selection -----
    for cfg in configs:
        logger.info(f"\n  Running: {cfg['name']}...")
        fold_metrics = []
        all_fold_ids, all_test_indices, all_predictions, all_truths = [], [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # *** Per-fold SelectKBest, fit on training only ***
            selector = SelectKBest(f_classif, k=k_select)
            X_tr_s = selector.fit_transform(X_tr_raw, y_tr)
            X_te_s = selector.transform(X_te_raw)

            # Record selected features once per fold (only on first config)
            if cfg["name"] == configs[0]["name"]:
                for f, m in zip(feature_cols, selector.get_support()):
                    if m:
                        fold_selection_count[f] += 1

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_s)
            X_te_s = scaler.transform(X_te_s)

            if cfg.get("smote"):
                try:
                    smote = SMOTE(k_neighbors=cfg["smote_k"], random_state=random_state)
                    X_tr_s, y_tr = smote.fit_resample(X_tr_s, y_tr)
                except Exception as e:
                    logger.warning(f"    SMOTE failed on fold {fold_idx}: {e}")

            model = cfg["model"].__class__(**cfg["model"].get_params())
            if cfg.get("sample_weights"):
                sw = np.array([cfg["sample_weights"][yi] for yi in y_tr])
                model.fit(X_tr_s, y_tr, sample_weight=sw)
            else:
                model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)

            all_fold_ids.extend([fold_idx] * len(test_idx))
            all_test_indices.extend(test_idx.tolist())
            all_predictions.extend(y_pred.tolist())
            all_truths.extend(y_te.tolist())
            fold_metrics.append({
                "fold": fold_idx,
                "macro_f1": f1_score(y_te, y_pred, average="macro"),
                "accuracy": accuracy_score(y_te, y_pred),
            })

        preds_arr, truths_arr = np.array(all_predictions), np.array(all_truths)
        v4._save_predictions(
            dataset_name, cfg["name"], all_fold_ids, all_test_indices,
            all_truths, all_predictions, class_names,
        )

        per_class_f1 = f1_score(truths_arr, preds_arr, average=None, labels=range(len(class_names)))
        per_class_p = precision_score(truths_arr, preds_arr, average=None, labels=range(len(class_names)))
        per_class_r = recall_score(truths_arr, preds_arr, average=None, labels=range(len(class_names)))
        macro_scores = [fm["macro_f1"] for fm in fold_metrics]

        result = {
            "dataset": dataset_name,
            "model": cfg["name"],
            "macro_f1_mean": float(np.mean(macro_scores)),
            "macro_f1_std": float(np.std(macro_scores)),
            "accuracy": accuracy_score(truths_arr, preds_arr),
        }
        for i, cls in enumerate(class_names):
            cs = cls.replace(" credibility", "").strip()
            result[f"f1_{cs}"] = per_class_f1[i]
            result[f"precision_{cs}"] = per_class_p[i]
            result[f"recall_{cs}"] = per_class_r[i]
        all_results.append(result)
        logger.info(f"    Macro F1: {result['macro_f1_mean']:.4f} (+/- {result['macro_f1_std']:.4f})")
        for i, cls in enumerate(class_names):
            cs = cls.replace(" credibility", "").strip()
            logger.info(f"    {cs} F1: {per_class_f1[i]:.4f}")
        cm = confusion_matrix(truths_arr, preds_arr)
        v4._save_confusion_matrix(cm, class_names,
                                  f"{dataset_name}__{cfg['name']}",
                                  v4.CONFIG["output_dir"])

    # ----- Feature-selection stability report -----
    sel_df = pd.DataFrame([
        {"feature": f, "n_folds_selected": c}
        for f, c in fold_selection_count.items()
    ]).sort_values("n_folds_selected", ascending=False)
    sel_df.to_csv(
        os.path.join(v4.CONFIG["output_dir"],
                     f"feature_selection_stability_{dataset_name}.csv"),
        index=False,
    )
    n_all_folds = int((sel_df["n_folds_selected"] == n_folds).sum())
    n_some_folds = int(((sel_df["n_folds_selected"] > 0) &
                        (sel_df["n_folds_selected"] < n_folds)).sum())
    n_never = int((sel_df["n_folds_selected"] == 0).sum())
    logger.info(
        f"\n  Feature-selection stability ({dataset_name}): "
        f"{n_all_folds} always, {n_some_folds} sometimes, {n_never} never selected"
    )

    # ----- Stacking with per-fold selection -----
    logger.info("\n  Running: Stacking (RF+XGB meta-learner)...")
    s_fold_ids, s_test_idx, s_preds, s_truths = [], [], [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        selector = SelectKBest(f_classif, k=k_select)
        X_tr_s = selector.fit_transform(X_tr_raw, y_tr)
        X_te_s = selector.transform(X_te_raw)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_s)
        X_te_s = scaler.transform(X_te_s)

        if smote_viable:
            try:
                k_smote = min(5, int(min(np.bincount(y_tr))) - 1)
                smote = SMOTE(k_neighbors=max(1, k_smote), random_state=random_state)
                X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_s, y_tr)
            except Exception:
                X_tr_sm, y_tr_sm = X_tr_s, y_tr
        else:
            X_tr_sm, y_tr_sm = X_tr_s, y_tr

        rf = RandomForestClassifier(**best_rf_params, class_weight="balanced",
                                    random_state=random_state, n_jobs=-1)
        rf.fit(X_tr_sm, y_tr_sm)
        sw = np.array([weight_dict[yi] for yi in y_tr_sm])
        xgb_m = xgb.XGBClassifier(**best_xgb_params, random_state=random_state,
                                  eval_metric="mlogloss", use_label_encoder=False, verbosity=0)
        xgb_m.fit(X_tr_sm, y_tr_sm, sample_weight=sw)
        meta_tr = np.hstack([rf.predict_proba(X_tr_sm), xgb_m.predict_proba(X_tr_sm)])
        meta_te = np.hstack([rf.predict_proba(X_te_s), xgb_m.predict_proba(X_te_s)])
        meta = LogisticRegression(class_weight="balanced", max_iter=1000,
                                  random_state=random_state)
        meta.fit(meta_tr, y_tr_sm)
        y_pred = meta.predict(meta_te)
        s_fold_ids.extend([fold_idx] * len(test_idx))
        s_test_idx.extend(test_idx.tolist())
        s_preds.extend(y_pred.tolist())
        s_truths.extend(y_te.tolist())

    s_preds_arr, s_truths_arr = np.array(s_preds), np.array(s_truths)
    v4._save_predictions(dataset_name, "Stacking_RF_XGB",
                         s_fold_ids, s_test_idx, s_truths, s_preds, class_names)
    per_class_f1 = f1_score(s_truths_arr, s_preds_arr, average=None, labels=range(len(class_names)))
    per_class_p = precision_score(s_truths_arr, s_preds_arr, average=None, labels=range(len(class_names)))
    per_class_r = recall_score(s_truths_arr, s_preds_arr, average=None, labels=range(len(class_names)))
    macro_f1 = f1_score(s_truths_arr, s_preds_arr, average="macro")
    stack_result = {
        "dataset": dataset_name, "model": "Stacking_RF_XGB",
        "macro_f1_mean": float(macro_f1), "macro_f1_std": 0.0,
        "accuracy": accuracy_score(s_truths_arr, s_preds_arr),
    }
    for i, cls in enumerate(class_names):
        cs = cls.replace(" credibility", "").strip()
        stack_result[f"f1_{cs}"] = per_class_f1[i]
        stack_result[f"precision_{cs}"] = per_class_p[i]
        stack_result[f"recall_{cs}"] = per_class_r[i]
    all_results.append(stack_result)
    logger.info(f"    Macro F1: {macro_f1:.4f}")
    for i, cls in enumerate(class_names):
        cs = cls.replace(" credibility", "").strip()
        logger.info(f"    {cs} F1: {per_class_f1[i]:.4f}")
    cm = confusion_matrix(s_truths_arr, s_preds_arr)
    v4._save_confusion_matrix(cm, class_names,
                              f"{dataset_name}__Stacking_RF_XGB",
                              v4.CONFIG["output_dir"])

    # ----- Final XGB for SHAP -----
    # One selector fit on full data is OK here: this model is for
    # post-hoc interpretation, not a generalisation estimate.
    if save_final_xgb_model and smote_viable:
        logger.info(f"\n  Fitting final XGB_SMOTE_weights on full {dataset_name} for SHAP...")
        final_sel = SelectKBest(f_classif, k=k_select)
        X_sel = final_sel.fit_transform(X, y)
        sel_feat_names = [f for f, m in zip(feature_cols, final_sel.get_support()) if m]
        final_scaler = StandardScaler()
        X_full = final_scaler.fit_transform(X_sel)
        try:
            k_smote = min(5, min_class_count - 1)
            smote = SMOTE(k_neighbors=k_smote, random_state=random_state)
            X_full_sm, y_full_sm = smote.fit_resample(X_full, y)
        except Exception as e:
            logger.warning(f"  SMOTE failed when fitting final model: {e}")
            X_full_sm, y_full_sm = X_full, y
        sw = np.array([weight_dict[yi] for yi in y_full_sm])
        final_xgb = xgb.XGBClassifier(**best_xgb_params, random_state=random_state,
                                      eval_metric="mlogloss", use_label_encoder=False, verbosity=0)
        final_xgb.fit(X_full_sm, y_full_sm, sample_weight=sw)
        joblib.dump({
            "model": final_xgb,
            "scaler": final_scaler,
            "selector": final_sel,
            "label_encoder": le,
            "feature_names_engineered": feature_cols,
            "feature_names_selected": sel_feat_names,
            "class_names": list(class_names),
        }, os.path.join(v4.CONFIG["output_dir"], f"final_xgb_{dataset_name}.joblib"))
        logger.info(f"  Saved final model bundle to final_xgb_{dataset_name}.joblib")

    return all_results


# ===========================================================================
# Sanity check: compare v5 results to v4 if available
# ===========================================================================
def compare_against_v4(v5_results_path):
    v4_path = os.path.join(
        os.path.dirname(V5_DIR),
        "model_results_v4",
        "results_latest.csv",
    )
    if not os.path.exists(v4_path):
        logger.info("\n(No v4 results_latest.csv found at expected path; "
                    "skipping before/after comparison.)")
        return
    try:
        v4_df = pd.read_csv(v4_path)
        v5_df = pd.read_csv(v5_results_path)
        merged = v5_df.merge(
            v4_df, on=["dataset", "model"], suffixes=("_v5", "_v4"),
        )
        merged["delta_macro_f1"] = merged["macro_f1_mean_v5"] - merged["macro_f1_mean_v4"]
        if "f1_low_v5" in merged.columns and "f1_low_v4" in merged.columns:
            merged["delta_f1_low"] = merged["f1_low_v5"] - merged["f1_low_v4"]
        out_cols = ["dataset", "model",
                    "macro_f1_mean_v4", "macro_f1_mean_v5", "delta_macro_f1"]
        if "delta_f1_low" in merged.columns:
            out_cols += ["f1_low_v4", "f1_low_v5", "delta_f1_low"]
        comp = merged[out_cols].round(4)
        comp_path = os.path.join(V5_DIR, "v4_vs_v5_comparison.csv")
        comp.to_csv(comp_path, index=False)
        logger.info(f"\n{'=' * 80}")
        logger.info("V4 -> V5 COMPARISON (positive delta = v5 higher)")
        logger.info(f"{'=' * 80}")
        logger.info(f"\n{comp.to_string(index=False)}\n")
        logger.info(f"Comparison saved to: {comp_path}")
    except Exception as e:
        logger.warning(f"Could not produce v4/v5 comparison: {e}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"\n{'=' * 80}")
    logger.info(f"V5 PIPELINE (per-fold SelectKBest) started at {timestamp}")
    logger.info(f"Output directory: {V5_DIR}")
    logger.info(f"{'=' * 80}\n")

    # *** Patch v4 ***
    v4.run_experiment = run_experiment_perfold
    logger.info("Patched v4.run_experiment -> run_experiment_perfold\n")

    # Run v4's main (uses our patched function)
    v4.main()

    # Compare against v4 if available
    v5_results = os.path.join(V5_DIR, "results_latest.csv")
    if os.path.exists(v5_results):
        compare_against_v4(v5_results)


if __name__ == "__main__":
    print("""
    +---------------------------------------------------------------+
    |  MODEL COMPARISON v5                                          |
    |  Per-fold SelectKBest (no feature-selection leakage)          |
    |  Outputs to model_results_v5/                                 |
    +---------------------------------------------------------------+
    """)
    main()
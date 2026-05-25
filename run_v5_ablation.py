"""
============================================================================
RUN v5 ABLATION — Feature-tier comparison WITHOUT feature-selection leakage
============================================================================
Master Thesis - Wilbert | Tilburg University
Supervisor: Dr. Chris Emmery

This wrapper:
  1. Defines a per-fold-SelectKBest replacement for v4.run_experiment.
  2. Monkey-patches v4 so the ablation logic uses the clean version.
  3. Reuses model_ablation_v4.py for the tier-filtering / output logic.
  4. Skips the "Full" tier because it's identical to v5 XGB_SMOTE_weights
     on the enriched dataset (already in thesis_output/model_results_v5/).
  5. Redirects all outputs to thesis_output/model_results_v5_ablation/.
  6. Copies the v4 ablation's data_cache.pkl across to avoid re-extracting
     features from HTML (~75 min saved).

Outputs (in thesis_output/model_results_v5_ablation/):
    results_ablation_<timestamp>.csv
    results_ablation_latest.csv
    ablation_summary.csv
    ablation_tier_comparison.png
    features_<tier>.csv                   (per-tier feature lists)
    feature_stability_<tier>.csv          (NEW: per-fold selection stability)
    cm_Ablation_<tier>__*.png
    predictions/
    run_v5_ablation.log

Place this script in the SAME directory as model_comparison_v4.py
and model_ablation_v4.py, then:
    python run_v5_ablation.py
============================================================================
"""

import os
import sys
import json
import shutil
import logging
import warnings
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score,
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Locate the v4/ablation modules
# ----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ============================================================================
# STEP 1: Import v4 (this triggers module-level setup including logging)
# ============================================================================
import model_comparison_v4 as v4


# ============================================================================
# STEP 2: Define the per-fold replacement for v4.run_experiment
# ============================================================================
# The ONLY difference from v4's original run_experiment is that SelectKBest
# is fitted on each training fold separately (in both the inner Optuna CV
# loops and the outer 5-fold evaluation loop), rather than once on the full
# (X, y) up front. StandardScaler and SMOTE are per-fold in both versions.
# ============================================================================
def run_experiment_perfold(df, dataset_name, feature_cols, n_folds=5,
                           random_state=42, save_final_xgb_model=False):
    logger = v4.logger
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EXPERIMENT (per-fold SelectKBest): {dataset_name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Samples: {len(df)}, Candidate features: {len(feature_cols)}")

    df, feature_cols = v4.engineer_features(df, feature_cols)
    le = LabelEncoder()
    X = df[feature_cols].fillna(0).values
    y = le.fit_transform(df["credibility_class"])
    class_names = le.classes_
    logger.info(f"Classes: {dict(zip(class_names, np.bincount(y)))}")
    logger.info(f"After feature engineering: {len(feature_cols)} candidate features")

    # Same k formula as v4: floor at 80, otherwise 60% of candidates
    n_features_to_keep = min(len(feature_cols), max(80, int(len(feature_cols) * 0.6)))
    logger.info(f"Per-fold SelectKBest target: k = {n_features_to_keep}")

    min_class_count = min(np.bincount(y))
    smote_viable = min_class_count >= 6
    skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # -------- Inner CV objectives (per-fold SelectKBest inside each trial) --
    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.1, 0.8),
        }
        scores = []
        for train_idx, val_idx in skf_inner.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            selector = SelectKBest(f_classif, k=n_features_to_keep)
            X_tr = selector.fit_transform(X_tr, y_tr)
            X_val = selector.transform(X_val)
            model = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds, average="macro"))
        return np.mean(scores)

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
        for train_idx, val_idx in skf_inner.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            selector = SelectKBest(f_classif, k=n_features_to_keep)
            X_tr = selector.fit_transform(X_tr, y_tr)
            X_val = selector.transform(X_val)
            model = xgb.XGBClassifier(
                **params, random_state=random_state,
                eval_metric="mlogloss", use_label_encoder=False, verbosity=0
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds, average="macro"))
        return np.mean(scores)

    # -------- Optuna tuning ----------------------------------------------
    configs = []
    logger.info(f"\n  Tuning RF with Optuna (50 trials, per-fold SelectKBest inside each trial)...")
    rf_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    rf_study.optimize(rf_objective, n_trials=50, show_progress_bar=True)
    best_rf_params = rf_study.best_params
    logger.info(f"  Best RF params: {best_rf_params} (CV score: {rf_study.best_value:.4f})")

    logger.info(f"  Tuning XGB with Optuna (50 trials, per-fold SelectKBest inside each trial)...")
    xgb_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
    best_xgb_params = xgb_study.best_params
    logger.info(f"  Best XGB params: {best_xgb_params} (CV score: {xgb_study.best_value:.4f})")

    with open(os.path.join(v4.CONFIG["output_dir"], f"best_params_{dataset_name}.json"), "w") as f:
        json.dump({"rf": best_rf_params, "xgb": best_xgb_params}, f, indent=2)

    # -------- Config list (identical to v4) ------------------------------
    configs.append({
        "name": "RF_no_balance",
        "model": RandomForestClassifier(**best_rf_params, random_state=random_state, n_jobs=-1),
        "smote": False,
    })
    configs.append({
        "name": "XGB_no_balance",
        "model": xgb.XGBClassifier(
            **best_xgb_params, random_state=random_state,
            eval_metric="mlogloss", use_label_encoder=False, verbosity=0
        ),
        "smote": False,
    })

    if smote_viable:
        k = min(5, min_class_count - 1)
        configs.append({
            "name": "RF_SMOTE",
            "model": RandomForestClassifier(**best_rf_params, random_state=random_state, n_jobs=-1),
            "smote": True,
            "smote_k": k,
        })
        configs.append({
            "name": "XGB_SMOTE",
            "model": xgb.XGBClassifier(
                **best_xgb_params, random_state=random_state,
                eval_metric="mlogloss", use_label_encoder=False, verbosity=0
            ),
            "smote": True,
            "smote_k": k,
        })

    class_counts = np.bincount(y)
    total = len(y)
    n_classes = len(class_counts)
    weight_dict = {i: total / (n_classes * class_counts[i]) for i in range(n_classes)}

    rf_weighted_params = {**best_rf_params, "class_weight": "balanced"}
    configs.append({
        "name": "RF_class_weights",
        "model": RandomForestClassifier(**rf_weighted_params, random_state=random_state, n_jobs=-1),
        "smote": False,
    })
    configs.append({
        "name": "XGB_class_weights",
        "model": xgb.XGBClassifier(
            **best_xgb_params, random_state=random_state,
            eval_metric="mlogloss", use_label_encoder=False, verbosity=0
        ),
        "smote": False,
        "sample_weights": weight_dict,
    })

    if smote_viable:
        k = min(5, min_class_count - 1)
        configs.append({
            "name": "RF_SMOTE_weights",
            "model": RandomForestClassifier(**rf_weighted_params, random_state=random_state, n_jobs=-1),
            "smote": True,
            "smote_k": k,
        })
        configs.append({
            "name": "XGB_SMOTE_weights",
            "model": xgb.XGBClassifier(
                **best_xgb_params, random_state=random_state,
                eval_metric="mlogloss", use_label_encoder=False, verbosity=0
            ),
            "smote": True,
            "smote_k": k,
            "sample_weights": weight_dict,
        })

    # -------- Outer 5-fold CV with per-fold SelectKBest ------------------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    all_results = []

    # Track per-fold selected feature names for stability reporting
    fold_selected_features = []

    for cfg in configs:
        logger.info(f"\n  Running: {cfg['name']}...")

        fold_metrics = []
        all_fold_ids = []
        all_test_indices = []
        all_predictions = []
        all_truths = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # *** PER-FOLD SELECTION (the v5 fix) ***
            selector = SelectKBest(f_classif, k=n_features_to_keep)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

            # Record selected feature names (only for the first config to avoid duplicates)
            if cfg["name"] == configs[0]["name"]:
                mask = selector.get_support()
                fold_selected_features.append(
                    [f for f, s in zip(feature_cols, mask) if s]
                )

            if cfg.get("smote"):
                try:
                    smote = SMOTE(k_neighbors=cfg["smote_k"], random_state=random_state)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                except Exception as e:
                    logger.warning(f"    SMOTE failed on fold {fold_idx}: {e}")

            model = cfg["model"].__class__(**cfg["model"].get_params())
            if cfg.get("sample_weights"):
                sw = np.array([cfg["sample_weights"][yi] for yi in y_train])
                model.fit(X_train, y_train, sample_weight=sw)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            all_fold_ids.extend([fold_idx] * len(test_idx))
            all_test_indices.extend(test_idx.tolist())
            all_predictions.extend(y_pred.tolist())
            all_truths.extend(y_test.tolist())

            fold_metrics.append({
                "fold": fold_idx,
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
                "accuracy": accuracy_score(y_test, y_pred),
            })

        all_preds_arr = np.array(all_predictions)
        all_truths_arr = np.array(all_truths)

        v4._save_predictions(
            dataset_name, cfg["name"],
            all_fold_ids, all_test_indices,
            all_truths, all_predictions, class_names,
        )

        per_class_f1 = f1_score(all_truths_arr, all_preds_arr, average=None, labels=range(len(class_names)))
        per_class_precision = precision_score(all_truths_arr, all_preds_arr, average=None, labels=range(len(class_names)))
        per_class_recall = recall_score(all_truths_arr, all_preds_arr, average=None, labels=range(len(class_names)))
        macro_f1_scores = [fm["macro_f1"] for fm in fold_metrics]

        result = {
            "dataset": dataset_name,
            "model": cfg["name"],
            "macro_f1_mean": np.mean(macro_f1_scores),
            "macro_f1_std": np.std(macro_f1_scores),
            "accuracy": accuracy_score(all_truths_arr, all_preds_arr),
        }
        for i, cls in enumerate(class_names):
            cls_short = cls.replace(" credibility", "").strip()
            result[f"f1_{cls_short}"] = per_class_f1[i]
            result[f"precision_{cls_short}"] = per_class_precision[i]
            result[f"recall_{cls_short}"] = per_class_recall[i]

        all_results.append(result)

        logger.info(f"    Macro F1: {result['macro_f1_mean']:.4f} (+/- {result['macro_f1_std']:.4f})")
        for i, cls in enumerate(class_names):
            cls_short = cls.replace(" credibility", "").strip()
            logger.info(f"    {cls_short} F1: {per_class_f1[i]:.4f}")

        cm = confusion_matrix(all_truths_arr, all_preds_arr)
        v4._save_confusion_matrix(
            cm, class_names, f"{dataset_name}__{cfg['name']}",
            v4.CONFIG["output_dir"],
        )

    # -------- Feature-selection stability (per-fold) ---------------------
    if fold_selected_features:
        feature_counts = Counter()
        for selected in fold_selected_features:
            feature_counts.update(selected)
        n_folds_actual = len(fold_selected_features)
        always = [f for f, c in feature_counts.items() if c == n_folds_actual]
        sometimes = [f for f, c in feature_counts.items() if 0 < c < n_folds_actual]
        never = [f for f in feature_cols if f not in feature_counts]
        logger.info(
            f"\n  Feature-selection stability ({dataset_name}): "
            f"{len(always)} always, {len(sometimes)} sometimes, "
            f"{len(never)} never selected"
        )

        stability_df = pd.DataFrame([
            {"feature": f, "folds_selected": feature_counts.get(f, 0)}
            for f in feature_cols
        ]).sort_values("folds_selected", ascending=False)
        stability_df.to_csv(
            os.path.join(v4.CONFIG["output_dir"], f"feature_stability_{dataset_name}.csv"),
            index=False,
        )

    # -------- Stacking (also per-fold SelectKBest) -----------------------
    logger.info(f"\n  Running: Stacking (RF+XGB meta-learner)...")
    stack_fold_ids = []
    stack_test_indices = []
    stack_preds = []
    stack_truths = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # *** PER-FOLD SELECTION ***
        selector = SelectKBest(f_classif, k=n_features_to_keep)
        X_train_s = selector.fit_transform(X_train_s, y_train)
        X_test_s = selector.transform(X_test_s)

        if smote_viable:
            try:
                k = min(5, min(np.bincount(y_train)) - 1)
                smote = SMOTE(k_neighbors=max(1, k), random_state=random_state)
                X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)
            except Exception:
                X_train_sm, y_train_sm = X_train_s, y_train
        else:
            X_train_sm, y_train_sm = X_train_s, y_train

        rf = RandomForestClassifier(**best_rf_params, class_weight="balanced",
                                    random_state=random_state, n_jobs=-1)
        rf.fit(X_train_sm, y_train_sm)
        sw = np.array([weight_dict[yi] for yi in y_train_sm])
        xgb_model = xgb.XGBClassifier(
            **best_xgb_params, random_state=random_state,
            eval_metric="mlogloss", use_label_encoder=False, verbosity=0
        )
        xgb_model.fit(X_train_sm, y_train_sm, sample_weight=sw)
        rf_proba_train = rf.predict_proba(X_train_sm)
        xgb_proba_train = xgb_model.predict_proba(X_train_sm)
        meta_train = np.hstack([rf_proba_train, xgb_proba_train])
        rf_proba_test = rf.predict_proba(X_test_s)
        xgb_proba_test = xgb_model.predict_proba(X_test_s)
        meta_test = np.hstack([rf_proba_test, xgb_proba_test])
        meta_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
        meta_model.fit(meta_train, y_train_sm)
        y_pred = meta_model.predict(meta_test)

        stack_fold_ids.extend([fold_idx] * len(test_idx))
        stack_test_indices.extend(test_idx.tolist())
        stack_preds.extend(y_pred.tolist())
        stack_truths.extend(y_test.tolist())

    stack_preds_arr = np.array(stack_preds)
    stack_truths_arr = np.array(stack_truths)
    v4._save_predictions(
        dataset_name, "Stacking_RF_XGB",
        stack_fold_ids, stack_test_indices,
        stack_truths, stack_preds, class_names,
    )

    per_class_f1 = f1_score(stack_truths_arr, stack_preds_arr, average=None, labels=range(len(class_names)))
    per_class_precision = precision_score(stack_truths_arr, stack_preds_arr, average=None, labels=range(len(class_names)))
    per_class_recall = recall_score(stack_truths_arr, stack_preds_arr, average=None, labels=range(len(class_names)))
    macro_f1 = f1_score(stack_truths_arr, stack_preds_arr, average="macro")
    stack_result = {
        "dataset": dataset_name,
        "model": "Stacking_RF_XGB",
        "macro_f1_mean": macro_f1,
        "macro_f1_std": 0.0,
        "accuracy": accuracy_score(stack_truths_arr, stack_preds_arr),
    }
    for i, cls in enumerate(class_names):
        cls_short = cls.replace(" credibility", "").strip()
        stack_result[f"f1_{cls_short}"] = per_class_f1[i]
        stack_result[f"precision_{cls_short}"] = per_class_precision[i]
        stack_result[f"recall_{cls_short}"] = per_class_recall[i]
    all_results.append(stack_result)
    logger.info(f"    Macro F1: {macro_f1:.4f}")
    for i, cls in enumerate(class_names):
        cls_short = cls.replace(" credibility", "").strip()
        logger.info(f"    {cls_short} F1: {per_class_f1[i]:.4f}")

    cm = confusion_matrix(stack_truths_arr, stack_preds_arr)
    v4._save_confusion_matrix(
        cm, class_names, f"{dataset_name}__Stacking_RF_XGB",
        v4.CONFIG["output_dir"],
    )

    # Ablation never asks to save the final model (save_final_xgb_model=False),
    # so we don't need the final-model branch here.

    return all_results


# ============================================================================
# STEP 3: Apply the monkey-patch BEFORE importing the ablation module
# ============================================================================
v4.run_experiment = run_experiment_perfold
v4.logger.info("Patched v4.run_experiment -> run_experiment_perfold (per-fold SelectKBest)")


# ============================================================================
# STEP 4: Import the ablation module
# (this triggers its module-level setup, which would otherwise write to
#  model_results_v4_ablation/; we override paths immediately after)
# ============================================================================
import model_ablation_v4 as ablation


# ============================================================================
# STEP 5: Redirect ALL outputs to model_results_v5_ablation/
# ============================================================================
V5_ABL_DIR = os.path.join(
    os.path.dirname(v4.CONFIG["output_dir"].rstrip(os.sep)),
    "model_results_v5_ablation",
)
os.makedirs(V5_ABL_DIR, exist_ok=True)

# (a) ablation module's own constants
ablation.ABLATION_DIR = V5_ABL_DIR
ablation.CACHE_PATH = os.path.join(V5_ABL_DIR, "data_cache.pkl")

# (b) v4 CONFIG used inside the patched run_experiment
v4.CONFIG["output_dir"] = V5_ABL_DIR
v4.PREDS_DIR = os.path.join(V5_ABL_DIR, "predictions")
os.makedirs(v4.PREDS_DIR, exist_ok=True)


# ============================================================================
# STEP 6: Copy the v4 ablation data cache (avoids the ~75-min rebuild)
# ============================================================================
V4_ABL_DIR = os.path.join(
    os.path.dirname(V5_ABL_DIR.rstrip(os.sep)),
    "model_results_v4_ablation",
)
V4_CACHE = os.path.join(V4_ABL_DIR, "data_cache.pkl")

if os.path.exists(V4_CACHE) and not os.path.exists(ablation.CACHE_PATH):
    shutil.copy2(V4_CACHE, ablation.CACHE_PATH)
    print(f"[v5-ablation] Reused v4 ablation cache -> {ablation.CACHE_PATH}")
elif not os.path.exists(V4_CACHE):
    print(f"[v5-ablation] WARN: v4 ablation cache not found at {V4_CACHE}")
    print(f"[v5-ablation] The script will rebuild features from HTML (~75 min)")
else:
    print(f"[v5-ablation] Using existing v5 ablation cache at {ablation.CACHE_PATH}")


# ============================================================================
# STEP 7: Drop the "Full" tier
# (already computed in v5 main run as Enriched / XGB_SMOTE_weights)
# ============================================================================
ablation.TIERS = {
    "NoMBFC":         ablation.MBFC_PATTERNS,
    "PureStructural": ablation.MBFC_PATTERNS + ablation.CONTENT_PATTERNS,
}


# ============================================================================
# STEP 8: Reconfigure logging to write into the v5 ablation folder
# ============================================================================
log_path = os.path.join(V5_ABL_DIR, "run_v5_ablation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ],
    force=True,
)


# ============================================================================
# STEP 9: Run
# ============================================================================
if __name__ == "__main__":
    print("""
    +---------------------------------------------------------------+
    |  V5 ABLATION (per-fold SelectKBest, leakage-free)             |
    |  Tiers: NoMBFC + PureStructural                               |
    |  (Full tier already covered by v5 main run)                   |
    |  Outputs: thesis_output/model_results_v5_ablation/            |
    +---------------------------------------------------------------+
    """)
    ablation.main()
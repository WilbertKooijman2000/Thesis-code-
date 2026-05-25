import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from model_comparison_v4 import engineer_features, get_feature_columns

DATA_FILE = r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\model_results_v5\enriched_df_cache.pkl"

df = pd.read_pickle(DATA_FILE)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# DON'T filter — keep all rows. "baseline" + "enriched" together = the full enriched config (n=3942)
print(f"Dataset value counts: {df['dataset'].value_counts().to_dict()}")
print(f"Credibility class counts: {df['credibility_class'].value_counts().to_dict()}")

# Apply engineering
feature_cols = get_feature_columns(df)
df_eng, all_cols = engineer_features(df, feature_cols)
engineered = [c for c in all_cols if c not in feature_cols]
print(f"Engineered features: {len(engineered)}")

# Run f_classif on engineered features against the 3-class label
X = df_eng[engineered].fillna(0).replace([np.inf, -np.inf], 0)
y = df_eng["credibility_class"]
f_scores, p_values = f_classif(X, y)

out = pd.DataFrame({
    "feature_name": engineered,
    "f_score": f_scores,
    "p_value": p_values,
}).sort_values("f_score", ascending=False, na_position="last")

out.to_csv("engineered_fstats_35.csv", index=False)
print("\nSaved engineered_fstats_35.csv")
print(out.to_string(index=False))
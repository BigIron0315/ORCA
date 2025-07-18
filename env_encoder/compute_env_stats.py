# compute_env_stats.py

import os
import json
import glob
import pandas as pd

INPUT_FOLDER = "../dataset"
OUTPUT_JSON  = "global_env_stats.json"

NUM_FEATURES = ["Users", "DL_Buffer", "Avg_SNR_dB"]
CAT_FEATURE  = "Slice"

# --- Accumulate raw values ---------------------------------------------
values_num = {c: [] for c in NUM_FEATURES}
categories = set()

for csv_path in glob.glob(os.path.join(INPUT_FOLDER, "ORAN_log_*.csv")):
    df = pd.read_csv(csv_path)

    for col in NUM_FEATURES:
        if col in df.columns:
            values_num[col].extend(df[col].dropna().tolist())

    if CAT_FEATURE in df.columns:
        categories.update(df[CAT_FEATURE].dropna().unique())

# --- Compute mean / std -------------------------------------------------
stats = {}
for col in NUM_FEATURES:
    if values_num[col]:
        series = pd.Series(values_num[col])
        stats[col] = {
            "mean": float(series.mean()),
            "std":  float(series.std())
        }

# --- Add categorical list ----------------------------------------------
stats[CAT_FEATURE] = {
    "values": sorted(categories)
}

# --- Save JSON ----------------------------------------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(stats, f, indent=2)

print(f"✅ Global stats (pruned) saved to {OUTPUT_JSON}")

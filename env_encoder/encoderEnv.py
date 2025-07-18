import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""
Environment‑vector encoder (pruned version)
-------------------------------------------
* **Keeps only**: Users, DL_Buffer, Avg_SNR_dB (numeric) and Slice (categorical).
* **No stability test** – every selected feature is always encoded.
* Uses the same global stats JSON for z‑score normalisation of numeric features.
"""

# === Paths ===
input_folder  = "../dataset"
output_folder = "encoded_env"
os.makedirs(output_folder, exist_ok=True)

stats_path = "./global_env_stats.json"
with open(stats_path, "r") as f:
    global_stats = json.load(f)

# === Feature definitions (pruned) ===
NUM_FEATURES = ["Users", "DL_Buffer", "Avg_SNR_dB"]
CAT_FEATURES = ["Slice"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

for path in sorted(glob.glob(f"{input_folder}/ORAN_log_*.csv")):
    df = pd.read_csv(path)
    fname = os.path.splitext(os.path.basename(path))[0]

    print(f"\n📂 Encoding {fname} ...")

    encoded_vec   = []
    feature_dims  = {}

    # --- numeric features ------------------------------------------------
    for col in NUM_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing expected numeric column: {col}")
        mean, std = global_stats[col]["mean"], global_stats[col]["std"]
        col_mean  = df[col].mean()
        norm_val  = (col_mean - mean) / std if std > 0 else 0.0
        print(f"  🔢 {col}: raw={col_mean:.4f}, norm={norm_val:.4f}")
        encoded_vec.append(norm_val)
        feature_dims[col] = 1

    # --- categorical: Slice ---------------------------------------------
    if "Slice" not in df.columns:
        raise ValueError("Missing expected categorical column: Slice")
    cats = sorted(global_stats["Slice"]["values"])
    val  = df["Slice"].iloc[0]
    encoder = OneHotEncoder(categories=[[c for c in cats]], sparse_output=False, handle_unknown="ignore")
    onehot  = encoder.fit_transform(df[["Slice"]].head(1))[0]
    print(f"  🏷️  Slice: value={val}, one‑hot={onehot.tolist()}")
    encoded_vec.extend(onehot)
    feature_dims["Slice"] = len(cats)

    vec_np = np.array(encoded_vec, dtype=float)
    print(f"  ✅ Encoded vector shape: {vec_np.shape}, values: {vec_np}")

    np.save(os.path.join(output_folder, f"{fname}_env_vector.npy"), vec_np)

    with open(os.path.join(output_folder, f"{fname}_env_vector.txt"), "w") as f:
        f.write(f"Feature dims: {feature_dims}\n")
        f.write("Encoded vector:\n" + np.array2string(vec_np, precision=4))

print("\n✅ Environment encoding (pruned) completed.")

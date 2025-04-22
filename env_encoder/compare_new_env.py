import os
import glob
import numpy as np
import json

"""
compare_all_new_envs_v2.py
---------------------------
* For each encoded ORAN_log_new*.npy vector:
  - Compare it against all other environment vectors (excluding other new logs).
  - Compute numeric + soft slice distance.
  - Save top-3 closest environments (excluding new ones) into a JSON file.
"""

ENCODED_DIR = "encoded_env"
NEW_PREFIX = "ORAN_log_new"
output_folder = "compare_env"
os.makedirs(output_folder, exist_ok=True)

NUM_DIMS = 3  # Users_z, DL_Buffer_z, Avg_SNR_z

# --- Load all new environment vectors -----------------------------
new_files = sorted(glob.glob(os.path.join(ENCODED_DIR, f"{NEW_PREFIX}*_env_vector.npy")))

if not new_files:
    raise FileNotFoundError(f"No files found with prefix '{NEW_PREFIX}' in {ENCODED_DIR}")

for new_path in new_files:
    NEW_NAME = os.path.basename(new_path).replace("_env_vector.npy", "")
    vec_new = np.load(new_path)
    print(f"\n📄 Loaded {NEW_NAME}: shape={vec_new.shape}, values={np.round(vec_new, 4)}")

    results = []
    for other_path in glob.glob(os.path.join(ENCODED_DIR, "*_env_vector.npy")):
        other_name = os.path.basename(other_path).replace("_env_vector.npy", "")

        # Skip self and other new logs
        if other_name.startswith(NEW_PREFIX):
            continue

        vec_other = np.load(other_path)
        if vec_other.shape != vec_new.shape:
            print(f"⚠️  Skipping {other_name}: shape mismatch {vec_other.shape}")
            continue

        # --- numeric distance ---------------------------------------
        dist_num = np.linalg.norm(vec_new[:NUM_DIMS] - vec_other[:NUM_DIMS])

        # --- soft slice penalty -------------------------------------
        same_slice = np.allclose(vec_new[NUM_DIMS:], vec_other[NUM_DIMS:])
        dist = dist_num if same_slice else dist_num * 2.0

        results.append((other_name, dist))

    # --- Report top 3 ----------------------------------------------
    topk = sorted(results, key=lambda x: x[1])[:3]

    print(f"\n🔍 Top 3 closest to {NEW_NAME}:")
    for name, dist in topk:
        print(f"   • {name:25s} → Distance: {dist:.4f}")

    # --- Save to file ----------------------------------------------
    out_json = os.path.join(output_folder, f"topk_similar_envs_{NEW_NAME}.json")
    with open(out_json, "w") as f:
        json.dump([{"name": n, "distance": d} for n, d in topk], f, indent=2)

    print(f"📁 Saved to {out_json}")

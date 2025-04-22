import numpy as np
import os
import json
from pathlib import Path

INPUT_ENV_ID = os.environ.get("INPUT_ENV_ID", "0")
# === CONFIG ===
SHAP_DIR = "./shap_outputs"
OUTPUT_PATH = f"./interim_results/_9_shap_output/env_new{INPUT_ENV_ID}/extrapolated.json"

# === Reference environments for extrapolation
# === Load top-2 reference environments based on decoded difference file
json_path = f"./env_encoder/decoded_differences/decoded_differences_ORAN_log_new{INPUT_ENV_ID}.json"
with open(json_path, "r") as f:
    decoded = json.load(f)

# Sort environments by distance (ascending)
sorted_envs = sorted(decoded, key=lambda x: x["distance"])
envs = [sorted_envs[0]["env_name"], sorted_envs[1]["env_name"]]

print(f"🔍 Using top-2 reference environments for extrapolation: {envs}")

kpm_targets = ["Throughput_Mbps", "Avg_Delay_ms", "user_throughput"]

# === Load SHAP vectors and extrapolate with 2x projection ===
extrapolated_shap = {}

for kpm in kpm_targets:
    vectors = []
    features_list = []

    for env in envs:
        shap_path = f"{SHAP_DIR}/{env}_{kpm}_mean_abs.npy"
        if not os.path.exists(shap_path):
            print(f"❌ Missing SHAP file: {shap_path}")
            continue

        loaded = np.load(shap_path, allow_pickle=True)
        features = loaded[0]
        values = loaded[1].astype(float)
        vectors.append(values)
        features_list = features

    if len(vectors) != 2:
        print(f"⚠️ Could not extrapolate {kpm} — missing vectors")
        continue
    # Print SHAP values from env1 and env2 before extrapolation
    print(f"\n🔍 SHAP values for {kpm} from reference environments:")
    print(f"   • {envs[0]}:")
    for feat, val in zip(features_list, vectors[0]):
        print(f"     - {feat:16s}: {val:.6f}")

    print(f"\n   • {envs[1]}:")
    for feat, val in zip(features_list, vectors[1]):
        print(f"     - {feat:16s}: {val:.6f}")
    # Directional extrapolation: v_new = v2 + 2 * (v2 - v1)
    vec1, vec2 = vectors
    d_new_to_env1 = sorted_envs[0]["distance"]
    d_env2_to_env1 = abs(sorted_envs[1]["distance"] - sorted_envs[0]["distance"])

    if d_env2_to_env1 == 0:
        print("⚠️ env1 and env2 have equal distance — falling back to vec1")
        extrapolated = vec1
    else:
        alpha = d_new_to_env1 / d_env2_to_env1
        print(f"📐 Computed α = {alpha:.4f}")

        # Extrapolate or interpolate: move α units along (vec1 - vec2)
        extrapolated = vec1 + alpha * (vec1 - vec2)

        if alpha < 1:
            print("📊 Interpolating between env1 and env2")
        else:
            print("📈 Extrapolating beyond env2 toward new")


    result = {feat: round(max(val, 0), 6) for feat, val in zip(features_list, extrapolated)}
    extrapolated_shap[kpm] = result

    print(f"\n📈 Extrapolated SHAP (from 1→2→new) for {kpm}")
    for f, v in result.items():
        print(f"   • {f:16s}: {v:.6f}")
    print(f"✅ Total Sum: {np.sum(list(result.values())):.6f}")

# === Save to file ===
Path(os.path.dirname(OUTPUT_PATH)).mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(extrapolated_shap, f, indent=2)

print(f"\n💾 Saved to {OUTPUT_PATH}")

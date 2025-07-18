import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

INPUT_ENV_ID = os.environ.get("INPUT_ENV_ID", "0")

# === Paths ===
actual_prefix = f"shap_outputs/ORAN_log_new{INPUT_ENV_ID}"
llm_prefix = f"interim_results/_9_shap_output/env_new{INPUT_ENV_ID}/LLM_shap_output"
extrap_path = f"interim_results/_9_shap_output/env_new{INPUT_ENV_ID}/extrapolated.json"
no_know_path = f"interim_results/_9_shap_output/env_new{INPUT_ENV_ID}/no_external_knowledge_llm.json"
SA_RAG_path = f"interim_results/_9_shap_output/env_new{INPUT_ENV_ID}/llm_SA_RAG.json"
pureLLM_path = f"interim_results/_9_shap_output/env_new{INPUT_ENV_ID}/pureLLM.json"
decoded_path = f"./env_encoder/decoded_differences/decoded_differences_ORAN_log_new{INPUT_ENV_ID}.json"

# === Detect slice type ===
with open(decoded_path, "r") as f:
    decoded = json.load(f)

slice_str = decoded[0]["env_name"].lower()
slice_type = "urllc" if "urllc" in slice_str else "embb" if "embb" in slice_str else "unknown"
print(f"📂 Detected slice type: {slice_type}")

if slice_type == "embb":
    targets = ["Throughput_Mbps"]
elif slice_type == "urllc":
    targets = ["Avg_Delay_ms"]
else:
    targets = ["user_throughput", "Avg_Delay_ms", "Throughput_Mbps"]

# === Load extrapolated and no-knowledge results
def normalize_keys(d):
    new_d = {}
    for kpm, val_dict in d.items():
        new_d[kpm] = {}
        for feat, val in val_dict.items():
            key = "Scheduling" if feat.startswith("Scheduling") else feat
            new_d[kpm][key] = val
    return new_d

with open(extrap_path, "r") as f:
    extrap_data = normalize_keys(json.load(f))
with open(no_know_path, "r") as f:
    no_know_data = normalize_keys(json.load(f))
with open(SA_RAG_path, "r") as f:
    SA_RAG_data = normalize_keys(json.load(f))
with open(pureLLM_path, "r") as f:
    pureLLM_data = normalize_keys(json.load(f))

# === Load LLM shap outputs (up to 10 files)
llm_raw = [{} for _ in range(10)]
for i in range(1, 10):
    path = f"{llm_prefix}_{i}.json"
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        data = json.load(f)
        llm_raw[i] = normalize_keys(data)

# === Evaluation metrics
def cosine_error(a, b):
    sim = cosine_similarity([a], [b])[0][0]
    return 1 - sim

def rmse(a, b):
    a = np.array(a)
    b = np.array(b)
    mse = np.mean((a - b) ** 2)
    return np.sqrt(mse)

def nrmse_max(a, b):
    """RMSE divided by largest |actual| value (0‑safe)."""
    denom = np.abs(a).max()
    return rmse(a, b) / denom if denom > 0 else 0.0

# === Evaluate per target
for target in targets:
    actual_path = f"{actual_prefix}_{target}_mean_abs.npy"
    if not os.path.exists(actual_path):
        print(f"❌ Missing actual SHAP: {actual_path}")
        continue

    # === Load actual
    arr = np.load(actual_path, allow_pickle=True)
    raw_feats = arr[0].tolist()
    raw_vals = arr[1].astype(float).tolist()

    actual_map = {}
    for feat, val in zip(raw_feats, raw_vals):
        key = "Scheduling" if feat.startswith("Scheduling") else feat
        actual_map[key] = actual_map.get(key, 0.0) + val

    # === Union of all features
    feature_union = set(actual_map.keys())
    for i in range(1, 10):
        if target in llm_raw[i]:
            feature_union.update(llm_raw[i][target].keys())
    feature_union.update(extrap_data.get(target, {}).keys())
    feature_union.update(no_know_data.get(target, {}).keys())
    feature_union.update(SA_RAG_data.get(target, {}).keys())
    feature_union.update(pureLLM_data.get(target, {}).keys())
    
    feature_list = sorted(feature_union)

    # === Create actual vector
    actual_vec = [actual_map.get(f, 0.0) for f in feature_list]

    # === Compare variants
    print(f"\n📊 [Target: {target}] Evaluation vs Actual SHAP:")
    def evaluate_and_print(name, pred_dict):
        if target not in pred_dict:
            return
        pred_vec = [pred_dict[target].get(f, 0.0) for f in feature_list]
        cos_err = cosine_error(actual_vec, pred_vec)
        error_rmse = rmse(actual_vec, pred_vec)
        nr = nrmse_max(actual_vec, pred_vec)
        print(f"   • {name:<20s} | Cosine Error: {cos_err:.4f} | RMSE: {error_rmse:.4f} | NRMSE_max: {nr:.4f}")

    for i in range(1, 10):
        if target in llm_raw[i]:
            evaluate_and_print(f"LLM_{i}", llm_raw[i])

    evaluate_and_print("Extrapolated", extrap_data)
    evaluate_and_print("LLM_No_Knowledge", no_know_data)
    evaluate_and_print("SA_RAG", SA_RAG_data)
    evaluate_and_print("pureLLM", pureLLM_data)

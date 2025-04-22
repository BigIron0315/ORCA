import os
import json
import numpy as np
import matplotlib.pyplot as plt

INPUT_ENV_ID = os.environ.get("INPUT_ENV_ID", "0")

# === Configuration ===
actual_prefix = f"shap_outputs/ORAN_log_new{INPUT_ENV_ID}"
llm_prefix = f"interim_results/_9_shap_output/env_new{INPUT_ENV_ID}/LLM_shap_output"
extrap_path = f"interim_results/_9_shap_output/env_new{INPUT_ENV_ID}/extrapolated.json"
no_know_path = f"interim_results/_9_shap_output/env_new{INPUT_ENV_ID}/no_external_knowledge_llm.json"
targets = ["user_throughput", "Avg_Delay_ms", "Throughput_Mbps"]

# === Load LLM outputs (files: LLM_shap_output_1.json ~ _9.json) ===
llm_raw = [{} for _ in range(10)]  # support up to 10 files
for i in range(1, 10):
    path = f"{llm_prefix}_{i}.json"
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        data = json.load(f)
        llm_raw[i] = {}
        for target in data:
            llm_raw[i][target] = {}
            for feat, val in data[target].items():
                feat_key = "Scheduling" if feat.startswith("Scheduling") else feat
                llm_raw[i][target][feat_key] = val

# === Load extrapolated and no-knowledge LLM SHAP values ===
with open(extrap_path, "r") as f:
    extrap_data = json.load(f)
with open(no_know_path, "r") as f:
    no_know_data = json.load(f)

# Normalize keys for extrap and no-knowledge responses
def normalize_keys(d):
    new_d = {}
    for kpm, val_dict in d.items():
        new_d[kpm] = {}
        for feat, val in val_dict.items():
            key = "Scheduling" if feat.startswith("Scheduling") else feat
            new_d[kpm][key] = val
    return new_d

extrap_data = normalize_keys(extrap_data)
no_know_data = normalize_keys(no_know_data)

# === Load and plot per target ===
for target in targets:
    actual_path = f"{actual_prefix}_{target}_mean_abs.npy"
    if not os.path.exists(actual_path):
        print(f"❌ Missing actual SHAP: {actual_path}")
        continue

    # === Load actual SHAP ===
    data = np.load(actual_path, allow_pickle=True)
    raw_feats = data[0].tolist()
    raw_vals = data[1].astype(float).tolist()

    # Merge "Scheduling_*" into "Scheduling"
    actual_vals_map = {}
    for feat, val in zip(raw_feats, raw_vals):
        key = "Scheduling" if feat.startswith("Scheduling") else feat
        actual_vals_map[key] = actual_vals_map.get(key, 0.0) + val

    # === Get full feature set from all sources
    all_feats = set(actual_vals_map.keys())
    for llm in llm_raw:
        if target in llm:
            all_feats.update(llm[target].keys())
    all_feats.update(extrap_data.get(target, {}).keys())
    all_feats.update(no_know_data.get(target, {}).keys())

    features = sorted(all_feats)

    # === Get actual SHAP vector
    actual_vals = [actual_vals_map.get(f, 0.0) for f in features]

    # === Collect LLM variants
    llm_vals_by_file = []
    for i in range(1, 10):
        if target in llm_raw[i]:
            vals = [llm_raw[i][target].get(f, 0.0) for f in features]
            llm_vals_by_file.append((f"LLM_{i}", vals))

    # Add extrapolation
    if target in extrap_data:
        extrap_vals = [extrap_data[target].get(f, 0.0) for f in features]
        llm_vals_by_file.append(("Extrapolated", extrap_vals))

    # Add no-knowledge LLM
    if target in no_know_data:
        no_know_vals = [no_know_data[target].get(f, 0.0) for f in features]
        llm_vals_by_file.append(("LLM_No_Knowledge", no_know_vals))

    # === Plot
    num_bars = 1 + len(llm_vals_by_file)  # Actual + N LLMs
    x = np.arange(len(features))
    width = 0.8 / num_bars

    plt.figure(figsize=(12, 5))
    plt.bar(x - width * (num_bars - 1) / 2, actual_vals, width, label="Actual SHAP")
    for j, (label, llm_vals) in enumerate(llm_vals_by_file):
        offset = (j + 1) * width - width * (num_bars - 1) / 2
        plt.bar(x + offset, llm_vals, width, label=label)

    plt.xticks(x, features, rotation=45, ha="right")
    plt.ylabel("Mean |SHAP value|")
    plt.title(f"SHAP Comparison — {target}")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    out_path = f"interim_results/_9_shap_output/env_new{INPUT_ENV_ID}/compare_{target}_shap.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")

import os
import json
import numpy as np
import pandas as pd
import glob

# === Configurations ===
encoded_dir = "encoded_env"
dataset_dir = "../dataset"
new_env_csvs = sorted(glob.glob("../dataset/ORAN_log_new*.csv"))
top_k = 3
compare_results_dir = "./compare_env"
output_folder = "./decoded_differences"
os.makedirs(output_folder, exist_ok=True)

numerical_features = [
    "Users", "PRB_num", "DL_Buffer",
    "TxPower", "Avg_SNR_dB", "ant_tilt_deg", "CIO"
]
categorical_features = ["Slice", "Scheduling"]

if not new_env_csvs:
    raise FileNotFoundError("❌ No new environment CSVs found with pattern: ../dataset/ORAN_log_new*.csv")

# === Process each new environment file
for new_env_csv in new_env_csvs:
    new_env_name = os.path.splitext(os.path.basename(new_env_csv))[0]

    results_from_compare_script = os.path.join(compare_results_dir, f"topk_similar_envs_{new_env_name}.json")
    output_path = os.path.join(output_folder, f"decoded_differences_{new_env_name}.json")
    output_txt_path = os.path.join(output_folder, f"decoded_differences_{new_env_name}.txt")

    if not os.path.exists(results_from_compare_script):
        print(f"⚠️ Skipping {new_env_name}: Missing compare results file {results_from_compare_script}")
        continue

    new_df = pd.read_csv(new_env_csv)

    with open(results_from_compare_script, "r") as f:
        topk_envs = json.load(f)

    all_differences = []
    print_lines = []

    for entry in topk_envs[:top_k]:
        name = entry["name"]
        dist = entry["distance"]
        compare_csv = os.path.join(dataset_dir, name + ".csv")
        if not os.path.exists(compare_csv):
            print(f"⚠️ Missing file: {compare_csv}")
            continue

        old_df = pd.read_csv(compare_csv)

        print(f"\n🔍 Comparing {new_env_name} with: {name} (Distance = {dist:.4f})")
        print_lines.append(f"🔍 Comparing with: {name} (Distance = {dist:.4f})\n")
        differences = {}

        for col in numerical_features:
            if col in new_df.columns and col in old_df.columns:
                new_val = new_df[col].mean()
                old_val = old_df[col].mean()
                delta = new_val - old_val
                differences[col] = {
                    "new": round(new_val, 4),
                    "old": round(old_val, 4),
                    "delta": round(delta, 4)
                }

        for col in categorical_features:
            if col in new_df.columns and col in old_df.columns:
                new_val = str(new_df[col].iloc[0])
                old_val = str(old_df[col].iloc[0])
                differences[col] = {
                    "new": new_val,
                    "old": old_val,
                    "changed": new_val != old_val
                }

        result = {
            "env_name": name,
            "distance": dist,
            "differences": differences
        }

        result_str = json.dumps(result, indent=2)
        print(result_str)
        print_lines.append(result_str + "\n")
        all_differences.append(result)

    # === Save results
    with open(output_txt_path, "w") as f:
        f.writelines(print_lines)

    with open(output_path, "w") as f:
        json.dump(all_differences, f, indent=2)

    print(f"\n✅ Saved decoded differences for {new_env_name} to:\n  {output_path}\n  {output_txt_path}")

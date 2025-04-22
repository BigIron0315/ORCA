import glob
import json
import os
import numpy as np
import re
from pathlib import Path
from queryGPT import Query_GPT4
import math

# === CONFIG ===
INPUT_TXT_GLOB = "./interim_results/_8_shap_inference_prompt/LLM_shap_inference_prompt_*.txt"
output_dir = Path("./interim_results/_9_shap_output")
output_dir.mkdir(exist_ok=True)

# Get current environment ID (default to "0" for safety)
INPUT_ENV_ID = os.environ.get("INPUT_ENV_ID", "0")

# Create subfolder for each env
env_output_dir = output_dir / f"env_new{INPUT_ENV_ID}"
env_output_dir.mkdir(parents=True, exist_ok=True)

# Output file prefix
OUTPUT_PREFIX = env_output_dir / "LLM_shap_output"


SHAP_DIR = "./shap_outputs"
shap_outputs = []

def extract_clean_json(response: str):
    import json
    from json.decoder import JSONDecodeError

    lines = response.strip().splitlines()
    for line in reversed(lines):  # start from the bottom
        line = line.strip()
        # ✅ Extract only the JSON part if there's a prefix like "Line 6:"
        if "{" in line and "}" in line:
            json_start = line.find("{")
            json_part = line[json_start:].strip()
            try:
                cleaned = json_part.replace("'", '"')
                cleaned = re.sub(r'(?<!["{,\s])([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', cleaned)
                cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
                cleaned = re.sub(r"\n", " ", cleaned)
                cleaned = evaluate_expressions_in_json(cleaned)
                parsed = json.loads(cleaned)
                return parsed
            except JSONDecodeError as e:
                print(f"⚠️ Failed to parse line as JSON:\n{json_part}\n→ {e}")
                continue
    raise ValueError("No valid JSON block found.")



def evaluate_expressions_in_json(s):
    def try_eval(match):
        expr = match.group(1)
        expr_clean = expr.replace("max", "max_")
        local_env = {"max_": max, "math": math}
        try:
            result = eval(expr_clean, {"__builtins__": {}}, local_env)
            return f": {round(result, 6)}"
        except Exception as e:
            print(f"⚠️ Failed to eval: {expr} → {e}")
            return f": {expr}"

    pattern = r':\s*([^"\n\r:,]+?)\s*(?=[,}])'
    return re.sub(pattern, try_eval, s)

# === MAIN LOOP ===
for i, prompt_file in enumerate(sorted(glob.glob(INPUT_TXT_GLOB)), start=1):
    with open(prompt_file, "r") as f:
        prompt = f.read()

    print(f"\n🔍 Querying GPT-4 with {prompt_file}...")
    system_msg = "You are a reasoning agent for SHAP value estimation."
    response = Query_GPT4(system_msg, prompt, 0)
    print(response)

    raw_path = OUTPUT_PREFIX.parent / f"LLM_shap_output_raw_{i}.txt"
    with open(raw_path, "w") as f:
        f.write(response)

    try:
        parsed = extract_clean_json(response)
        print("✅ JSON parsed successfully.")
    except Exception as e:
        print(f"❌ JSON parse failed: {e}")
        print("🔎 Full LLM Response:\n", response)
        raise e

    env_match = re.search(r"### Reference Environment: ([^\n]+)", prompt)
    kpm_match = re.search(r"KPM is: ([^\n]+)", prompt)

    if env_match and kpm_match:
        env_name = env_match.group(1).strip()
        target_kpm = kpm_match.group(1).strip()

        if target_kpm not in parsed:
            print(f"⚠️ Target KPM '{target_kpm}' not found in parsed output")
            continue

        # ✅ Only append after KPM check passes
        shap_outputs.append(parsed)

    if env_match and kpm_match:
        env_name = env_match.group(1).strip()
        target_kpm = kpm_match.group(1).strip()

        if target_kpm not in parsed:
            print(f"⚠️ Target KPM '{target_kpm}' not found in parsed output")
            continue

        try:
            past_file = f"{SHAP_DIR}/{env_name}_{target_kpm}_ytest.npy"
            curr_file = f"{SHAP_DIR}/ORAN_log_new{INPUT_ENV_ID}_{target_kpm}_ytest.npy"
            if not (Path(past_file).exists() and Path(curr_file).exists()):
                print(f"⚠️ Missing KPM files for {target_kpm} in {env_name}")
                continue

            past_vals = np.load(past_file)
            curr_vals = np.load(curr_file)
            past_kpm = float(np.mean(past_vals))
            curr_kpm = float(np.mean(curr_vals))
            if past_kpm == 0:
                print(f"⚠️ KPM_past is zero for {target_kpm}, skipping")
                continue
            kpm_scale = curr_kpm / past_kpm

            mean_abs_path = f"{SHAP_DIR}/{env_name}_{target_kpm}_mean_abs.npy"
            if not Path(mean_abs_path).exists():
                print(f"⚠️ Missing past SHAP file for {target_kpm}")
                continue
            loaded = np.load(mean_abs_path, allow_pickle=True)
            past_shap_vals = dict(zip(loaded[0], map(float, loaded[1])))
            past_total = sum(past_shap_vals.values())

            shap_path = f"{SHAP_DIR}/{env_name}_{target_kpm}_shap.npy"
            feature_path = f"{SHAP_DIR}/{env_name}_{target_kpm}_features.npy"
            if not (Path(shap_path).exists() and Path(feature_path).exists()):
                print(f"⚠️ Missing normalized SHAP files for {target_kpm}")
                continue

            shap_raw = np.load(shap_path)
            feature_names = list(np.load(feature_path, allow_pickle=True))
            ϕ_past_norm = np.mean(np.abs(shap_raw), axis=0)
            ϕ_past_norm = ϕ_past_norm / np.sum(ϕ_past_norm)

            sched_sum = 0.0
            print(f"\n📌 Normalized SHAP values from the reference environment ({env_name}) for {target_kpm}:")
            for i_f, name in enumerate(feature_names):
                if "Scheduling" in name:
                    sched_sum += ϕ_past_norm[i_f]
                else:
                    print(f"   • {name:16s}: {ϕ_past_norm[i_f]:.6f}")
            if sched_sum > 0:
                print(f"   • Scheduling       : {sched_sum:.6f}")

            ϕ_dict = {}
            sched_sum = 0.0
            for i_f, name in enumerate(feature_names):
                if "Scheduling" in name:
                    sched_sum += ϕ_past_norm[i_f]
                else:
                    ϕ_dict[name] = ϕ_past_norm[i_f]
            if sched_sum > 0:
                ϕ_dict["Scheduling"] = sched_sum

            beta = parsed[target_kpm]
            ϕ_new_unnorm = {}
            print(f"\n🧪 βᵢ (importance shift) for {env_name} → {target_kpm}:")
            for f in ϕ_dict:
                b = beta.get(f, 1.0)
                print(f"   • {f:16s}: β = {b:.3f}")
                ϕ_new_unnorm[f] = ϕ_dict[f] * b

            ϕ_sum = sum(ϕ_new_unnorm.values())
            if ϕ_sum == 0:
                print(f"⚠️ Sum of new unnormalized SHAP is zero for {target_kpm}")
                continue
            ϕ_new_norm = {f: v / ϕ_sum for f, v in ϕ_new_unnorm.items()}

            ϕ_abs = {
                f: round(v * past_total * kpm_scale, 6)
                for f, v in ϕ_new_norm.items()
            }

            #print(f"\n🔧 Recovery details for {target_kpm} in environment {env_name}:")
            #print(f"   - Past KPM mean     = {past_kpm:.4f}")
            #print(f"   - Current KPM mean  = {curr_kpm:.4f}")
            #print(f"   - KPM scale factor  = {kpm_scale:.4f}")
            #print(f"   - Past SHAP total   = {past_total:.4f}")
            #print(f"   - Raw features used = {list(ϕ_dict.keys())}")
            #print(f"   - βᵢ keys           = {list(beta.keys())}")

            print("\n📉 Intermediate unnormalized SHAP × βᵢ values:")
            for f, v in sorted(ϕ_new_unnorm.items(), key=lambda x: -x[1]):
                print(f"   • {f:16s}: {v:.6f}")

            print("\n📊 Final normalized SHAP values:")
            for f, v in sorted(ϕ_new_norm.items(), key=lambda x: -x[1]):
                print(f"   • {f:16s}: {v:.6f}")

            print(f"\n📈 Recovered final absolute SHAP for {env_name} → {target_kpm}")
            for f, v in sorted(ϕ_abs.items(), key=lambda x: -x[1]):
                print(f"   • {f:16s}: {v:.6f}")
            #print(f"✅ Total SHAP sum: {sum(ϕ_abs.values()):.6f}")

            parsed[target_kpm] = ϕ_abs

            output_path = output_dir / f"LLM_shap_output_{i}_{target_kpm}.json"
            with open(output_path, "w") as f:
                json.dump({target_kpm: ϕ_abs}, f, indent=2)
            print(f"✅ Saved GPT-4 SHAP output to {output_path}")
        except Exception as e:
            print(f"❌ Error in SHAP recovery for {target_kpm}: {e}")

# === STEP 2: Collect SHAP values grouped by KPM ===

merged_by_kpm = {}

for entry in shap_outputs:
    for kpm in entry:
        if kpm not in merged_by_kpm:
            merged_by_kpm[kpm] = {}
        for feature, value in entry[kpm].items():
            merged_by_kpm[kpm][feature] = merged_by_kpm[kpm].get(feature, 0.0) + value

merged_output_path = env_output_dir / "LLM_shap_output_1.json"
with open(merged_output_path, "w") as f:
    json.dump(merged_by_kpm, f, indent=2)

print(f"📊 Merged SHAP values grouped by KPM saved to {merged_output_path}")

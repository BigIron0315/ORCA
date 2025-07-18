import numpy as np
import json
import os
import re
from pathlib import Path
from queryGPT import Query_GPT4

INPUT_ENV_ID = os.environ.get("INPUT_ENV_ID", "0")
# === Reference environments for extrapolation
# === Load top-2 reference environments based on decoded difference file
json_path = f"./env_encoder/decoded_differences/decoded_differences_ORAN_log_new{INPUT_ENV_ID}.json"
with open(json_path, "r") as f:
    decoded = json.load(f)

# Sort environments by distance (ascending)
sorted_envs = sorted(decoded, key=lambda x: x["distance"])
env1 = sorted_envs[0]["env_name"]
env2 = sorted_envs[1]["env_name"]
print(f"🔍 Using top-2 reference environments for extrapolation: {env1} {env2}")

slice_str = env1.lower()
slice_type = "urllc" if "urllc" in slice_str else "embb" if "embb" in slice_str else "unknown"
print(f"📂 Detected slice type from {env1}: {slice_type}")

# === CONFIG ===
shap_dir = Path("./shap_outputs")
output_dir = Path(f"./interim_results/_9_shap_output/env_new{INPUT_ENV_ID}")
output_dir.mkdir(parents=True, exist_ok=True)


kpm_keys = ["Throughput_Mbps", "Avg_Delay_ms", "user_throughput"]
output_json_file = output_dir / "no_external_knowledge_llm.json"
output_raw_file = output_dir / "no_external_knowledge_llm.txt"

# === Load SHAP values for env1 and env2 ===
shap_env1 = {}
shap_env2 = {}
for kpm in kpm_keys:
    if slice_type == "embb" and kpm != "Throughput_Mbps":
        print(f"⏭️ Skipping {slice_type} with KPM {kpm} (Only Throughput_Mbps for eMBB)")
        continue
    elif slice_type == "urllc" and kpm != "Avg_Delay_ms":
        print(f"⏭️ Skipping {slice_type} with KPM {kpm} (Only Avg_Delay_ms for urllc)")
        continue
    else:
        kpm_keys = ["Throughput_Mbps", "Avg_Delay_ms", "user_throughput"] 
        
    f1 = shap_dir / f"{env1}_{kpm}_mean_abs.npy"
    f2 = shap_dir / f"{env2}_{kpm}_mean_abs.npy"

    if not (f1.exists() and f2.exists()):
        raise FileNotFoundError(f"Missing SHAP files for {kpm}")

    arr1 = np.load(f1, allow_pickle=True)
    arr2 = np.load(f2, allow_pickle=True)

    features = arr1[0]
    values1 = arr1[1].astype(float)
    values2 = arr2[1].astype(float)

    if not np.allclose(values1, values2, atol=1e-3):
        print(f"⚠️ SHAP values for {kpm} are not exactly identical between {env1} and {env2}.")

    shap_env1[kpm] = {f: round(v, 4) for f, v in zip(features, values1)}
    shap_env2[kpm] = {f: round(v, 4) for f, v in zip(features, values2)}

# === Load environment differences ===
diff_txt = []
diff_json_path = f"./env_encoder/decoded_differences/decoded_differences_ORAN_log_new{INPUT_ENV_ID}.json"
with open(diff_json_path, "r") as f:
    env_diffs = json.load(f)

# Pick the reference that matches env1
ref_entry = next((e for e in env_diffs if e["env_name"] == env1), None)

diff_txt.append(f"Reference environment: {env1}")
diff_txt.append(f"New environment: ORAN_log_new{INPUT_ENV_ID}")
diff_txt.append("\nDifferences between new and reference environment:")
if ref_entry and "differences" in ref_entry:
    for k, v in ref_entry["differences"].items():
        if "delta" in v:
            delta = round(v["delta"], 4)
            old = round(v["old"], 4)
            new = round(v["new"], 4)
            diff_txt.append(f"- {k}: Δ = {delta} (old = {old}, new = {new})")
        elif "changed" in v and v["changed"]:
            diff_txt.append(f"- {k}: changed from {v['old']} to {v['new']}")
else:
    diff_txt.append("- ⚠️ No difference info found for reference environment.")


# === Build extrapolation prompt (NO triple backticks) ===
prompt = f"""
You are an expert in SHAP-based feature attribution modeling in 5G wireless systems.

You are given:
- SHAP values for two environments: env1 and env2.
- A new environment (env3), which differs from env1 based on the parameters below.
- The SHAP values in env1 and env2 illustrate how feature importance shifts as the environment changes.

Use the SHAP trends and your domain knowledge to infer what will happen in env3. 
Do NOT interpolate or extrapolate numerically. Use reasoning and patterns.

{diff_txt}

Here are the SHAP values for env1:
{json.dumps(shap_env1, indent=2)}

Here are the SHAP values for env2:
{json.dumps(shap_env2, indent=2)}

Please return your inferred SHAP values for env3 in this JSON format only:
For scheduling, integrate one feature to scheduling.
Do not include comments or explanations.
{{
  "Throughput_Mbps": {{ "Feature1": value, ... }},
  "Avg_Delay_ms": {{ "Feature1": value, ... }},
  "user_throughput": {{ "Feature1": value, ... }}
}}
"""



# === Query LLM ===
system_msg = "You are a reasoning agent for directional SHAP extrapolation."
response = Query_GPT4(system_msg, prompt, 0)
print(prompt)
print("===============================================================================================")
print(response)

# === Save raw response ===
with open(output_raw_file, "w") as f:
    f.write(response)

# === Extract JSON ===
try:
    json_match = re.search(r"\{[\s\S]*\}", response)
    if not json_match:
        raise ValueError("No JSON found in LLM response.")

    json_str = json_match.group(0)
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r"//.*", "", json_str)
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    parsed = json.loads(json_str)

    with open(output_json_file, "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"✅ Saved LLM extrapolated SHAP to {output_json_file}")
except Exception as e:
    print(f"❌ Failed to parse JSON: {e}")
    print("🔍 Response content:\n", response[:300])

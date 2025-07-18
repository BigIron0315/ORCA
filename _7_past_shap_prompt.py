import os
import re
import json
import numpy as np
from pathlib import Path
from queryGPT import Query_GPT4
from _7_1_shap_prompt import get_symbolic_form_prompt, get_math_equation_prompt, get_importance_form

# === CONFIG ===
INPUT_ENV_ID = os.environ.get("INPUT_ENV_ID", "0")
INPUT_TXT_FILE = f"./env_encoder/decoded_differences/decoded_differences_ORAN_log_new{INPUT_ENV_ID}.txt"
SHAP_DIR = "./shap_outputs"
KPM_LOG_DIR = "./dataset"
output_dir = Path("./interim_results/_7_past_shap_prompt")
output_dir.mkdir(parents=True, exist_ok=True)


RAG_DOCUMENT = """
# Shannon Capacity Expression

The Shannon Capacity (maximum theoretical throughput) of a communication channel is given by the formula:

C = B * log2(1 + SNR)

where:
- C = Channel capacity in Megabits per second (Mbps)
- B = Bandwidth in Megahertz (MHz)
- SNR = Signal-to-noise ratio (unitless, linear scale)

# Important Notes
- log2() denotes logarithm base 2.
- SNR must be in **linear scale**, not dB.
    - If SNR is in dB, convert by:  SNR_linear = 10^(SNR_dB/10)
- Bandwidth (B) must be in Megahertz (MHz).
- Final capacity (C) is in Megabits per second (Mbps).

# Conversion Factors
- 1 MHz = 1,000,000 Hz
- 1 Mbps = 1,000,000 bits per second

Thus, if B is in MHz and C is calculated directly in Mbps, the formula is cleanly usable.

# Example: PRB to Bandwidth Mapping in 5G

- 273 Physical Resource Blocks (PRBs) correspond to approximately 100 MHz bandwidth.
- Therefore, if you are given 273 PRBs, you can set:

B = 100 MHz

# Simplified Expression for 273 PRBs

Given 273 PRBs:

C = 100 * log2(1 + SNR)

where SNR is in linear scale.

# Quick Conversion for SNR
- SNR (linear) = 10^(SNR_dB / 10)

Thus, the full computation is:

C = 100 * log2(1 + 10^(SNR_dB/10))

- [IMPORTANT] Follow below equations. Check partial derivatives. 
- a = k·log₂(1+q)
- ∂a/∂q = k/(1+q)
"""


TOP_K = 1
#KPM_KEYS = ['Throughput_Mbps', 'Avg_Delay_ms', 'user_throughput']
KPM_KEYS = ['Throughput_Mbps', 'Avg_Delay_ms']
SHAP_KPM_KEYS = ['Throughput_Mbps', 'Avg_Delay_ms', 'user_throughput', 'Avg_SNR_dB']
# === Load top-k environments from decoded differences ===
with open(INPUT_TXT_FILE, "r") as f:
    raw_text = f.read()
json_blocks = re.findall(r'\{\s+"env_name":.*?\n\}', raw_text, re.DOTALL)
env_data = [json.loads(block) for block in json_blocks]
TOP_K = min(TOP_K, len(env_data))

def RAG_needed(user_query):
    # Simple heuristic: if "derivative" or "partial derivative" is mentioned, RAG is needed
    trigger_keywords = ["Reformulate", "derivative"]
    for keyword in trigger_keywords:
        if keyword.lower() in user_query.lower():
            return True  # Needs RAG
    return False  # Proceed directly

def retrieve_context():
    # For now, just return our simple cheat sheet
    return RAG_DOCUMENT

# === Generate prompt for each top-k environment and KPM ===
for variant_id in range(1, TOP_K + 1):
    env = env_data[variant_id - 1]
    env_name = env["env_name"]
    differences = env["differences"]

    for kpm_target in KPM_KEYS:
        symbolic_form_prompt = get_symbolic_form_prompt(kpm_target)
        #print(symbolic_form_prompt)
        system_msg = "You are an expert in wireless networks and Open RAN."
        symbolic_form = Query_GPT4(system_msg, symbolic_form_prompt, 0)
        print(symbolic_form)

        math_equation_prompt = get_math_equation_prompt(symbolic_form)
        needs_rag = RAG_needed(math_equation_prompt)

        retrieved_context = retrieve_context() if needs_rag else None
        #print("[Planner Decision] Needs RAG?", needs_rag, retrieved_context)

        math_equation = Query_GPT4(system_msg, retrieved_context+math_equation_prompt, 0)
        print(math_equation)

        importance_form_prompt = get_importance_form(symbolic_form)
        #importance_form = Query_GPT4(system_msg, importance_form_prompt, 0)
        
        #print(importance_form)


        prompt_lines = []
        prompt_lines.append(importance_form_prompt)
        prompt_lines.append(math_equation)
        #prompt_lines.append(importance_form)

 
        get_beta_prompt = f"""
β_A and β_B are given, find the value and what A and B are.
**Step 3**: If A or B is applied to an **abstraction layer** (e.g., SystemCapacity), distribute the shift into the relevant features
using **mathematical sensitivity (partial derivatives)** rather than SHAP weights.
Check the [IMPORTANT] before calculating partial derivatives.

feature importance α = ΔKPM / Δxi

e.g., a = f(x1, x2), where xi = {{PRB_num, Avg_SNR_dB, etc.}}
- Then distribute β into x1 and x2 using:
- ∂a / ∂xi
- u1 = (∂a / ∂x1)_curr / (∂a / ∂x1)_past
- u2 = (∂a / ∂x2)_curr / (∂a / ∂x2)_past
Do not forget to normalize u1 and u2, u1 + u2 = 1.
Then, β_x1 = a^u1 , β_x2 = a^u2

**Step 4**: Insert values from new environment to the Step 3 and derive β.
If the feature of derived β has another SHAP weights, apply β to each feature proportional to the weights.
For example, Avg_SNR_dB has another SHAP weights e.g., (w3, w4, w5) for feature x3, x4, x5.
If x3 is same with x1 or x2, then discard x3.
Make sure that β_x3 = β_SNR^w3, β_x4 = β_SNR^w4, β_x5 = β_SNR^w5, be sure that β_SNR keeps the value.

Repeat this recursively for x3, x4, and x5 if the feature has another Normalized SHAP values.
Be sure that Avg_SNR_dB needs to be redistributed.

**Output Format**:
1: Original mathematical formulation of feature importance A and B
2: Approximated mathematical formulation feature importance A and B
3: Explain why you approximated like this.
4: Check the value and what A and B are for β_A and β_B (e.g., β_DL_Buffer = , β_SystemCapacity = )
5: Step 3 – make the mathematical formulation for how to distribute importance shift
6: Step 4 – Repeat the process recursively if the feature has another normalized SHAP values. Check the Normalizd SHAP values and show the process.
7: Results display – show me the values of β for all features in json format. Strictly follow below format and Only output final numeric values. DO NOT include expressions or equal signs.
{{"{kpm_target}": {{"ant_tilt_deg": , "CIO": , "TxPower": , "PRB_num": , "DL_Buffer": , "Avg_SNR_dB": , "Scheduling": }}}}
KPM is: {kpm_target}"""

        prompt_lines.append(get_beta_prompt)

        prompt_lines.append(f"\n### Reference Environment: {env_name}")

        prompt_lines.append("\n• Differences from new environment:")
        users_old = differences.get("Users", {}).get("old", None)
        users_new = differences.get("Users", {}).get("new", None)

        for key, val in differences.items():
            if "delta" in val and abs(val["delta"]) > 0.001:
                # Apply per-user normalization for Delay-related analysis
                if kpm_target == "Avg_Delay_ms" and key in ["DL_Buffer", "Throughput_Mbps"] and users_old and users_new:
                    per_user_old = val["old"] / users_old
                    per_user_new = val["new"] / users_new
                    delta = per_user_new - per_user_old
                    prompt_lines.append(
                        f"  - {key}: Δ = {delta:.4f} (old = {per_user_old:.4f}, new = {per_user_new:.4f})"
                    )
                else:
                    prompt_lines.append(
                        f"  - {key}: Δ = {val['delta']:.4f} (old = {val['old']}, new = {val['new']})"
                    )
            elif "changed" in val and val["changed"]:
                prompt_lines.append(f"  - {key}: changed from {val['old']} to {val['new']}")
        if not differences:
            prompt_lines.append("  - No meaningful changes.")

        prompt_lines.append("\n• Past vs. Current KPMs:")
        for kpm in KPM_KEYS:
            past_csv = f"{KPM_LOG_DIR}/{env_name}.csv"
            curr_csv = f"{KPM_LOG_DIR}/ORAN_log_new{INPUT_ENV_ID}.csv"
            if os.path.exists(past_csv) and os.path.exists(curr_csv):
                import pandas as pd
                past_df = pd.read_csv(past_csv)
                curr_df = pd.read_csv(curr_csv)

                if kpm in past_df.columns and kpm in curr_df.columns and "Users" in past_df.columns and "Users" in curr_df.columns:
                    if kpm_target == "Avg_Delay_ms" and kpm in ["Throughput_Mbps", "DL_Buffer"]:
                        past_val = (past_df[kpm] / past_df["Users"]).mean()
                        curr_val = (curr_df[kpm] / curr_df["Users"]).mean()
                        prompt_lines.append(f"  - {kpm}: past = {past_val:.4f}, current = {curr_val:.4f}")
                    else:
                        past_val = past_df[kpm].mean()
                        curr_val = curr_df[kpm].mean()
                        prompt_lines.append(f"  - {kpm}: past = {past_val:.4f}, current = {curr_val:.4f}")
                else:
                    prompt_lines.append(f"  - {kpm}: ⚠️ column missing in CSV, skipped")
            else:
                prompt_lines.append(f"  - {kpm}: ⚠️ full‑log missing, skipped")

        prompt_lines.append("\n• Normalized SHAP values:")
        for kpm in SHAP_KPM_KEYS:
            shap_path = f"{SHAP_DIR}/{env_name}_{kpm}_shap.npy"
            feature_path = f"{SHAP_DIR}/{env_name}_{kpm}_features.npy"
            if not os.path.exists(shap_path) or not os.path.exists(feature_path):
                prompt_lines.append(f"  - ⚠️ Missing SHAP for {kpm}")
                continue

            shap_values = np.load(shap_path)
            feature_names = np.load(feature_path, allow_pickle=True)
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            norm_shap = mean_abs / np.sum(mean_abs) if np.sum(mean_abs) > 0 else mean_abs

            feat, vals = list(feature_names), list(norm_shap)
            sched_idx = [i for i, n in enumerate(feat) if 'Scheduling' in n]
            if sched_idx:
                sched_total = sum(vals[i] for i in sched_idx)
                for i in sorted(sched_idx, reverse=True):
                    del feat[i]; del vals[i]
                feat.append("Scheduling")
                vals.append(sched_total)

            prompt_lines.append(f"  - {kpm}:")
            for f, v in zip(feat, vals):
                prompt_lines.append(f"     • {f}: {v:.4f}")

        prompt = "\n".join(prompt_lines)   # Join the list into a single string
        needs_rag = RAG_needed(prompt)
        retrieved_context = retrieve_context() if needs_rag else None
        new_prompt = retrieved_context+prompt
        print("##########################################################################")
        #print(new_prompt)
        print("##########################################################################")
        #response = Query_GPT4(system_msg, prompt, 0)
        #print(response)

        outname = f"LLM_shap_prompt_{variant_id}_{kpm_target}.txt"
        output_path = output_dir / outname
        with open(output_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write("\n".join(prompt_lines))
        print(f"✅ Saved SHAP prompt variant {variant_id}, KPM {kpm_target} to {output_path}")

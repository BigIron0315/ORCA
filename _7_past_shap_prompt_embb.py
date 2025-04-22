import os
import re
import json
import numpy as np
from pathlib import Path

# === CONFIG ===
INPUT_TXT_FILE = "./env_encoder/decoded_differences.txt"
SHAP_DIR = "./shap_outputs"
KPM_LOG_DIR = "./dataset"
output_dir = Path("./interim_results/_7_past_shap_prompt")
output_dir.mkdir(parents=True, exist_ok=True)

TOP_K = 1
KPM_KEYS = ['Throughput_Mbps', 'Avg_Delay_ms', 'user_throughput']
SHAP_KPM_KEYS = ['Throughput_Mbps', 'Avg_Delay_ms', 'user_throughput', 'Avg_SNR_dB']
# === Load top-k environments from decoded differences ===
with open(INPUT_TXT_FILE, "r") as f:
    raw_text = f.read()
json_blocks = re.findall(r'\{\s+"env_name":.*?\n\}', raw_text, re.DOTALL)
env_data = [json.loads(block) for block in json_blocks]
TOP_K = min(TOP_K, len(env_data))

# === Generate prompt for each top-k environment and KPM ===
for variant_id in range(1, TOP_K + 1):
    env = env_data[variant_id - 1]
    env_name = env["env_name"]
    differences = env["differences"]

    for kpm_target in KPM_KEYS:
        prompt_lines = [
            "You are an expert in wireless networks and Open RAN.",
            "You are given a target Key Performance Metric (KPM), denoted as k_i.",
            "",
            "You are also provided with:",
            "- A list of available metrics/intermediate variables:",
            "  [DL_Buffer, Avg_SNR_dB, Throughput_Mbps, Avg_Delay_ms, user_throughput]",
            "- SHAP-based feature importance values from a **past environment**",
            "- The corresponding values for a and b in both **past** and **current environments**:",
            "  a_past, b_past, a_curr, b_curr",
            "",
            "Your task has four parts:",
            "",
            "---",
            "**Step 1**: Express k_i using one of the following abstract forms:",
            "   - k_i = max(a, b)",
            "   - k_i = min(a, b)",
            "   - k_i = a / b",
            "   - k_i = a + b",
            "   - k_i = a - b",
            "   - k_i = a * b",
            "",
            "Use only high-level concepts here, such as:",
            "   - SystemCapacity",
            "   - TrafficDemand",
            "   - BufferStatus",
            "   - SignalQuality",
            "",
            "Then, in the next line, express k_i using only the available candidate variables:",
            "  [DL_Buffer, Avg_SNR_dB, Throughput_Mbps, Avg_Delay_ms, user_throughput]",
            "",
            "DL_Buffer : total amount of traffic demand of the gNB",
            "Avg_SNR_dB : average SNR across users in the gNB",
            "Throughput_Mbps : sum throughput of all users in the gNB (approximation of system capacity)",
            "Avg_Delay_ms : average packet end-to-end delay (queuing delay + transmission delay) across users in the gNB",
            "user_throughput : average throughput per user in the gNB",
            "Users : number of users in the gNB",
            "",
            "---",
            "**Step 2**: If the form in Step 1 is `min(a, b)` or `max(a, b)`, compute the importance shift ratio:",
            "",
            "If you don't have exact a and b, you can approximate a, b using available variables.",
            "For example, SystemCapacity ≈ Throughput_Mbps, and TrafficDemand ≈ DL_Buffer.",
            "",
            "**(i) If k_i = min(a, b)**:",
            "  - Compute: importance_shift_ratio Δγ = (a_curr / b_curr) / (a_past / b_past)",
            "  - For a: β_a = 1 / Δγ",
            "  - For b: β_b = Δγ",
            "",
            "**(ii) If k_i = max(a, b)**:",
            "  - Compute: importance_shift_ratio Δγ= (a_curr / b_curr) / (a_past / b_past)",
            "  - For a: β_a = Δγ",
            "  - For b: β_b = 1 / Δγ",
            "",
            "If the expression is **not min or max**, do not proceed to Step 2.",
            "",
            "---",
            "**Step 3**: If a or b is applied to an **abstraction layer** (e.g., SystemCapacity), distribute the shift into the relevant features",
            "using **mathematical sensitivity (partial derivatives)** rather than SHAP weights.",
            "",
            "feature importance α = ΔKPM / Δxi",
            "For example, Shannon's theorem for capacity, Little's law for delay, and so on.",
            "e.g., a = f(x1, x2), where xi = {PRB_num, Avg_SNR_dB, etc.}",
            "- Then distribute β into x1 and x2 using:",
            "  - ∂a / ∂xi",
            "  - or inferred impact from the analytical model",
            "- Check partial derivatives.",
            "  - SNR_lin = 10^(SNR_dB/10)",
            "  - ∂C/∂PRB_num = log2(1+SNR_lin)",
            "  - ∂C/∂Avg_SNR_dB = PRB_num/(1+SNR_lin)",
            "- Derive the shift ratio",
            "  - u1 = (∂a / ∂x1)_curr / (∂a / ∂x1)_past",
            "  - u2 = (∂a / ∂x2)_curr / (∂a / ∂x2)_past",
            "Normalize to get contribution ratios between u1 and u2, u1 + u2 = 1.",
            "Then, β_x1 = a^u1 , β_x2 = a^u2",
            "",
            "---",
            "**Step 4**: Insert values from new environment to the Step 3 and derive β.",
            "Never normalize anything.",
            "If the feature of derived β has another SHAP weights, apply β to each feature proportional to the weights.",
            "For example, Avg_SNR_dB has another SHAP weights e.g., (w3, w4, w5) for feature x3, x4, x5.",
            "If x3 is same with x1 or x2, then discard x3.",
            "Make sure that β_x3 = β_SNR^w3, β_x4 = β_SNR^w4, β_x5 = β_SNR^w5",
            "",
            "---",
            "**Output Format**:",
            "Line 1: Step 1 abstract form",
            "Line 2: Step 1 approximated with available candidates",
            "Line 3: Step 2 – importance shift ratio (if applicable)",
            "Line 4: Step 2 – updated rule for adjusting importance (if applicable)",
            "Line 5: Step 3 – make the mathematical formulation for how to distribute importance shift",
            "Line 6: Step 4 – show me the values of β for all features in json format. Strictly follow below format and Only output final numeric values. DO NOT include expressions or equal signs.",
            f'{{"{kpm_target}": {{"ant_tilt_deg": , "CIO": , "TxPower": , "PRB_num": , "DL_Buffer": , "Avg_SNR_dB": , "Scheduling": }}}}'
            "",
            f"KPM is: {kpm_target}",
            "",
            "Note:",
            "- Throughput_Mbps refers to delivered data rate limited by demand and capacity (not UL + DL sum).",
            "- You must not introduce any variables outside the given list.",
            "",
            "📌 Important: Δγ reflects imbalance pressure on other control knobs, not only on buffer.",
            "📌 For other KPMs (Avg_Delay_ms, user_throughput):",
            "  - Derive your own formulas or heuristics using domain knowledge.",
            "  - Consider effects of congestion, latency, user load, or radio conditions.",
            ""
        ]

        prompt_lines.append(f"\n### Reference Environment: {env_name}")

        prompt_lines.append("\n• Differences from new environment:")
        for key, val in differences.items():
            if "delta" in val and abs(val["delta"]) > 0.001:
                prompt_lines.append(f"  - {key}: Δ = {val['delta']:.4f} (old = {val['old']}, new = {val['new']})")
            elif "changed" in val and val["changed"]:
                prompt_lines.append(f"  - {key}: changed from {val['old']} to {val['new']}")
        if not differences:
            prompt_lines.append("  - No meaningful changes.")

        #prompt_lines.append("\n• Past vs. Current KPMs:")
        #for kpm in KPM_KEYS:
        #    past_path = f"{SHAP_DIR}/{env_name}_{kpm}_ytest.npy"
        #    curr_path = f"{SHAP_DIR}/ORAN_log_new_{kpm}_ytest.npy"
        #    past_val, curr_val = None, None
        #    if os.path.exists(past_path):
        #        past_val = float(np.mean(np.load(past_path)))
        #    if os.path.exists(curr_path):
        #        curr_val = float(np.mean(np.load(curr_path)))
        #    if past_val and curr_val:
        #        prompt_lines.append(f"  - {kpm}: past = {past_val:.4f}, current = {curr_val:.4f}")

        prompt_lines.append("\n• Past vs. Current KPMs:")
        for kpm in KPM_KEYS:
            past_csv = f"{KPM_LOG_DIR}/{env_name}.csv"
            curr_csv = f"{KPM_LOG_DIR}/ORAN_log_new.csv"
            if os.path.exists(past_csv) and os.path.exists(curr_csv):
                import pandas as pd
                past_df = pd.read_csv(past_csv)
                curr_df = pd.read_csv(curr_csv)
                if kpm in past_df.columns and kpm in curr_df.columns:
                    past_val = past_df[kpm].mean()
                    curr_val = curr_df[kpm].mean()
                    prompt_lines.append(f"  - {kpm}: past = {past_val:.4f}, current = {curr_val:.4f}")
                else:
                    prompt_lines.append(f"  - {kpm}: ⚠️ column '{kpm}' missing in CSV, skipped")
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

        outname = f"LLM_shap_prompt_{variant_id}_{kpm_target}.txt"
        output_path = output_dir / outname
        with open(output_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write("\n".join(prompt_lines))
        print(f"✅ Saved SHAP prompt variant {variant_id}, KPM {kpm_target} to {output_path}")

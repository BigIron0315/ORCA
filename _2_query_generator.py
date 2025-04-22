import os
import re
from queryGPT import Query_GPT4
import json

base_dir = "interim_results"
query_dir = os.path.join(base_dir, "_2_query_gen")
INPUT_ENV_ID = os.environ.get("INPUT_ENV_ID", "0")
INPUT_TXT_FILE = f"./env_encoder/decoded_differences/decoded_differences_ORAN_log_new{INPUT_ENV_ID}.txt"


OUTPUT_PROMPT_PREFIX = "LLM_query_prompt"
OUTPUT_QUERY_PREFIX = "LLM_generated_queries"

TOP_K = 1
TARGET_KPMS = ["Total_Throughput", "Avg_Delay"]

os.makedirs(query_dir, exist_ok=True)

def load_environment_differences(path, top_k):
    with open(path, "r") as f:
        raw_text = f.read()
    json_blocks = re.findall(r'\{\s+"env_name":.*?\n\}', raw_text, re.DOTALL)
    env_data = [json.loads(block) for block in json_blocks]
    return env_data[:top_k]


def generate_llm_prompt_for_query_generation(kpm_list, single_env):
    lines = [
        "You are a domain expert in ORAN/3GPP systems helping to retrieve relevant information for performance analysis.",
        "You are given differences between a new environment and one reference environment.",
        "Your task is to generate up to 5 meaningful information retrieval queries that will be used in a hybrid RAG system (dense + sparse retrieval).",
        "",
        "Each query should explore:",
        "1. How the changes in environment (feature differences) may affect the key performance metrics (KPMs),",
        "2. What factors become bottlenecks under the new conditions,",
        "3. Which parameters are most influential on each KPM in the new environment.",
        "",
        f"Target KPMs: {', '.join(kpm_list)}",
        "",
        f"Reference environment: {single_env['env_name']}",
        "Differences from new environment:"
    ]

    for key, val in single_env["differences"].items():
        if "delta" in val and abs(val["delta"]) > 0.001:
            lines.append(f"- {key} has changed by Δ = {val['delta']:.4f} (old: {val['old']}, new: {val['new']})")
        elif "changed" in val and val["changed"]:
            lines.append(f"- {key} changed from {val['old']} to {val['new']}")

    lines += [
        "",
        "Generate no more than 5 concise and precise retrieval queries that focus only on KPMs (e.g., throughput, delay),",
        "the important control parameters, and the key environmental changes listed above.",
        "Each query should be answerable using 3GPP or ORAN specifications or simulation reports.",
        "",
        "Example format:",
        "- What are typical bottlenecks in throughput when traffic load is low?",
        "- How does decreasing buffer occupancy affect average delay in 5G networks?",
        "- When is SNR not the limiting factor for throughput in URLLC scenarios?"
    ]

    return "\n".join(lines)


# === Run full pipeline for each top-k environment ===
if __name__ == "__main__":
    envs = load_environment_differences(INPUT_TXT_FILE, TOP_K)
    for i, env in enumerate(envs, start=1):
        prompt = generate_llm_prompt_for_query_generation(TARGET_KPMS, env)

        prompt_path = os.path.join(query_dir, f"{OUTPUT_PROMPT_PREFIX}_{i}.txt")
        with open(prompt_path, "w") as f:
            f.write(prompt)

        print(f"🧠 Querying GPT-4 for Env #{i}...")
        system_msg = "You are a helpful assistant for domain-aware RAG query generation."
        user_msg = prompt
        result = Query_GPT4(user_msg, user_msg, 0)

        result_path = os.path.join(query_dir, f"{OUTPUT_QUERY_PREFIX}_{i}.txt")
        with open(result_path, "w") as f:
            f.write(result)

        print(f"✅ Saved query prompt to {prompt_path}")
        print(f"✅ Saved generated queries to {result_path}\n")


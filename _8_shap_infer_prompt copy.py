import glob
from pathlib import Path

output_dir = Path("./interim_results/_8_shap_inference_prompt")
output_dir.mkdir(exist_ok=True)


def generate_shap_inference_prompts(shap_prompt_glob, domain_knowledge_glob, output_prefix):
    """
    Generate SHAP inference prompts by pairing each SHAP file with its corresponding domain knowledge file.
    """
    shap_prompt_files = sorted(glob.glob(shap_prompt_glob))
    domain_knowledge_files = sorted(glob.glob(domain_knowledge_glob))

    for i, (shap_file, domain_file) in enumerate(zip(shap_prompt_files, domain_knowledge_files), start=1):
        with open(shap_file, "r") as f:
            past_shap_section = f.read()

        with open(domain_file, "r") as f:
            domain_knowledge = f.read()

        final_prompt_lines = [
            "--- Past SHAP Environments and Differences ---",
            past_shap_section,
            "\n--- Domain Knowledge Retrieved via RAG ---",
            domain_knowledge
        ]

        output_path = f"{output_prefix}_{i}.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(final_prompt_lines))

        print(f"✅ Prompt {i} saved to: {output_path}")


if __name__ == "__main__":
    out_name = "LLM_shap_inference_prompt"
    generate_shap_inference_prompts(
        shap_prompt_glob="./interim_results/_7_past_shap_prompt/LLM_shap_prompt_*.txt",
        domain_knowledge_glob="./interim_results/_4_reranked_results/reranked_results_*.txt",
        output_prefix=output_dir / out_name
    )

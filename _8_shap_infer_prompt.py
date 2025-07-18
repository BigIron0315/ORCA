import glob
from pathlib import Path

output_dir = Path("./interim_results/_8_shap_inference_prompt")
output_dir.mkdir(exist_ok=True)

def generate_shap_inference_prompts(shap_prompt_glob, output_prefix):
    """
    Generate SHAP inference prompts by processing each SHAP file individually.
    """
    shap_prompt_files = sorted(glob.glob(shap_prompt_glob))

    for shap_file in shap_prompt_files:
        with open(shap_file, "r") as f:
            past_shap_section = f.read()

        final_prompt_lines = [
            "--- Past SHAP Environments and Differences ---",
            past_shap_section
        ]

        filename = Path(shap_file).stem.replace("LLM_shap_prompt_", "")  # e.g., 1_Throughput_Mbps
        output_path = output_dir / f"{output_prefix}_{filename}.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(final_prompt_lines))

        print(f"✅ Prompt saved to: {output_path}")

if __name__ == "__main__":
    out_name = "LLM_shap_inference_prompt"
    generate_shap_inference_prompts(
        shap_prompt_glob="./interim_results/_7_past_shap_prompt/LLM_shap_prompt_*.txt",
        output_prefix=out_name
    )

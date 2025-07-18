#!/bin/bash
# Go to project root first
cd /home/dc/_UCSD/Research/Conflict_mitigation_ORAN_slice_done || { echo "❌ Failed to cd into project root"; exit 1; }

# Step 1: Encode and Compare Environments (using system Python)
echo -e "\n📦 [Step 1] Encoding and comparing environments..."
cd env_encoder || { echo "❌ Failed to enter env_encoder"; exit 1; }

LD_LIBRARY_PATH= python3 compute_env_stats.py || exit 1
LD_LIBRARY_PATH= python3 encoderEnv.py || exit 1
LD_LIBRARY_PATH= python3 compare_new_env.py || exit 1
LD_LIBRARY_PATH= python3 env_difference_decoder.py || exit 1
cd ..

# Step 2: Generate Recursive SHAP and Queries (requires conda)
echo -e "\n🔍 [Step 2] Generating recursive SHAP + RAG queries..."

/home/dc/anaconda3/envs/conf_mit/bin/python _0_shap_recursive.py || exit 1
# === Loop over each new environment ===
for f in ./env_encoder/decoded_differences/decoded_differences_ORAN_log_new0.txt; do
    # Extract numeric ID
    fname=$(basename "$f")
    id="${fname//[!0-9]/}"   # Extract digits from filename
    id="${id:-0}"            # Fallback to 0 if nothing extracted

    #skip_ids=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31" "32" )  # Add more as needed

    if [[ " ${skip_ids[@]} " =~ " ${id} " ]]; then
        echo "⏭️  Skipping ORAN_log_new${id} (already processed)"
        continue
    fi
    
    echo -e "\\n==============================="
    echo -e "🌍 Starting pipeline for new$id"
    echo "📌 Extracted ID = $id"
    echo -e "===============================\\n"
    start_time=$(date +%s)
    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _7_past_shap_prompt.py || exit 1
    #INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _7_past_shap_prompt_wo_knowledge.py || exit 1
    LD_LIBRARY_PATH= python3 _8_shap_infer_prompt.py || exit 1
    echo -e "\\n🧠 ===================================================================[Step 5] Running SHAP reasoner..."
    # Step 5: Reasoning
    echo -e "\\n🧠 [Step 5] Running SHAP reasoner..."
    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _9_Reasoner.py || exit 1
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo -e "\n⏳ ================================================================= Total execution time: ${elapsed} seconds"
    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _extrapolation_method.py || exit 1
    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _no_external_knowledge_llm.py || exit 1
    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _compare_shap_actual_vs_llm.py || exit 1

    echo -e "\\n✅ Finished pipeline for ORAN_log_new${id}"
done

echo -e "\\n🏁 All new environments processed."


#!/bin/bash

# === Loop over each new environment ===
for f in ./env_encoder/decoded_differences/decoded_differences_ORAN_log_new*.txt; do
    # Extract numeric ID
    fname=$(basename "$f")
    id="${fname//[!0-9]/}"   # Extract digits from filename
    id="${id:-0}"            # Fallback to 0 if nothing extracted

    skip_ids=("0") #("0" "1")  # Add more as needed

    if [[ " ${skip_ids[@]} " =~ " ${id} " ]]; then
        echo "⏭️  Skipping ORAN_log_new${id} (already processed)"
        continue
    fi
    
    echo -e "\\n==============================="
    echo -e "🌍 Starting pipeline for new$id"
    echo "📌 Extracted ID = $id"
    echo -e "===============================\\n"

    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _7_past_shap_prompt.py || exit 1
    LD_LIBRARY_PATH= python3 _8_shap_infer_prompt.py || exit 1

    # Step 5: Reasoning
    echo -e "\\n🧠 [Step 5] Running SHAP reasoner..."
    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _9_Reasoner.py || exit 1
    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _extrapolation_method.py || exit 1
    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _no_external_knowledge_llm.py || exit 1
    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _compare_shap_actual_vs_llm.py || exit 1

    echo -e "\\n✅ Finished pipeline for ORAN_log_new${id}"
done

echo -e "\\n🏁 All new environments processed."


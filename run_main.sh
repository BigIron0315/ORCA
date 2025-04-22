#!/bin/bash

# === Loop over each new environment ===
for f in ./env_encoder/decoded_differences/decoded_differences_ORAN_log_new*.txt; do
    # Extract numeric ID
    fname=$(basename "$f")
    id="${fname//[!0-9]/}"   # Extract digits from filename
    id="${id:-0}"            # Fallback to 0 if nothing extracted

    echo -e "\\n==============================="
    echo -e "🌍 Starting pipeline for new$id"
    echo "📌 Extracted ID = $id"
    echo -e "===============================\\n"

    # Step 2: Query Generation
    echo -e "💬 [Step 2] Generating query prompts..."
    INPUT_ENV_ID="$id" LD_LIBRARY_PATH= python3 _2_query_generator.py || exit 1

    # Step 3: Run Hybrid RAG Retrieval
    echo -e "\\n🔎 [Step 3] Running hybrid RAG retrieval..."
    /home/dc/anaconda3/envs/conf_mit/bin/python _3_hybrid_rag_retrieval.py || exit 1

    # Step 4: Rerank and organize answers
    echo -e "\\n⚙️ [Step 4] Reranking and organizing answers..."
    LD_LIBRARY_PATH= python3 _4_rerank_with_llm.py || exit 1
    LD_LIBRARY_PATH= python3 _5_split_reranked_queries.py || exit 1
    LD_LIBRARY_PATH= python3 _6_generate_organized_answer.py || exit 1
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


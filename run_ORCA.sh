
#!/bin/bash

echo "🚀 Starting full ORCA pipeline..."

# Go to project root first
cd /home/dc/_UCSD/Research/Conflict_mitigation_ORAN || { echo "❌ Failed to cd into project root"; exit 1; }

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

# Step 2.5: Query generation (system Python)
LD_LIBRARY_PATH= python3 _2_query_generator.py || exit 1

# Step 3: Run Hybrid RAG Retrieval (requires conda)
echo -e "\n🔎 [Step 3] Running hybrid RAG retrieval in conda env..."

/home/dc/anaconda3/envs/conf_mit/bin/python _3_hybrid_rag_retrieval.py || exit 1

# Step 4: Rerank and Process RAG Results (system Python)
echo -e "\n⚙️ [Step 4] Reranking and organizing answers..."
LD_LIBRARY_PATH= python3 _4_rerank_with_llm.py || exit 1
LD_LIBRARY_PATH= python3 _5_split_reranked_queries.py || exit 1
LD_LIBRARY_PATH= python3 _6_generate_organized_answer.py || exit 1
LD_LIBRARY_PATH= python3 _7_past_shap_prompt.py || exit 1
LD_LIBRARY_PATH= python3 _8_shap_infer_prompt.py || exit 1

# Step 5: Reasoning (system Python)
echo -e "\n🧠 [Step 5] Running SHAP reasoner..."
LD_LIBRARY_PATH= python3 _9_Reasoner.py || exit 1

echo -e "\n✅ Pipeline completed successfully!"

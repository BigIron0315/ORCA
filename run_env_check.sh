
#!/bin/bash

echo "🚀 Starting Env Check"

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
echo "✅ Pipeline completed successfully!"

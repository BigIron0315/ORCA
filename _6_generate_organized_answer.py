import os
import glob
from queryGPT import Query_GPT4  # Make sure this is your working wrapper

# Input/output paths
query_files = sorted(glob.glob("./interim_results/_5_split_queries/query_*.txt"))
output_dir = "./interim_results/_6_organized_answers"
os.makedirs(output_dir, exist_ok=True)

# System prompt
system_msg = (
    "You are a 5G network researcher who deeply understands 3GPP and ORAN architecture. "
    "You are given technical knowledge retrieved from Hybrid RAG. "
    "Your job is to interpret it and generate a structured, academic-level answer that directly addresses the query."
)

# Build user prompt with the query focus
def build_user_msg(retrieved_knowledge):
    query_line = next((line for line in retrieved_knowledge.splitlines() if line.startswith("=== Query:")), "")
    query = query_line.replace("=== Query:", "").replace("===", "").strip()

    return f"""
You are answering the following query based on retrieved domain knowledge:

**{query}**

The retrieved content below comes from 3GPP and ORAN documents. It may be fragmented, redundant, or partially relevant.

Please:
- Interpret the key ideas from the retrieved text,
- Synthesize them into an academic-style paragraph or structured list,
- Do not include general background — focus only on what's supported by the evidence.

Avoid bullet-point summaries unless it's necessary for clarity. Focus on clarity, completeness, and directness.

=== Retrieved Knowledge ===
{retrieved_knowledge}
=== End of Knowledge ===

Write your answer below:
"""

# Loop through queries
for i, filepath in enumerate(query_files):
    with open(filepath, "r", encoding="utf-8") as f:
        retrieved_knowledge = f.read()

    query_line = next((line for line in retrieved_knowledge.splitlines() if line.startswith("=== Query:")), "")
    user_msg = build_user_msg(retrieved_knowledge)
    print(f"🟡 Processing: {filepath}")

    try:
        answer = Query_GPT4(system_msg, user_msg, temp=0.3)
    except Exception as e:
        print(f"❌ Error in {filepath}: {e}")
        continue

    # Save both the query and the answer in the output file
    output_path = os.path.join(output_dir, f"organized_answer_{i:03d}.txt")
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"{query_line}\n\n=== Answer ===\n{answer}\n")

    print(f"✅ Saved answer with query to: {output_path}")

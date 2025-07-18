from pathlib import Path
import re
from queryGPT import Query_GPT4
import os
from glob import glob

output_dir = "./interim_results/_4_reranked_results"
os.makedirs(output_dir, exist_ok=True)

# === Helper: Parse .txt files ===
def parse_chunks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = re.split(r'=== Query: (.*?) ===', content)[1:]
    parsed = []
    for i in range(0, len(entries), 2):
        query = entries[i].strip()
        results = re.findall(r'--- Result #\d+ ---\n\n(.*?)\n\n📌 Source: (.*?)\n', entries[i+1], re.DOTALL)
        chunks = [{'content': r[0].strip(), 'source': r[1].strip()} for r in results]
        parsed.append((query, chunks))
    return parsed

# === Prompt builder (ask for re-ordered contents) ===
def build_prompt(query, chunks):
    prompt = [
        "You are a 5G systems expert. Given the following query and retrieved document chunks from 3GPP/ORAN specs, re-rank the chunks from most to least relevant based on how useful they are to answer the query.\n\n",
        f"Query: \"{query}\"\n\nChunks:\n"
    ]
    for i, chunk in enumerate(chunks, 1):
        prompt.append(f"--- Chunk #{i} ---\n{chunk['content']}\n📌 Source: {chunk['source']}\n\n")

    prompt.append(
        "Please return the reordered chunks (from most to least relevant), including their full content and source. Do not return a list of numbers or summaries—just return the full chunks in the new order.\n"
    )
    return ''.join(prompt)

# === Main script ===
retrieved_files = sorted(glob("./interim_results/_3_retrieved_chunks/retrieved_chunks_*.txt"))
if not retrieved_files:
    print("❌ No retrieved chunk files found. Check the directory or Top_K setting.")

for file_path in retrieved_files:
    output_lines = []
    parsed_entries = parse_chunks(file_path)
    system_msg = ""

    for query, chunks in parsed_entries:
        if not chunks:
            continue
        prompt = build_prompt(query, chunks)
        try:
            response_text = Query_GPT4(system_msg, prompt, 0)

            output_lines.append(f"=== Query: {query} ===\n")
            output_lines.append("--- Re-ranked Results (Full Chunks) ---\n")
            output_lines.append(response_text + "\n\n")

        except Exception as e:
            output_lines.append(f"Error processing query: {query}\n{e}\n\n")

    # Save to separate file based on index
    suffix = re.findall(r'\d+', file_path)[-1]  # Extract '1', '2', '3' from file name
    out_path = os.path.join(output_dir, f"reranked_results_{suffix}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

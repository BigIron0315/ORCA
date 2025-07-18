import os
import glob

input_files = sorted(glob.glob("./interim_results/_4_reranked_results/reranked_results_*.txt"))
output_dir = "./interim_results/_5_split_queries"
os.makedirs(output_dir, exist_ok=True)

query_id = 0
file_id = 0

for filename in input_files:
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_query_lines = []
    query_in_file = 0

    for line in lines:
        # Check for new top-level query block
        if line.strip().startswith("=== Query:"):
            # Save previous query
            if current_query_lines:
                out_path = os.path.join(output_dir, f"query_{query_id:03d}.txt")
                with open(out_path, "w", encoding="utf-8") as out:
                    out.writelines(current_query_lines)
                print(f"Saved query {query_id:03d} from file {filename}")
                query_id += 1
                query_in_file += 1
                current_query_lines = []

        current_query_lines.append(line)

    # Save the final query in the file
    if current_query_lines:
        out_path = os.path.join(output_dir, f"query_{query_id:03d}.txt")
        with open(out_path, "w", encoding="utf-8") as out:
            out.writelines(current_query_lines)
        print(f"Saved query {query_id:03d} from file {filename}")
        query_id += 1
        query_in_file += 1

    print(f"✔️ {query_in_file} queries extracted from {filename}")
    file_id += 1

print(f"\n✅ Total queries extracted: {query_id}")

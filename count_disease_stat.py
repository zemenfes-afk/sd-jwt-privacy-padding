import os
import statistics
from pathlib import Path
import tqdm

def analyze_tildes_in_folder(folder_path):
    """
    Traverses a folder, counts tildes (~) in each text file,
    and computes max, mean, median, and 75th percentile.
    """
    tilde_counts = []
    file_results = {}

    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    txt_files = list(folder.rglob("*.txt"))
    
    if not txt_files:
        print("No text files found in the specified folder.")
        return

    for file_path in tqdm.tqdm(txt_files):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                count = content.count('~')
                tilde_counts.append(count)
                file_results[str(file_path)] = count
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")

    if not tilde_counts:
        print("No readable text files found.")
        return

    max_val    = max(tilde_counts)
    mean_val   = statistics.mean(tilde_counts)
    median_val = statistics.median(tilde_counts)
    p25        = statistics.quantiles(tilde_counts, n=4)[0] if len(tilde_counts) >= 2 else tilde_counts[0]

    p75        = statistics.quantiles(tilde_counts, n=4)[2] if len(tilde_counts) >= 2 else tilde_counts[0]

    # Per-file results
    print("\n📄 Tilde counts per file:")
    print("-" * 60)

    # Summary
    print("\n📊 Summary Statistics:")
    print("-" * 60)
    print(f"  Total files analyzed : {len(tilde_counts)}")
    print(f"  Max                  : {max_val}")
    print(f"  Mean                 : {mean_val:.2f}")
    print(f"  Median               : {median_val}")
    print(f"  25th Percentile (Q1) : {p25}")
    print(f"  75th Percentile (Q3) : {p75}")

   


if __name__ == "__main__":
    analyze_tildes_in_folder('./output/final_sd_jwt_unpadded_disclosure')
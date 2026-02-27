import os

# --- CONFIGURATION ---
DIRS = {
    "1. Raw FHIR JSON": "output/fhir/",
    "2. Standard SD-JWT (Leaky)": "output/final_sd_jwt_unpadded/",
    "3. Padded SD-JWT (Non leaky)": "output/final_sd_jwt_padded/"
}

def calculate_directory_stats(directory):
    if not os.path.exists(directory):
        return None, 0
        
    total_size = 0
    count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith(".json") or filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            total_size += os.path.getsize(filepath)
            count += 1
            
    if count == 0:
        return 0, 0
        
    avg_size_bytes = total_size / count
    return avg_size_bytes, count

def run_evaluation():
    print("⚖️  EVALUATING PERFORMANCE OVERHEAD ⚖️")
    print("="*60)
    
    baseline_size = 0
    
    for name, path in DIRS.items():
        avg_size, count = calculate_directory_stats(path)
        
        if count == 0:
            print(f"❌ {name:<35} | Directory not found or empty.")
            continue
            
        avg_kb = avg_size / 1024  # Convert to Kilobytes
        
        # Save baseline to calculate percentage increase
        if "Standard" in name:
            baseline_size = avg_size
            overhead = "0.00% (Baseline)"
        elif baseline_size > 0:
            increase = ((avg_size - baseline_size) / baseline_size) * 100
            overhead = f"+{increase:.2f}% overhead"
        else:
            overhead = "N/A"

        print(f"📁 {name:<35} | Avg Size: {avg_kb:>6.2f} KB | {overhead}")

    print("="*60)

if __name__ == "__main__":
    run_evaluation()
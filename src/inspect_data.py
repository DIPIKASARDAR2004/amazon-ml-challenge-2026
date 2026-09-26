import os
import pandas as pd
import subprocess

def inspect_dataset():
    data_dir = os.path.join("student_resource", "dataset")
    
    # Files to process
    source_files = [
        os.path.join(data_dir, "train", "train_source1.tsv"),
        os.path.join(data_dir, "train", "train_source2.tsv"),
        os.path.join(data_dir, "train", "train_source3.tsv"),
        os.path.join(data_dir, "test", "test_source1.tsv"),
        os.path.join(data_dir, "test", "test_source2.tsv"),
        os.path.join(data_dir, "test", "test_source3.tsv")
    ]
    
    print("=== File Summaries ===\n")
    
    for file_path in source_files:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}, file not found.")
            continue
            
        print(f"Processing {os.path.basename(file_path)}...")
        
        record_count = 0
        columns = None
        country_counts = {}
        missing_name_count = 0
        missing_address_count = 0
        
        chunk_iter = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, chunksize=50000)
        
        for chunk in chunk_iter:
            if columns is None:
                columns = list(chunk.columns)
            
            record_count += len(chunk)
            
            # Country counts
            if 'country' in chunk.columns:
                counts = chunk['country'].value_counts().to_dict()
                for c, count in counts.items():
                    country_counts[c] = country_counts.get(c, 0) + count
            
            # Missing fields
            if 'business_name' in chunk.columns:
                missing_name_count += (chunk['business_name'] == "").sum()
            
            if 'business_address' in chunk.columns:
                missing_address_count += (chunk['business_address'] == "").sum()
                
        print(f"  Record count: {record_count}")
        print(f"  Columns: {columns}")
        print(f"  Country counts: {country_counts}")
        print(f"  Missing business_name: {missing_name_count}")
        print(f"  Missing business_address: {missing_address_count}")
        print("-" * 40)
        
    print("\n=== Ground Truth Summary ===")
    gt_file = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    if os.path.exists(gt_file):
        print(f"Processing {os.path.basename(gt_file)}...")
        gt_record_count = 0
        zero_matches = 0
        total_matches = 0
        max_matches = 0
        
        gt_chunk_iter = pd.read_csv(gt_file, sep="\t", dtype=str, keep_default_na=False, chunksize=50000)
        for chunk in gt_chunk_iter:
            gt_record_count += len(chunk)
            
            # Calculate matches per row
            # If matched_entity_ids is empty, matches = 0
            # Else matches = number of commas + 1
            def count_matches(s):
                if not s:
                    return 0
                return len(s.split(","))
                
            matches = chunk['matched_entity_ids'].apply(count_matches)
            
            zero_matches += (matches == 0).sum()
            total_matches += matches.sum()
            max_matches = max(max_matches, matches.max())
            
        mean_matches = total_matches / gt_record_count if gt_record_count > 0 else 0
        
        print(f"  Number of Source 1 businesses with zero matches: {zero_matches}")
        print(f"  Mean matches per business: {mean_matches:.2f}")
        print(f"  Max matches per business: {max_matches}")
    else:
        print(f"{gt_file} not found.")

    print("\n=== System RAM Summary (using built-in Windows tools) ===")
    try:
        # systeminfo is slow, maybe try wmic or gcim
        # wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value
        result = subprocess.run(["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/Value"], 
                                capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith("FreePhysicalMemory="):
                    free_kb = int(line.split("=")[1])
                    print(f"  Available RAM: {free_kb / 1024 / 1024:.2f} GB")
                elif line.startswith("TotalVisibleMemorySize="):
                    total_kb = int(line.split("=")[1])
                    print(f"  Total RAM: {total_kb / 1024 / 1024:.2f} GB")
    except Exception as e:
        print(f"Failed to get RAM info: {e}")

if __name__ == "__main__":
    inspect_dataset()

import csv
import argparse
import sys
import json
import os
import re
from typing import Dict, Any, Optional

def extract_kernel_stats(file_path: str, kernel_name_fragment: str) -> Dict[str, Any]:
    """
    Finds TotalCalls, TotalDuration_us, and Ave_us for a specific kernel 
    function in a CSV file where the kernel name contains the fragment.
    """
    stats = {}
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Read header row
            try:
                header = next(reader)
            except StopIteration:
                print(f"Error: File {file_path} is empty.", file=sys.stderr)
                return stats
            
            # Find required column indices
            try:
                name_idx = header.index("Name")
                calls_idx = header.index("TotalCalls")
                duration_idx = header.index("TotalDuration_us")
                # Modified to look for Ave_us as required by the prompt
                ave_us_idx = header.index("Ave_us") 
            except ValueError as e:
                print(f"Error: CSV header in file {file_path} must contain 'Name', 'TotalCalls', 'TotalDuration_us', and 'Ave_us'. Missing: {e}", file=sys.stderr)
                return stats

            # Iterate through data rows
            for row in reader:
                if len(row) > max(name_idx, calls_idx, duration_idx, ave_us_idx):
                    name = row[name_idx]
                    
                    # Check if the name contains the target fragment
                    if kernel_name_fragment in name:
                        # Only return the required Ave_us and name for the main logic
                        stats = {
                            "Name": name,
                            "Ave_us": row[ave_us_idx],
                            "TotalCalls": row[calls_idx],
                            "TotalDuration_us": row[duration_idx],
                            "SourceFile": file_path
                        }
                        # We don't return immediately here, as the main function 
                        # will call this function twice with different fragments ('<0' and '<1').
                        # The original function *did* return immediately, but for 
                        # the new requirement, we need to adapt the calling logic in main.
                        # For simplicity, let's keep the return as is, and adjust main.
                        return stats 
                        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        
    return stats

def load_json(file_path: str) -> Dict[str, Any]:
    """Loads JSON data from a file, returning an empty dict if the file doesn't exist."""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: JSON file {file_path} is corrupted. Starting with empty data.", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}", file=sys.stderr)
        return {}

def save_json(file_path: str, data: Dict[str, Any]):
    """Saves dictionary data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Extracts and compares Ave_us for GOLDEN (kernel<0) and EXPERIMENTAL (kernel<1) kernels from an RPD trace CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Define the required CSV file argument
    parser.add_argument(
        '--csv', 
        type=str, 
        required=True, # Added required=True
        help="Path to the CSV file which records rpd kernels."
    )
    
    # Define optional base kernel name fragment argument
    parser.add_argument(
        '-k', '--kernel', 
        type=str, 
        default="paged_attention_ll4mi_QKV_mfma16_kernel",
        help="Base fragment of the kernel function name to look up (e.g., 'paged_attention_...').\n(Default: paged_attention_ll4mi_QKV_mfma16_kernel)"
    )

    args = parser.parse_args()

    base_kernel_fragment = args.kernel
    csv_path = args.csv
    
    # 1. Extract GOLDEN (kernel<0) stats
    golden_fragment = f"{base_kernel_fragment}<0"
    golden_stats = extract_kernel_stats(csv_path, golden_fragment)

    # 2. Extract EXPERIMENTAL (kernel<1) stats
    experimental_fragment = f"{base_kernel_fragment}<1"
    experimental_stats = extract_kernel_stats(csv_path, experimental_fragment)
    
    result_data = {
        "GOLDEN": None,
        "EXPERIMENTAL": None
    }
    
    # 3. Print Results
    print(f"\n✨ Kernel Analysis for CSV: **{os.path.basename(csv_path)}**")
    print("-" * 30)

    if golden_stats:
        golden_ave_us = golden_stats.get("Ave_us")
        print(f"**GOLDEN** ({golden_fragment}): **Ave_us = {golden_ave_us}** (Calls: {golden_stats.get('TotalCalls', 'N/A')}, Duration: {golden_stats.get('TotalDuration_us', 'N/A')} us)")
        result_data["GOLDEN"] = golden_ave_us
    else:
        print(f"**GOLDEN** ({golden_fragment}): Not found.")

    if experimental_stats:
        experimental_ave_us = experimental_stats.get("Ave_us")
        print(f"**EXPERIMENTAL** ({experimental_fragment}): **Ave_us = {experimental_ave_us}** (Calls: {experimental_stats.get('TotalCalls', 'N/A')}, Duration: {experimental_stats.get('TotalDuration_us', 'N/A')} us)")
        result_data["EXPERIMENTAL"] = experimental_ave_us
    else:
        print(f"**EXPERIMENTAL** ({experimental_fragment}): Not found.")
        
    # 4. Save to json
    summary_file_path = "UT_rpd.json" # Changed filename to UT_rpd.json as requested
    summary_data = load_json(summary_file_path)
    
    # Create the key from the CSV filename (e.g., 'trace_PS1_BS512_ILEN256_GOLDEN.csv' -> 'trace_PS1_BS512_ILEN256_GOLDEN.csv_Ave_us')
    pattern = r'PS(\d+)_BS(\d+)_ILEN(\d+)'
    match = re.search(pattern, csv_path)
    if match:
        ps_value = str(match.group(1))
        bs_value = str(match.group(2))
        ilen_value = str(match.group(3))
    else:
        print("[ERROR] csv-path cannot be parsed.")
        return
    PS_Key = f"PS{ps_value}"
    key = f"BS{bs_value}_ILEN{ilen_value}" + "_Ave_us" 
    if PS_Key not in summary_data:
        summary_data[PS_Key] = {}  # Initialize the nested dictionary

    # Save the extracted data
    if result_data["GOLDEN"] is not None or result_data["EXPERIMENTAL"] is not None:
        summary_data[PS_Key][key] = {
            "GOLDEN": result_data["GOLDEN"],
            "EXPERIMENTAL": result_data["EXPERIMENTAL"]
        }
        save_json(summary_file_path, summary_data)
        print(f"\n✅ Result saved to **{summary_file_path}** under key: **{key}**")
    else:
        print("\n⚠️ No kernel data found. Skipping JSON save.")


if __name__ == "__main__":
    main()
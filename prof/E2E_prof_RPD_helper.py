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
    
    Args:
        file_path (str): Path to the RPD trace CSV file.
        kernel_name_fragment (str): The specific name fragment (e.g., 'kernel<0') 
                                    to look for in the 'Name' column.
                                    
    Returns:
        Dict[str, Any]: A dictionary containing "Name", "Ave_us", "TotalCalls", 
                        "TotalDuration_us", and "SourceFile" if found, otherwise an empty dict.
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
                # Ensure all required columns are present in the header
                name_idx = header.index("Name")
                calls_idx = header.index("TotalCalls")
                duration_idx = header.index("TotalDuration_us")
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
                        stats = {
                            "Name": name,
                            "Ave_us": row[ave_us_idx],
                            "TotalCalls": row[calls_idx],
                            "TotalDuration_us": row[duration_idx],
                            "SourceFile": file_path
                        }
                        # Return the first matching kernel found
                        return stats 
                        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        
    return stats

def load_json(file_path: str) -> Dict[str, Any]:
    """Loads JSON data from a file, returning an empty dict if the file doesn't exist or is corrupted."""
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
        description="Extracts and compares Ave_us for GOLDEN (kernel<0) and EXPERIMENTAL (kernel<1) kernels from an RPD trace CSV file. Designed for incremental updates to the JSON summary.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Define the required CSV file argument
    parser.add_argument(
        '--csv', 
        type=str, 
        required=True,
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
    
    # 1. Extract GOLDEN (kernel<0) statistics
    golden_fragment = f"{base_kernel_fragment}<0"
    golden_stats = extract_kernel_stats(csv_path, golden_fragment)

    # 2. Extract EXPERIMENTAL (kernel<1) statistics
    experimental_fragment = f"{base_kernel_fragment}<1"
    experimental_stats = extract_kernel_stats(csv_path, experimental_fragment)
    
    # Initialize result_data with extracted Ave_us values (if found) or None
    result_data = {
        "GOLDEN": golden_stats.get("Ave_us") if golden_stats else None,
        "EXPERIMENTAL": experimental_stats.get("Ave_us") if experimental_stats else None
    }
    
    # 3. Print Results
    print(f"\n✨ Kernel Analysis for CSV: **{os.path.basename(csv_path)}**")
    print("-" * 30)

    # Print results for GOLDEN
    if golden_stats:
        golden_ave_us = result_data["GOLDEN"]
        print(f"**GOLDEN** ({golden_fragment}): **Ave_us = {golden_ave_us}** (Calls: {golden_stats.get('TotalCalls', 'N/A')}, Duration: {golden_stats.get('TotalDuration_us', 'N/A')} us)")

    # Print results for EXPERIMENTAL
    if experimental_stats:
        experimental_ave_us = result_data["EXPERIMENTAL"]
        print(f"**EXPERIMENTAL** ({experimental_fragment}): **Ave_us = {experimental_ave_us}** (Calls: {experimental_stats.get('TotalCalls', 'N/A')}, Duration: {experimental_stats.get('TotalDuration_us', 'N/A')} us)")
        
    # 4. JSON Handling Logic (Ensuring Incremental Update)
    summary_file_path = "E2E_rpd.json" 
    
    # Only proceed if at least one piece of data was found
    if result_data["GOLDEN"] is not None or result_data["EXPERIMENTAL"] is not None:
        
        # Load existing JSON data
        summary_data = load_json(summary_file_path)
        
        # Parse CSV path to get PS, BS, ILEN keys
        pattern = r'E2E_bs(\d+)_ilen(\d+)_olen(\d+)_PageSize(\d+)_'
        match = re.search(pattern, csv_path)
        
        if not match:
            print("[ERROR] csv-path cannot be parsed (missing PSx_BSx_ILENx pattern).")
            return
            
        bs_value = str(match.group(1))
        ilen_value = str(match.group(2))
        olen_value = str(match.group(3))
        ps_value = str(match.group(3))
        
        PS_Key = f"PS{ps_value}"
        key = f"BS{bs_value}_ILEN{ilen_value}_OLEN{olen_value}" + "_Ave_us" 
        
        # Ensure the top-level PS_Key exists
        if PS_Key not in summary_data:
            summary_data[PS_Key] = {}  

        # Get or initialize the current key's data from the JSON file.
        # This preserves existing GOLDEN/EXPERIMENTAL values that might be missing in this CSV run.
        current_entry = summary_data[PS_Key].get(key, {
            "GOLDEN": None,
            "EXPERIMENTAL": None
        })

        # Update only the found values to ensure incremental update
        updated = False
        
        if result_data["GOLDEN"] is not None:
            current_entry["GOLDEN"] = result_data["GOLDEN"]
            updated = True
        
        if result_data["EXPERIMENTAL"] is not None:
            current_entry["EXPERIMENTAL"] = result_data["EXPERIMENTAL"]
            updated = True
            
        # Write the updated data back to summary_data and save
        if updated:
            summary_data[PS_Key][key] = current_entry
            save_json(summary_file_path, summary_data)
            print(f"\n✅ Result successfully updated and saved to {summary_file_path} ")
        else:
            # Should not be reached if the outer 'if' check passed
            print("\n⚠️ Found data but no update was necessary. Skipping JSON save.")

    else:
        print("\n⚠️ No kernel data found (neither <0 nor <1). Skipping JSON update.")


if __name__ == "__main__":
    main()
import csv
import argparse
import sys
import json
import os
from typing import Dict, Any, Optional

def extract_kernel_stats(file_path: str, kernel_name_fragment: str) -> Dict[str, Any]:
    """
    Finds TotalCalls and TotalDuration_us for a specific kernel function in a CSV file.
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
            except ValueError:
                print(f"Error: CSV header in file {file_path} must contain 'Name', 'TotalCalls', and 'TotalDuration_us'.", file=sys.stderr)
                return stats

            # Iterate through data rows
            for row in reader:
                if len(row) > max(name_idx, calls_idx, duration_idx):
                    name = row[name_idx]
                    
                    # Check if the name contains the target fragment
                    if kernel_name_fragment in name:
                        stats = {
                            "TotalCalls": row[calls_idx],
                            "TotalDuration_us": row[duration_idx],
                            "SourceFile": file_path
                        }
                        return stats # Return immediately upon finding the kernel
                        
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
        description="Compares TotalCalls and TotalDuration_us for a specific kernel function across two RPD trace CSV files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Define two required CSV file arguments
    parser.add_argument(
        '--warmup-csv', 
        type=str, 
        help="Path to the first CSV file which only runs warmup."
    )
    parser.add_argument(
        '--allruns-csv', 
        type=str, 
        help="Path to the second CSV file which runs all runs"
    )
    
    # Define optional kernel name fragment argument
    parser.add_argument(
        '-k', '--kernel', 
        type=str, 
        default="paged_attention_ll4mi_QKV_mfma16_kernel",
        help="Fragment of the kernel function name to look up.\n(Default: paged_attention_ll4mi_QKV_mfma16_kernel)"
    )

    args = parser.parse_args()

    kernel_fragment = args.kernel
    
    # Execute extraction
    stats1 = extract_kernel_stats(args.allruns_csv, kernel_fragment)
    stats2 = extract_kernel_stats(args.warmup_csv, kernel_fragment)

    # Output results
    print(f"--- Kernel: {kernel_fragment} Comparison Report ---\n")

    def print_stats(stats: Dict[str, Any], label: str):
        print(f"--- {label} ({stats.get('SourceFile', 'File not found')}) ---")
        if stats:
            print(f"TotalCalls:       {stats.get('TotalCalls')}")
            print(f"TotalDuration_us: {stats.get('TotalDuration_us')}")
        else:
            print("Target kernel statistics not found.")

    print_stats(stats1, "File 1")
    print()
    print_stats(stats2, "File 2")

    # Perform comparison
    if stats1 and stats2:
        try:
            calls_allruns = int(stats1.get('TotalCalls'))
            calls_warmup  = int(stats2.get('TotalCalls'))
            duration1 = float(stats1.get('TotalDuration_us'))
            duration2 = float(stats2.get('TotalDuration_us'))
            assert calls_allruns > calls_warmup
            duration_diff = duration1 - duration2
            calls_diff    = calls_allruns - calls_warmup
            
            cold_run_avg = duration_diff/calls_diff
            print(f"Avg of cold run is: {cold_run_avg}us.")
                
        except ValueError:
            print("\nError: Could not compare durations, as data is not valid floating point numbers.", file=sys.stderr)
        except AssertionError as e:
            print(f"\nFATAL ASSERTION: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Save to json:
    summary_file = load_json("decode_rpd_runtime.json")
    allruns_basename = os.path.basename(args.allruns_csv)
    key = allruns_basename.replace('trace_', '').replace('_AllRuns.csv', '')
    summary_file[key] = cold_run_avg
    save_json("decode_rpd_runtime.json", summary_file)
    print(f"Result saved to decode_rpd_runtime.json under key: {key}")

if __name__ == "__main__":
    main()


'''
Example:
python ~/PR/aiter/prof/UT_prof_RPD_helper.py \
        --warmup-csv   trace_PS1_BS512_ILEN256_GOLDEN_WARMUP.csv \
        --allruns-csv  trace_PS1_BS512_ILEN256_GOLDEN_AllRuns.csv

'''
import json
import sys
import argparse
from typing import List, Dict, Any, Union

def find_kernel_durations(
    file_path: str, 
    kernel_name_substring: str = "paged_attention_ll4mi_QKV_mfma16_kernel"
) -> Dict[str, Union[float, int]]:
    """
    Reads data from a Chrome Tracing JSON file and filters for the execution
    durations ('dur') of a specific Kernel. Calculates min, max, avg, P90, and P95.

    Args:
        file_path (str): Path to the JSON file.
        kernel_name_substring (str): Substring of the Kernel name to look for.

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing statistics,
        including the P95 duration under the key 'p95_duration'.
    """
    try:
        # 1. Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data: Union[Dict[str, Any], List[Any]] = json.load(f)
            
            # Attempt to extract the traceEvents list
            if isinstance(data, dict) and 'traceEvents' in data:
                events = data['traceEvents']
            elif isinstance(data, list):
                events = data # If the file is directly a list of events
            else:
                print("Warning: JSON structure does not contain 'traceEvents' key or is malformed.", file=sys.stderr)
                return {}
                
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.", file=sys.stderr)
        return {}
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not a valid JSON format.", file=sys.stderr)
        return {}

    durations = []
    
    # Counter for durations > 500 (microsecond threshold)
    duration_500_count = 0 
    
    # 2. Iterate through the events list
    for event in events:
        # 3. Filtering conditions: 'X' (Complete) event, must have 'name', 'dur' fields, and name must match
        if (event.get('ph') == 'X' and 
            'name' in event and 
            'dur' in event and
            kernel_name_substring in event['name']):
            
            # 4. Record the duration
            try:
                # Ensure conversion to float (durations are typically in microseconds)
                duration = float(event['dur'])
                durations.append(duration)
                
                # Check for duration > 500 (original logic)
                if duration > 500:
                    duration_500_count += 1
                    # Real-time printing commented out as requested previously
                    # print(f"duration={duration}, duration_500_count={duration_500_count}")
            except ValueError:
                print(f"Warning: Found non-numeric 'dur' field: {event['dur']}, skipped.", file=sys.stderr)

    
    # 5. Output Results Summary and Calculations
    stats = {}
    
    print(f"\n--- Analysis Summary ({kernel_name_substring}) ---")
    print(f"Found {len(durations)} matching Kernel execution records.")
    
    if durations:
        # Sort for percentile calculation
        sorted_durations = sorted(durations)
        total_count = len(sorted_durations)
        
        # --- Percentile Calculation Helper Function ---
        def calculate_percentile(data: List[float], percentile: float) -> float:
            """Calculates the specific percentile (e.g., 0.9 for P90, 0.95 for P95)."""
            n = len(data)
            if n == 0:
                return 0.0
                
            # Use the N*P/100 index method, rounded down to get the 0-based index
            index = int(n * percentile)
            
            # Ensure index is within [0, n-1] range
            index = max(0, min(index, n - 1))
            return data[index]
        
        # Calculate P90
        p90_duration = calculate_percentile(sorted_durations, 0.90)
        
        # Calculate P95
        p95_duration = calculate_percentile(sorted_durations, 0.95)

        # Store statistics
        stats['min_duration'] = min(durations)
        stats['max_duration'] = max(durations)
        stats['avg_duration'] = sum(durations) / total_count
        stats['p90_duration'] = p90_duration
        stats['p95_duration'] = p95_duration
        stats['total_count'] = total_count
        stats['count_dur_over_500us'] = duration_500_count # Include the 500us count

        
        print(f"Min Duration (min dur): {stats['min_duration']:.4f} us")
        print(f"Max Duration (max dur): {stats['max_duration']:.4f} us")
        print(f"Avg Duration (avg dur): {stats['avg_duration']:.4f} us")
        print(f"**90th Percentile (P90): {stats['p90_duration']:.4f} us**")
        print(f"**95th Percentile (P95): {stats['p95_duration']:.4f} us**")
        print(f"Count > 500 us: {stats['count_dur_over_500us']}") # Display the 500us count
        
    print("---------------------------------------------")
    
    return stats

# --- Use argparse to handle command-line arguments ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyzes Chrome Tracing JSON files to extract specific Kernel durations and calculate statistics (including P90 and P95).",
        epilog="Default Kernel substring is: paged_attention_ll4mi_QKV_mfma16_kernel"
    )
    
    # -i or --input argument for file path (required)
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True,
        help="Input Chrome Tracing JSON file path."
    )
    
    # -k or --kernel argument for the substring (optional)
    parser.add_argument(
        '-k', '--kernel', 
        type=str, 
        default="paged_attention_ll4mi_QKV_mfma16_kernel",
        help="Kernel name substring to filter for."
    )

    args = parser.parse_args()
    
    file_path = args.input
    target_name = args.kernel
    
    # Execute analysis
    stats = find_kernel_durations(file_path, target_name)

    # --- Save P95 to a JSON file ---
    P95_OUTPUT_FILE = "E2E_decode_kernel_95th.json"
    
    if 'p95_duration' in stats:
        # Load existing data or initialize
        try:
            with open(P95_OUTPUT_FILE, 'r') as f:
                output_data = json.load(f)
            if not isinstance(output_data, dict):
                 # Handle case where file is corrupted or not a dict
                output_data = {}
        except (FileNotFoundError, json.JSONDecodeError):
            output_data = {}

        # Use the input file path as the key and the P95 duration as the value
        output_data[file_path] = stats['p95_duration']
        
        # Save the updated data back to the JSON file
        try:
            with open(P95_OUTPUT_FILE, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"\n✅ Successfully saved P95 duration to '{P95_OUTPUT_FILE}' under key: '{file_path}'.")
        except IOError as e:
             print(f"Error: Could not write to output JSON file '{P95_OUTPUT_FILE}': {e}", file=sys.stderr)
    
    # Print a sample of the durations (optional, removed the full list print for brevity)
    if 'total_count' in stats and stats['total_count'] > 0:
        # Note: To print the actual durations, you would need to return the list 
        # from find_kernel_durations, but the current design returns only stats.
        print(f"\n{target_name} analysis complete. Total samples: {stats['total_count']}.")
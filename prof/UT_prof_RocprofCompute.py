import re
import argparse
import subprocess
import sys
import os
from itertools import product 

# --- Configuration ---
# Define constants outside of main for easy modification
PAGE_SIZES = [1, 16]
BS = [1024, 512, 128, 16]      # Batch Size                          [1024, 512, 128, 16] 
CL = [256, 512, 1024, 2048, 4096] # Prefill Length (Context Length)  [256, 512, 1024, 4096]
WARM_UP = 30
ITERS = 30

# Determine HOME directory dynamically
HOME_DIR = os.path.expanduser("~")
UT_SCRIPT = f"{HOME_DIR}/PR/aiter/op_tests/test_pa_ragged_experimental.py"

# Fixed search strings for dispatch IDs
SEARCH_STRING_GOLDEN      = "void paged_attention_ll4mi_QKV_mfma16_kernel<0"
SEARCH_STRING_EXPERIMENT  = "void paged_attention_ll4mi_QKV_mfma16_kernel<1"

# Regex to extract the Dispatch ID (the second number in the log line)
DISPATCH_ID_REGEX = r'^\s*│\s*\d+\s*│\s*(\d+)\s*│'

# --- Utility Functions ---

def run_command(cmd, log_output=True, check_success=True, capture=False, env=None):
    """
    Executes an external command using subprocess.
    All execution logs are printed to sys.stderr.
    """
    cmd_str = ' '.join(cmd)
    # Log command execution to stderr
    print(f"\n[EXEC] {cmd_str}", file=sys.stderr)
    
    try:
        if capture:
            # Capture stdout for later parsing
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check_success,
                encoding='utf-8',
                env=env
            )
            return result.stdout
        else:
            # Used for profile, prints output directly to console (which is now stdout)
            # Since we want to keep logs separate from CSV, redirect profile output to stderr too
            subprocess.run(
                cmd,
                check=check_success,
                stdout=sys.stderr, # Redirect profile's stdout to stderr
                stderr=sys.stderr, # Redirect profile's stderr to stderr
            )
            return None

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}.", file=sys.stderr)
        if capture:
            print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print(f"ERROR: Command '{cmd[0]}' not found. Is it in PATH?", file=sys.stderr)
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}", file=sys.stderr)
        raise

def find_dispatch_id(log_content, search_string):
    """
    Searches log content for a line containing the specific string and 
    उत्सextracts the maximum Dispatch ID.
    """
    id = -1
    for line in log_content.splitlines():
        if search_string in line:
            match = re.match(DISPATCH_ID_REGEX, line)
            if match:
                try:
                    extracted_id = int(match.group(1).strip())
                    id = max(id, extracted_id)
                except ValueError:
                    pass

    return str(id) if id != -1 else None

def parse_analysis_output(raw_output):
    """
    Parses the Mean(us) from the rocprof-compute analyze output table.
    """
    if not raw_output:
        return "N/A"
    
    # Regex to capture the Mean(us) value (the third numeric column in the stats line)
    # Target: │ ... kernel_name ... │    1.00 │   118.19 │      (118.19) │ ...
    regex = r'QKV_mfma16_ke.*?│\s*[\d\.]+\s*│\s*[\d\.]+\s*│\s*([\d\.]+)\s*│'
    
    match = re.search(regex, raw_output, re.DOTALL) 
    
    if match:
        return match.group(1).strip()
    else:
        return "N/A"

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Integrated runner for rocprof profiling and analysis of aiter kernels."
    )
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install required dependencies from the ROCm libexec path (requires sudo/root).'
    )
    parser.add_argument(
        '--csv-output',
        type=str,
        default="UT_prof.csv",
        help='The path to the output CSV file where results will be written.'
    )
    args = parser.parse_args()

    # Install Dependencies (Skipping actual execution if False)
    if args.install_deps:
        try:
            req_path = "/opt/rocm-7.0.0/libexec/rocprofiler-compute/requirements.txt"
            run_command(["pip", "install", "-r", req_path], check_success=True, capture=False)
        except Exception:
            print("Dependency installation failed. Please run manually if necessary.", file=sys.stderr)
            sys.exit(1)

    csv_filepath = args.csv_output
    try:
        with open(csv_filepath, 'w') as csv_file:
            csv_file.write("page_size,bs,context_length,Golden runtime(us),Experimental runtime(us),Speedup (%)\n")
            print(f"[INFO] Writing results to CSV file: {csv_filepath}", file=sys.stderr)
            
            # Iterate over all configurations
            for ps, bs, cl in product(PAGE_SIZES, BS, CL):
                config_name = f"PS={ps}, BS={bs}, CL={cl}"

                try:
                    # 1. Configuration Naming
                    OUT = f"UT_RPC_bs{bs}_c{cl}_PageSize{ps}"
                    ANALYSIS_PATH = f"workloads/{OUT}/MI355/"
                    # Save the log content for debugging (optional)
                    dispatch_log = f"{OUT}.log"
                    
                    print(f"-----------------------------------------------------------")
                    print(f"\n[START] Analyzing config: {config_name}", file=sys.stderr)

                    # 2. Run Profiling
                    if os.path.exists(dispatch_log)==False:
                        cmd_profile = [
                            "rocprof-compute", "profile", "-n", OUT, #"--no-roof",
                            "--", "python", UT_SCRIPT, "-n", str(bs), "-c", str(cl), 
                            "--warmup", str(WARM_UP), "--num-iters", str(ITERS),
                            "--page-size", str(ps)
                        ]

                        env_vars = os.environ.copy()
                        env_vars["HIP_VISIBLE_DEVICES"] = "5"

                        run_command(cmd_profile, env=env_vars, check_success=True, capture=False)
                    else:
                        print(f"{dispatch_log} already exists. Skip profile. Go to analyze step.")

                    # 3. Analyze and Generate Dispatch Log (Capture output for ID lookup)
                    cmd_list_stats = [
                        "rocprof-compute", "analyze", "-p", ANALYSIS_PATH, "--list-stats"
                    ]
                    list_stats_output = run_command(cmd_list_stats, check_success=True, capture=True)
                    with open(dispatch_log, 'w') as f:
                         f.write(list_stats_output)
                    print(f"[LOG] Log saved to {dispatch_log}", file=sys.stderr)


                    # 4. Find Dispatch IDs
                    golden_id = find_dispatch_id(list_stats_output, SEARCH_STRING_GOLDEN)
                    experiment_id = find_dispatch_id(list_stats_output, SEARCH_STRING_EXPERIMENT)

                    if golden_id is None or experiment_id is None:
                        print(f"[ERROR] ID not found for: {config_name}", file=sys.stderr)
                        csv_file.write(f"{ps},{bs},ilen-{cl},N/A,N/A,N/A\n")
                        continue
                    
                    print(f"[{config_name}] Found IDs: Golden={golden_id}, Exp={experiment_id}", file=sys.stderr)
                    
                    # 5. Analyze Golden ID
                    cmd_golden_analyze = [
                        "rocprof-compute", "analyze", "-p", ANALYSIS_PATH, 
                        "-d", golden_id, "-t", "us", "-b", "0"
                    ]
                    golden_raw_output = run_command(cmd_golden_analyze, check_success=True, capture=True)
                    golden_avg_time = parse_analysis_output(golden_raw_output)

                    # 6. Analyze Experiment ID
                    cmd_experi_analyze = [
                        "rocprof-compute", "analyze", "-p", ANALYSIS_PATH, 
                        "-d", experiment_id, "-t", "us", "-b", "0"
                    ]
                    experi_raw_output = run_command(cmd_experi_analyze, check_success=True, capture=True)
                    experi_avg_time = parse_analysis_output(experi_raw_output)

                    # 7. Calculate Speedup and Write Results (to file)
                    speedup_str = "N/A"
                    if golden_avg_time != "N/A" and experi_avg_time != "N/A":
                        try:
                            g_time = float(golden_avg_time)
                            e_time = float(experi_avg_time)
                            # Speedup formula: (Golden / Experimental) * 100%
                            speedup = (g_time / e_time) * 100 
                            speedup_str = f"{speedup:.2f}"
                            print(f"golden_avg_time={golden_avg_time}, experi_avg_time={experi_avg_time}")
                        except ValueError:
                            print(f"[WARNING] Could not convert parsed times to float for calculation.", file=sys.stderr)
                            pass

                    # Save csv
                    csv_file.write(f"{ps},{bs},{cl},{golden_avg_time},{experi_avg_time},{speedup_str}\n")
                    csv_file.flush()

                except Exception as e:
                    # Log fatal error for this configuration to stderr
                    print(f"[FATAL ERROR] {config_name}: {type(e).__name__} - {e}", file=sys.stderr)
                    csv_file.write(f"{ps},{bs},{cl},ERROR,ERROR,ERROR\n")
                    continue
            
        print(f"[INFO] CSV data successfully written to {csv_filepath}", file=sys.stderr)
    
    except Exception as e:
        print(f"\n[GLOBAL ERROR] Script failed to open or write to CSV file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[GLOBAL ERROR] Script failed during initialization: {e}", file=sys.stderr)
        sys.exit(1)

        
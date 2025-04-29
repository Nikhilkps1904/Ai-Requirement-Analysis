from functools import lru_cache
import os
import pandas as pd
import requests
import json
import logging
from queue import Queue, Empty
from threading import Thread, Event
import warnings
from dotenv import load_dotenv
import itertools
import sys
import tkinter as tk
from tkinter import filedialog
import time
from collections import deque
import concurrent.futures
import hashlib
import pickle

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Configure logging with less verbosity for better performance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conflict_detection.log"),
        logging.StreamHandler(sys.stdout)
    ],
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Detect if running in restricted environment
request_log_file = "request_counts.log"
try:
    with open(request_log_file, "a"):
        pass
    use_file_logging = True
except:
    logger.warning("File system access unavailable. Using console logging only.")
    use_file_logging = False

# Enhanced request tracking with batch logging
total_requests = 0
requests_per_minute = deque()
daily_request_limit = 1500
rpm_limit = 15  
max_workers = 4  # Increased from 2
request_log_queue = Queue()
last_log_flush = time.time()
log_flush_interval = 5  # Flush logs every 5 seconds

# Global stop event
stop_event = Event()

# API setup for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY1")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables. Please set it in .env or environment.")
    sys.exit(1)

GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Persistent cache implementation
CACHE_FILE = "api_response_cache.pkl"

def load_cache():
    if use_file_logging and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
    return {}

def save_cache(cache_dict):
    if use_file_logging and cache_dict:
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache_dict, f)
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")

# Load persistent cache
api_response_cache = load_cache()

# Log writer thread
def log_writer():
    last_flush = time.time()
    batch = []
    
    while not stop_event.is_set():
        try:
            # Get logs with a timeout to allow checking stop_event
            item = request_log_queue.get(timeout=0.5)
            batch.append(item)
            request_log_queue.task_done()
            
            # Flush logs if batch size exceeds 10 or time threshold reached
            if len(batch) >= 10 or (time.time() - last_flush > log_flush_interval):
                if use_file_logging and batch:
                    with open(request_log_file, "a") as f:
                        for log_item in batch:
                            f.write(f"{log_item}\n")
                batch = []
                last_flush = time.time()
        except Empty:
            # Flush remaining logs if there's a timeout
            if batch and use_file_logging:
                with open(request_log_file, "a") as f:
                    for log_item in batch:
                        f.write(f"{log_item}\n")
                batch = []
                last_flush = time.time()

# Start log writer thread
log_writer_thread = Thread(target=log_writer, daemon=True)
log_writer_thread.start()

def hash_prompt(prompt):
    """Generate a stable hash for a prompt"""
    return hashlib.md5(prompt.encode()).hexdigest()

def call_inference_api(prompt, api_key=GEMINI_API_KEY, api_url=GEMINI_API_URL, max_retries=5, initial_delay=2):
    """Call Gemini API for inference with improved request tracking and rate limit enforcement"""
    global total_requests, requests_per_minute, api_response_cache
    
    # Check daily limit
    if total_requests >= daily_request_limit:
        logger.error("Daily limit of 1500 requests reached")
        return "Inference failed: Daily request limit reached"
    
    # Check for cached response first (using hash for stability)
    prompt_hash = hash_prompt(prompt)
    if prompt_hash in api_response_cache:
        return api_response_cache[prompt_hash]
    
    # Check RPM limit with adaptive delay
    current_time = time.time()
    # Clean up old timestamps
    while requests_per_minute and current_time - requests_per_minute[0] > 60:
        requests_per_minute.popleft()
    
    # Enforce RPM limit with backoff
    if len(requests_per_minute) >= rpm_limit:
        wait_time = 60 - (current_time - requests_per_minute[0]) + 0.5  # Add small buffer
        # Use exponential backoff if close to limit
        if len(requests_per_minute) >= rpm_limit * 0.9:
            wait_time *= 1.5
        logger.warning(f"RPM limit reached. Waiting {wait_time:.2f} seconds")
        time.sleep(wait_time)
        # Refresh timestamps after waiting
        current_time = time.time()
        while requests_per_minute and current_time - requests_per_minute[0] > 60:
            requests_per_minute.popleft()
    
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    retries = 0
    consecutive_429_count = 0
    
    short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt

    while retries < max_retries and not stop_event.is_set():
        try:
            response = requests.post(f"{api_url}?key={api_key}", headers=headers, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                output = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                # Increment counters
                total_requests += 1
                requests_per_minute.append(time.time())
                
                # Batch log request
                log_msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Request {total_requests}: Prompt={short_prompt}, RPM={len(requests_per_minute)}"
                request_log_queue.put(log_msg)
                
                # Update cache
                api_response_cache[prompt_hash] = output
                if len(api_response_cache) % 50 == 0:  # Save cache every 50 new entries
                    save_cache(api_response_cache)
                    
                consecutive_429_count = 0
                return output
            else:
                error_msg = result.get("error", {}).get("message", "Unknown API error")
                logger.error(f"API error: {error_msg}")
                consecutive_429_count = 0
                return f"Inference failed: {error_msg}"

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                consecutive_429_count += 1
                logger.warning(f"Rate limit exceeded (429). Consecutive 429 count: {consecutive_429_count}")
                
                if consecutive_429_count >= 3:
                    delay = 30
                    logger.warning(f"Three consecutive 429 errors. Pausing for {delay} seconds...")
                    time.sleep(delay)
                    consecutive_429_count = 0
                
                delay = initial_delay * (2 ** retries)
                logger.warning(f"Retrying in {delay} seconds...")
                retries += 1
                time.sleep(delay)
                continue
            
            else:
                logger.error(f"HTTP error: {str(e)}")
                consecutive_429_count = 0
                return f"Inference failed: HTTP error - {str(e)}"

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.error(f"Connection error: {str(e)}")
            consecutive_429_count = 0
            delay = initial_delay * (2 ** retries)
            logger.warning(f"Connection issue. Retrying in {delay} seconds...")
            retries += 1
            time.sleep(delay)
            continue

        except requests.exceptions.RequestException as e:
            logger.error(f"Unexpected request error: {str(e)}")
            consecutive_429_count = 0
            return f"Inference failed: Unexpected error - {str(e)}"

    if stop_event.is_set():
        logger.warning("API call interrupted by stop event.")
        return "Inference failed: Interrupted"
    
    logger.error("Max retries exceeded for API call.")
    consecutive_429_count = 0
    return "Inference failed: Max retries exceeded"

@lru_cache(maxsize=2000)  # Increased cache size
def cached_call_inference_api(prompt):
    """Cached version of API call"""
    result = call_inference_api(prompt)
    return result.strip()

def parse_api_output(full_output):
    """Parse API output with improved robustness"""
    if not full_output or "Inference failed" in full_output:
        logger.warning(f"Blank or failed output: {full_output}")
        return "Unknown", "Requires manual review", "Not applicable"

    full_output = full_output.strip()
    
    try:
        # First try the expected format with || separator
        if "||" in full_output:
            parts = [p.strip() for p in full_output.split("||") if p.strip()]
            if len(parts) >= 2:
                type_part = parts[0].strip()
                reason_part = parts[1].strip()
                if type_part.startswith("Conflict_Type:"):
                    conflict_type = type_part.replace("Conflict_Type:", "").strip()
                    conflict_reason = reason_part.replace("Reason:", "").strip() if reason_part.startswith("Reason:") else reason_part
                    if conflict_type:
                        return conflict_type, conflict_reason, "Not applicable"

        # Try alternative format with ": " separator
        if ": " in full_output:
            parts = [p.strip() for p in full_output.split(": ", 1) if p.strip()]
            if len(parts) == 2:
                conflict_type = parts[0]
                conflict_reason = parts[1]
                if conflict_type and conflict_reason:
                    return conflict_type, conflict_reason, "Not applicable"
        
        # Try to extract conflict type and reason more aggressively
        lines = full_output.split('\n')
        for line in lines:
            if "conflict" in line.lower() and not line.lower().startswith("no conflict"):
                return "Potential Conflict", line.strip(), "Not applicable"
        
        # Check for "No Conflict" specifically
        if "no conflict" in full_output.lower():
            return "No Conflict", "Requirements are compatible", "Not applicable"
    
    except Exception as e:
        logger.warning(f"Error parsing output: {e} - '{full_output}'")
    
    # Fallback
    logger.warning(f"Malformed output, unable to parse: '{full_output}'")
    return "Unknown", "Requires manual review", "Not applicable"

def ensure_directories():
    """Create Results directory with CSV and XLSX subdirectories if they don't exist"""
    base_dir = "Results"
    csv_dir = os.path.join(base_dir, "CSV")
    xlsx_dir = os.path.join(base_dir, "XLSX")
    
    if use_file_logging:
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(xlsx_dir, exist_ok=True)
    return csv_dir, xlsx_dir

def compute_requirement_checksum(requirements):
    """Generate a checksum for a list of requirements to allow incremental processing"""
    requirements_str = "||".join(sorted(requirements))
    return hashlib.md5(requirements_str.encode()).hexdigest()

def check_requirements_batch(req_pairs, batch_size=25):
    """Process a batch of requirement pairs efficiently using ThreadPoolExecutor"""
    global stop_event
    results = []
    
    prompt_template = (
        "Analyze the following requirements: Requirement 1: '{req1}' and Requirement 2: '{req2}'. Identify any conflicts between them. "
        "For each pair, determine if there is a conflict, and if so, specify a descriptive conflict type (e.g., 'Performance Conflict', 'Cost Conflict', etc.) and the reason (as a one-line sentence). "
        "The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is your determined conflict type, and <reason> is a one-line explanation. "
        "If no conflict exists, output: \"Conflict_Type: No Conflict||Reason: Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )
    
    def process_pair(pair_data):
        if stop_event.is_set():
            return None
        req1, req2 = pair_data
        input_text = prompt_template.format(req1=req1, req2=req2)
        full_output = cached_call_inference_api(input_text)
        if stop_event.is_set():
            return None
        conflict_type, conflict_reason, _ = parse_api_output(full_output)
        if conflict_type != "No Conflict":
            return {
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason
            }
        return None
    
    # Process req_pairs in smaller batches
    for i in range(0, len(req_pairs), batch_size):
        if stop_event.is_set():
            break
        
        batch = req_pairs[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(req_pairs) + batch_size - 1)//batch_size} ({len(batch)} pairs)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {executor.submit(process_pair, pair): pair for pair in batch}
            
            for future in concurrent.futures.as_completed(future_to_pair):
                if stop_event.is_set():
                    break
                
                result = future.result()
                if result:
                    results.append(result)
        
        # Save intermediate results every batch
        if results and use_file_logging:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f"Results/temp_results_{int(time.time())}.csv", index=False)
    
    return results

def check_new_requirement(new_req, all_existing_requirements, checked_pairs=None):
    """Check a new requirement against existing ones using the efficient batch processing"""
    if checked_pairs is None:
        checked_pairs = set()
    
    pairs = []
    for existing_req in all_existing_requirements:
        pair_key = f"{new_req}||{existing_req}"
        if pair_key in checked_pairs:
            continue
        pairs.append((new_req, existing_req))
        checked_pairs.add(pair_key)
    
    results = check_requirements_batch(pairs)
    if results:
        return pd.DataFrame(results)
    return pd.DataFrame(columns=["Requirement_1", "Requirement_2", "Conflict_Type", "Conflict_Reason"])

def predict_conflicts(input_file, new_requirement=None):
    """Analyze requirements for conflicts with improved batch processing and caching"""
    start_time = time.time()
    
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found. Exiting.")
        sys.exit(1)

    try:
        df_input = pd.read_csv(input_file, encoding='utf-8')
        if "Requirements" not in df_input.columns:
            logger.error("Input file must contain a 'Requirements' column")
            sys.exit(1)
        logger.info(f"Loaded {len(df_input)} requirements from {input_file}")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        sys.exit(1)

    all_original_requirements = df_input["Requirements"].tolist()
    if not all_original_requirements:
        logger.error("No requirements found in the input file.")
        sys.exit(1)
    
    # Check if we have previously processed results to avoid redundant work
    requirements_checksum = compute_requirement_checksum(all_original_requirements)
    incremental_file = f"Results/CSV/results_{requirements_checksum}.csv"
    
    results = []
    if new_requirement:
        new_results = check_new_requirement(new_requirement, all_original_requirements)
        if not new_results.empty:
            results = new_results.to_dict('records')
    else:
        # Check if we have already processed these requirements
        if use_file_logging and os.path.exists(incremental_file) and not new_requirement:
            logger.info(f"Found existing results for these requirements. Loading from {incremental_file}")
            output_df = pd.read_csv(incremental_file)
            
            # Generate output files with timestamp
            csv_dir, xlsx_dir = ensure_directories()
            suffix = f"_{int(time.time())}"
            csv_output = os.path.join(csv_dir, f"results{suffix}.csv")
            xlsx_output = os.path.join(xlsx_dir, f"results{suffix}.xlsx")
            
            if use_file_logging:
                output_df.to_csv(csv_output, index=False)
                output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
                logger.info(f"Reused existing results. Files saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
            return output_df
        
        # Determine most efficient processing strategy
        num_requirements = len(all_original_requirements)
        if num_requirements <= 10:
            logger.info(f"Dataset has {num_requirements} requirements. Using full pairwise comparison.")
            req_pairs = list(itertools.combinations(all_original_requirements, 2))
        elif num_requirements <= 25:
            logger.info(f"Dataset has {num_requirements} requirements. Using baseline mode.")
            # Use first requirement as baseline for smaller datasets
            baseline_req = all_original_requirements[0]
            req_pairs = [(baseline_req, req) for req in all_original_requirements[1:]]
        else:
            logger.info(f"Dataset has {num_requirements} requirements. Using smart sampling.")
            # For larger datasets, use a combination of baseline and strategic sampling
            baseline_reqs = all_original_requirements[:3]  # Use first 3 as baselines
            baseline_pairs = [(base, req) for base in baseline_reqs 
                             for req in all_original_requirements if base != req]
            
            # Add a strategic sample of other pairs (25% of possible combinations)
            remaining_pairs = list(itertools.combinations(all_original_requirements[3:], 2))
            import random
            random.seed(42)  # For reproducibility
            sample_size = min(len(remaining_pairs) // 4, 1000)  # At most 1000 extra pairs
            sampled_pairs = random.sample(remaining_pairs, sample_size)
            
            req_pairs = baseline_pairs + sampled_pairs
        
        logger.info(f"Will analyze {len(req_pairs)} pairs in batches")
        results = check_requirements_batch(req_pairs)

    output_df = pd.DataFrame(results) if results else pd.DataFrame(columns=["Requirement_1", "Requirement_2", "Conflict_Type", "Conflict_Reason"])
    
    # Save results with checksum for incremental processing
    if not new_requirement and use_file_logging:
        output_df.to_csv(incremental_file, index=False)
    
    # Save regular outputs
    csv_dir, xlsx_dir = ensure_directories()
    suffix = f"_{int(time.time())}"
    csv_output = os.path.join(csv_dir, f"results{suffix}.csv")
    xlsx_output = os.path.join(xlsx_dir, f"results{suffix}.xlsx")
    
    if use_file_logging:
        try:
            output_df.to_csv(csv_output, index=False)
            output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
            logger.info(f"Analysis complete. Results saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")
        except Exception as e:
            logger.error(f"Failed to save results to files: {str(e)}")
            logger.info("Results (not saved to file):\n" + output_df.to_string())
    else:
        logger.info("File saving skipped (no file system access). Results:\n" + output_df.to_string())
    
    # Save final cache state
    save_cache(api_response_cache)
    
    # Summarize performance
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Total API requests made: {total_requests}")
    logger.info(f"Cache hits: {len(api_response_cache) - total_requests}")
    
    # Final summary for log
    summary_msg = f"Run completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_msg += f"Total processing time: {elapsed_time:.2f} seconds\n"
    summary_msg += f"Total API requests: {total_requests}/{daily_request_limit}\n"
    summary_msg += f"Requests in last minute: {len(requests_per_minute)}/{rpm_limit}\n"
    summary_msg += f"Cache entries: {len(api_response_cache)}\n"
    if use_file_logging:
        request_log_queue.put("="*50)
        request_log_queue.put(summary_msg)
        request_log_queue.put("="*50)
    
    return output_df

def display_menu():
    """Display the simplified menu and handle user input"""
    try:
        root = tk.Tk()
        root.withdraw()
    except:
        logger.warning("Tkinter unavailable. Using console input for file selection.")
        return console_menu()

    while True:
        print("\n=== Requirements Conflict Detection Menu ===")
        print("1. Analyze Requirements")
        print("2. Enter a new requirement")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            input_file = filedialog.askopenfilename(title="Select CSV file for analysis", filetypes=[("CSV files", "*.csv")])
            if not input_file:
                logger.error("No file selected.")
                continue
            predict_conflicts(input_file)

        elif choice == "2":
            new_requirement = input("Enter a new requirement to analyze (or 'back' to return): ").strip()
            if new_requirement.lower() == 'back':
                continue
            if not new_requirement:
                logger.error("Requirement cannot be empty.")
                continue
            input_file = filedialog.askopenfilename(title="Select CSV file to compare against", filetypes=[("CSV files", "*.csv")])
            if not input_file:
                logger.error("No file selected.")
                continue
            predict_conflicts(input_file, new_requirement=new_requirement)

        elif choice == "3":
            logger.info("Exiting the program.")
            stop_event.set()
            # Wait for log writer to finish
            if log_writer_thread.is_alive():
                request_log_queue.join()
                log_writer_thread.join(timeout=2.0)
            sys.exit(0)

        else:
            logger.error("Invalid choice. Please enter a number between 1 and 3.")

def console_menu():
    """Fallback menu for environments without Tkinter"""
    while True:
        print("\n=== Requirements Conflict Detection Menu ===")
        print("1. Analyze Requirements")
        print("2. Enter a new requirement")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            input_file = input("Enter the path to the CSV file: ").strip()
            if not os.path.exists(input_file):
                logger.error("File not found.")
                continue
            predict_conflicts(input_file)

        elif choice == "2":
            new_requirement = input("Enter a new requirement to analyze (or 'back' to return): ").strip()
            if new_requirement.lower() == 'back':
                continue
            if not new_requirement:
                logger.error("Requirement cannot be empty.")
                continue
            input_file = input("Enter the path to the CSV file: ").strip()
            if not os.path.exists(input_file):
                logger.error("File not found.")
                continue
            predict_conflicts(input_file, new_requirement=new_requirement)

        elif choice == "3":
            logger.info("Exiting the program.")
            stop_event.set()
            # Wait for log writer to finish
            if log_writer_thread.is_alive():
                request_log_queue.join()
                log_writer_thread.join(timeout=2.0)
            sys.exit(0)

        else:
            logger.error("Invalid choice. Please enter a number between 1 and 3.")

def main():
    """Main function with enhanced interrupt handling and performance tracking"""
    start_time = time.time()
    try:
        display_menu()
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user. Cleaning up...")
        stop_event.set()
        # Wait for log writer to finish
        if log_writer_thread.is_alive():
            request_log_queue.join()
            log_writer_thread.join(timeout=2.0)
        # Save cache on exit
        save_cache(api_response_cache)
        elapsed_time = time.time() - start_time
        logger.info(f"Run interrupted. Total execution time: {elapsed_time:.2f} seconds")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        stop_event.set()
        # Save cache on error
        save_cache(api_response_cache)
        sys.exit(1)

if __name__ == "__main__":
    main()
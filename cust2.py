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

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conflict_detection.log"),
        logging.StreamHandler()
    ],
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Detect if running in Google AI Studio
request_log_file = "request_counts.log"
try:
    with open(request_log_file, "a"):
        pass
    use_file_logging = True
except:
    logger.warning("File system access unavailable (likely Google AI Studio). Using console logging only.")
    use_file_logging = False

# Global request tracking
total_requests = 0
requests_per_minute = deque()
daily_request_limit = 1500
rpm_limit = 15

# Global stop event
stop_event = Event()

# API setup for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY1")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables. Please set it in .env or environment.")
    sys.exit(1)

GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

def call_inference_api(prompt, api_key=GEMINI_API_KEY, api_url=GEMINI_API_URL, max_retries=5, initial_delay=2):
    """Call Gemini API for inference with request tracking, rate limit enforcement, and interrupt support"""
    global total_requests, requests_per_minute
    
    # Check daily limit
    if total_requests >= daily_request_limit:
        logger.error("Daily limit of 1500 requests reached")
        return "Inference failed: Daily request limit reached"
    
    # Check RPM limit
    current_time = time.time()
    while requests_per_minute and current_time - requests_per_minute[0] > 60:
        requests_per_minute.popleft()
    
    if len(requests_per_minute) >= rpm_limit:
        wait_time = 60 - (current_time - requests_per_minute[0])
        logger.warning(f"RPM limit reached. Waiting {wait_time:.1f} seconds")
        time.sleep(wait_time)
        while requests_per_minute and time.time() - requests_per_minute[0] > 60:
            requests_per_minute.popleft()
    
    # Warn at 90% of limits
    if total_requests >= daily_request_limit * 0.9:
        logger.warning(f"Approaching daily limit: {total_requests}/{daily_request_limit} requests")
    if len(requests_per_minute) >= rpm_limit * 0.9:
        logger.warning(f"Approaching RPM limit: {len(requests_per_minute)}/{rpm_limit} requests/minute")
    
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    retries = 0
    consecutive_429_count = 0

    while retries < max_retries and not stop_event.is_set():
        try:
            logger.debug(f"Sending request to {api_url} with prompt: {prompt[:100]}...")
            response = requests.post(f"{api_url}?key={api_key}", headers=headers, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Raw API response: {json.dumps(result, indent=2)[:500]}...")

            if "candidates" in result and len(result["candidates"]) > 0:
                output = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Increment counters
                total_requests += 1
                requests_per_minute.append(time.time())
                # Log request
                log_msg = f"Request {total_requests}: Prompt={prompt[:50]}..., RPM={len(requests_per_minute)}"
                logger.info(log_msg)
                if use_file_logging:
                    with open(request_log_file, "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {log_msg}\n")
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

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            consecutive_429_count = 0
            delay = initial_delay * (2 ** retries)
            logger.warning(f"Connection issue. Retrying in {delay} seconds...")
            retries += 1
            time.sleep(delay)
            continue

        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {str(e)}")
            consecutive_429_count = 0
            delay = initial_delay * (2 ** retries)
            logger.warning(f"Timeout occurred. Retrying in {delay} seconds...")
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

@lru_cache(maxsize=1000)
def cached_call_inference_api(prompt):
    """Cached version of API call"""
    result = call_inference_api(prompt)
    return result.strip()

def parse_api_output(full_output):
    """Parse API output robustly, expecting 'Conflict_Type: <type>||Reason: <reason>' or similar"""
    if not full_output or "Inference failed" in full_output:
        logger.warning(f"Blank or failed output: {full_output}")
        return "Unknown", "Requires manual review", "Not applicable"

    full_output = full_output.strip()
    logger.debug(f"Parsing API output: '{full_output}'")

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
                else:
                    logger.warning(f"Empty conflict type in: '{full_output}'")
                    return "Unknown", "Requires manual review", "Not applicable"
            else:
                logger.warning(f"No 'Conflict_Type:' prefix in: '{full_output}'")
                return "Unknown", "Requires manual review", "Not applicable"

    if ": " in full_output:
        parts = [p.strip() for p in full_output.split(": ", 1) if p.strip()]
        if len(parts) == 2:
            conflict_type = parts[0]
            conflict_reason = parts[1]
            if conflict_type and conflict_reason:
                return conflict_type, conflict_reason, "Not applicable"
            else:
                logger.warning(f"Invalid type or reason in fallback format: '{full_output}'")
                return "Unknown", "Requires manual review", "Not applicable"

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

def check_new_requirement(new_req, all_existing_requirements, checked_pairs=None):
    """Check a new requirement against existing ones for conflicts using a queue with robust interrupt handling"""
    if checked_pairs is None:
        checked_pairs = set()

    prompt_template = (
        "Analyze the following requirements: Requirement 1: '{req1}' and Requirement 2: '{req2}'. Identify any conflicts between them. "
        "For each pair, determine if there is a conflict, and if so, specify a descriptive conflict type (e.g., 'Performance Conflict', 'Cost Conflict', etc.) and the reason (as a one-line sentence). "
        "The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is your determined conflict type, and <reason> is a one-line explanation. "
        "If no conflict exists, output: \"Conflict_Type: No Conflict||Reason: Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )

    def process_task(task, result_queue):
        if stop_event.is_set():
            return
        existing_req, input_text = task
        full_output = cached_call_inference_api(input_text)
        if stop_event.is_set():
            return
        conflict_type, conflict_reason, _ = parse_api_output(full_output)
        if conflict_type != "No Conflict":
            result_queue.put({
                "Requirement_1": new_req,
                "Requirement_2": existing_req,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason
            })

    tasks = []
    for existing_req in all_existing_requirements:
        pair_key = f"{new_req}||{existing_req}"
        if pair_key in checked_pairs:
            continue
        input_text = prompt_template.format(req1=new_req, req2=existing_req)
        tasks.append((existing_req, input_text))
        checked_pairs.add(pair_key)

    result_queue = Queue()
    task_queue = Queue()
    for task in tasks:
        if stop_event.is_set():
            break
        task_queue.put(task)

    def worker():
        while not stop_event.is_set():
            try:
                task = task_queue.get(timeout=0.1)
                process_task(task, result_queue)
                task_queue.task_done()
                time.sleep(4)  # Stricter delay for 15 RPM
            except Empty:
                break

    num_workers = 2
    threads = [Thread(target=worker) for _ in range(num_workers)]
    for t in threads:
        t.daemon = True
        t.start()

    try:
        task_queue.join()
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Stopping queue processing...")
        stop_event.set()
        while not task_queue.empty():
            try:
                task_queue.get_nowait()
                task_queue.task_done()
            except Empty:
                break
        for t in threads:
            t.join(timeout=2.0)
        logger.info("Queue processing stopped.")

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    return pd.DataFrame(results) if results else None

def predict_conflicts(input_file, new_requirement=None):
    """Analyze requirements for conflicts using hybrid approach with queue and robust interrupt handling"""
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

    prompt_template_single = (
        "Analyze the following requirements: Requirement 1: '{req1}' and Requirement 2: '{req2}'. Identify any conflicts between them. "
        "For each pair, determine if there is a conflict, and if so, specify a descriptive conflict type (e.g., 'Performance Conflict', 'Cost Conflict', etc.) and the reason (as a one-line sentence). "
        "The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is your determined conflict type, and <reason> is a one-line explanation. "
        "If no conflict exists, output: \"Conflict_Type: No Conflict||Reason: Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )

    results = []
    checked_pairs = set()

    if new_requirement:
        new_results = check_new_requirement(new_requirement, all_original_requirements, checked_pairs)
        if new_results is not None and not new_results.empty:
            results.extend(new_results.to_dict('records'))
    else:
        num_requirements = len(all_original_requirements)
        if num_requirements <= 15:
            logger.info(f"Dataset has {num_requirements} requirements. Using baseline mode.")
            baseline_req = all_original_requirements[0]
            req_pairs = [(baseline_req, req) for req in all_original_requirements[1:]]
        else:
            logger.info(f"Dataset has {num_requirements} requirements. Using exhaustive mode.")
            req_pairs = list(itertools.combinations(all_original_requirements, 2))
        logger.info(f"Analyzing {len(req_pairs)} pairs")

        def process_task(task, result_queue):
            if stop_event.is_set():
                return
            req1, req2, input_text = task
            full_output = cached_call_inference_api(input_text)
            if stop_event.is_set():
                return
            conflict_type, conflict_reason, _ = parse_api_output(full_output)
            if conflict_type != "No Conflict":
                result_queue.put({
                    "Requirement_1": req1,
                    "Requirement_2": req2,
                    "Conflict_Type": conflict_type,
                    "Conflict_Reason": conflict_reason
                })

        tasks = []
        for req1, req2 in req_pairs:
            pair_key = f"{req1}||{req2}"
            if pair_key in checked_pairs:
                continue
            input_text = prompt_template_single.format(req1=req1, req2=req2)
            tasks.append((req1, req2, input_text))
            checked_pairs.add(pair_key)

        result_queue = Queue()
        task_queue = Queue()
        for task in tasks:
            if stop_event.is_set():
                break
            task_queue.put(task)

        def worker():
            while not stop_event.is_set():
                try:
                    task = task_queue.get(timeout=0.1)
                    process_task(task, result_queue)
                    task_queue.task_done()
                    time.sleep(4)  # Stricter delay for 15 RPM
                except Empty:
                    break

        num_workers = 2
        threads = [Thread(target=worker) for _ in range(num_workers)]
        for t in threads:
            t.daemon = True
            t.start()

        try:
            task_queue.join()
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt received. Stopping queue processing...")
            stop_event.set()
            while not task_queue.empty():
                try:
                    task_queue.get_nowait()
                    task_queue.task_done()
                except Empty:
                    break
            for t in threads:
                t.join(timeout=2.0)
            logger.info("Queue processing stopped.")

        while not result_queue.empty():
            results.append(result_queue.get())

    output_results = results
    output_df = pd.DataFrame(output_results) if output_results else pd.DataFrame(columns=["Requirement_1", "Requirement_2", "Conflict_Type", "Conflict_Reason"])

    # Summarize API usage
    logger.info(f"Total API requests made: {total_requests}")
    logger.info(f"Requests in last minute (at end): {len(requests_per_minute)}")
    summary_msg = f"Run completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_msg += f"Total API requests: {total_requests}/{daily_request_limit}\n"
    summary_msg += f"Requests in last minute: {len(requests_per_minute)}/{rpm_limit}\n"
    if use_file_logging:
        with open(request_log_file, "a") as f:
            f.write("="*50 + "\n" + summary_msg + "="*50 + "\n")

    csv_dir, xlsx_dir = ensure_directories()
    suffix = f"_{int(time.time())}" if new_requirement else ""
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

    return output_df

def display_menu():
    """Display the simplified menu and handle user input"""
    try:
        root = tk.Tk()
        root.withdraw()
    except:
        logger.warning("Tkinter unavailable (likely Google AI Studio). Using console input for file selection.")
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
            sys.exit(0)

        else:
            logger.error("Invalid choice. Please enter a number between 1 and 3.")

def main():
    """Main function with interrupt handling"""
    try:
        display_menu()
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user. Exiting...")
        stop_event.set()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
import os
import pandas as pd
import requests
import json
import logging
from tqdm import tqdm
import warnings
from dotenv import load_dotenv
import itertools
import time
from functools import lru_cache
from inputimeout import inputimeout, TimeoutOccurred
import sys

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

# API setup for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables. Please set it in .env or environment.")
    sys.exit(1)

GEMINI_MODEL = "gemini-2.0-flash-lite"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

def call_inference_api(prompt, api_key=GEMINI_API_KEY, api_url=GEMINI_API_URL, max_retries=5, initial_delay=4):
    """Call Gemini API for inference with exponential backoff and improved error handling"""
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    retries = 0

    while retries < max_retries:
        try:
            logger.debug(f"Sending request to {api_url} with prompt: {prompt[:100]}...")
            response = requests.post(f"{api_url}?key={api_key}", headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Raw API response: {json.dumps(result, indent=2)[:500]}...")

            if "candidates" in result and len(result["candidates"]) > 0:
                output = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                return output
            else:
                error_msg = result.get("error", {}).get("message", "Unknown API error")
                logger.error(f"API error: {error_msg}")
                return f"Inference failed: {error_msg}"

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                delay = initial_delay * (2 ** retries)
                logger.warning(f"Rate limit exceeded (429). Retrying in {delay} seconds...")
                time.sleep(delay)
                retries += 1
                continue
            else:
                logger.error(f"HTTP error: {str(e)}")
                return f"Inference failed: HTTP error - {str(e)}"

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            delay = initial_delay * (2 ** retries)
            logger.warning(f"Connection issue. Retrying in {delay} seconds...")
            time.sleep(delay)
            retries += 1
            continue

        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {str(e)}")
            delay = initial_delay * (2 ** retries)
            logger.warning(f"Timeout occurred. Retrying in {delay} seconds...")
            time.sleep(delay)
            retries += 1
            continue

        except requests.exceptions.RequestException as e:
            logger.error(f"Unexpected request error: {str(e)}")
            return f"Inference failed: Unexpected error - {str(e)}"

    logger.error("Max retries exceeded for API call.")
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

    # Try standard format: "Conflict_Type: <type>||Reason: <reason>"
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

    # Fallback: Try "<type>: <reason>" format
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
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(xlsx_dir, exist_ok=True)
    return csv_dir, xlsx_dir

def api_pseudo_train(test_file, iterations=2):
    """Train using provided Conflict_Type as ground truth, refining API predictions"""
    if not os.path.exists(test_file):
        logger.error(f"Input file {test_file} not found. Exiting.")
        sys.exit(1)

    try:
        original_df = pd.read_csv(test_file)
        if "Requirement_1" not in original_df.columns or "Requirement_2" not in original_df.columns:
            logger.error("Input file must contain 'Requirement_1' and 'Requirement_2' columns")
            sys.exit(1)
        if "Conflict_Type" not in original_df.columns:
            logger.warning("No 'Conflict_Type' column found; assuming 'Unknown' for training.")
            original_df["Conflict_Type"] = "Unknown"
        if "Conflict_Reason" not in original_df.columns:
            logger.warning("No 'Conflict_Reason' column found; assuming empty reasons.")
            original_df["Conflict_Reason"] = ""
    except Exception as e:
        logger.error(f"Error reading test file: {e}")
        sys.exit(1)
    
    logger.info(f"Starting pseudo-training with {len(original_df)} examples using {GEMINI_MODEL} via Gemini API")
    
    prompt_template = (
        "Analyze the following requirements: Requirement 1: '{req1}' and Requirement 2: '{req2}'. Identify any conflicts between them. "
        "For each pair, determine if there is a conflict, and if so, specify a descriptive conflict type (e.g., 'Performance Conflict', 'Cost Conflict', etc.) and the reason (as a one-line sentence). "
        "The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is your determined conflict type, and <reason> is a one-line explanation. "
        "If no conflict exists, output: \"Conflict_Type: No Conflict||Reason: Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )

    results = []
    iteration = 0
    max_iterations = iterations
    conflict_type_weights = {}

    df_train = original_df.copy()

    for _ in range(max_iterations):
        iteration += 1
        logger.info(f"Pseudo-training iteration {iteration}/{max_iterations}")
        
        prompt_template_iter = prompt_template
        if iteration > 1 and results:
            weighted_conflicts = ', '.join([f"{k} (weight: {v:.2f})" for k, v in sorted(conflict_type_weights.items(), key=lambda x: x[1], reverse=True)])
            prompt_template_iter += f" Consider these previously identified conflict types and their weights for consistency: {weighted_conflicts}."
        
        results.clear()

        for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Iteration {iteration}"):
            req1, req2 = row["Requirement_1"], row["Requirement_2"]
            expected_conflict = row["Conflict_Type"]
            expected_reason = row["Conflict_Reason"]
            
            input_text = prompt_template_iter.format(req1=req1, req2=req2)
            
            full_output = cached_call_inference_api(input_text)
            predicted_type, predicted_reason, _ = parse_api_output(full_output)
            
            # Use expected conflict type as ground truth, update weights based on matches
            if predicted_type == expected_conflict:
                conflict_type_weights[expected_conflict] = conflict_type_weights.get(expected_conflict, 0) + 1.0
            elif predicted_type != "Unknown":
                conflict_type_weights[predicted_type] = conflict_type_weights.get(predicted_type, 0) + 0.5  # Lower weight for API-derived types
            
            results.append({
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": predicted_type,  # API prediction
                "Conflict_Reason": predicted_reason,
                "Expected_Conflict": expected_conflict,  # Ground truth from CSV
                "Expected_Reason": expected_reason
            })
            time.sleep(1)

        correct = sum(1 for r in results if r["Conflict_Type"] == r["Expected_Conflict"])
        accuracy = correct / len(results) if results else 0
        logger.info(f"Iteration {iteration} accuracy (against Expected_Conflict): {accuracy:.2%}")
        
        df_train = pd.DataFrame(results)

        logger.info(f"Learned conflict types after iteration {iteration}: {list(conflict_type_weights.keys())}")

    output_results = [
        {
            "Requirement_1": r["Requirement_1"],
            "Requirement_2": r["Requirement_2"],
            "Conflict_Type": r["Conflict_Type"],
            "Conflict_Reason": r["Conflict_Reason"],
            "Expected_Conflict": r["Expected_Conflict"],
            "Expected_Reason": r["Expected_Reason"]
        } for r in results
    ]
    output_df = pd.DataFrame(output_results)

    csv_dir, xlsx_dir = ensure_directories()
    csv_output = os.path.join(csv_dir, "training_results.csv")
    xlsx_output = os.path.join(xlsx_dir, "training_results.xlsx")
    
    output_df.to_csv(csv_output, index=False)
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    logger.info(f"Pseudo-training complete after {max_iterations} iterations. Results saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")
    return output_df, conflict_type_weights

def check_new_requirement(new_req, all_existing_requirements, checked_pairs=None, conflict_type_weights=None):
    if checked_pairs is None:
        checked_pairs = set()

    prompt_template = (
        "Analyze the following requirements: Requirement 1: '{req1}' and Requirement 2: '{req2}'. Identify any conflicts between them. "
        "For each pair, determine if there is a conflict, and if so, specify a descriptive conflict type (e.g., 'Performance Conflict', 'Cost Conflict', etc.) and the reason (as a one-line sentence). "
        "The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is your determined conflict type, and <reason> is a one-line explanation. "
        "If no conflict exists, output: \"Conflict_Type: No Conflict||Reason: Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )
    if conflict_type_weights:
        weighted_conflicts = ', '.join([f"{k} (weight: {v:.2f})" for k, v in sorted(conflict_type_weights.items(), key=lambda x: x[1], reverse=True)])
        prompt_template += f" Consider these previously identified conflict types and their weights for consistency: {weighted_conflicts}."

    results = []
    for existing_req in all_existing_requirements:
        pair_key = f"{new_req}||{existing_req}"
        if pair_key in checked_pairs:
            continue

        input_text = prompt_template.format(req1=new_req, req2=existing_req)

        full_output = cached_call_inference_api(input_text)
        conflict_type, conflict_reason, _ = parse_api_output(full_output)

        if conflict_type != "No Conflict":
            results.append({
                "Requirement_1": new_req,
                "Requirement_2": existing_req,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason
            })
        checked_pairs.add(pair_key)
        time.sleep(1)

    return pd.DataFrame(results) if results else None

def predict_conflicts(input_file, new_requirement=None, conflict_type_weights=None, exhaustive=False):
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

    prompt_template_single = (
        "Analyze the following requirements: Requirement 1: '{req1}' and Requirement 2: '{req2}'. Identify any conflicts between them. "
        "For each pair, determine if there is a conflict, and if so, specify a descriptive conflict type (e.g., 'Performance Conflict', 'Cost Conflict', etc.) and the reason (as a one-line sentence). "
        "The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is your determined conflict type, and <reason> is a one-line explanation. "
        "If no conflict exists, output: \"Conflict_Type: No Conflict||Reason: Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )
    if conflict_type_weights:
        weighted_conflicts = ', '.join([f"{k} (weight: {v:.2f})" for k, v in sorted(conflict_type_weights.items(), key=lambda x: x[1], reverse=True)])
        prompt_template_single += f" Consider these previously identified conflict types and their weights for consistency: {weighted_conflicts}."

    results = []
    checked_pairs = set()

    if new_requirement:
        new_results = check_new_requirement(new_requirement, all_original_requirements, checked_pairs, conflict_type_weights)
        if new_results is not None and not new_results.empty:
            results.extend(new_results.to_dict('records'))
    else:
        baseline_req = all_original_requirements[0]
        req_pairs = [(baseline_req, req) for req in all_original_requirements[1:]] if not exhaustive else list(itertools.combinations(all_original_requirements, 2))
        logger.info(f"Analyzing {len(req_pairs)} pairs (exhaustive={exhaustive})")

        for req1, req2 in tqdm(req_pairs, desc="Analyzing conflicts"):
            pair_key = f"{req1}||{req2}"
            if pair_key in checked_pairs:
                continue

            input_text = prompt_template_single.format(req1=req1, req2=req2)
            full_output = cached_call_inference_api(input_text)
            conflict_type, conflict_reason, _ = parse_api_output(full_output)

            if conflict_type != "No Conflict":
                results.append({
                    "Requirement_1": req1,
                    "Requirement_2": req2,
                    "Conflict_Type": conflict_type,
                    "Conflict_Reason": conflict_reason
                })
            checked_pairs.add(pair_key)
            time.sleep(1)

    output_results = results
    output_df = pd.DataFrame(output_results) if output_results else pd.DataFrame(columns=["Requirement_1", "Requirement_2", "Conflict_Type", "Conflict_Reason"])

    csv_dir, xlsx_dir = ensure_directories()
    suffix = f"_{int(time.time())}" if new_requirement else ""
    csv_output = os.path.join(csv_dir, f"results{suffix}.csv")
    xlsx_output = os.path.join(xlsx_dir, f"results{suffix}.xlsx")
    
    output_df.to_csv(csv_output, index=False)
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    logger.info(f"Analysis complete. Results saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")
    return output_df

def get_csv_files(directory="/workspaces/PC-user-Task3"):
    """List all CSV files in the specified directory"""
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def display_menu():
    """Display the menu and handle user input"""
    conflict_type_weights = None

    while True:
        print("\n=== Requirements Conflict Detection Menu ===")
        print("1. Select a file for prediction (baseline mode)")
        print("2. Select a file for prediction (exhaustive mode)")
        print("3. Enter a new requirement")
        print("4. Load a file for training")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()

        if choice in ["1", "2"]:
            exhaustive = (choice == "2")
            csv_files = get_csv_files()
            if not csv_files:
                logger.error("No CSV files found in the directory.")
                continue
            
            print(f"\nAvailable CSV files for prediction ({'exhaustive' if exhaustive else 'baseline'} mode):")
            for i, file in enumerate(csv_files, 1):
                print(f"{i}. {file}")
            
            file_choice = input("Select a file by number (or 'back' to return): ").strip()
            if file_choice.lower() == 'back':
                continue
            try:
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(csv_files):
                    input_file = os.path.join("/workspaces/PC-user-Task3", csv_files[file_idx])
                    predict_conflicts(input_file, conflict_type_weights=conflict_type_weights, exhaustive=exhaustive)
                else:
                    logger.error("Invalid file number.")
            except ValueError:
                logger.error("Please enter a valid number.")

        elif choice == "3":
            new_requirement = input("Enter a new requirement to analyze (or 'back' to return): ").strip()
            if new_requirement.lower() == 'back':
                continue
            if not new_requirement:
                logger.error("Requirement cannot be empty.")
                continue
            
            csv_files = get_csv_files()
            if not csv_files:
                logger.error("No CSV files found to compare against.")
                continue
            
            print("\nSelect a file to compare the new requirement against:")
            for i, file in enumerate(csv_files, 1):
                print(f"{i}. {file}")
            
            file_choice = input("Select a file by number (or 'back' to return): ").strip()
            if file_choice.lower() == 'back':
                continue
            try:
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(csv_files):
                    input_file = os.path.join("/workspaces/PC-user-Task3", csv_files[file_idx])
                    predict_conflicts(input_file, new_requirement=new_requirement, conflict_type_weights=conflict_type_weights)
                else:
                    logger.error("Invalid file number.")
            except ValueError:
                logger.error("Please enter a valid number.")

        elif choice == "4":
            csv_files = get_csv_files()
            if not csv_files:
                logger.error("No CSV files found in the directory.")
                continue
            
            print("\nAvailable CSV files for training:")
            for i, file in enumerate(csv_files, 1):
                print(f"{i}. {file}")
            
            file_choice = input("Select a file by number (or 'back' to return): ").strip()
            if file_choice.lower() == 'back':
                continue
            try:
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(csv_files):
                    test_file = os.path.join("/workspaces/PC-user-Task3", csv_files[file_idx])
                    iterations = input("Enter number of training iterations (default is 2): ").strip()
                    iterations = int(iterations) if iterations.isdigit() else 2
                    _, conflict_type_weights = api_pseudo_train(test_file, iterations)
                    logger.info("Training completed. Conflict type weights updated.")
                else:
                    logger.error("Invalid file number.")
            except ValueError:
                logger.error("Please enter a valid number.")

        elif choice == "5":
            logger.info("Exiting the program.")
            sys.exit(0)

        else:
            logger.error("Invalid choice. Please enter a number between 1 and 5.")

def main():
    try:
        display_menu()
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
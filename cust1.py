import os
import argparse
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
    level=logging.DEBUG,  # Changed to DEBUG as per your latest code
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conflict_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API setup for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Your Gemini API key
GEMINI_MODEL = "gemini-2.0-flash"  # Model with 15 RPM, 1M TPM, 1500 RPD
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Global predefined conflict types
PREDEFINED_CONFLICTS = {
    "Performance Conflict", "Compliance Conflict", "Safety Conflict", "Cost Conflict", "Battery Conflict",
    "Environmental Conflict", "Structural Conflict", "Comfort Conflict", "Power Source Conflict", "Reliability Conflict",
    "Scalability Conflict", "Security Conflict", "Usability Conflict", "Maintenance Conflict", "Weight Conflict",
    "Time-to-Market Conflict", "Compatibility Conflict", "Aesthetic Conflict", "Noise Conflict", "Other Conflict",
    "Sustainability Conflict", "Regulatory Conflict", "Resource Conflict", "Technology Conflict", "Design Conflict", "Contradiction",
    "No Conflict"  # Added "No Conflict" to the predefined conflicts
}

def call_inference_api(prompt, api_key=GEMINI_API_KEY, api_url=GEMINI_API_URL):
    """Call Gemini API for inference with a delay respecting 15 RPM"""
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        logger.debug(f"Sending request to {api_url} with prompt: {prompt[:100]}...")
        response = requests.post(f"{api_url}?key={api_key}", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"Raw API response: {json.dumps(result, indent=2)}")
        
        if "candidates" in result and len(result["candidates"]) > 0:
            output = result["candidates"][0]["content"]["parts"][0]["text"]
            time.sleep(4)  # 4-second delay to respect 15 RPM
            return output.strip()
        elif "error" in result:
            logger.error(f"API error message: {result['error']}")
            return f"Inference failed: {result['error']}"
        else:
            logger.error(f"Unexpected response format: {result}")
            return "Inference failed: Unexpected response format"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return f"Inference failed: Request error - {str(e)}"

@lru_cache(maxsize=1000)
def cached_call_inference_api(prompt):
    """Cached version of API call"""
    result = call_inference_api(prompt)
    return result.strip()

def parse_api_output(full_output):
    """Parse API output expecting 'Conflict_Type: <type>||Reason: <reason>'"""
    if not full_output or "Inference failed" in full_output:
        logger.warning(f"Blank or failed output: {full_output}")
        return "Other", full_output or "No response from API", "Not applicable"
    
    full_output = full_output.strip()
    logger.debug(f"Parsing API output: '{full_output}'")
    
    # Split by "||" as per the required format
    parts = [p.strip() for p in full_output.split("||") if p.strip()]
    
    if len(parts) >= 2:
        # Extract conflict type and reason
        conflict_type_part = parts[0].replace("Conflict_Type: ", "").strip()
        conflict_reason = parts[1].replace("Reason: ", "").strip()
        
        if conflict_type_part in PREDEFINED_CONFLICTS:
            conflict_type = conflict_type_part
            logger.debug(f"Parsed - Type: '{conflict_type}', Reason: '{conflict_reason}'")
        else:
            logger.warning(f"Unexpected conflict type '{conflict_type_part}' not in predefined list, defaulting to 'Other'")
            conflict_type = "Other"
            conflict_reason = full_output
    else:
        logger.warning(f"Malformed output, expected 'Conflict_Type: <type>||Reason: <reason>': '{full_output}'")
        # Fallback to check if it's a simple "<type>: <reason>" format
        colon_parts = [p.strip() for p in full_output.split(":", 1) if p.strip()]
        if len(colon_parts) == 2 and colon_parts[0] in PREDEFINED_CONFLICTS:
            conflict_type = colon_parts[0]
            conflict_reason = colon_parts[1]
            logger.debug(f"Parsed fallback - Type: '{conflict_type}', Reason: '{conflict_reason}'")
        else:
            conflict_type = "Other"
            conflict_reason = full_output or "Malformed or incomplete output"
    
    resolution = "Not applicable"
    return conflict_type, conflict_reason, resolution

def ensure_directories():
    """Create Results directory with CSV and XLSX subdirectories if they don't exist"""
    base_dir = "Results"
    csv_dir = os.path.join(base_dir, "CSV")
    xlsx_dir = os.path.join(base_dir, "XLSX")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(xlsx_dir, exist_ok=True)
    return csv_dir, xlsx_dir

def api_pseudo_train(args):
    """Use Gemini API to iteratively 'train' on the dataset"""
    if not os.path.exists(args.test_file):
        logger.error(f"Input file {args.test_file} not found. Exiting.")
        sys.exit(1)

    try:
        original_df = pd.read_csv(args.test_file)
        if "Requirement_1" not in original_df.columns or "Requirement_2" not in original_df.columns:
            logger.error("Input file must contain 'Requirement_1' and 'Requirement_2' columns")
            sys.exit(1)
        
        # Check if Conflict_Type exists, if not, create it with default value
        if "Conflict_Type" not in original_df.columns:
            logger.warning("'Conflict_Type' column not found, creating with default value 'Other'")
            original_df["Conflict_Type"] = "Other"
            
        # Make sure to have the Expected_Conflict column for accuracy calculation
        if "Expected_Conflict" not in original_df.columns:
            logger.info("Creating 'Expected_Conflict' column from 'Conflict_Type'")
            original_df["Expected_Conflict"] = original_df["Conflict_Type"]
    except Exception as e:
        logger.error(f"Error reading test file: {e}")
        sys.exit(1)
    
    logger.info(f"Starting pseudo-training with {len(original_df)} examples using {GEMINI_MODEL} via Gemini API")
    
    prompt_template = (
        "Analyze the provided {req1} and {req2} from the uploaded CSV or Excel file. Identify any conflicts between {req1} and {req2}. You MUST choose the conflict type ONLY from the following list: {conflict_types}. For each pair, determine if there is a conflict, and if so, specify the type of conflict and the reason (as a one-line sentence). The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is exactly one of {conflict_types}, <reason> is a one-line explanation. If no conflict exists, output: \"Conflict_Type: No Conflict||Reason: Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )

    results = []
    iteration = 0
    max_iterations = args.iterations
    conflict_type_weights = {conflict: 1.0 for conflict in PREDEFINED_CONFLICTS}

    # Use original_df to preserve Expected_Conflict across iterations
    df_train = original_df.copy()

    for _ in range(max_iterations):
        iteration += 1
        logger.info(f"Pseudo-training iteration {iteration}/{max_iterations}")
        
        prompt_template_iter = prompt_template
        if iteration > 1 and results:
            # Make sure all results have Expected_Conflict key before calculating accuracy
            for r in results:
                if "Expected_Conflict" not in r:
                    # Find the corresponding row in original_df to get Expected_Conflict
                    matching_row = original_df[(original_df["Requirement_1"] == r["Requirement_1"]) & 
                                             (original_df["Requirement_2"] == r["Requirement_2"])]
                    if not matching_row.empty:
                        r["Expected_Conflict"] = matching_row.iloc[0]["Expected_Conflict"]
                    else:
                        r["Expected_Conflict"] = "Other"  # Default if not found
            
            # Now calculate accuracy
            correct = sum(1 for r in results if r["Conflict_Type"] == r["Expected_Conflict"])
            accuracy = correct / len(results) if results else 0
            logger.info(f"Accuracy before iteration {iteration}: {accuracy:.2%}")
            
            if accuracy < 0.8:
                for r in results:
                    if r["Conflict_Type"] != r["Expected_Conflict"]:
                        conflict_type_weights[r["Expected_Conflict"]] += 0.1
                    else:
                        conflict_type_weights[r["Conflict_Type"]] += 0.05
                weighted_conflicts = ', '.join([f"{k} (weight: {v:.2f})" for k, v in sorted(conflict_type_weights.items(), key=lambda x: x[1], reverse=True)])
                prompt_template_iter += f" Prioritize conflict types based on these weights: {weighted_conflicts}."
        
        results.clear()

        for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Iteration {iteration}"):
            req1, req2 = row["Requirement_1"], row["Requirement_2"]
            expected_conflict = row["Expected_Conflict"] if "Expected_Conflict" in row else row["Conflict_Type"]
            
            input_text = prompt_template_iter.format(req1=req1, req2=req2, conflict_types=', '.join(sorted(PREDEFINED_CONFLICTS)))
            
            full_output = cached_call_inference_api(input_text)
            conflict_type, conflict_reason, _ = parse_api_output(full_output)
            
            if conflict_type not in PREDEFINED_CONFLICTS:
                logger.warning(f"Unexpected conflict type '{conflict_type}' detected, defaulting to 'Other'")
                conflict_type = "Other"
            
            results.append({
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason,
                "Expected_Conflict": expected_conflict
            })
        
        # Now calculate accuracy for this iteration
        correct = sum(1 for r in results if r["Conflict_Type"] == r["Expected_Conflict"])
        logger.info(f"Iteration {iteration} accuracy: {correct / len(results):.2%}")
        
        df_train = pd.DataFrame(results)

    # Process results for output, excluding Expected_Conflict
    output_results = [{k: v for k, v in r.items() if k != "Expected_Conflict"} for r in results]
    output_df = pd.DataFrame(output_results)

    csv_dir, xlsx_dir = ensure_directories()
    csv_output = os.path.join(csv_dir, "training_results.csv")
    xlsx_output = os.path.join(xlsx_dir, "training_results.xlsx")
    
    # Save without Expected_Conflict
    output_df.to_csv(csv_output, index=False)
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    logger.info(f"Pseudo-training complete after {max_iterations} iterations. Results saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")
    return output_df, conflict_type_weights

def check_new_requirement(new_req, all_existing_requirements, predefined_conflicts, checked_pairs=None, conflict_type_weights=None):
    if checked_pairs is None:
        checked_pairs = set()

    results = []
    prompt_template = (
        "Analyze the provided {req1} and {req2} from the uploaded CSV or Excel file. Identify any conflicts between {req1} and {req2}. You MUST choose the conflict type ONLY from the following list: {conflict_types}. For each pair, determine if there is a conflict, and if so, specify the type of conflict and the reason (as a one-line sentence). The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is exactly one of {conflict_types}, <reason> is a one-line explanation. If no conflict exists, output: \"Conflict_Type: No Conflict||Reason: Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )
    if conflict_type_weights:
        weighted_conflicts = ', '.join([f"{k} (weight: {v:.2f})" for k, v in sorted(conflict_type_weights.items(), key=lambda x: x[1], reverse=True)])
        prompt_template += f" Prioritize conflict types based on these weights: {weighted_conflicts}."

    for existing_req in all_existing_requirements:
        pair_key = f"{new_req}||{existing_req}"
        if pair_key in checked_pairs:
            continue

        pairs_str = f"Requirement 1: \"{new_req}\" - Requirement 2: \"{existing_req}\""
        input_text = prompt_template.format(req1=new_req, req2=existing_req, conflict_types=', '.join(sorted(predefined_conflicts)))

        full_output = cached_call_inference_api(input_text)
        conflict_type, conflict_reason, _ = parse_api_output(full_output)

        if conflict_type not in predefined_conflicts:
            logger.warning(f"Unexpected conflict type '{conflict_type}' detected, defaulting to 'Other'")
            conflict_type = "Other"

        if conflict_type != "No Conflict":
            results.append({
                "Requirement_1": new_req,
                "Requirement_2": existing_req,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason
            })
        checked_pairs.add(pair_key)

    return pd.DataFrame(results) if results else None

def predict_conflicts(args, conflict_type_weights=None):
    if not os.path.exists(args.input_file):
        logger.error(f"Input file {args.input_file} not found. Exiting.")
        sys.exit(1)

    try:
        df_input = pd.read_csv(args.input_file, encoding='utf-8')
        if "Requirements" not in df_input.columns:
            logger.error("Input file must contain a 'Requirements' column")
            sys.exit(1)
        logger.info(f"Loaded {len(df_input)} requirements from {args.input_file}")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        sys.exit(1)

    all_original_requirements = df_input["Requirements"].tolist()

    prompt_template_single = (
        "Analyze the provided {req1} and {req2} from the uploaded CSV or Excel file. Identify any conflicts between {req1} and {req2}. You MUST choose the conflict type ONLY from the following list: {conflict_types}. For each pair, determine if there is a conflict, and if so, specify the type of conflict and the reason (as a one-line sentence). The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is exactly one of {conflict_types}, <reason> is a one-line explanation. If no conflict exists, output: \"Conflict_Type: No Conflict||Reason: Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )
    if conflict_type_weights:
        weighted_conflicts = ', '.join([f"{k} (weight: {v:.2f})" for k, v in sorted(conflict_type_weights.items(), key=lambda x: x[1], reverse=True)])
        prompt_template_single += f" Prioritize conflict types based on these weights: {weighted_conflicts}."

    results = []
    checked_pairs = set()

    for req1, req2 in tqdm(list(itertools.combinations(all_original_requirements, 2)), desc="Analyzing conflicts"):
        pair_key = f"{req1}||{req2}"
        if pair_key in checked_pairs:
            continue

        input_text = prompt_template_single.format(req1=req1, req2=req2, conflict_types=', '.join(sorted(PREDEFINED_CONFLICTS)))
        full_output = cached_call_inference_api(input_text)
        conflict_type, conflict_reason, _ = parse_api_output(full_output)

        if conflict_type not in PREDEFINED_CONFLICTS:
            logger.warning(f"Unexpected conflict type '{conflict_type}' detected, defaulting to 'Other'")
            conflict_type = "Other"

        if conflict_type != "No Conflict":
            results.append({
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason
            })
        checked_pairs.add(pair_key)

    # Process results for output, excluding Expected_Conflict
    output_results = [{k: v for k, v in r.items() if k != "Expected_Conflict"} for r in results]
    output_df = pd.DataFrame(output_results) if output_results else pd.DataFrame(columns=["Requirement_1", "Requirement_2", "Conflict_Type", "Conflict_Reason"])

    csv_dir, xlsx_dir = ensure_directories()
    csv_output = os.path.join(csv_dir, "results.csv")
    xlsx_output = os.path.join(xlsx_dir, "results.xlsx")
    
    # Save without Expected_Conflict
    output_df.to_csv(csv_output, index=False)
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    logger.info(f"Initial analysis complete. Results saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")

    while True:
        try:
            new_requirement = inputimeout(prompt="New Requirement: ", timeout=30).strip()
        except TimeoutOccurred:
            logger.info("No input received within 30 seconds. Exiting new requirement input phase.")
            break

        if new_requirement.lower() == 'exit':
            break

        new_results = check_new_requirement(new_requirement, all_original_requirements, PREDEFINED_CONFLICTS, checked_pairs, conflict_type_weights)

        if new_results is not None and not new_results.empty:
            # Exclude Expected_Conflict from new_results before saving (though check_new_requirement doesn't include it)
            new_output_results = new_results.drop(columns=["Expected_Conflict"] if "Expected_Conflict" in new_results.columns else [], errors='ignore')
            new_csv_output = os.path.join(csv_dir, f"new_results_{int(time.time())}.csv")
            new_xlsx_output = os.path.join(xlsx_dir, f"new_results_{int(time.time())}.xlsx")
            new_output_results.to_csv(new_csv_output, index=False)
            new_output_results.to_excel(new_xlsx_output, index=False, engine='openpyxl')
            logger.info(f"New conflicts found and saved to {new_csv_output} (CSV) and {new_xlsx_output} (XLSX)")
            print(new_output_results)
        else:
            logger.info("No conflicts found with existing requirements.")

    return output_df

def main():
    parser = argparse.ArgumentParser(description="Requirements Conflict Detection with Gemini API")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "both"], default="train")
    parser.add_argument("--input_file", type=str, default="/workspaces/PC-user-Task3/InputData.csv")  # from user input file
    parser.add_argument("--test_file", type=str, default="/workspaces/PC-user-Task3/reduced_requirements.csv")  # for training the pseudo code
    parser.add_argument("--output_file", type=str, default="results.csv") # for default output file
    parser.add_argument("--iterations", type=int, default=2, help="Number of pseudo-training iterations")
    
    args = parser.parse_args()
    
    conflict_type_weights = None
    if args.mode in ["train", "both"]:
        _, conflict_type_weights = api_pseudo_train(args)
        logger.info("Gemini API pseudo-training completed!")
    
    if args.mode in ["predict", "both"]:
        predict_conflicts(args, conflict_type_weights)
        logger.info("Prediction and new requirement checking completed!")

if __name__ == "__main__":
    main()
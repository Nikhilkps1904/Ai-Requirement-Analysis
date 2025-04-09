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
import random
import sys
from functools import lru_cache
from inputimeout import inputimeout, TimeoutOccurred  # Requires `pip install inputimeout`

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
    ]
)
logger = logging.getLogger(__name__)

# API setup for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY1")  # Your Gemini API key
GEMINI_MODEL = "gemini-2.0-flash"  # Model with 15 RPM, 1M TPM, 1500 RPD
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

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
    """Cached version of API call to reduce redundant calls"""
    return call_inference_api(prompt)

def load_sample_data():
    """Load initial sample training data"""
    train_data = [
        ("The vehicle must achieve a fuel efficiency of at least 50 km/l.",
         "The engine should have a minimum power output of 25 HP.",
         "Performance Conflict"),
        ("The bike should include an always-on headlight for safety compliance.",
         "Users should be able to turn off the headlight manually.",
         "Compliance Conflict"),
        ("The car's doors should automatically lock when the vehicle is in motion.",
         "Passengers must be able to open the doors at any time.",
         "Safety Conflict"),
    ]
    return pd.DataFrame(train_data, columns=["Requirement_1", "Requirement_2", "Conflict_Type"])

def parse_api_output(full_output):
    """Robustly parse API output, expecting two parts"""
    if not full_output or "Inference failed" in full_output:
        logger.warning(f"Blank or failed output: {full_output}")
        return "Other", full_output or "No response from API", "Not applicable"
    
    parts = [p.strip() for p in full_output.split("||") if p.strip()]
    if len(parts) < 2:
        logger.warning(f"Malformed output: {full_output}")
        conflict_type = parts[0] if parts else "Other"
        conflict_reason = "Malformed or incomplete output"
    else:
        conflict_type = parts[0].replace("Conflict_Type: ", "").strip()
        conflict_reason = parts[1].replace("Reason: ", "").strip()
    
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
    """Use Gemini API to iteratively 'train' on the dataset without handling new requirements"""
    if os.path.exists(args.test_file):
        df_train = pd.read_csv(args.test_file)
    else:
        logger.warning(f"Input file {args.test_file} not found. Using sample data.")
        df_train = load_sample_data()
    
    logger.info(f"Starting pseudo-training with {len(df_train)} examples using {GEMINI_MODEL} via Gemini API")
    
    PREDEFINED_CONFLICTS = {
        "Performance Conflict", "Compliance Conflict", "Safety Conflict", "Cost Conflict", "Battery Conflict",
        "Environmental Conflict", "Structural Conflict", "Comfort Conflict", "Power Source Conflict", "Reliability Conflict",
        "Scalability Conflict", "Security Conflict", "Usability Conflict", "Maintenance Conflict", "Weight Conflict",
        "Time-to-Market Conflict", "Compatibility Conflict", "Aesthetic Conflict", "Noise Conflict", "Other Conflict",
        "Sustainability Conflict", "Regulatory Conflict", "Resource Conflict", "Technology Conflict", "Design Conflict", "Contradiction"
    }

    # Base prompt template with all placeholders
    base_prompt_template = (
        "Analyze the provided {req1} and {req2} from the uploaded CSV or Excel file. Identify any conflicts between {req1} and {req2}. You MUST choose the conflict type ONLY from the following list: {conflict_types}. For each pair, determine if there is a conflict, and if so, specify the type of conflict and the reason (as a one-line sentence). The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is exactly one of {conflict_types}, <reason> is a one-line explanation. If no conflict exists, output: \"No Conflict||Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )

    results = []
    iteration = 0
    max_iterations = args.iterations
    conflict_type_weights = {conflict: 1.0 for conflict in PREDEFINED_CONFLICTS}  # Initial weights

    for _ in range(max_iterations):
        iteration += 1
        logger.info(f"Pseudo-training iteration {iteration}/{max_iterations}")
        
        # Adjust prompt based on previous iteration's accuracy
        prompt_template = base_prompt_template
        if iteration > 1 and results:
            correct = sum(1 for r in results if r["Conflict_Type"] == r["Expected_Conflict"])
            accuracy = correct / len(results)
            if accuracy < 0.8:  # If accuracy is low, adjust weights
                for r in results:
                    if r["Conflict_Type"] != r["Expected_Conflict"]:
                        conflict_type_weights[r["Expected_Conflict"]] += 0.1  # Boost expected type
                    else:
                        conflict_type_weights[r["Conflict_Type"]] += 0.05  # Reinforce correct type
                weighted_conflicts = ', '.join([f"{k} (weight: {v:.2f})" for k, v in sorted(conflict_type_weights.items(), key=lambda x: x[1], reverse=True)])
                prompt_template += f" Prioritize conflict types based on these weights: {weighted_conflicts}."
        
        results.clear()  # Reset results for new iteration
        
        for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Iteration {iteration}"):
            req1 = row["Requirement_1"]
            req2 = row["Requirement_2"]
            expected_conflict = row["Conflict_Type"]
            
            # Format prompt with all required placeholders inside the loop
            input_text = prompt_template.format(req1=req1, req2=req2, conflict_types=', '.join(sorted(PREDEFINED_CONFLICTS)))
            
            full_output = cached_call_inference_api(input_text)
            conflict_type, conflict_reason, _ = parse_api_output(full_output)
            
            if conflict_type not in PREDEFINED_CONFLICTS and conflict_type != "No Conflict":
                conflict_type = "Other"
                logger.warning(f"Unexpected conflict type '{conflict_type}' detected, defaulting to 'Other'")
            
            results.append({
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason,
                "Expected_Conflict": expected_conflict
            })
        
        correct = sum(1 for r in results if r["Conflict_Type"] == r["Expected_Conflict"])
        logger.info(f"Iteration {iteration} accuracy: {correct / len(results):.2%}")
        
        df_train = pd.DataFrame(results)

    output_df = pd.DataFrame(df_train)
    
    csv_dir, xlsx_dir = ensure_directories()
    csv_output = os.path.join(csv_dir, "training_results.csv")
    output_df.to_csv(csv_output, index=False)
    xlsx_output = os.path.join(xlsx_dir, "training_results.xlsx")
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    logger.info(f"Pseudo-training complete after {max_iterations} iterations. Results saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")
    return output_df, conflict_type_weights

def check_new_requirement(new_req, all_existing_requirements, predefined_conflicts, checked_pairs=None, conflict_type_weights=None):
    """Efficiently check if a new requirement conflicts with all existing requirements."""
    results = []
    prompt_template = (
        "Analyze the following pairs of requirements to identify any conflicts: {pairs}. You MUST choose the conflict type ONLY from: {conflict_types}. For each pair, output: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is one of {conflict_types}, <reason> is a one-line explanation. If no conflict exists, output: \"No Conflict||Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )
    if conflict_type_weights:
        weighted_conflicts = ', '.join([f"{k} (weight: {v:.2f})" for k, v in sorted(conflict_type_weights.items(), key=lambda x: x[1], reverse=True)])
        prompt_template += f" Prioritize conflict types based on these weights: {weighted_conflicts}."

    if checked_pairs is None:
        checked_pairs = set()

    existing_reqs = set(all_existing_requirements)
    new_pairs = [(new_req, existing_req) for existing_req in existing_reqs if (new_req, existing_req) not in checked_pairs]
    if not new_pairs:
        return pd.DataFrame(results)

    pairs_str = "\n- ".join([f"Requirement 1: \"{new_req}\" - Requirement 2: \"{existing_req}\"" for new_req, existing_req in new_pairs])
    input_text = prompt_template.format(conflict_types=', '.join(sorted(predefined_conflicts)), pairs=pairs_str)
    
    full_output = cached_call_inference_api(input_text)
    output_lines = [line.strip() for line in full_output.split("\n") if line.strip()]
    for line in output_lines:
        conflict_type, conflict_reason, _ = parse_api_output(line)
        if conflict_type not in predefined_conflicts and conflict_type != "No Conflict":
            conflict_type = "Other"
            logger.warning(f"Unexpected conflict type '{conflict_type}' detected, defaulting to 'Other'")
        
        if conflict_type != "No Conflict":
            for new_r, existing_r in new_pairs:
                results.append({
                    "Requirement_1": new_r,
                    "Requirement_2": existing_r,
                    "Conflict_Type": conflict_type,
                    "Conflict_Reason": conflict_reason
                })
                checked_pairs.add((new_r, existing_r))

    return pd.DataFrame(results) if results else None

def predict_conflicts(args, conflict_type_weights=None):
    """Predict conflicts and allow new requirement input with timeout."""
    try:
        df_input = pd.read_csv(args.test_file, encoding='utf-8')
        if "Requirements" not in df_input.columns:
            logger.error("Input file must contain a 'Requirements' column")
            return
        logger.info(f"Loaded {len(df_input)} requirements from {args.test_file}")
    except Exception as e:
        logger.error(f"Error reading test file: {e}")
        return

    PREDEFINED_CONFLICTS = {
        "Performance Conflict", "Compliance Conflict", "Safety Conflict", "Cost Conflict", "Battery Conflict",
        "Environmental Conflict", "Structural Conflict", "Comfort Conflict", "Power Source Conflict", "Reliability Conflict",
        "Scalability Conflict", "Security Conflict", "Usability Conflict", "Maintenance Conflict", "Weight Conflict",
        "Time-to-Market Conflict", "Compatibility Conflict", "Aesthetic Conflict", "Noise Conflict", "Other Conflict",
        "Sustainability Conflict", "Regulatory Conflict", "Resource Conflict", "Technology Conflict", "Design Conflict", "Contradiction"
    }

    all_original_requirements = df_input["Requirements"].tolist()

    prompt_template_single = (
        "Analyze the provided {req1} and {req2} to identify any conflicts based on: {conflict_types}. You MUST choose the conflict type ONLY from: {conflict_types}. Input: - Requirement 1: \"{req1}\" - Requirement 2: \"{req2}\". Output: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is one of {conflict_types}, <reason> is a one-line explanation. If no conflict exists, output: \"No Conflict||Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )
    if conflict_type_weights:
        weighted_conflicts = ', '.join([f"{k} (weight: {v:.2f})" for k, v in sorted(conflict_type_weights.items(), key=lambda x: x[1], reverse=True)])
        prompt_template_single += f" Prioritize conflict types based on these weights: {weighted_conflicts}."

    results = []
    checked_pairs = set()

    all_pairs = list(itertools.combinations(all_original_requirements, 2))
    logger.info(f"Generated {len(all_pairs)} unique pairwise comparisons for analysis")
    
    random.shuffle(all_pairs)
    logger.info("Shuffled all possible combinations for random order processing")

    for req1, req2 in tqdm(all_pairs, desc="Analyzing conflicts"):
        if (req1, req2) not in checked_pairs:
            input_text = prompt_template_single.format(req1=req1, req2=req2, conflict_types=', '.join(sorted(PREDEFINED_CONFLICTS)))
            full_output = cached_call_inference_api(input_text)
            conflict_type, conflict_reason, _ = parse_api_output(full_output)
            
            if conflict_type not in PREDEFINED_CONFLICTS and conflict_type != "No Conflict":
                conflict_type = "Other"
                logger.warning(f"Unexpected conflict type '{conflict_type}' detected, defaulting to 'Other'")
            
            results.append({
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason
            })
            checked_pairs.add((req1, req2))

    output_df = pd.DataFrame(results)
    
    csv_dir, xlsx_dir = ensure_directories()
    csv_output = os.path.join(csv_dir, "results.csv")
    output_df.to_csv(csv_output, index=False)
    xlsx_output = os.path.join(xlsx_dir, "results.xlsx")
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    logger.info(f"Initial analysis complete. Results saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")

    logger.info("Prediction completed. Enter a new requirement within 30 seconds (or type 'exit' to finish):")
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
            new_csv_output = os.path.join(csv_dir, f"new_results_{int(time.time())}.csv")
            new_xlsx_output = os.path.join(xlsx_dir, f"new_results_{int(time.time())}.xlsx")
            new_results.to_csv(new_csv_output, index=False)
            new_results.to_excel(new_xlsx_output, index=False, engine='openpyxl')
            logger.info(f"New conflicts found and saved to {new_csv_output} (CSV) and {new_xlsx_output} (XLSX)")
            print(new_results)
        else:
            logger.info("No conflicts found with existing requirements.")

        logger.info("Enter another new requirement within 30 seconds (or type 'exit' to finish):")

    return output_df

def main():
    parser = argparse.ArgumentParser(description="Requirements Conflict Detection with Gemini API")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "both"], default="train")
    parser.add_argument("--input_file", type=str, default="reduced_requirements.csv")
    parser.add_argument("--test_file", type=str, default="/workspaces/PC-user-Task3/Test_data/data.csv")
    parser.add_argument("--output_file", type=str, default="results.csv")
    parser.add_argument("--iterations", type=int, default=3, help="Number of pseudo-training iterations")
    
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
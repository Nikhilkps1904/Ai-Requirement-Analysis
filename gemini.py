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
import shutil

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Your Gemini API key
GEMINI_MODEL = "gemini-2.0-flash"  # Model with 15 RPM, 1M TPM, 1500 RPD
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

def call_inference_api(prompt, api_key=GEMINI_API_KEY, api_url=GEMINI_API_URL):
    """Call Gemini API for inference with a delay respecting 15 RPM"""
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    try:
        logger.debug(f"Sending request to {api_url} with prompt: {prompt[:100]}...")
        response = requests.post(f"{api_url}?key={api_key}", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"Raw API response: {json.dumps(result, indent=2)}")
        
        # Gemini response structure: extract the generated text
        if "candidates" in result and len(result["candidates"]) > 0:
            output = result["candidates"][0]["content"]["parts"][0]["text"]
            time.sleep(4)  # 4-second delay to respect 15 RPM (60/15 = 4)
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
        conflict_type = parts[0].replace("Conflict_Type: ", "")
        conflict_reason = parts[1].replace("Reason: ", "")
    
    # Resolution is not expected
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
    if os.path.exists(args.input_file):
        df_train = pd.read_csv(args.input_file)
    else:
        logger.warning(f"Input file {args.input_file} not found. Using sample data.")
        df_train = load_sample_data()
    
    logger.info(f"Starting pseudo-training with {len(df_train)} examples using {GEMINI_MODEL} via Gemini API")
    
    PREDEFINED_CONFLICTS = {
        "Performance Conflict", "Compliance Conflict", "Safety Conflict", "Cost Conflict", "Battery Conflict",
        "Environmental Conflict", "Structural Conflict", "Comfort Conflict", "Power Source Conflict", "Reliability Conflict",
        "Scalability Conflict", "Security Conflict", "Usability Conflict", "Maintenance Conflict", "Weight Conflict",
        "Time-to-Market Conflict", "Compatibility Conflict", "Aesthetic Conflict", "Noise Conflict", "Other Conflict",
        "Sustainability Conflict", "Regulatory Conflict", "Resource Conflict", "Technology Conflict", "Design Conflict", "Contradiction"
    }

    prompt_template = (
        "Analyze the provided {req1} and {req2} from the uploaded CSV or Excel file, which must contain a single column labeled 'Requirements.' Identify any conflicts between {req1} and {req2}. For each pair, determine if there is a conflict, and if so, specify the type of conflict (including all possible types) and the reason for the conflict (as a one-line sentence). The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is one of {conflict_types}, <reason> is a one-line explanation. If no conflict exists, output: \"No Conflict||Requirements are compatible\". Ensure the output is concise and follows the exact format specified, using angle brackets < > to indicate placeholders."
    )

    results = []
    iteration = 0
    max_iterations = args.iterations
    
    for _ in range(max_iterations):
        iteration += 1
        logger.info(f"Pseudo-training iteration {iteration}/{max_iterations}")
        
        for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Iteration {iteration}"):
            req1 = row["Requirement_1"]
            req2 = row["Requirement_2"]
            expected_conflict = row["Conflict_Type"]
            
            input_text = prompt_template.format(req1=req1, req2=req2, conflict_types=', '.join(PREDEFINED_CONFLICTS))
            
            full_output = call_inference_api(input_text)
            conflict_type, conflict_reason, resolution = parse_api_output(full_output)
            
            if conflict_type not in PREDEFINED_CONFLICTS and conflict_type != "No Conflict":
                conflict_type = "Other"
            
            results.append({
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason,
                "Resolution_Suggestion": resolution,  # Will be "Not applicable"
                "Expected_Conflict": expected_conflict
            })
        
        correct = sum(1 for r in results if r["Conflict_Type"] == r["Expected_Conflict"])
        logger.info(f"Iteration {iteration} accuracy: {correct / len(results):.2%}")
        
        df_train = pd.DataFrame(results)
        df_train["Conflict_Type"] = df_train["Conflict_Type"]
        results = []
    
    output_df = pd.DataFrame(df_train)
    
    csv_dir, xlsx_dir = ensure_directories()
    
    csv_output = os.path.join(csv_dir, "results.csv")
    output_df.to_csv(csv_output, index=False)
    
    xlsx_output = os.path.join(xlsx_dir, "results.xlsx")
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    logger.info(f"Pseudo-training complete after {max_iterations} iterations. Results saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")
    return output_df

def predict_conflicts(args):
    """Predict conflicts for a single-column requirements file by analyzing pairwise combinations"""
    try:
        df_input = pd.read_csv(args.input_file, encoding='utf-8')
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

    prompt_template = (
        "Analyze the provided {req1} and {req2} from the uploaded CSV or Excel file, which must contain a single column labeled 'Requirements,' to identify any conflicts between them based on the following conflict types: {conflict_types}. Input: - Requirement 1: \"{req1}\" - Requirement 2: \"{req2}\". Task: 1. Determine if there is a conflict using vehicle engineering principles and provide a one-line expert explanation. 2. If a conflict exists, output: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is one of the conflict types, <reason> is a one-line explanation. 3. If no conflict exists, output: \"No Conflict||Requirements are compatible\". 4. Ensure the output is concise and follows the exact format specified, using angle brackets < > to indicate placeholders."
    )

    requirements = df_input["Requirements"].tolist()
    results = []

    pairs = list(itertools.combinations(requirements, 2))
    logger.info(f"Generated {len(pairs)} unique pairwise combinations for analysis")

    for req1, req2 in tqdm(pairs, desc="Analyzing conflicts"):
        input_text = prompt_template.format(req1=req1, req2=req2, conflict_types=', '.join(PREDEFINED_CONFLICTS))
        
        full_output = call_inference_api(input_text)
        conflict_type, conflict_reason, resolution = parse_api_output(full_output)
        
        if conflict_type not in PREDEFINED_CONFLICTS and conflict_type != "No Conflict":
            conflict_type = "Other"
        
        results.append({
            "Requirement_1": req1,
            "Requirement_2": req2,
            "Conflict_Type": conflict_type,
            "Conflict_Reason": conflict_reason,
            "Resolution_Suggestion": resolution  # Will be "Not applicable"
        })
    
    output_df = pd.DataFrame(results)
    
    csv_dir, xlsx_dir = ensure_directories()
    
    csv_output = os.path.join(csv_dir, "results.csv")
    output_df.to_csv(csv_output, index=False)
    
    xlsx_output = os.path.join(xlsx_dir, "results.xlsx")
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    logger.info(f"Analysis complete. Results saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Requirements Conflict Detection with Gemini API")
    
    parser.add_argument("--mode", type=str, choices=["train", "predict", "both"], default="train")
    parser.add_argument("--input_file", type=str, default="/workspaces/PC-user-Task3/Test_data/data.csv")
    parser.add_argument("--test_file", type=str, default="./test_data.csv")
    parser.add_argument("--output_file", type=str, default="/workspaces/PC-user-Task3/Test_data/data.csv")
    parser.add_argument("--iterations", type=int, default=2, help="Number of pseudo-training iterations")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        api_pseudo_train(args)
        logger.info("Gemini API pseudo-training completed!")
    
    if args.mode in ["predict", "both"]:
        predict_conflicts(args)
        logger.info("Prediction completed!")

if __name__ == "__main__":
    main()
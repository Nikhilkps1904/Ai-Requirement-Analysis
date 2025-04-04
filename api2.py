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

# API setup for OpenRouter
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Your OpenRouter API key
HF_MODEL = "mistralai/mistral-7b-instruct"  # Model for inference
HF_API_URL = "https://openrouter.ai/api/v1/chat/completions"  # Correct OpenRouter endpoint

def call_inference_api(prompt, api_token=HF_API_TOKEN, api_url=HF_API_URL):
    """Call OpenRouter API for inference"""
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": HF_MODEL,
        "prompt": prompt,
        "max_tokens": 256,
        "num_beams": 5
    }
    try:
        logger.debug(f"Sending request to {api_url} with prompt: {prompt[:100]}...")  # Truncate for readability
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)  # Added timeout
        response.raise_for_status()
        result = response.json()
        logger.debug(f"Raw API response: {json.dumps(result, indent=2)}")
        
        if "choices" in result and len(result["choices"]) > 0:
            output = result["choices"][0].get("text", result["choices"][0].get("message", {}).get("content", "No output"))
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
    """Robustly parse API output, handling malformed or blank responses"""
    if not full_output or "Inference failed" in full_output:
        logger.warning(f"Blank or failed output: {full_output}")
        return "Other", full_output or "No response from API", "Requires manual review"
    
    # Split by "||" and handle cases where format varies
    parts = [p.strip() for p in full_output.split("||") if p.strip()]
    
    if len(parts) < 3:
        logger.warning(f"Malformed output: {full_output}")
        conflict_type = parts[0] if parts else "Other"
        conflict_reason = parts[1] if len(parts) > 1 else "Malformed or incomplete output"
        resolution = "Requires manual review"
    else:
        conflict_type, conflict_reason, resolution = parts[0], parts[1], parts[2]
    
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
    """Use OpenRouter API to iteratively 'train' on the dataset"""
    if os.path.exists(args.input_file):
        df_train = pd.read_csv(args.input_file)
    else:
        logger.warning(f"Input file {args.input_file} not found. Using sample data.")
        df_train = load_sample_data()
    
    logger.info(f"Starting pseudo-training with {len(df_train)} examples using {HF_MODEL} via OpenRouter")
    
    PREDEFINED_CONFLICTS = {
        "Performance Conflict", "Compliance Conflict", "Safety Conflict", "Cost Conflict", "Battery Conflict",
        "Environmental Conflict", "Structural Conflict", "Comfort Conflict", "Power Source Conflict", "Reliability Conflict",
        "Scalability Conflict", "Security Conflict", "Usability Conflict", "Maintenance Conflict", "Weight Conflict",
        "Time-to-Market Conflict", "Compatibility Conflict", "Aesthetic Conflict", "Noise Conflict", "Other Conflict",
        "Sustainability Conflict", "Regulatory Conflict", "Resource Conflict", "Technology Conflict", "Design Conflict", "Contradiction"
    }

    # Corrected prompt template with escaped curly braces
    prompt_template = (
        "Analyze the provided {req1} and {req2} from the uploaded CSV or Excel file, which must contain a single column labeled 'Requirements.' Identify any conflicts between {req1} and {req2}. For each pair, determine if there is a conflict, and if so, specify the type of conflict (including all possible types), the reason for the conflict (as a one-line sentence), and a suggested resolution. The output should be in Excel format, displaying the following columns: Requirement 1, Requirement 2, Conflict Type, and Reason. Include all pairs in the output, even those with no conflicts. This analysis is intended for engineers and analysts. The analysis should also include specific metrics or performance indicators related to the conflicts.\n\n"
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
            
            # Use the corrected prompt template
            input_text = prompt_template.format(req1=req1, req2=req2)
            
            full_output = call_inference_api(input_text)
            conflict_type, conflict_reason, resolution = parse_api_output(full_output)
            
            if conflict_type not in PREDEFINED_CONFLICTS:
                conflict_type = "Other"
            
            results.append({
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason,
                "Resolution_Suggestion": resolution,
                "Expected_Conflict": expected_conflict
            })
        
        correct = sum(1 for r in results if r["Conflict_Type"] == r["Expected_Conflict"])
        logger.info(f"Iteration {iteration} accuracy: {correct / len(results):.2%}")
        
        df_train = pd.DataFrame(results)
        df_train["Conflict_Type"] = df_train["Conflict_Type"]
        results = []
    
    output_df = pd.DataFrame(df_train)
    
    # Ensure directories exist
    csv_dir, xlsx_dir = ensure_directories()
    
    # Save to CSV in Results/CSV
    csv_output = os.path.join(csv_dir, "results.csv")
    output_df.to_csv(csv_output, index=False)
    
    # Save to XLSX in Results/XLSX
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

    # Define the prompt template
    prompt_template = (
    "Analyze two vehicle-related requirements for potential conflicts based on these conflict types: "
    f"{', '.join(PREDEFINED_CONFLICTS)}.\n\n"
    "Input:\n- Requirement 1: \"{req1}\"\n- Requirement 2: \"{req2}\"\n\n"
    "Task:\n1. Identify any conflict and give one line expert answer for all, using vehicle engineering principles.\n"
    "2. If a conflict exists, output: \"Conflict_Type: {{type}}||Reason: {{reason}}||Resolution: {{resolution}}\"\n"
    "3. If no conflict, output: \"\"\n"
    "4. If analysis fails, output: \"Inference failed||Reason unknown||Manual review required\""
)

    requirements = df_input["Requirements"].tolist()
    results = []

    # Generate all unique pairwise combinations of requirements
    pairs = list(itertools.combinations(requirements, 2))
    logger.info(f"Generated {len(pairs)} unique pairwise combinations for analysis")

    for req1, req2 in tqdm(pairs, desc="Analyzing conflicts"):
        # Use the prompt template
        input_text = prompt_template.format(req1=req1, req2=req2)
        
        full_output = call_inference_api(input_text)
        conflict_type, conflict_reason, resolution = parse_api_output(full_output)
        
        if conflict_type not in PREDEFINED_CONFLICTS:
            conflict_type = "Other"
        
        results.append({
            "Requirement_1": req1,
            "Requirement_2": req2,
            "Conflict_Type": conflict_type,
            "Conflict_Reason": conflict_reason,
            "Resolution_Suggestion": resolution
        })
    
    output_df = pd.DataFrame(results)
    
    # Ensure directories exist
    csv_dir, xlsx_dir = ensure_directories()
    
    # Save to CSV in Results/CSV
    csv_output = os.path.join(csv_dir, "results.csv")
    output_df.to_csv(csv_output, index=False)
    
    # Save to XLSX in Results/XLSX
    xlsx_output = os.path.join(xlsx_dir, "results.xlsx")
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    logger.info(f"Analysis complete. Results saved to {csv_output} (CSV) and {xlsx_output} (XLSX)")
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Requirements Conflict Detection with OpenRouter API")
    
    parser.add_argument("--mode", type=str, choices=["train", "predict", "both"], default="train")
    parser.add_argument("--input_file", type=str, default="/workspaces/PC-user-Task3/Test_data/data.csv")
    parser.add_argument("--test_file", type=str, default="./test_data.csv")
    parser.add_argument("--output_file", type=str, default="/workspaces/PC-user-Task3/Test_data/data.csv")  # Kept for compatibility, but ignored for fixed paths
    parser.add_argument("--iterations", type=int, default=2, help="Number of pseudo-training iterations")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        api_pseudo_train(args)
        logger.info("API pseudo-training completed!")
    
    if args.mode in ["predict", "both"]:
        predict_conflicts(args)
        logger.info("Prediction completed!")

if __name__ == "__main__":
    main()
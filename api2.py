import os
import argparse
import pandas as pd
import numpy as np
import requests
import json
import logging
from tqdm import tqdm
import warnings
from dotenv import load_dotenv
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conflict_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face Inference API setup
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Your token
HF_MODEL = "google/flan-t5-base"  # Model for API inference
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def huggingface_inference(prompt, api_token=HF_API_TOKEN, api_url=HF_API_URL):
    """Call Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_length": 256, "num_beams": 5}}
    try:
        logger.debug(f"Sending request to {api_url} with prompt: {prompt}")
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"Raw API response: {json.dumps(result, indent=2)}")
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "error" in result:
            logger.error(f"API error message: {result['error']}")
            return f"Inference failed: {result['error']}"
        else:
            logger.error(f"Unexpected response format: {result}")
            return "Inference failed due to unexpected response format"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return f"Inference failed due to request error: {str(e)}"
    finally:
        time.sleep(2)  # Rate limit delay
# Load Sample Data
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

# API Pseudo-Training Function
def api_pseudo_train(args):
    """Use Hugging Face API to iteratively 'train' on the dataset"""
    if os.path.exists(args.input_file):
        df_train = pd.read_csv(args.input_file)
    else:
        logger.warning(f"Input file {args.input_file} not found. Using sample data.")
        df_train = load_sample_data()
    
    logger.info(f"Starting pseudo-training with {len(df_train)} examples using Hugging Face API")
    
    PREDEFINED_CONFLICTS = {
        "Performance Conflict", "Compliance Conflict", "Safety Conflict",
        "Cost Conflict", "Battery Conflict", "Environmental Conflict",
        "Structural Conflict", "Comfort Conflict", "Power Source Conflict",
        "Other"
    }

    results = []
    iteration = 0
    max_iterations = args.iterations  # Number of iterations for refinement
    
    for _ in range(max_iterations):
        iteration += 1
        logger.info(f"Pseudo-training iteration {iteration}/{max_iterations}")
        
        for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Iteration {iteration}"):
            req1 = row["Requirement_1"]
            req2 = row["Requirement_2"]
            expected_conflict = row["Conflict_Type"]
            
            # Refine prompt with expected conflict as a hint for better alignment
            input_text = (
                f"Analyze requirements conflict: {req1} AND {req2} "
                f"Expected conflict type: {expected_conflict}. "
                "Generate output format: Conflict_Type||Conflict_Reason||Resolution_Suggestion:"
            )
            
            full_output = huggingface_inference(input_text)
            
            # Handle newline truncation
            if "\n" in full_output:
                full_output = full_output.split("\n")[0].strip()
            
            parts = full_output.split("||")
            conflict_type = parts[0].strip() if len(parts) > 0 else "Other"
            if conflict_type not in PREDEFINED_CONFLICTS:
                conflict_type = "Other"
            conflict_reason = parts[1].strip() if len(parts) > 1 else "Needs manual analysis"
            resolution = parts[2].strip() if len(parts) > 2 else "Requires engineering review"
            
            results.append({
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason,
                "Resolution_Suggestion": resolution,
                "Expected_Conflict": expected_conflict
            })
        
        # Update df_train with API predictions for next iteration
        df_train = pd.DataFrame(results)
        df_train["Conflict_Type"] = df_train["Conflict_Type"]  # Use API predictions as new "ground truth"
        results = []  # Reset for next iteration
    
    output_df = pd.DataFrame(df_train)
    output_df.to_csv(args.output_file, index=False)
    logger.info(f"Pseudo-training complete after {max_iterations} iterations. Results saved to {args.output_file}")
    return output_df

# Prediction Function
def predict_conflicts(args):
    """Predict conflicts with structured output using Hugging Face API"""
    try:
        df_input = pd.read_csv(args.test_file, encoding='utf-8')
        logger.info(f"Loaded {len(df_input)} requirement pairs")
    except Exception as e:
        logger.error(f"Error reading test file: {e}")
        return

    PREDEFINED_CONFLICTS = {
        "Performance Conflict", "Compliance Conflict", "Safety Conflict",
        "Cost Conflict", "Battery Conflict", "Environmental Conflict",
        "Structural Conflict", "Comfort Conflict", "Power Source Conflict",
        "Other"
    }

    results = []
    
    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Analyzing conflicts"):
        req1 = row["Requirement_1"]
        req2 = row["Requirement_2"]
        
        input_text = (
            f"Analyze requirements conflict: {req1} AND {req2} "
            "Generate output format: Conflict_Type||Conflict_Reason||Resolution_Suggestion:"
        )
        
        full_output = huggingface_inference(input_text)
        
        # Handle newline truncation
        if "\n" in full_output:
            full_output = full_output.split("\n")[0].strip()
        
        parts = full_output.split("||")
        conflict_type = parts[0].strip() if len(parts) > 0 else "Other"
        if conflict_type not in PREDEFINED_CONFLICTS:
            conflict_type = "Other"
        conflict_reason = parts[1].strip() if len(parts) > 1 else "Needs manual analysis"
        resolution = parts[2].strip() if len(parts) > 2 else "Requires engineering review"
        
        results.append({
            "Requirement_1": req1,
            "Requirement_2": req2,
            "Conflict_Type": conflict_type,
            "Conflict_Reason": conflict_reason,
            "Resolution_Suggestion": resolution
        })
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output_file, index=False)
    logger.info(f"Analysis complete. Results saved to {args.output_file}")
    return output_df

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Requirements Conflict Detection with Hugging Face API")
    
    parser.add_argument("--mode", type=str, choices=["train", "predict", "both"], default="train")
    parser.add_argument("--input_file", type=str, default="TwoWheeler_Requirement_Conflicts.csv")
    parser.add_argument("--test_file", type=str, default="./test_data.csv")
    parser.add_argument("--output_file", type=str, default="conflict_results.csv")
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
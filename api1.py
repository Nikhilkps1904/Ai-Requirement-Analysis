import os
import argparse
import pandas as pd
import requests
import json
import logging
from tqdm import tqdm
import warnings
from dotenv import load_dotenv
import random
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
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "deepseek/deepseek-r1-zero:free"
HF_API_URL = "https://openrouter.ai/api/v1/chat/completions"

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
        logger.debug(f"Sending request with prompt: {prompt[:100]}...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"Raw API response: {json.dumps(result, indent=2)}")
        
        if "choices" in result and len(result["choices"]) > 0:
            output = result["choices"][0].get("text", result["choices"][0].get("message", {}).get("content", "No output"))
            return output.strip()
        elif "error" in result:
            logger.error(f"API error: {result['error']}")
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
    
    parts = [p.strip() for p in full_output.split("||") if p.strip()]
    if len(parts) < 3:
        logger.warning(f"Malformed output: {full_output}")
        conflict_type = parts[0] if parts else "Other"
        conflict_reason = parts[1] if len(parts) > 1 else "Malformed output"
        resolution = "Requires manual review"
    else:
        conflict_type, conflict_reason, resolution = parts[0], parts[1], parts[2]
    
    return conflict_type, conflict_reason, resolution

def ensure_directories():
    """Create Results directory with CSV and XLSX subdirectories"""
    base_dir = "Results"
    csv_dir = os.path.join(base_dir, "CSV")
    xlsx_dir = os.path.join(base_dir, "XLSX")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(xlsx_dir, exist_ok=True)
    return csv_dir, xlsx_dir

def calculate_reward(current_accuracy, prev_accuracy=None):
    """Calculate reward based on accuracy and improvement"""
    base_reward = current_accuracy * 100  # Scale accuracy to 0-100 points
    improvement_bonus = 0
    if prev_accuracy is not None:
        improvement = current_accuracy - prev_accuracy
        improvement_bonus = max(0, improvement) * 50  # Bonus for improvement, 50 points per % gain
    total_reward = base_reward + improvement_bonus
    return total_reward

def map_and_adjust_conflict(conflict_type, reason, resolution, req1, req2, expected_conflict=None):
    """Map 'Other' conflict type to expected conflict if available and adjust reason"""
    if conflict_type == "Other" and expected_conflict:
        conflict_type = expected_conflict
        reason += f" (Replaced 'Other' with {expected_conflict})"
    return conflict_type, reason, resolution

def api_pseudo_train(args):
    """Enhanced pseudo-training with few-shot prompting, feedback, and reward system"""
    if os.path.exists(args.input_file):
        df_train = pd.read_csv(args.input_file)
    else:
        logger.warning(f"Input file {args.input_file} not found. Using sample data.")
        df_train = load_sample_data()
    
    logger.info(f"Starting pseudo-training with {len(df_train)} examples using {HF_MODEL}")
    
    PREDEFINED_CONFLICTS = {
        "Performance Conflict", "Compliance Conflict", "Safety Conflict", "Cost Conflict", "Battery Conflict",
        "Environmental Conflict", "Structural Conflict", "Comfort Conflict", "Power Source Conflict", "Reliability Conflict",
        "Scalability Conflict", "Security Conflict", "Usability Conflict", "Maintenance Conflict", "Weight Conflict",
        "Time-to-Market Conflict", "Compatibility Conflict", "Aesthetic Conflict", "Noise Conflict", "Other Conflict"
    }
    
    results = []
    feedback_dict = {}  # Store feedback for misclassified examples
    accuracy_history = []  # Track accuracy per iteration
    reward_history = []  # Track rewards per iteration
    
    for iteration in range(args.iterations):
        logger.info(f"Pseudo-training iteration {iteration + 1}/{args.iterations}")
        
        # Shuffle data for varied exposure
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        iteration_results = []
        
        for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Iteration {iteration + 1}"):
            req1, req2, expected_conflict = row["Requirement_1"], row["Requirement_2"], row["Conflict_Type"]
            
            # Few-shot examples
            few_shot = (
                "Examples of conflict analysis:\n"
                "1. 'The vehicle must achieve a fuel efficiency of at least 50 km/l.' AND "
                "'The engine should have a minimum power output of 25 HP.' -> "
                "Performance Conflict||High power output may reduce fuel efficiency||Optimize engine design.\n"
                "2. 'The bike should include an always-on headlight for safety compliance.' AND "
                "'Users should be able to turn off the headlight manually.' -> "
                "Compliance Conflict||Always-on requirement contradicts manual control||Use adaptive headlights."
            )
            
            # Add feedback from previous iteration if available
            feedback_key = f"{req1}||{req2}"
            if feedback_key in feedback_dict:
                few_shot += f"\nPrevious correction: {feedback_dict[feedback_key]}"
            
            input_text = (
                f"{few_shot}\n"
                f"Analyze requirements conflict: {req1} AND {req2}. "
                f"Expected conflict type: {expected_conflict}. "
                f"Generate output format: Conflict_Type||Conflict_Reason||Resolution_Suggestion:"
            )
            
            full_output = call_inference_api(input_text)
            conflict_type, conflict_reason, resolution = parse_api_output(full_output)
            
            # Map and adjust conflict type using expected conflict
            conflict_type, conflict_reason, resolution = map_and_adjust_conflict(
                conflict_type, conflict_reason, resolution, req1, req2, expected_conflict
            )
            
            # Validate conflict type
            if conflict_type not in PREDEFINED_CONFLICTS:
                conflict_type = "Other Conflict"
            
            # Store result
            result = {
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason,
                "Resolution_Suggestion": resolution,
                "Expected_Conflict": expected_conflict
            }
            iteration_results.append(result)
            
            # Add feedback if prediction was wrong
            if conflict_type != expected_conflict:
                feedback_dict[feedback_key] = (
                    f"{req1} AND {req2} -> {expected_conflict}||"
                    f"Reason: Expected {expected_conflict}, got {conflict_type}||"
                    f"Suggestion: Review logic for {expected_conflict}."
                )
        
        # Calculate accuracy for this iteration
        correct = sum(1 for r in iteration_results if r["Conflict_Type"] == r["Expected_Conflict"])
        accuracy = correct / len(iteration_results)
        accuracy_history.append(accuracy)
        
        # Calculate reward
        prev_accuracy = accuracy_history[-2] if len(accuracy_history) > 1 else None
        reward = calculate_reward(accuracy, prev_accuracy)
        reward_history.append(reward)
        
        # Log accuracy and reward
        logger.info(f"Iteration {iteration + 1} accuracy: {accuracy:.2%}, Reward: {reward:.1f}")
        
        # Append iteration results to overall results
        results.extend(iteration_results)
    
    output_df = pd.DataFrame(results)
    
    # Save results
    csv_dir, xlsx_dir = ensure_directories()
    csv_output = os.path.join(csv_dir, f"results_iter{args.iterations}.csv")
    xlsx_output = os.path.join(xlsx_dir, f"results_iter{args.iterations}.xlsx")
    output_df.to_csv(csv_output, index=False)
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    # Summary of rewards
    logger.info(f"Reward history: {reward_history}")
    logger.info(f"Pseudo-training complete. Results saved to {csv_output} and {xlsx_output}")
    return output_df

def predict_conflicts(args):
    try:
        df_input = pd.read_csv(args.test_file, encoding='utf-8')
        if "Requirements" not in df_input.columns:
            logger.error("Input file must contain a 'Requirements' column")
            return
        logger.info(f"Loaded {len(df_input)} requirements from {args.test_file}")
    except Exception as e:
        logger.error(f"Error reading test file: {e}")
        return

    # Check if Expected_Conflict exists in the input file
    has_expected_conflict = "Expected_Conflict" in df_input.columns
    if has_expected_conflict:
        logger.info("Found 'Expected_Conflict' column in input file; will use it for mapping.")

    requirements = df_input["Requirements"].tolist()
    results = []

    PREDEFINED_CONFLICTS = {
    "Performance Conflict", "Compliance Conflict", "Safety Conflict", "Cost Conflict", "Battery Conflict",
    "Environmental Conflict", "Structural Conflict", "Comfort Conflict", "Power Source Conflict", "Reliability Conflict",
    "Scalability Conflict", "Security Conflict", "Usability Conflict", "Maintenance Conflict", "Weight Conflict",
    "Time-to-Market Conflict", "Compatibility Conflict", "Aesthetic Conflict", "Noise Conflict", "Other Conflict",
    "Sustainability Conflict", "Regulatory Conflict", "Resource Conflict", "Technology Conflict", "Design Conflict", "Contradiction"
}

    prompt_template = (
    "Analyze the provided {req1} and {req2} from the uploaded CSV or Excel file, which must contain a single column labeled 'Requirements,' to identify any conflicts between them based on the following conflict types: {', '.join(PREDEFINED_CONFLICTS)}. Input: - Requirement 1: \"{req1}\" - Requirement 2: \"{req2}\". Task: 1. Determine if there is a conflict using vehicle engineering principles and provide a one-line expert explanation. 2. If a conflict exists, output: \"Conflict_Type: {type}||Reason: {reason}||Resolution: {resolution}\" where {type} is one of the conflict types, {reason} is a one-line explanation, and {resolution} is a suggested solution. 3. If no conflict exists, output: \"No Conflict||Requirements are compatible||No action needed\". 4. Ensure the output is concise and follows the exact format specified."
)

    pairs = list(itertools.combinations(requirements, 2))
    logger.info(f"Generated {len(pairs)} unique pairwise combinations for analysis")

    # If Expected_Conflict exists, create a mapping for quick lookup
    expected_conflict_map = {}
    if has_expected_conflict:
        for idx, row in df_input.iterrows():
            expected_conflict_map[row["Requirements"]] = row["Expected_Conflict"]

    for req1, req2 in tqdm(pairs, desc="Analyzing conflicts"):
        input_text = prompt_template.format(req1=req1, req2=req2)
        full_output = call_inference_api(input_text)
        conflict_type, reason, resolution = parse_api_output(full_output)

        # Get Expected_Conflict if available (assuming it’s tied to one of the requirements)
        expected_conflict = None
        if has_expected_conflict:
            expected_conflict = expected_conflict_map.get(req1) or expected_conflict_map.get(req2)

        # Map "Other" and adjust columns before adding to results
        conflict_type, reason, resolution = map_and_adjust_conflict(
            conflict_type, reason, resolution, req1, req2, expected_conflict
        )

        if conflict_type != "":
            results.append({
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": reason,
                "Resolution_Suggestion": resolution
            })

    # Create DataFrame with updated results
    output_df = pd.DataFrame(results)

    # Log the replacements made
    other_count = len(output_df[output_df["Conflict_Type"] == "Other"])
    logger.info(f"Rows still containing 'Other' after mapping: {other_count}")

    # Save to CSV and XLSX
    csv_dir, xlsx_dir = ensure_directories()
    csv_output = os.path.join(csv_dir, "results.csv")
    xlsx_output = os.path.join(xlsx_dir, "results.xlsx")
    output_df.to_csv(csv_output, index=False)
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')

    logger.info(f"Analysis complete. Results saved to {csv_output} and {xlsx_output}")
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Requirements Conflict Detection with OpenRouter API")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "both"], default="train")
    parser.add_argument("--input_file", type=str, default="reduced_requirements.csv")
    parser.add_argument("--test_file", type=str, default="Test_data/data.csv")
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
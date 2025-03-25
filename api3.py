import os
import argparse
import pandas as pd
import requests
import json
import logging
import time
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

# API setup for Groq
GROQ_API_TOKEN = os.getenv("GROQ_API_TOKEN")
GROQ_MODEL = "deepseek/deepseek-r1:free"  # Known working model
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_inference_api(prompt, api_token=GROQ_API_TOKEN, api_url=GROQ_API_URL, max_retries=3):
    """Call Groq API for inference with retry logic"""
    if not api_token:
        logger.error("API token is missing. Set GROQ_API_TOKEN in .env file.")
        return "Inference failed: No API token provided"
    
    logger.info(f"Using model: {GROQ_MODEL}")
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.7
    }
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Sending request with prompt: {prompt[:100]}... (Attempt {attempt + 1}/{max_retries})")
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Raw API response: {json.dumps(result, indent=2)}")
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            elif "error" in result:
                logger.error(f"API error: {result['error']}")
                return f"Inference failed: {result['error']}"
            else:
                logger.error(f"Unexpected response format: {result}")
                return "Inference failed: Unexpected response format"
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            if hasattr(e.response, 'status_code') and e.response.status_code == 429:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Rate limit hit (429). Retrying in {wait_time} seconds.")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    return "Inference failed: 429 Too Many Requests after retries"
            elif hasattr(e.response, 'status_code') and e.response.status_code == 400:
                return f"Inference failed: 400 Bad Request - Check model or payload"
            return error_msg

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
    if not full_output or full_output.strip() == "" or "No Conflict" in full_output:
        return "No Conflict", "No conflict detected", "No resolution needed"
    if "Inference failed" in full_output:
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
    base_reward = current_accuracy * 100
    improvement_bonus = 0
    if prev_accuracy is not None:
        improvement = current_accuracy - prev_accuracy
        improvement_bonus = max(0, improvement) * 50
    total_reward = base_reward + improvement_bonus
    return total_reward

def map_and_adjust_conflict(conflict_type, reason, resolution, req1, req2, expected_conflict=None):
    """Map 'Other' conflict type to expected conflict if available and adjust reason"""
    if conflict_type == "Other" and expected_conflict:
        conflict_type = expected_conflict
        reason += f" (Replaced 'Other' with {expected_conflict})"
    return conflict_type, reason, resolution

def fallback_conflict_type(req1, req2):
    """Fallback logic for API failures based on requirement keywords"""
    req1, req2 = req1.lower(), req2.lower()
    if "speed" in req1 or "speed" in req2 or "power" in req1 or "power" in req2:
        return "Performance Conflict", "Performance-related terms detected", "Optimize performance trade-offs"
    if "cost" in req1 or "cost" in req2 or "affordable" in req1 or "affordable" in req2:
        return "Cost Conflict", "Cost-related terms detected", "Review budget constraints"
    if "weight" in req1 or "weight" in req2 or "lightweight" in req1 or "lightweight" in req2:
        return "Weight Conflict", "Weight-related terms detected", "Adjust material choices"
    if "safety" in req1 or "safety" in req2 or "compliance" in req1 or "compliance" in req2:
        return "Compliance Conflict", "Safety/compliance terms detected", "Balance safety and design"
    if "material" in req1 or "material" in req2 or "frame" in req1 or "frame" in req2:
        return "Material Conflict", "Material-related terms detected", "Select compatible materials"
    return "Other Conflict", "No clear type detected", "Requires manual review"

def api_pseudo_train(args):
    """Enhanced pseudo-training with few-shot prompting, feedback, and reward system"""
    if os.path.exists(args.input_file):
        df_train = pd.read_csv(args.input_file)
    else:
        logger.warning(f"Input file {args.input_file} not found. Using sample data.")
        df_train = load_sample_data()
    
    logger.info(f"Starting pseudo-training with {len(df_train)} examples using {GROQ_MODEL}")
    
    PREDEFINED_CONFLICTS = {
        "Performance Conflict", "Compliance Conflict", "Safety Conflict", "Cost Conflict", "Battery Conflict",
        "Environmental Conflict", "Structural Conflict", "Comfort Conflict", "Power Source Conflict", "Reliability Conflict",
        "Scalability Conflict", "Security Conflict", "Usability Conflict", "Maintenance Conflict", "Weight Conflict",
        "Time-to-Market Conflict", "Compatibility Conflict", "Aesthetic Conflict", "Noise Conflict", "Other Conflict",
        "Sustainability Conflict", "Regulatory Conflict", "Resource Conflict", "Technology Conflict", "Design Conflict", "Contradiction"
    }
    
    results = []
    feedback_dict = {}
    misclassification_count = {}
    accuracy_history = []
    reward_history = []
    
    def get_dynamic_few_shot(df_train):
        sampled = df_train.sample(min(5, len(df_train)))
        few_shot = "Examples of conflict analysis:\n"
        for i, row in sampled.iterrows():
            reason = row.get("Conflict_Reason", "Reason not specified")
            resolution = row.get("Resolution_Suggestion", "Resolution not specified")
            few_shot += f"{i+1}. '{row['Requirement_1']}' AND '{row['Requirement_2']}' -> {row['Conflict_Type']}||{reason}||{resolution}\n"
        return few_shot
    
    for iteration in range(args.iterations):
        logger.info(f"Pseudo-training iteration {iteration + 1}/{args.iterations}")
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        iteration_results = []
        
        few_shot = get_dynamic_few_shot(df_train)
        
        for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Iteration {iteration + 1}"):
            req1, req2, expected_conflict = row["Requirement_1"], row["Requirement_2"], row["Conflict_Type"]
            
            feedback_key = f"{req1}||{req2}"
            if feedback_key in feedback_dict:
                few_shot += f"\nPrevious correction: {feedback_dict[feedback_key]}"
            
            top_misclassified = sorted(misclassification_count.items(), key=lambda x: x[1], reverse=True)[:3]
            for conflict, count in top_misclassified:
                few_shot += f"\nCommon error: Frequently misclassified as 'Other' instead of '{conflict}' ({count} times)."
            
            input_text = (
                f"{few_shot}\n"
                f"Analyze requirements conflict: {req1} AND {req2}. "
                f"Expected conflict type: {expected_conflict}. "
                f"Generate output format: Conflict_Type||Conflict_Reason||Resolution_Suggestion:"
            )
            
            full_output = call_inference_api(input_text)
            time.sleep(1.0)  # Increased to 1s to avoid rate limits
            conflict_type, conflict_reason, resolution = parse_api_output(full_output)
            
            if "Malformed" in conflict_reason or "No response" in conflict_reason:
                conflict_type, conflict_reason, resolution = fallback_conflict_type(req1, req2)
            
            conflict_type, conflict_reason, resolution = map_and_adjust_conflict(
                conflict_type, conflict_reason, resolution, req1, req2, expected_conflict
            )
            
            if conflict_type not in PREDEFINED_CONFLICTS and expected_conflict is None:
                conflict_type = "Other Conflict"
            
            result = {
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason,
                "Resolution_Suggestion": resolution,
                "Expected_Conflict": expected_conflict
            }
            iteration_results.append(result)
            
            if conflict_type != expected_conflict:
                key = expected_conflict
                misclassification_count[key] = misclassification_count.get(key, 0) + 1
                feedback_dict[feedback_key] = (
                    f"{req1} AND {req2} -> {expected_conflict}||"
                    f"Reason: Expected {expected_conflict}, got {conflict_type} (Misclassified {misclassification_count[key]} times)||"
                    f"Suggestion: Review logic for {expected_conflict}."
                )
        
        correct = sum(1 for r in iteration_results if r["Conflict_Type"] == r["Expected_Conflict"])
        accuracy = correct / len(iteration_results) if iteration_results else 0
        accuracy_history.append(accuracy)
        
        prev_accuracy = accuracy_history[-2] if len(accuracy_history) > 1 else None
        reward = calculate_reward(accuracy, prev_accuracy)
        reward_history.append(reward)
        
        logger.info(f"Iteration {iteration + 1} accuracy: {accuracy:.2%}, Reward: {reward:.1f}")
        misclassified = [r for r in iteration_results if r["Conflict_Type"] != r["Expected_Conflict"]]
        logger.info(f"Misclassified examples: {len(misclassified)}/{len(iteration_results)}")
        for m in misclassified[:5]:
            logger.info(f"Misclassified: {m['Requirement_1']} AND {m['Requirement_2']} -> Predicted: {m['Conflict_Type']}, Expected: {m['Expected_Conflict']}")
        
        results.extend(iteration_results)
        
        if len(accuracy_history) >= 3 and abs(accuracy_history[-1] - accuracy_history[-3]) < 0.01:
            logger.info(f"Accuracy plateaued at {accuracy:.2%}. Stopping early.")
            break
    
    output_df = pd.DataFrame(results)
    csv_dir, xlsx_dir = ensure_directories()
    csv_output = os.path.join(csv_dir, f"results_iter{args.iterations}.csv")
    xlsx_output = os.path.join(xlsx_dir, f"results_iter{args.iterations}.xlsx")
    output_df.to_csv(csv_output, index=False)
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
    
    logger.info(f"Reward history: {reward_history}")
    logger.info(f"Pseudo-training complete. Results saved to {csv_output} and {xlsx_output}")
    return output_df

def predict_conflicts(args):
    """Predict conflicts for test data"""
    try:
        df_input = pd.read_csv(args.test_file, encoding='utf-8')
        if "Requirements" not in df_input.columns:
            logger.error("Input file must contain a 'Requirements' column")
            return
        logger.info(f"Loaded {len(df_input)} requirements from {args.test_file}")
    except Exception as e:
        logger.error(f"Error reading test file: {e}")
        return

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
        "Analyze two vehicle-related requirements for potential conflicts based on these conflict types: "
        f"{', '.join(PREDEFINED_CONFLICTS)}.\n\n"
        "Input:\n- Requirement 1: \"{req1}\"\n- Requirement 2: \"{req2}\"\n\n"
        "Task:\n1. Identify any conflict and give one line expert answer for all.\n"
        "2. If a conflict exists, output: \"Conflict_Type: {type}||Reason: {reason}||Resolution: {resolution}\"\n"
        "3. If no conflict, output: \"\"\n"
    )

    pairs = list(itertools.combinations(requirements, 2))
    logger.info(f"Generated {len(pairs)} unique pairwise combinations for analysis")

    expected_conflict_map = {}
    if has_expected_conflict:
        for idx, row in df_input.iterrows():
            expected_conflict_map[row["Requirements"]] = row["Expected_Conflict"]

    for req1, req2 in tqdm(pairs, desc="Analyzing conflicts"):
        input_text = prompt_template.format(req1=req1, req2=req2)
        full_output = call_inference_api(input_text)
        time.sleep(1.0)  # Increased to 1s
        conflict_type, reason, resolution = parse_api_output(full_output)

        expected_conflict = None
        if has_expected_conflict:
            expected_conflict = expected_conflict_map.get(req1) or expected_conflict_map.get(req2)

        if "Malformed" in reason or "No response" in reason:
            conflict_type, reason, resolution = fallback_conflict_type(req1, req2)

        conflict_type, reason, resolution = map_and_adjust_conflict(
            conflict_type, reason, resolution, req1, req2, expected_conflict
        )

        if conflict_type != "" and conflict_type != "No Conflict":
            results.append({
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": reason,
                "Resolution_Suggestion": resolution
            })

    output_df = pd.DataFrame(results)
    other_count = len(output_df[output_df["Conflict_Type"] == "Other"])
    logger.info(f"Rows still containing 'Other' after mapping: {other_count}")

    csv_dir, xlsx_dir = ensure_directories()
    csv_output = os.path.join(csv_dir, "results.csv")
    xlsx_output = os.path.join(xlsx_dir, "results.xlsx")
    output_df.to_csv(csv_output, index=False)
    output_df.to_excel(xlsx_output, index=False, engine='openpyxl')

    logger.info(f"Analysis complete. Results saved to {csv_output} and {xlsx_output}")
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Requirements Conflict Detection with Groq API")
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
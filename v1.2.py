'''
In this version there is implementation of the new api and removed all the constraints which was using
in gemini api config and new logging system the script takes around 5mins to for 1346 pairs to analyse.

IMPROVEMENTS:-
1.Command Line interface should be improved.
2.new Logic need to be added properly.
3.Implementation proper guideline interface.
4.Logic of interface should be changed.
'''


import os
import pandas as pd
import logging
import sys
import tkinter as tk
from tkinter import filedialog
import concurrent.futures
import itertools
import pickle
from dotenv import load_dotenv
from openai import OpenAI

# Check if file system access is available
try:
    with open("conflict_detection.log", "a"):
        pass
    use_file_logging = True
except Exception as e:
    logger.warning(f"File system access unavailable: {e}. Using console logging only.")
    use_file_logging = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conflict_detection.log") if use_file_logging else logging.NullHandler(),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key="dummy",
    base_url="https://temp.com/api/openai/deployments/google-gemini-1-5-flash",
    default_headers={"genaiplatform-farm-subscription-key": "dummy"}
)

# File to store file analysis requirements
FILE_ANALYSIS_FILE = "file_analysis.pkl"

def save_file_analysis(requirements, results):
    """Save file-based requirements and analysis results"""
    if use_file_logging:
        try:
            with open(FILE_ANALYSIS_FILE, "wb") as f:
                pickle.dump({"requirements": requirements, "results": results}, f)
            logger.info(f"Saved file analysis to {FILE_ANALYSIS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save file analysis: {e}")

def load_file_analysis():
    """Load file-based requirements and analysis results"""
    if use_file_logging and os.path.exists(FILE_ANALYSIS_FILE):
        try:
            with open(FILE_ANALYSIS_FILE, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded file analysis from {FILE_ANALYSIS_FILE}")
            return data.get("requirements", []), data.get("results", [])
        except Exception as e:
            logger.error(f"Failed to load file analysis: {e}")
    return [], []

def call_inference_api(prompt):
    """Call API using OpenAI client"""
    try:
        response = client.chat.completions.create(
            model="gemini-1.5-flash",
            n=1,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return f"Inference failed: {str(e)}"

def parse_api_output(full_output):
    """Parse API output"""
    if not full_output or "Inference failed" in full_output:
        logger.warning(f"Blank or failed output: {full_output}")
        return "Unknown", "Requires manual review", "Not applicable"

    full_output = full_output.strip()
    
    try:
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

        if ": " in full_output:
            parts = [p.strip() for p in full_output.split(": ", 1) if p.strip()]
            if len(parts) == 2:
                conflict_type = parts[0]
                conflict_reason = parts[1]
                if conflict_type and conflict_reason:
                    return conflict_type, conflict_reason, "Not applicable"
        
        lines = full_output.split('\n')
        for line in lines:
            if "conflict" in line.lower() and not line.lower().startswith("no conflict"):
                return "Potential Conflict", line.strip(), "Not applicable"
        
        if "no conflict" in full_output.lower():
            return "No Conflict", "Requirements are compatible", "Not applicable"
    
    except Exception as e:
        logger.warning(f"Error parsing output: {e} - '{full_output}'")
    
    logger.warning(f"Malformed output, unable to parse: '{full_output}'")
    return "Unknown", "Requires manual review", "Not applicable"

def ensure_directories(is_new_requirement=False):
    """Create results directories for file or new requirement analysis"""
    base_dir = "Results"
    if is_new_requirement:
        results_dir = os.path.join(base_dir, "NewRequirementAnalysis")
    else:
        results_dir = os.path.join(base_dir, "FileAnalysis")
    csv_dir = os.path.join(results_dir, "CSV")
    xlsx_dir = os.path.join(results_dir, "XLSX")
    
    if use_file_logging:
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(xlsx_dir, exist_ok=True)
    return csv_dir, xlsx_dir

def check_requirements_batch(req_pairs):
    """Process requirement pairs using ThreadPoolExecutor"""
    results = []
    
    prompt_template = (
        "Analyze the following requirements: Requirement 1: '{req1}' and Requirement 2: '{req2}'. Identify any conflicts between them. Always give a Engineering reason."
        "For each pair, determine if there is a conflict, and if so, specify a descriptive conflict type and the reason (as a one-line sentence). "
        "The output should be in the format: \"Conflict_Type: <type>||Reason: <reason>\" where <type> is your determined conflict type, and <reason> is a one-line explanation. "
        "If no conflict exists, output: \"Conflict_Type: No Conflict||Reason: Requirements are compatible\". Ensure the output is concise and follows the exact format."
    )
    
    def process_pair(pair_data):
        req1, req2 = pair_data
        input_text = prompt_template.format(req1=req1, req2=req2)
        full_output = call_inference_api(input_text)
        conflict_type, conflict_reason, _ = parse_api_output(full_output)
        if conflict_type != "No Conflict":
            return {
                "Requirement_1": req1,
                "Requirement_2": req2,
                "Conflict_Type": conflict_type,
                "Conflict_Reason": conflict_reason
            }
        return None
    
    logger.info(f"Processing {len(req_pairs)} pairs")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_pair = {executor.submit(process_pair, pair): pair for pair in req_pairs}
        for future in concurrent.futures.as_completed(future_to_pair):
            result = future.result()
            if result:
                results.append(result)
    
    return results

def check_new_requirement(new_req, all_existing_requirements):
    """Check a new requirement against existing ones"""
    pairs = [(new_req, existing_req) for existing_req in all_existing_requirements]
    results = check_requirements_batch(pairs)
    if results:
        return pd.DataFrame(results)
    return pd.DataFrame(columns=["Requirement_1", "Requirement_2", "Conflict_Type", "Conflict_Reason"])

def predict_conflicts(input_file=None, new_requirement=None):
    """Analyze requirements for conflicts"""
    # Delete old log file if it exists
    if use_file_logging and os.path.exists("conflict_detection.log"):
        try:
            os.remove("conflict_detection.log")
            logger.info("Deleted old log file: conflict_detection.log")
        except Exception as e:
            logger.error(f"Failed to delete old log file: {e}")

    # Load requirements
    all_original_requirements = []
    previous_results = []
    if new_requirement and not input_file:
        # Try loading from file analysis
        all_original_requirements, previous_results = load_file_analysis()
        if not all_original_requirements:
            logger.warning("No previous file analysis found. Please provide a requirements file.")
            return pd.DataFrame(columns=["Requirement_1", "Requirement_2", "Conflict_Type", "Conflict_Reason"])
    elif input_file:
        if not os.path.exists(input_file):
            logger.error(f"Input file {input_file} not found. Exiting.")
            sys.exit(1)
        try:
            df_input = pd.read_csv(input_file, encoding='utf-8')
            if "Requirements" not in df_input.columns:
                logger.error("Input file must contain a 'Requirements' column")
                sys.exit(1)
            logger.info(f"Loaded {len(df_input)} requirements from {input_file}")
            all_original_requirements = df_input["Requirements"].tolist()
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            sys.exit(1)
    
    if not all_original_requirements:
        logger.error("No requirements found.")
        sys.exit(1)

    results = []
    if new_requirement:
        new_results = check_new_requirement(new_requirement, all_original_requirements)
        if not new_results.empty:
            results = new_results.to_dict('records')
    else:
        req_pairs = list(itertools.combinations(all_original_requirements, 2))
        results = check_requirements_batch(req_pairs)
        # Save file analysis for reuse
        save_file_analysis(all_original_requirements, results)

    output_df = pd.DataFrame(results) if results else pd.DataFrame(columns=["Requirement_1", "Requirement_2", "Conflict_Type", "Conflict_Reason"])
    
    if use_file_logging:
        csv_dir, xlsx_dir = ensure_directories(is_new_requirement=bool(new_requirement))
        csv_output = os.path.join(csv_dir, "results.csv")
        xlsx_output = os.path.join(xlsx_dir, "results.xlsx")
        try:
            output_df.to_csv(csv_output, index=False)
            output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
            logger.info(f"Analysis complete. Results saved to {csv_output} and {xlsx_output}")
        except Exception as e:
            logger.error(f"Failed to save results to files: {e}")
            logger.info("Results (not saved to file):\n" + output_df.to_string())
    else:
        logger.info("File saving skipped (no file system access). Results:\n" + output_df.to_string())
    
    # Ensure logs are flushed to file
    if use_file_logging:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        logger.info("Execution completed. Logs saved to conflict_detection.log")
    
    return output_df

def display_menu():
    """Display the menu"""
    try:
        root = tk.Tk()
        root.withdraw()
    except:
        logger.warning("Tkinter unavailable. Using console input.")
        return console_menu()

    while True:
        print("\n=== Requirements Conflict Detection Menu ===")
        print("1. Analyze Requirements")
        print("2. Enter a new requirement")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            input_file = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
            if not input_file:
                logger.error("No file selected.")
                continue
            predict_conflicts(input_file=input_file)

        elif choice == "2":
            new_requirement = input("Enter a new requirement (or 'back' to return): ").strip()
            if new_requirement.lower() == 'back':
                continue
            if not new_requirement:
                logger.error("Requirement cannot be empty.")
                continue
            # Try using file analysis; prompt for file if none exists
            requirements, _ = load_file_analysis()
            if requirements:
                logger.info("Using requirements from previous file analysis.")
                predict_conflicts(new_requirement=new_requirement)
            else:
                input_file = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
                if not input_file:
                    logger.error("No file selected.")
                    continue
                predict_conflicts(input_file=input_file, new_requirement=new_requirement)

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
            predict_conflicts(input_file=input_file)

        elif choice == "2":
            new_requirement = input("Enter a new requirement (or 'back' to return): ").strip()
            if new_requirement.lower() == 'back':
                continue
            if not new_requirement:
                logger.error("Requirement cannot be empty.")
                continue
            requirements, _ = load_file_analysis()
            if requirements:
                logger.info("Using requirements from previous file analysis.")
                predict_conflicts(new_requirement=new_requirement)
            else:
                input_file = input("Enter the path to the CSV file: ").strip()
                if not os.path.exists(input_file):
                    logger.error("File not found.")
                    continue
                predict_conflicts(input_file=input_file, new_requirement=new_requirement)

        elif choice == "3":
            logger.info("Exiting the program.")
            sys.exit(0)

        else:
            logger.error("Invalid choice. Please enter a number between 1 and 3.")

def main():
    """Main function"""
    try:
        display_menu()
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
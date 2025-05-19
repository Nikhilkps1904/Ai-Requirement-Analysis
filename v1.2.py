import os
import pandas as pd
import logging
import sys
import streamlit as st
import concurrent.futures
import itertools
from dotenv import load_dotenv
from openai import OpenAI
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Check if file system access is available
try:
    with open("conflict_detection.log", "a"):
        pass
    use_file_logging = True
except:
    logger.warning("File system access unavailable. Using console logging only.")
    use_file_logging = False

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key="dummy",
    base_url="https://temp.com/api/openai/deployments/google-gemini-1-5-flash",
    default_headers={"genaiplatform-farm-subscription-key": "dummy"}
)

def call_inference_api(prompt):
    """Call API using OpenAI client"""
    try:
        response = client.chat.completions.create(
            model="gemini-1.5-flash",
            n=1,
            messages=[
                {"role": "system", "content": "You are a highly knowledgeable assistant with expertise in physics, mathematics, logic, and engineering."},
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

def ensure_directories():
    """Create Results directory with CSV and XLSX subdirectories"""
    base_dir = "Results"
    csv_dir = os.path.join(base_dir, "CSV")
    xlsx_dir = os.path.join(base_dir, "XLSX")
    
    if use_file_logging:
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(xlsx_dir, exist_ok=True)
    return csv_dir, xlsx_dir

def check_requirements_batch(req_pairs):
    """Process requirement pairs using ThreadPoolExecutor"""
    results = []
    
    prompt_template = """
You are an expert in embedded systems, microcontroller (MCU) architecture, electrical engineering, and logical reasoning. Analyze the following technical requirements for conflicts, particularly in the context of MCU design, power systems, and electronic equipment integration.

Instructions:
1. Identify the domain of each requirement (e.g., embedded hardware, electrical interfaces, firmware behavior, timing, power management, communication protocols).
2. Apply relevant engineering and scientific principles (e.g., Kirchhoff’s laws, voltage/current compatibility, thermal limits, signal integrity, real-time constraints, logic consistency).
3. Use chain-of-thought reasoning: decompose each requirement, compare their implications, and verify design feasibility or detect logical inconsistencies.
4. Flag any ambiguous requirement for manual review and explain the ambiguity (e.g., undefined parameters, vague constraints, or missing units).
5. Determine if a conflict exists. If so, specify the conflict type and a one-line explanation grounded in system-level design principles.
6. Use the following format:
   "Conflict_Type: <type>||Reason: <reason>"
   You may define the conflict type as appropriate to the context. There are no restrictions on accepted types.
7. If no conflict exists, respond with:
   "Conflict_Type: No Conflict||Reason: Requirements are compatible in electronic system design context"

Ensure the output is concise, technically grounded, and follows the exact format.
Now analyze: Requirement 1: '{req1}' and Requirement 2: '{req2}'.
"""

    
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

def predict_conflicts(input_file, new_requirement=None):
    """Analyze requirements for conflicts"""
    try:
        try:
            df_input = pd.read_csv(input_file, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed. Trying ISO-8859-1 encoding.")
            df_input = pd.read_csv(input_file, encoding='ISO-8859-1')
        
        if "Requirements" not in df_input.columns:
            logger.error("Input file must contain a 'Requirements' column")
            return None
        logger.info(f"Loaded {len(df_input)} requirements from input file")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return None
    
    all_original_requirements = df_input["Requirements"].tolist()
    if not all_original_requirements:
        logger.error("No requirements found in the input file.")
        return None
    
    results = []
    if new_requirement:
        new_results = check_new_requirement(new_requirement, all_original_requirements)
        if not new_results.empty:
            results = new_results.to_dict('records')
    else:
        req_pairs = list(itertools.combinations(all_original_requirements, 2))
        results = check_requirements_batch(req_pairs)

    output_df = pd.DataFrame(results) if results else pd.DataFrame(columns=["Requirement_1", "Requirement_2", "Conflict_Type", "Conflict_Reason"])
    
    csv_dir, xlsx_dir = ensure_directories()
    csv_output = os.path.join(csv_dir, "results.csv")
    xlsx_output = os.path.join(xlsx_dir, "results.xlsx")
    
    if use_file_logging:
        try:
            output_df.to_csv(csv_output, index=False)
            output_df.to_excel(xlsx_output, index=False, engine='openpyxl')
            logger.info(f"Analysis complete. Results saved to {csv_output} and {xlsx_output}")
        except Exception as e:
            logger.error(f"Failed to save results to files: {e}")
            logger.info("Results (not saved to file):\n" + output_df.to_string())
    else:
        logger.info("Results:\n" + output_df.to_string())
    
    return output_df

def main():
    """Main Streamlit app"""
    st.set_page_config(page_title="Requirements Conflict Detection", layout="wide")
    
    st.title("Requirements Conflict Detection")
    st.markdown("""
    ### Instructions
    - **Analyze Requirements**: Upload a CSV file containing a "Requirements" column to check for conflicts between all pairs.
    - **Check New Requirement**: Enter a new requirement and upload a CSV file to check it against existing requirements.
    - The CSV file must have a column named "Requirements" with the requirements to analyze.
    - Results are saved in `Results/CSV/results.csv` and `Results/XLSX/results.xlsx` if file system access is available.
    - Conflicts are displayed in a table below.
    - Use the **Exit** button in the sidebar to stop the application, or press **Ctrl+C** in the terminal.
    """)

    # Sidebar for mode selection and exit button
    st.sidebar.header("Options")
    mode = st.sidebar.selectbox("Select Mode", ["Analyze Requirements", "Check New Requirement"])
    
    # Exit button with confirmation
    st.sidebar.markdown("### Exit Application")
    if st.sidebar.button("Exit"):
        confirm_exit = st.sidebar.checkbox("Confirm Exit (check to proceed)")
        if confirm_exit:
            st.sidebar.success("Exiting application...")
            pid = os.getpid()
            os.kill(pid, signal.SIGINT)  # Send SIGINT to kill Streamlit process

    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = None

    if mode == "Analyze Requirements":
        st.subheader("Analyze All Requirements")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file:
            # Save uploaded file temporarily
            with open("temp.csv", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if st.button("Analyze"):
                with st.spinner("Analyzing requirements..."):
                    st.session_state.results = predict_conflicts("temp.csv")
                if st.session_state.results is not None:
                    if not st.session_state.results.empty:
                        st.success("Analysis complete!")
                        st.dataframe(st.session_state.results, use_container_width=True)
                    else:
                        st.info("No conflicts found.")
                else:
                    st.error("Failed to process the file. Check logs for details.")
            
            # Clean up temp file
            if os.path.exists("temp.csv"):
                os.remove("temp.csv")

    elif mode == "Check New Requirement":
        st.subheader("Check New Requirement")
        new_requirement = st.text_area("Enter New Requirement", height=100)
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="new_req_uploader")
        
        if uploaded_file and new_requirement:
            # Save uploaded file temporarily
            with open("temp.csv", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if st.button("Check"):
                with st.spinner("Checking new requirement..."):
                    st.session_state.results = predict_conflicts("temp.csv", new_requirement=new_requirement)
                if st.session_state.results is not None:
                    if not st.session_state.results.empty:
                        st.success("Analysis complete!")
                        st.dataframe(st.session_state.results, use_container_width=True)
                    else:
                        st.info("No conflicts found with the new requirement.")
                else:
                    st.error("Failed to process the file or requirement. Check logs for details.")
            
            # Clean up temp file
            if os.path.exists("temp.csv"):
                os.remove("temp.csv")
        elif not new_requirement and uploaded_file:
            st.warning("Please enter a new requirement.")
        elif new_requirement and not uploaded_file:
            st.warning("Please upload a CSV file.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
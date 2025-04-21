import os
import argparse
import pandas as pd
import requests
import json
import logging
from tqdm import tqdm
import itertools
import time
import random
from typing import List, Tuple, Dict
from dotenv import load_dotenv

# Configuration Constants
RATE_LIMIT_DELAY = 4  # 15 RPM = 4 seconds between requests
MAX_RETRIES = 3
REQUEST_TIMEOUT = 15
BASE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

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

# Predefined conflict types
PREDEFINED_CONFLICTS = {
    "Performance", "Compliance", "Safety", "Cost", "Battery",
    "Environmental", "Structural", "Comfort", "Power Source", "Reliability",
    "Scalability", "Security", "Usability", "Maintenance", "Weight",
    "Time-to-Market", "Compatibility", "Aesthetic", "Noise", "Other",
    "Sustainability", "Regulatory", "Resource", "Technology", "Design", "Contradiction"
}

load_dotenv()

class ConflictDetector:
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = model
        self.api_url = BASE_API_URL.format(model=model)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        if not self.api_key:
            raise ValueError("Gemini API key not found in environment variables")

    def call_inference_api(self, prompt: str) -> str:
        """Call Gemini API with retry logic and rate limiting"""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    f"{self.api_url}?key={self.api_key}",
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                result = response.json()
                
                if "candidates" in result and result["candidates"]:
                    output = result["candidates"][0]["content"]["parts"][0]["text"]
                    time.sleep(RATE_LIMIT_DELAY)
                    return output.strip()
                
                logger.error(f"API error: {result.get('error', 'Unknown error')}")
                return f"Inference failed: {result.get('error', 'Unknown error')}"
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        return "Inference failed: Maximum retries exceeded"

    @staticmethod
    def parse_api_output(full_output: str) -> Tuple[str, str]:
        """Parse API output with improved validation"""
        if not full_output or "Inference failed" in full_output:
            return "Other", full_output or "No response from API"
        
        parts = [p.strip() for p in full_output.split("||") if p.strip()]
        
        if len(parts) < 2:
            return "Other", "Malformed response"
            
        conflict_type = parts[0].replace("Conflict_Type: ", "").strip()
        reason = parts[1].replace("Reason: ", "").strip()
        
        return (
            conflict_type if conflict_type in PREDEFINED_CONFLICTS else "Other",
            reason
        )

class DataHandler:
    @staticmethod
    def ensure_directories() -> Tuple[str, str]:
        """Create output directories with path validation"""
        base_dir = "Results"
        subdirs = ("CSV", "XLSX")
        
        paths = [os.path.join(base_dir, d) for d in subdirs]
        for path in paths:
            os.makedirs(path, exist_ok=True)
            
        return tuple(paths)

    @staticmethod
    def load_requirements(file_path: str) -> List[str]:
        """Load requirements from file with validation"""
        try:
            df = pd.read_csv(file_path)
            if "Requirements" not in df.columns:
                raise ValueError("Missing 'Requirements' column")
            return df["Requirements"].tolist()
        except Exception as e:
            logger.error(f"Error loading requirements: {e}")
            raise

class ConflictAnalyzer:
    def __init__(self, detector: ConflictDetector):
        self.detector = detector
        self.prompt_template = (
            "Analyze the requirements pair and identify conflicts using these types: {types}.\n"
            "Requirement 1: {req1}\nRequirement 2: {req2}\n"
            "Output format: \"Conflict_Type: <type>||Reason: <reason>\" or \"No Conflict||Compatible\""
        )

    def analyze_pair(self, req1: str, req2: str) -> Dict:
        """Analyze a single requirement pair"""
        prompt = self.prompt_template.format(
            types=", ".join(PREDEFINED_CONFLICTS),
            req1=req1,
            req2=req2
        )
        
        response = self.detector.call_inference_api(prompt)
        conflict_type, reason = self.detector.parse_api_output(response)
        
        return {
            "Requirement_1": req1,
            "Requirement_2": req2,
            "Conflict_Type": conflict_type,
            "Conflict_Reason": reason
        }

def train_model(args):
    """Optimized training workflow"""
    csv_dir, xlsx_dir = DataHandler.ensure_directories()
    
    try:
        detector = ConflictDetector()
        analyzer = ConflictAnalyzer(detector)
        
        # Load training data
        df_train = pd.read_csv(args.input_file) if os.path.exists(args.input_file) else load_sample_data()
        logger.info(f"Starting training with {len(df_train)} examples")
        
        results = []
        for iteration in range(1, args.iterations + 1):
            logger.info(f"Iteration {iteration}/{args.iterations}")
            
            for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
                result = analyzer.analyze_pair(row["Requirement_1"], row["Requirement_2"])
                result["Expected"] = row["Conflict_Type"]
                results.append(result)
            
            accuracy = sum(r["Conflict_Type"] == r["Expected"] for r in results) / len(results)
            logger.info(f"Iteration {iteration} accuracy: {accuracy:.2%}")
        
        # Save results
        output_df = pd.DataFrame(results)
        output_df.to_csv(os.path.join(csv_dir, "training_results.csv"), index=False)
        output_df.to_excel(os.path.join(xlsx_dir, "training_results.xlsx"), index=False)
        
        logger.info("Training completed successfully")
        return output_df
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def predict_conflicts(args):
    """Optimized prediction workflow with batch processing"""
    csv_dir, xlsx_dir = DataHandler.ensure_directories()
    
    try:
        detector = ConflictDetector()
        analyzer = ConflictAnalyzer(detector)
        
        requirements = DataHandler.load_requirements(args.test_file)
        pairs = list(itertools.combinations(requirements, 2))
        random.shuffle(pairs)
        
        logger.info(f"Analyzing {len(pairs)} pairs")
        results = [analyzer.analyze_pair(*pair) for pair in tqdm(pairs)]
        
        # Save results
        output_df = pd.DataFrame(results)
        output_df.to_csv(os.path.join(csv_dir, "predictions.csv"), index=False)
        output_df.to_excel(os.path.join(xlsx_dir, "predictions.xlsx"), index=False)
        
        logger.info("Prediction completed successfully")
        return output_df
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Optimized Requirements Conflict Detector")
    parser.add_argument("--mode", choices=["train", "predict", "both"], default="predict")
    parser.add_argument("--input", default="training_data.csv", help="Training data file")
    parser.add_argument("--test", default="requirements.csv", help="Test data file")
    parser.add_argument("--iterations", type=int, default=2, help="Training iterations")
    
    args = parser.parse_args()
    
    try:
        if args.mode in ["train", "both"]:
            train_model(args)
        
        if args.mode in ["predict", "both"]:
            predict_conflicts(args)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")

if __name__ == "__main__":
    main()
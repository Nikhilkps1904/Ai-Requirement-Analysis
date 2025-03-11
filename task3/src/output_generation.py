import os
import pickle
import json
import yaml
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='output_generation.log'
)
logger = logging.getLogger(__name__)

class OutputGeneration:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the OutputGeneration class with configuration."""
        self.config = self._load_config(config_path)
        self.output_dir = self.config["paths"]["output_files"]
        self.results_path = "analysis_results.pkl"
        self._create_directories()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found. Using default paths.")
            return {
                "paths": {
                    "output_files": "output_files/"
                }
            }
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Created/verified output directory: {self.output_dir}")

    def _load_results(self) -> List[Dict]:
        """Load the analysis results from the pickle file."""
        try:
            with open(self.results_path, "rb") as f:
                results = pickle.load(f)
            logger.info(f"Loaded analysis results from {self.results_path}")
            return results
        except FileNotFoundError:
            logger.error(f"Results file {self.results_path} not found.")
            raise
        except Exception as e:
            logger.error(f"Error loading results file: {e}")
            raise

    def _format_json(self, results: List[Dict]) -> List[Dict]:
        """Format the results into the required JSON structure."""
        formatted_results = []
        for result in results:
            formatted_result = {
                "Requirement ID 1": result["Requirement ID 1"],
                "Requirement 1": result["Requirement 1"],
                "Requirement ID 2": result["Requirement ID 2"],
                "Requirement 2": result["Requirement 2"],
                "Conflict": result["Conflict"],
                "Ambiguity": result["Ambiguity"]
            }
            formatted_results.append(formatted_result)
        return formatted_results

    def generate_output(self) -> None:
        """Generate the JSON output file from the analysis results."""
        logger.info("Starting output generation process...")

        # Load the results
        results = self._load_results()
        logger.info(f"Found {len(results)} requirement pairs to process")

        # Format the results into JSON structure
        formatted_results = self._format_json(results)

        # Save to JSON file
        output_file = os.path.join(self.output_dir, "conflicts_ambiguities.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Output saved to {output_file}")

def main():
    """Entry point for the output generation script."""
    try:
        generator = OutputGeneration()
        generator.generate_output()
    except Exception as e:
        logger.error(f"Output generation failed: {e}")
        raise

if __name__ == "__main__":
    main()
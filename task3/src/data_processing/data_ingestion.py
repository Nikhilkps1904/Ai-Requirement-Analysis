import os
import pandas as pd
from transformers import AutoTokenizer
import yaml
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_ingestion.log'
)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the DataIngestion class with configuration."""
        self.config = self._load_config(config_path)
        self.input_dir = self.config["paths"]["input_files"]
        self.dataset_dir = self.config["paths"]["dataset"]
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # Prepares text for RoBERTa
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
                    "input_files": "input_files/",
                    "dataset": "dataset/"
                }
            }
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.dataset_dir, exist_ok=True)
        logger.info(f"Created/verified dataset directory: {self.dataset_dir}")

    def _read_input_files(self) -> List[pd.DataFrame]:
        """Read all .xlsx and .csv files from the input directory."""
        dataframes = []
        for file in os.listdir(self.input_dir):
            if file.endswith(('.xlsx', '.csv')):
                file_path = os.path.join(self.input_dir, file)
                try:
                    if file.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    # Validate required columns
                    required_columns = ["Requirement ID", "Requirement"]
                    if not all(col in df.columns for col in required_columns):
                        logger.warning(f"File {file} missing required columns. Skipping.")
                        continue
                    dataframes.append(df)
                    logger.info(f"Successfully read {file}")
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
        return dataframes

    def _preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess text data (tokenization) for transformer compatibility."""
        df["Requirement"] = df["Requirement"].astype(str).apply(lambda x: x.strip().lower())
        df["Tokenized_Requirement"] = df["Requirement"].apply(
            lambda x: self.tokenizer.tokenize(x)
        )
        df["Encoded_Requirement"] = df["Requirement"].apply(
            lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length=128, truncation=True)
        )
        logger.info("Text preprocessing and tokenization completed.")
        return df

    def ingest_data(self) -> None:
        """Main method to ingest and process data."""
        logger.info("Starting data ingestion process...")
        dataframes = self._read_input_files()

        if not dataframes:
            logger.error("No valid input files found. Exiting.")
            raise ValueError("No valid input files to process.")

        # Combine all dataframes and remove duplicates
        combined_df = pd.concat(dataframes, ignore_index=True).drop_duplicates(subset=["Requirement ID"])
        processed_df = self._preprocess_text(combined_df)

        # Save processed data
        output_path = os.path.join(self.dataset_dir, "processed_requirements.csv")
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")

        # Optional: Save as pickle for faster loading later
        processed_df.to_pickle(os.path.join(self.dataset_dir, "processed_requirements.pkl"))
        logger.info(f"Data saved as pickle to {os.path.join(self.dataset_dir, 'processed_requirements.pkl')}")

def main():
    """Entry point for the data ingestion script."""
    try:
        ingestor = DataIngestion()
        ingestor.ingest_data()
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()
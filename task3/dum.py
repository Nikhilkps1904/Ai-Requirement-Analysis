import pandas as pd
import os

def inspect_pickle_file(file_path: str = "dataset/processed_requirements.pkl") -> None:
    """Inspect the processed_requirements.pkl file and print its shape and sequence lengths."""
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return

        # Load the pickle file
        df = pd.read_pickle(file_path)
        print(f"DataFrame shape: {df.shape}")

        # Check columns
        expected_columns = ["Requirement ID", "Requirement", "Tokenized_Requirement", "Encoded_Requirement"]
        if not all(col in df.columns for col in expected_columns):
            print(f"Error: DataFrame missing expected columns. Found columns: {df.columns.tolist()}")
            return

        # Print sequence lengths
        sequence_lengths = [len(row["Encoded_Requirement"]) for _, row in df.iterrows()]
        print(f"Sequence lengths: {sequence_lengths}")

        # Additional info
        print(f"Number of unique Requirement IDs: {df['Requirement ID'].nunique()}")
        print(f"Sample of Encoded_Requirement: {df['Encoded_Requirement'].iloc[0][:10]}")  # First 10 tokens of first row

    except pd.errors.EmptyDataError:
        print("Error: Pickle file is empty.")
    except Exception as e:
        print(f"Error loading pickle file: {e}")

if __name__ == "__main__":
    inspect_pickle_file()
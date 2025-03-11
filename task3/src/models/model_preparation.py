import os
import pickle
import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoTokenizer, T5Model
from typing import Dict, List, Tuple
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='model_preparation.log'
)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Implement positional encoding for transformer models."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    """Custom transformer model with encoder and decoder stacks."""
    def __init__(self, config: Dict, d_model: int = 768, nhead: int = 12, num_layers: int = 6):
        super(TransformerModel, self).__init__()
        self.config = config
        self.d_model = d_model
        self.embedding = nn.Embedding(50265, d_model)  # RoBERTa vocab size
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.positional_encoding = PositionalEncoding(d_model)
        self.roberta = AutoModel.from_pretrained(config["models"]["encoder"])
        self.t5 = T5Model.from_pretrained(config["models"]["decoder"])
        self.linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

        # Freeze pretrained weights initially
        for param in self.roberta.parameters():
            param.requires_grad = False
        for param in self.t5.parameters():
            param.requires_grad = False

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder with attention masks."""
        batch_size = src.size(0)
        seq_len = src.size(1)

        logger.debug(f"Forward pass: src shape={src.shape}, src_mask shape={src_mask.shape}")
        logger.debug(f"Forward pass: tgt shape={tgt.shape}, tgt_mask shape={tgt_mask.shape}")

        # Validate mask shapes
        if src_mask.size(0) != batch_size or src_mask.size(1) != seq_len:
            raise ValueError(f"src_mask shape mismatch: expected ({batch_size}, {seq_len}), got {src_mask.shape}")
        if tgt_mask.size(0) != batch_size or tgt_mask.size(1) != seq_len:
            raise ValueError(f"tgt_mask shape mismatch: expected ({batch_size}, {seq_len}), got {tgt_mask.shape}")

        # Embed source and target token IDs
        src = self.embedding(src) * (self.d_model ** 0.5)
        tgt = self.embedding(tgt) * (self.d_model ** 0.5)

        # Add positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Encoder and Decoder forward pass with masks
        memory = self.encoder(src, src_key_padding_mask=~src_mask)  # Invert mask
        output = self.decoder(tgt, memory, tgt_key_padding_mask=~tgt_mask)
        output = self.linear(output)
        output = self.softmax(output)
        return output

class RLIntegration:
    """Set up actor-critic RL framework for model refinement."""
    def __init__(self, model: TransformerModel, config: Dict):
        self.model = model
        self.config = config
        self.policy = ActorCriticPolicy
        self.rl_model = PPO(
            policy=self.policy,
            env=None,
            learning_rate=config["hyperparameters"]["learning_rate"],
            n_steps=2048,
            batch_size=config["hyperparameters"]["batch_size"],
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )
        logger.info("RL actor-critic model (PPO) initialized.")

    def save_rl_model(self, path: str) -> None:
        """Save the RL model for later use."""
        self.rl_model.save(path)
        logger.info(f"RL model saved to {path}")

class ModelPreparation:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the ModelPreparation class."""
        self.config = self._load_config(config_path)
        self.dataset_path = os.path.join(self.config["paths"]["dataset"], "processed_requirements.pkl")
        self.model_dir = self.config["paths"]["models"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["models"]["encoder"])
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
                    "models": "models/",
                    "dataset": "dataset/"
                },
                "models": {
                    "encoder": "roberta-base",
                    "decoder": "t5-base"
                },
                "hyperparameters": {
                    "learning_rate": 0.0001,
                    "batch_size": 16
                }
            }
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Created/verified model directory: {self.model_dir}")

    def _load_processed_data(self) -> pd.DataFrame:
        """Load processed data from the dataset directory."""
        try:
            with open(self.dataset_path, "rb") as f:
                df = pickle.load(f)
            logger.info(f"Loaded processed data from {self.dataset_path}")
            logger.info(f"DataFrame shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Processed data not found at {self.dataset_path}")
            raise

    def _pad_sequences(self, sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences to the same length and create attention masks."""
        batch_size = len(sequences)
        max_length = max(len(seq) for seq in sequences)
        
        padded_sequences = []
        attention_masks = []
        
        for seq in sequences:
            seq_len = len(seq)
            padded_seq = seq + [self.tokenizer.pad_token_id] * (max_length - seq_len)
            attention_mask = [1] * seq_len + [0] * (max_length - seq_len)
            padded_sequences.append(padded_seq)
            attention_masks.append(attention_mask)
        
        padded_tensor = torch.tensor(padded_sequences, dtype=torch.long)
        mask_tensor = torch.tensor(attention_masks, dtype=torch.bool)
        
        logger.debug(f"Padded sequences shape: {padded_tensor.shape}, mask shape: {mask_tensor.shape}")
        if padded_tensor.size(0) != batch_size or mask_tensor.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch in padding: expected {batch_size}, got {padded_tensor.size(0)} sequences, {mask_tensor.size(0)} masks")
        
        return padded_tensor, mask_tensor

    def prepare_model(self) -> Tuple[TransformerModel, RLIntegration]:
        """Prepare the transformer model and RL integration."""
        logger.info("Starting model preparation...")

        # Load processed data
        df = self._load_processed_data()

        # Verify DataFrame size
        logger.info(f"Number of requirements: {len(df)}")

        # Convert data to tensor format with padding
        sequences = [row["Encoded_Requirement"] for _, row in df.iterrows()]
        logger.info(f"Number of sequences to process: {len(sequences)}")

        # Use a smaller batch for the forward pass test
        batch_size = min(self.config["hyperparameters"]["batch_size"], len(sequences))
        logger.info(f"Using batch size for forward pass test: {batch_size}")
        
        # Take only the first batch_size sequences for the test
        test_sequences = sequences[:batch_size]
        src_data, src_mask = self._pad_sequences(test_sequences)
        tgt_data, tgt_mask = self._pad_sequences(test_sequences)  # Placeholder; same as src for now

        # Validate shapes
        if src_data.size(0) != src_mask.size(0) or tgt_data.size(0) != tgt_mask.size(0):
            raise ValueError(f"Batch size mismatch: src_data={src_data.size(0)}, src_mask={src_mask.size(0)}, tgt_data={tgt_data.size(0)}, tgt_mask={tgt_mask.size(0)}")

        # Initialize transformer model
        model = TransformerModel(self.config)
        logger.info("Transformer model initialized with RoBERTa and T5.")

        # Test forward pass to ensure compatibility
        try:
            with torch.no_grad():
                output = model(src_data, src_mask, tgt_data, tgt_mask)
            logger.info("Forward pass successful. Model is compatible with input data.")
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise

        # Initialize RL integration
        rl_integration = RLIntegration(model, self.config)
        logger.info("RL integration with actor-critic (PPO) set up.")

        # Save model setup
        model_path = os.path.join(self.model_dir, "prepared_model.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model state dict saved to {model_path}")

        rl_path = os.path.join(self.model_dir, "rl_model")
        rl_integration.save_rl_model(rl_path)
        logger.info(f"RL model saved to {rl_path}")

        return model, rl_integration

def main():
    """Entry point for the model preparation script."""
    try:
        preparer = ModelPreparation()
        model, rl_integration = preparer.prepare_model()
    except Exception as e:
        logger.error(f"Model preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()
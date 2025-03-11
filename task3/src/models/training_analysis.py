import os
import pickle
import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoTokenizer,T5Model
from typing import Dict, List, Tuple
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
import pandas as pd
import gym
from gym import spaces

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='training_analysis.log'
)
logger = logging.getLogger(__name__)

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
        self.linear = nn.Linear(d_model, 2)  # Output 2 classes: conflict and ambiguity
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        batch_size = src.size(0)
        seq_len = src.size(1)

        src = self.embedding(src) * (self.d_model ** 0.5)
        tgt = self.embedding(tgt) * (self.d_model ** 0.5)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        memory = self.encoder(src, src_key_padding_mask=~src_mask)
        output = self.decoder(tgt, memory, tgt_key_padding_mask=~tgt_mask)
        output = self.linear(output)
        output = self.softmax(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class ConflictAmbiguityEnv(gym.Env):
    """Custom RL environment for conflict and ambiguity detection."""
    def __init__(self, df: pd.DataFrame, model: nn.Module, tokenizer: AutoTokenizer, config: Dict):
        super(ConflictAmbiguityEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.current_idx = 0
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(2,), dtype=np.float32)  # Adjust conflict and ambiguity probs
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)  # Conflict and ambiguity probs
        self.requirement_pairs = self._generate_pairs()

    def _generate_pairs(self) -> List[Tuple[int, int]]:
        """Generate all possible pairs of requirements for conflict detection."""
        pairs = []
        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                pairs.append((i, j))
        return pairs

    def _pad_pair(self, seq1: List[int], seq2: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad two sequences to the same length and create attention masks."""
        max_length = max(len(seq1), len(seq2))
        padded_seq1 = seq1 + [self.tokenizer.pad_token_id] * (max_length - len(seq1))
        padded_seq2 = seq2 + [self.tokenizer.pad_token_id] * (max_length - len(seq2))
        mask1 = [1] * len(seq1) + [0] * (max_length - len(seq1))
        mask2 = [1] * len(seq2) + [0] * (max_length - len(seq2))
        padded_tensor = torch.tensor([padded_seq1, padded_seq2], dtype=torch.long)
        mask_tensor = torch.tensor([mask1, mask2], dtype=torch.bool)
        return padded_tensor, mask_tensor

    def reset(self) -> np.ndarray:
        """Reset the environment to the first pair."""
        self.current_idx = 0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get the current observation (conflict and ambiguity probabilities)."""
        if self.current_idx >= len(self.requirement_pairs):
            return np.zeros(2, dtype=np.float32)
        
        idx1, idx2 = self.requirement_pairs[self.current_idx]
        req1 = self.df.iloc[idx1]["Encoded_Requirement"]
        req2 = self.df.iloc[idx2]["Encoded_Requirement"]
        
        src, src_mask = self._pad_pair(req1, req2)
        tgt, tgt_mask = self._pad_pair(req1, req2)  # Placeholder
        
        with torch.no_grad():
            output = self.model(src, src_mask, tgt, tgt_mask)
        conflict_prob = output[0, 0, 0].item()  # First token, conflict prob
        ambiguity_prob = output[0, 0, 1].item()  # First token, ambiguity prob
        return np.array([conflict_prob, ambiguity_prob], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take an action (adjust logits) and return the next observation, reward, done, and info."""
        idx1, idx2 = self.requirement_pairs[self.current_idx]
        
        # Simulate ground truth (for demo; replace with actual labels)
        ground_truth_conflict = 1 if (idx1 == 0 and idx2 == 1) else 0  # R1 and R2 conflict (speed vs. accuracy)
        ground_truth_ambiguity = 0  # Placeholder

        # Get current observation
        obs = self._get_observation()
        conflict_prob, ambiguity_prob = obs

        # Apply action (adjust probabilities)
        adjusted_conflict_prob = conflict_prob + action[0]
        adjusted_ambiguity_prob = ambiguity_prob + action[1]
        adjusted_conflict_prob = np.clip(adjusted_conflict_prob, 0, 1)
        adjusted_ambiguity_prob = np.clip(adjusted_ambiguity_prob, 0, 1)

        # Compute reward
        predicted_conflict = 1 if adjusted_conflict_prob > 0.5 else 0
        predicted_ambiguity = 1 if adjusted_ambiguity_prob > 0.5 else 0
        reward = (self.config["rl"]["reward_correct"] if predicted_conflict == ground_truth_conflict else self.config["rl"]["reward_incorrect"]) + \
                 (self.config["rl"]["reward_correct"] if predicted_ambiguity == ground_truth_ambiguity else self.config["rl"]["reward_incorrect"])

        # Move to next pair
        self.current_idx += 1
        done = self.current_idx >= len(self.requirement_pairs)
        next_obs = self._get_observation() if not done else np.zeros(2, dtype=np.float32)
        info = {"idx1": idx1, "idx2": idx2, "conflict": predicted_conflict, "ambiguity": predicted_ambiguity}

        return next_obs, reward, done, info

    def render(self, mode='human'):
        pass

class TrainingAnalysis:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the TrainingAnalysis class."""
        self.config = self._load_config(config_path)
        self.dataset_path = os.path.join(self.config["paths"]["dataset"], "processed_requirements.pkl")
        self.model_dir = self.config["paths"]["models"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["models"]["encoder"])
        self._load_data_and_model()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found.")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    def _load_data_and_model(self) -> None:
        """Load processed data and prepared model."""
        try:
            with open(self.dataset_path, "rb") as f:
                self.df = pickle.load(f)
            logger.info(f"Loaded processed data from {self.dataset_path}")
        except FileNotFoundError:
            logger.error(f"Processed data not found at {self.dataset_path}")
            raise

        # Load the transformer model
        self.model = TransformerModel(self.config)
        model_path = os.path.join(self.model_dir, "prepared_model.pth")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        logger.info(f"Loaded transformer model from {model_path}")

        # Load the RL model
        rl_path = os.path.join(self.model_dir, "rl_model")
        self.rl_model = PPO.load(rl_path)
        logger.info(f"Loaded RL model from {rl_path}")

    def train_and_analyze(self) -> List[Dict]:
        """Fine-tune the transformer, train RL, and analyze requirements."""
        logger.info("Starting training and analysis...")

        # Step 1: Fine-tune the transformer (placeholder; assumes labeled data)
        # For demo, we'll skip fine-tuning and use the pretrained model directly

        # Step 2: Set up RL environment and train
        env = DummyVecEnv([lambda: ConflictAmbiguityEnv(self.df, self.model, self.tokenizer, self.config)])
        self.rl_model.set_env(env)
        self.rl_model.learn(total_timesteps=10000)  # Adjust timesteps for larger dataset
        logger.info("RL training completed.")

        # Step 3: Analyze requirements using the trained RL model
        results = []
        obs = env.reset()
        done = False
        while not done:
            action, _ = self.rl_model.predict(obs)
            obs, reward, done, info = env.step(action)
            if "idx1" in info[0]:
                idx1, idx2 = info[0]["idx1"], info[0]["idx2"]
                conflict = info[0]["conflict"]
                ambiguity = info[0]["ambiguity"]
                req1 = self.df.iloc[idx1]["Requirement"]
                req2 = self.df.iloc[idx2]["Requirement"]
                results.append({
                    "Requirement ID 1": self.df.iloc[idx1]["Requirement ID"],
                    "Requirement 1": req1,
                    "Requirement ID 2": self.df.iloc[idx2]["Requirement ID"],
                    "Requirement 2": req2,
                    "Conflict": bool(conflict),
                    "Ambiguity": bool(ambiguity)
                })
                logger.info(f"Analyzed pair ({idx1}, {idx2}): Conflict={conflict}, Ambiguity={ambiguity}")

        # Save RL model after training
        rl_path = os.path.join(self.model_dir, "trained_rl_model")
        self.rl_model.save(rl_path)
        logger.info(f"Trained RL model saved to {rl_path}")

        return results

def main():
    """Entry point for the training and analysis script."""
    try:
        trainer = TrainingAnalysis()
        results = trainer.train_and_analyze()
        # Save results for the next step
        with open("analysis_results.pkl", "wb") as f:
            pickle.dump(results, f)
        logger.info("Analysis results saved to analysis_results.pkl")
    except Exception as e:
        logger.error(f"Training and analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
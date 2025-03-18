import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import json

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

# Dataset Class
class RequirementConflictDataset(Dataset):
    """Dataset for requirement conflict detection"""
    
    def __init__(self, dataframe, tokenizer, max_input_length=128, max_target_length=64):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        input_text = f"Requirement: {row['Requirement_1']} | {row['Requirement_2']}"
        target_text = row["Conflict_Type"]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        target_ids = targets.input_ids.squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_ids
        }

# Load Sample Data
def load_sample_data():
    """Load initial sample training data"""
    train_data = [
        # Performance Conflict
        ("The vehicle must achieve a fuel efficiency of at least 50 km/l.",
         "The engine should have a minimum power output of 25 HP.",
         "Performance Conflict"),
        # Compliance Conflict
        ("The bike should include an always-on headlight for safety compliance.",
         "Users should be able to turn off the headlight manually.",
         "Compliance Conflict"),
        # Safety vs. Usability Conflict
        ("The car's doors should automatically lock when the vehicle is in motion.",
         "Passengers must be able to open the doors at any time.",
         "Safety Conflict"),
        # Cost vs. Material Quality Conflict
        ("The vehicle must be made from high-strength carbon fiber for durability.",
         "The manufacturing cost should not exceed $500 per unit.",
         "Cost Conflict"),
        # Battery vs. Weight Conflict
        ("The electric bike should have a battery range of 200 km on a single charge.",
         "The total weight of the bike should not exceed 100 kg.",
         "Battery Conflict"),
    ]
    
    # Convert to DataFrame
    return pd.DataFrame(train_data, columns=["Requirement_1", "Requirement_2", "Conflict_Type"])

# Data Augmentation
def augment_dataset(df, multiplier=2):
    """Augment dataset with simple techniques"""
    original_len = len(df)
    augmented_data = []
    
    # Simple word replacements for augmentation
    replacements = {
        "vehicle": ["machine", "transport", "automobile", "product"],
        "should": ["must", "needs to", "is required to", "has to"],
        "must": ["should", "is required to", "needs to", "has to"],
        "bike": ["motorcycle", "two-wheeler", "moped", "scooter"],
        "car": ["vehicle", "automobile", "transport", "machine"]
    }
    
    # 1. Swap requirement order
    for _, row in df.iterrows():
        augmented_data.append([row["Requirement_2"], row["Requirement_1"], row["Conflict_Type"]])
    
    # 2. Word replacements
    for _, row in df.iterrows():
        req1 = row["Requirement_1"]
        req2 = row["Requirement_2"]
        
        # Apply random replacements
        for word, alternates in replacements.items():
            if word in req1.lower():
                if np.random.random() > 0.5:
                    req1 = req1.replace(word, np.random.choice(alternates))
            if word in req2.lower():
                if np.random.random() > 0.5:
                    req2 = req2.replace(word, np.random.choice(alternates))
        
        augmented_data.append([req1, req2, row["Conflict_Type"]])
    
    # Combine and shuffle
    augmented_df = pd.DataFrame(augmented_data, columns=["Requirement_1", "Requirement_2", "Conflict_Type"])
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    logger.info(f"Augmented dataset from {original_len} to {len(combined_df)} examples")
    return combined_df

# Training Function
def train_model(args):
    """Train the conflict detection model"""
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    if os.path.exists(args.input_file):
        logger.info(f"Loading data from {args.input_file}")
        df_train = pd.read_csv(args.input_file)
    else:
        logger.warning(f"Input file {args.input_file} not found. Using sample data.")
        df_train = load_sample_data()
    
    # Augment dataset if needed
    if args.augment:
        df_train = augment_dataset(df_train, args.augment_multiplier)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df_train, test_size=args.validation_split, random_state=42)
    logger.info(f"Training on {len(train_df)} examples, validating on {len(val_df)} examples")
    
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    
    # Create datasets
    train_dataset = RequirementConflictDataset(train_df, tokenizer)
    val_dataset = RequirementConflictDataset(val_df, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    training_history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        training_history["train_loss"].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        training_history["val_loss"].append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            logger.info(f"Validation loss decreased from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
                json.dump(training_history, f)
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return model, tokenizer

# Prediction Function
def predict_conflicts(args, model=None, tokenizer=None):
    """Predict conflicts with structured output"""
    device = torch.device(args.device)
    
    if model is None or tokenizer is None:
        try:
            logger.info(f"Loading model from {args.output_dir}")
            model = T5ForConditionalGeneration.from_pretrained(args.output_dir).to(device)
            tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
    
    try:
        df_input = pd.read_csv(args.test_file)
        logger.info(f"Loaded {len(df_input)} requirement pairs")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return

    # Predefined conflict types for validation
    PREDEFINED_CONFLICTS = {
        "Performance Conflict", "Compliance Conflict", "Safety Conflict",
        "Cost Conflict", "Battery Conflict", "Environmental Conflict",
        "Structural Conflict", "Comfort Conflict", "Power Source Conflict",
        "Space Conflict", "Technology Conflict", "Material Conflict",
        "Regulatory Conflict", "Design Conflict", "Resource Conflict",
        "Other"
    }

    results = []
    
    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Analyzing conflicts"):
        req1 = row["Requirement_1"]
        req2 = row["Requirement_2"]
        
        # Create structured prompt
        input_text = (
            f"Analyze requirements conflict: {req1} AND {req2} "
            "Generate output format: "
            "Conflict_Type||Conflict_Reason||Resolution_Suggestion:"
        )
        
        # Generate prediction
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=256,
                num_beams=5,
                early_stopping=True
            )
        
        # Parse and validate output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        parts = full_output.split("||")
        
        # Default values
        conflict_type = "Other"
        conflict_reason = "Needs manual analysis"
        resolution = "Requires engineering review"
        
        # Extract components
        if len(parts) >= 1:
            conflict_type = parts[0].strip()
            if conflict_type not in PREDEFINED_CONFLICTS:
                conflict_type = "Other"
        
        if len(parts) >= 2:
            conflict_reason = parts[1].strip()
        
        if len(parts) >= 3:
            resolution = "||".join(parts[2:]).strip()  # Handle multi-part resolutions
        
        results.append({
            "Requirement_1": req1,
            "Requirement_2": req2,
            "Conflict_Type": conflict_type,
            "Conflict_Reason": conflict_reason,
            "Resolution_Suggestion": resolution
        })
    
    # Save results
    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output_file, index=False)
    logger.info(f"Analysis complete. Results saved to {args.output_file}")
    return output_df

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Requirements Conflict Detection")
    
    # General arguments
    parser.add_argument("--mode", type=str, choices=["train", "predict", "both"], default="both",
                        help="Mode: train, predict, or both (default: both)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (default: cuda if available, otherwise cpu)")
    
    # Training arguments
    parser.add_argument("--input_file", type=str, default="training_data.csv",
                        help="Path to training data CSV file (default: training_data.csv)")
    parser.add_argument("--output_dir", type=str, default="./trained_model",
                        help="Directory to save the trained model (default: ./trained_model)")
    parser.add_argument("--model_name", type=str, default="t5-base",
                        help="Base model name (default: t5-base)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate (default: 3e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer (default: 0.01)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping (default: 1.0)")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="Fraction of data to use for validation (default: 0.2)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (default: 3)")
    parser.add_argument("--augment", action="store_true",
                        help="Augment training data (default: False)")
    parser.add_argument("--augment_multiplier", type=int, default=2,
                        help="Data augmentation multiplier (default: 2)")
    
    # Prediction arguments
    parser.add_argument("--test_file", type=str, default="./TwoWheeler_Requirement_Conflicts.csv",
                        help="Path to test data CSV file (default: ./TwoWheeler_Requirement_Conflicts.csv)")
    parser.add_argument("--output_file", type=str, default="conflict_results.csv",
                        help="Path to output results CSV file (default: conflict_results.csv)")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        model, tokenizer = train_model(args)
        logger.info("Model training completed!")
    
    if args.mode in ["predict", "both"]:
        if args.mode == "both":
            predict_conflicts(args, model, tokenizer)
        else:
            predict_conflicts(args)
        logger.info("Prediction completed!")

if __name__ == "__main__":
    main()
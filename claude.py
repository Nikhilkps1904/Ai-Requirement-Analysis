import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback
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

class RequirementConflictDataset(Dataset):
    """Dataset for requirement conflict detection"""
    
    def __init__(self, dataframe, tokenizer, max_input_length=128, max_target_length=32):
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

        # Environmental vs. Performance Conflict
        ("The engine should meet the latest Euro 6 emission standards.",
         "The engine must provide a top speed of 220 km/h.",
         "Environmental Conflict"),

        # Structural Integrity Conflict
        ("The vehicle should be lightweight for better fuel efficiency.",
         "The vehicle must withstand crashes up to 80 km/h impact force.",
         "Structural Conflict"),

        # Comfort vs. Aerodynamics Conflict
        ("The seats should have extra thick padding for comfort.",
         "The vehicle should be designed for maximum aerodynamics.",
         "Comfort Conflict"),

        # Power Source Conflict
        ("The car should use only renewable energy sources.",
         "The vehicle must include a backup gasoline engine.",
         "Power Source Conflict"),

        # Space vs. Passenger Comfort Conflict
        ("The car should have a large trunk space for luggage.",
         "The rear passenger legroom should be maximized.",
         "Space Conflict"),
         
        # Additional examples for expanded dataset
        ("The two-wheeler must have a maximum speed of 120 km/h.",
         "The battery must be small enough to fit under the seat.",
         "Performance Conflict"),
         
        ("The vehicle must comply with all noise regulations.",
         "The engine must produce a distinctive sound for brand recognition.",
         "Compliance Conflict"),
         
        ("The bike should have regenerative braking to maximize range.",
         "The braking system should be as simple as possible for reliability.",
         "Battery Conflict"),
         
        ("The suspension should be tuned for off-road performance.",
         "The ride height should be optimized for highway aerodynamics.",
         "Performance Conflict"),
         
        ("The vehicle should have a quick-release battery for easy charging.",
         "The battery should be securely locked to prevent theft.",
         "Safety Conflict"),
         
        ("The helmet storage compartment should accommodate all helmet sizes.",
         "The rear design should be sleek with minimal protrusions.",
         "Space Conflict"),
         
        ("The vehicle must use eco-friendly manufacturing processes.",
         "Production costs must be kept below industry average.",
         "Cost Conflict"),
         
        ("The two-wheeler should include a premium sound system.",
         "The electrical system should prioritize motor performance.",
         "Power Source Conflict"),
         
        ("The frame should be made from recycled materials.",
         "The frame must last for at least 10 years of daily use.",
         "Environmental Conflict"),
         
        ("The suspension should provide maximum comfort on rough roads.",
         "The vehicle should have minimal body roll during cornering.",
         "Comfort Conflict")
    ]
    
    # Convert to DataFrame
    return pd.DataFrame(train_data, columns=["Requirement_1", "Requirement_2", "Conflict_Type"])

def augment_dataset(df, multiplier=2):
    """
    Augment dataset with simple techniques:
    - Switching order of requirements
    - Adding minor variations in wording
    """
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
    
    # 1. Swap requirement order (conflict type remains the same)
    for _, row in df.iterrows():
        augmented_data.append([
            row["Requirement_2"],
            row["Requirement_1"],
            row["Conflict_Type"]
        ])
    
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
        
        augmented_data.append([
            req1, 
            req2,
            row["Conflict_Type"]
        ])
    
    # Convert augmented data to DataFrame
    augmented_df = pd.DataFrame(augmented_data, columns=["Requirement_1", "Requirement_2", "Conflict_Type"])
    
    # Combine original and augmented data
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    # Shuffle the data
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    logger.info(f"Augmented dataset from {original_len} to {len(combined_df)} examples")
    return combined_df

def train_model(args):
    """Train the conflict detection model"""
    # Set device
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Training metrics
    best_val_loss = float('inf')
    early_stop_counter = 0
    training_history = {
        "train_loss": [],
        "val_loss": []
    }
    
    # Training loop
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
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
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        training_history["val_loss"].append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Check for best model
        if avg_val_loss < best_val_loss:
            logger.info(f"Validation loss decreased from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            
            # Save best model
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            # Save training history
            with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
                json.dump(training_history, f)
        else:
            early_stop_counter += 1
            logger.info(f"Validation loss did not decrease. Early stopping counter: {early_stop_counter}/{args.patience}")
            
            if early_stop_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return model, tokenizer

def predict_conflicts(args, model=None, tokenizer=None):
    """Predict conflicts for new requirement pairs"""
    # Set device
    device = torch.device(args.device)
    
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        try:
            logger.info(f"Loading model from {args.output_dir}")
            model = T5ForConditionalGeneration.from_pretrained(args.output_dir).to(device)
            tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
    
    # Check if input file exists
    if not os.path.exists(args.test_file):
        logger.error(f"Input file {args.test_file} not found.")
        return
    
    # Load input file
    try:
        df_input = pd.read_csv(args.test_file)
        logger.info(f"Loaded {len(df_input)} requirement pairs for prediction")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return
    
    # Detect conflicts
    conflict_results = []
    
    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Predicting conflicts"):
        req1, req2 = row["Requirement_1"], row["Requirement_2"]
        input_text = f"Requirement: {req1} | {req2}"
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        ).to(device)
        
        # Generate output
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=32,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode output
        conflict_type = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Append to results
        conflict_results.append({
            "Requirement_1": req1,
            "Requirement_2": req2,
            "Conflict_Type": conflict_type
        })
    
    # Convert to DataFrame
    output_df = pd.DataFrame(conflict_results)
    
    # Save results
    output_df.to_csv(args.output_file, index=False)
    logger.info(f"Conflict detection completed! Results saved to {args.output_file}")
    
    return output_df

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
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
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
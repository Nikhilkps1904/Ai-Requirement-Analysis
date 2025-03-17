import os
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model directory
MODEL_DIR = "./trained_model"

# Sample Training Data (Modify with real dataset)
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
     "Space Conflict")
]


# Convert to DataFrame
df_train = pd.DataFrame(train_data, columns=["Requirement_1", "Requirement_2", "Conflict_Type"])

# Initialize Model & Tokenizer
model_name = "t5-base"  # You can use "t5-base" for better results
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Prepare Training Data
train_inputs = []
train_labels = []

for _, row in df_train.iterrows():
    input_text = f"Requirement: {row['Requirement_1']} | {row['Requirement_2']}"
    label_text = row["Conflict_Type"]

    input_enc = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    label_enc = tokenizer(label_text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)

    train_inputs.append((input_enc.input_ids.to(device), label_enc.input_ids.to(device)))

# Train Model (10 Epochs)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    for input_ids, label_ids in train_inputs:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=label_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_inputs)}")

# Save Model & Tokenizer
os.makedirs(MODEL_DIR, exist_ok=True)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Model saved to {MODEL_DIR}")

# Load Model for Inference
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
print("Model loaded successfully!")

# **Hardcoded File Path**
INPUT_FILE_PATH = "./TwoWheeler_Requirement_Conflicts.csv"  # Update this path if needed

# Check if file exists
if not os.path.exists(INPUT_FILE_PATH):
    print(f"Error: File '{INPUT_FILE_PATH}' not found.")
    exit(1)

# Load Input File for Conflict Detection
df_input = pd.read_csv(INPUT_FILE_PATH)

# Detect Conflicts
conflict_results = []
for _, row in df_input.iterrows():
    req1, req2 = row["Requirement_1"], row["Requirement_2"]
    input_text = f"Requirement: {req1} | {req2}"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    conflict_result = tokenizer.decode(output[0], skip_special_tokens=True)

    conflict_results.append([req1, req2, conflict_result])

# Save Results
output_df = pd.DataFrame(conflict_results, columns=["Requirement_1", "Requirement_2", "Conflict_Result"])
output_df.to_csv("conflict_results.csv", index=False)

print(f"Conflict detection completed! Results saved to 'conflict_results.csv'.")

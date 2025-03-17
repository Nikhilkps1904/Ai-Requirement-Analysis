import os
import pandas as pd
import torch
import tkinter as tk
from tkinter import filedialog
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model directory
MODEL_DIR = "./trained_model"

# Sample Training Data (Modify with real dataset)
train_data = [
    ("The vehicle must achieve a fuel efficiency of at least 50 km/l.",
     "The engine should have a minimum power output of 25 HP.",
     "Performance Conflict"),
    
    ("The bike should include an always-on headlight for safety compliance.",
     "Users should be able to turn off the headlight manually.",
     "Compliance Conflict")
]

# Convert to DataFrame
df_train = pd.DataFrame(train_data, columns=["Requirement_1", "Requirement_2", "Conflict_Type"])

# Initialize Model & Tokenizer
model_name = "t5-small"  # You can use "t5-base" for better results
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

# Train Model (Increased Epochs to 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
epochs = 10  # Increased from 3 to 10
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

# File Upload Pop-up
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Requirements CSV File", filetypes=[("CSV files", "*.csv")])
    return file_path

# Get Input File from User
input_file = select_file()
if not input_file:
    print("No file selected. Exiting...")
    exit()

df_input = pd.read_csv(input_file)

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
output_file = "conflict_results.csv"
output_df = pd.DataFrame(conflict_results, columns=["Requirement_1", "Requirement_2", "Conflict_Result"])
output_df.to_csv(output_file, index=False)

print(f"Conflict detection completed! Results saved to '{output_file}'.")

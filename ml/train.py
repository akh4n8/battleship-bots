import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CognitiveBattleshipDataset
from model import CognitiveBattleshipModel, export_to_onnx

# ==========================================
# HYPERPARAMETERS & PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'Battleship_Export_20260729.csv')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'moriarty_defense_best.pth')
ONNX_EXPORT_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'moriarty_defense.onnx')
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

def train_model():
    print(f"Loading Universal Dataset from {CSV_FILE}...")
    dataset = CognitiveBattleshipDataset(
        csv_file_path=CSV_FILE, 
        is_offense_model=False # (True for train_offense.py)
    )
    
    # 80/20 Train-Test Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset loaded. Training samples: {train_size}, Validation samples: {val_size}")

    # Initialize Model, Loss, and Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = CognitiveBattleshipModel(in_channels=7, d_model=256, n_heads=4, n_layers=2).to(device)
    
    # CRITICAL: We use BCEWithLogitsLoss because placing a fleet covers 17 different cells.
    # Standard CrossEntropy is for single-cell targets; BCE allows independent probability per cell.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_loss = float('inf')

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        for batch_seq, batch_labels in train_loader:
            batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: outputs are 100 raw logits
            logits = model(batch_seq) 
            
            # Flatten the labels to [Batch, 100] to match logits
            batch_labels_flat = batch_labels.view(batch_labels.size(0), -1)
            
            # Calculate Loss
            loss = criterion(logits, batch_labels_flat)
            
            # Backward pass & optimize
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ==========================================
        # VALIDATION PHASE
        # ==========================================
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_seq, val_labels in val_loader:
                val_seq, val_labels = val_seq.to(device), val_labels.to(device)
                logits = model(val_seq)
                val_labels_flat = val_labels.view(val_labels.size(0), -1)
                loss = criterion(logits, val_labels_flat)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Saved new best model to {MODEL_SAVE_PATH}!")

    print("\nTraining Complete!")
    
    # ==========================================
    # EXPORT THE FINAL PIPELINE
    # ==========================================
    print("Loading best weights for ONNX export...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to('cpu') # Move to CPU for static export
    export_to_onnx(model, model_name=ONNX_EXPORT_PATH, in_channels=7)

if __name__ == "__main__":
    train_model()

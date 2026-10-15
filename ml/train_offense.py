import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CognitiveBattleshipDataset, get_grouped_train_test_split
from model import CognitiveBattleshipModel, export_to_onnx

# ==========================================
# HYPERPARAMETERS & PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_FILE = os.path.join(DATA_DIR, 'Battleship_Export_20260729.csv')
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

def train_personalized_model(target_player=None, only_wins=False, epochs=EPOCHS, batch_size=BATCH_SIZE):
    print(f"\n======================================")
    print(f"TRAINING MODEL: {target_player if target_player else 'Universal'} | Wins Only: {only_wins}")
    print(f"======================================")
    
    dataset = CognitiveBattleshipDataset(
        csv_file_path=CSV_FILE, 
        is_offense_model=True,
        target_player=target_player,
        only_wins=only_wins
    )
    
    if len(dataset) == 0:
        print(f"No sufficient data found for {target_player}. Skipping.")
        return
        
    train_dataset, val_dataset = get_grouped_train_test_split(dataset, test_ratio=0.2, seed=42)
    
    print(f"Dataset loaded. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CognitiveBattleshipModel(in_channels=5, d_model=256, n_heads=4, n_layers=2).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_val_loss = float('inf')
    
    # Use smaller batch size if dataset is tiny
    bs = min(BATCH_SIZE, max(1, len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    val_bs = min(BATCH_SIZE, max(1, len(val_dataset)))
    val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False) if len(val_dataset) > 0 else []

    player_str = target_player if target_player else 'universal'
    win_str = 'wins' if only_wins else 'all'
    best_model_name = os.path.join(DATA_DIR, f'moriarty_{player_str}_{win_str}_offense_best.pth')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch_seq, batch_labels in train_loader:
            batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_seq) 
            batch_labels_flat = batch_labels.view(batch_labels.size(0), -1)
            loss = criterion(logits, batch_labels_flat)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        model.eval()
        total_val_loss = 0.0
        if val_loader:
            with torch.no_grad():
                for val_seq, val_labels in val_loader:
                    val_seq, val_labels = val_seq.to(device), val_labels.to(device)
                    logits = model(val_seq)
                    val_labels_flat = val_labels.view(val_labels.size(0), -1)
                    loss = criterion(logits, val_labels_flat)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
        else:
            avg_val_loss = avg_train_loss
            
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_name)
            # If universal, also save standard alias moriarty_offense_best.pth
            if target_player is None:
                torch.save(model.state_dict(), os.path.join(DATA_DIR, 'moriarty_offense_best.pth'))
            print(f"  -> Saved new best model to {best_model_name}!")

    print(f"\nTraining Complete for {player_str} ({win_str})! Exporting ONNX...")
    model.load_state_dict(torch.load(best_model_name))
    model.to('cpu') 
    
    onnx_name = os.path.join(DATA_DIR, f"moriarty_{player_str}_{win_str}_offense.onnx")
    try:
        export_to_onnx(model, model_name=onnx_name, in_channels=5, seq_len=6)
        if target_player is None:
            standard_onnx = os.path.join(DATA_DIR, "moriarty_offense.onnx")
            export_to_onnx(model, model_name=standard_onnx, in_channels=5, seq_len=6)
            print(f"[SUCCESS] Exported standard production ONNX to {standard_onnx}")
    except Exception as e:
        print(f"ONNX export failed, but .pth is saved. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Moriarty Offense Model")
    parser.add_argument("--player", type=str, default=None, help="Target player name (default: None for Universal model)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--only-wins", action="store_true", help="Filter for player wins only (discouraged per AGENTS.md)")
    args = parser.parse_args()

    train_personalized_model(
        target_player=args.player,
        only_wins=args.only_wins,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
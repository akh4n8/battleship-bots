import os
import torch
import numpy as np
import time
from torch.utils.data import random_split
from dataset import CognitiveBattleshipDataset, SHIP_WEIGHTS
from model import CognitiveBattleshipModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'Battleship_Export_20260729.csv')

def simulate_game(model, history_sequence, true_target_ships):
    """
    Simulates a game from Turn 1 until all 17 ship pegs are sunk.
    Returns the number of turns taken to win.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    live_bot_misses = np.zeros((10, 10), dtype=np.float32)
    live_bot_hits = np.zeros((10, 10), dtype=np.float32)
    
    shot_mask = np.zeros((100,), dtype=bool)
    hits_found = 0
    total_pegs = 17
    turns = 0
    
    base_trauma_misses = history_sequence[-1, 1, :, :]
    base_trauma_hits = history_sequence[-1, 2, :, :]
    
    with torch.no_grad():
        while hits_found < total_pegs and turns < 100:
            turns += 1
            
            live_step = np.stack([
                np.zeros((10, 10), dtype=np.float32), 
                base_trauma_misses, 
                base_trauma_hits, 
                live_bot_misses, 
                live_bot_hits
            ])
            
            seq_tensor = np.concatenate([history_sequence, live_step[np.newaxis, ...]], axis=0)
            seq_tensor = torch.tensor(seq_tensor, dtype=torch.float32).unsqueeze(0).to(device)
            
            logits = model(seq_tensor)[0].cpu().numpy() # [100]
            logits[shot_mask] = -1e9
            
            best_idx = np.argmax(logits)
            shot_mask[best_idx] = True
            
            x = best_idx // 10
            y = best_idx % 10
            
            if true_target_ships[x, y] > 0:
                hits_found += 1
                live_bot_hits[x, y] += true_target_ships[x, y]
            else:
                live_bot_misses[x, y] += 1.0
                
    return turns

def evaluate_models():
    print(f"Loading Dataset for Evaluation...")
    from dataset import get_grouped_train_test_split
    dataset = CognitiveBattleshipDataset(csv_file_path=CSV_FILE, is_offense_model=True)
    
    _, val_dataset = get_grouped_train_test_split(dataset, test_ratio=0.2, seed=42)
    
    unique_target_games = []
    seen_labels = []
    
    print(f"Extracting Turn 0 sequences from {len(val_dataset)} test games...")
    for i in range(len(val_dataset)):
        seq, label = val_dataset[i]
        
        label_hash = hash(label.numpy().tobytes())
        if label_hash not in seen_labels:
            seen_labels.append(label_hash)
            history_seq = seq[:-1].numpy()
            unique_target_games.append((history_seq, label.numpy()))
                
    print(f"Found {len(unique_target_games)} unique games to simulate.")
    
    weights_file = os.path.join(SCRIPT_DIR, '..', 'data', 'moriarty_offense_best.pth')
    models_to_test = {
        "CognitiveTransformer": CognitiveBattleshipModel(in_channels=5)
    }
    weights = {
        "CognitiveTransformer": weights_file
    }
    
    for name, model in models_to_test.items():
        try:
            model.load_state_dict(torch.load(weights[name]))
            print(f"Loaded weights for {name}")
        except Exception as e:
            print(f"Could not load weights for {name}: {e}")
        
    results = {}
    
    for name, model in models_to_test.items():
        print(f"\nEvaluating {name}...")
        start_time = time.time()
        turns_list = []
        
        for idx, (hist, label) in enumerate(unique_target_games):
            turns = simulate_game(model, hist, label)
            turns_list.append(turns)
            if (idx+1) % 10 == 0:
                print(f"  Simulated {idx+1}/{len(unique_target_games)} games (Avg Turns: {np.mean(turns_list):.2f})")
                
        avg_turns = np.mean(turns_list)
        perfect_games = sum(1 for t in turns_list if t == 17)
        sub_35_games = sum(1 for t in turns_list if t <= 35)
        
        results[name] = {
            "Avg Turns": avg_turns,
            "Perfect Games (17 turns)": perfect_games,
            "Fast Wins (<= 35 turns)": f"{sub_35_games}/{len(unique_target_games)}",
            "Time (s)": time.time() - start_time
        }
        
    print("\n==============================")
    print("🏁 RACE RESULTS (TURNS TO WIN) 🏁")
    print("==============================")
    for name, res in results.items():
        print(f"Model: {name}")
        print(f"  Average Turns to Win: {res['Avg Turns']:.2f}")
        print(f"  Perfect Games (17-0): {res['Perfect Games (17 turns)']}")
        print(f"  Fast Wins (<= 35)   : {res['Fast Wins (<= 35 turns)']}")
        print(f"  Time taken          : {res['Time (s)']:.2f}s")
        print("------------------------------")
        
if __name__ == "__main__":
    evaluate_models()

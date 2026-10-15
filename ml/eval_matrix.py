import os
import sys
import time
import concurrent.futures
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import CognitiveBattleshipDataset, get_grouped_train_test_split, SHIP_WEIGHTS
from model import CognitiveBattleshipModel
from bots.mycroft import MycroftBot
from bots.sherlock import SherlockBot
from core.game_engine import BattleshipGame

# ==============================================================================
# ALGORITHMIC TOGGLES (THE 4 LEVELS OF FIDELITY)
# ==============================================================================
# 1. Pure NN: Argmax of heatmap
# 2. NN + Parity: Argmax of checkerboard-masked heatmap
# 3. NN + Assassin + Parity: Same as above, but uses SherlockBot for open hits
# 4. Full Fidelity: Same as above, but falls back to MycroftBot if confidence < 1.3

def get_pure_nn_target(logits, shot_mask):
    logits = logits.copy()
    logits[shot_mask] = -1e9
    best_idx = np.argmax(logits)
    return (best_idx // 10, best_idx % 10)

def get_parity_nn_target(logits, shot_mask, game_id):
    logits = logits.copy()
    logits[shot_mask] = -1e9
    
    # Parity Sweep (checkerboard)
    parity_offset = game_id % 2
    for x in range(10):
        for y in range(10):
            if (x + y) % 2 != parity_offset:
                idx = x * 10 + y
                logits[idx] -= 1e6 # Heavily penalize non-checkerboard, but allow it if no other options

    best_idx = np.argmax(logits)
    return (best_idx // 10, best_idx % 10)

def simulate_game(game_id, ships_list, history_sequence, model_weights_path, algo_level):
    """
    Worker function to simulate a single game.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CognitiveBattleshipModel(in_channels=5, d_model=256, n_heads=4, n_layers=2).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    game = BattleshipGame()
    # Inject exact historical ships
    for ship in ships_list:
        ship_id = ship['id']
        length = len(ship['coords'])
        game.ship_pegs[ship_id] = []
        game.ship_health[ship_id] = length
        for (x, y) in ship['coords']:
            game.truth_board[x][y] = ship_id
            game.ship_pegs[ship_id].append((x, y))

    live_bot_misses = np.zeros((10, 10), dtype=np.float32)
    live_bot_hits = np.zeros((10, 10), dtype=np.float32)
    
    shot_mask = np.zeros((100,), dtype=bool)
    base_trauma_misses = history_sequence[-1, 1, :, :]
    base_trauma_hits = history_sequence[-1, 2, :, :]
    
    sherlock_bot = SherlockBot()
    mycroft_bot = MycroftBot()
    
    with torch.no_grad():
        while not game.is_game_over() and game.shot_number < 100:
            # Check for Assassin Mode (Open Hits)
            open_hits = False
            for r in range(10):
                for c in range(10):
                    if game.tracking_board[r][c] == 1: # CELL_HIT
                        open_hits = True
                        break
            
            target = None
            
            # --- ALGO LEVEL 3/4: Assassin Mode ---
            if open_hits and algo_level >= 3:
                target_data = sherlock_bot.get_best_move(game.tracking_board, game.move_history)
                if target_data is not None:
                    target = target_data[0]
                # If Sherlock returns None, fallback below
            
            # --- GET NN HEATMAP ---
            if target is None:
                live_step = np.stack([
                    np.zeros((10, 10), dtype=np.float32), 
                    base_trauma_misses, 
                    base_trauma_hits, 
                    live_bot_misses, 
                    live_bot_hits
                ])
                seq_tensor = np.concatenate([history_sequence, live_step[np.newaxis, ...]], axis=0)
                seq_tensor = torch.tensor(seq_tensor, dtype=torch.float32).unsqueeze(0).to(device)
                
                logits = model(seq_tensor)[0].cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))
                
                # --- ALGO LEVEL 4: MCMC Fallback ---
                if algo_level == 4:
                    avg_prob = np.mean(probs)
                    max_prob = np.max(probs)
                    conf_ratio = max_prob / avg_prob if avg_prob > 0 else 1.0
                    
                    if conf_ratio < 1.3:
                        target_data = mycroft_bot.get_best_move(game.tracking_board, game.move_history)
                        if target_data is not None:
                            target = target_data[0]
                
                # --- ALGO LEVEL 1/2: NN Targeting ---
                if target is None:
                    if algo_level >= 2:
                        target = get_parity_nn_target(logits, shot_mask, game_id)
                    else:
                        target = get_pure_nn_target(logits, shot_mask)
            
            # --- EXECUTE TARGET ---
            x, y = target
            if x < 0 or x >= 10 or y < 0 or y >= 10 or shot_mask[x * 10 + y]:
                # Force fallback to pure random if everything broke (should never happen)
                valid_spots = np.where(~shot_mask)[0]
                if len(valid_spots) > 0:
                    best_idx = valid_spots[0]
                    x, y = best_idx // 10, best_idx % 10
                else:
                    break
                
            shot_mask[x * 10 + y] = True
            res = game.fire_shot(x, y)
            
            # Update live tensors for next turn
            if res in ["HIT", "SUNK"]:
                live_bot_hits[x, y] += 1.0 
            else:
                live_bot_misses[x, y] += 1.0
                
    return game.shot_number

# ==============================================================================
# MAIN MATRIX RUNNER
# ==============================================================================
def run_benchmark():
    print("Loading Battleship Dataset...")
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Battleship_Export_20260729.csv')
    df_raw = pd.read_csv(csv_path)
    
    models_to_test = {
        "Universal Model": os.path.join("data", "moriarty_offense_best.pth")
    }
    
    algo_names = {
        1: "Pure NN",
        2: "NN + Parity",
        3: "NN + Assassin + Parity",
        4: "Full Fidelity (MCMC)"
    }
    
    target_players = ["luki"]
    
    for tp in target_players:
        print(f"\n======================================")
        print(f"TESTING AGAINST: {tp.upper()}'S FLEET")
        print(f"======================================")
        
        # 1. Get the Test Set for this player
        dataset = CognitiveBattleshipDataset(csv_file_path=csv_path, is_offense_model=True, target_player=tp, only_wins=False)
        _, val_dataset = get_grouped_train_test_split(dataset, test_ratio=0.2, seed=42)
        
        # 2. Extract specific Game IDs from val_dataset hashes
        test_hashes = set()
        for i in range(len(val_dataset)):
            _, label = val_dataset[i]
            test_hashes.add(hash(label.numpy().tobytes()))
            
        test_games_data = []
        
        # Fast way to get seq: just index the full dataset
        for idx in range(len(dataset)):
            seq, label = dataset[idx]
            if hash(label.numpy().tobytes()) in test_hashes:
                ch_spatial = label.numpy()
                ships = []
                visited = np.zeros((10, 10), dtype=bool)
                ship_id = 1
                
                # Carrier, Battleship, Destroyer, Cruiser/Submarine
                for val in [0.4, 0.6, 1.0, 0.8]:
                    for r in range(10):
                        for c in range(10):
                            if not visited[r, c] and np.isclose(ch_spatial[r, c], val, atol=0.01):
                                coords = []
                                q = [(r, c)]
                                while q:
                                    curr_r, curr_c = q.pop(0)
                                    if not visited[curr_r, curr_c] and np.isclose(ch_spatial[curr_r, curr_c], val, atol=0.01):
                                        visited[curr_r, curr_c] = True
                                        coords.append((curr_r, curr_c))
                                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                                            nr, nc = curr_r + dr, curr_c + dc
                                            if 0 <= nr < 10 and 0 <= nc < 10:
                                                q.append((nr, nc))
                                                
                                # If two 0.8 ships touch, we split them roughly
                                if len(coords) > 5:
                                    ships.append({"id": ship_id, "coords": coords[:3]})
                                    ship_id += 1
                                    ships.append({"id": ship_id, "coords": coords[3:]})
                                    ship_id += 1
                                else:
                                    ships.append({"id": ship_id, "coords": coords})
                                    ship_id += 1
                
                test_games_data.append((idx, seq[:-1].numpy(), ships))
                
        print(f"Extracted {len(test_games_data)} test games for {tp.upper()}.")
        
        max_cores = 4
        
        for model_name, weights_path in models_to_test.items():
            full_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', weights_path))
            if not os.path.exists(full_weights_path):
                print(f"\nWarning: {weights_path} not found. Skipping {model_name}.")
                continue
                
            print(f"\n--- Model: {model_name} ---")
            
            for algo in [1, 2, 3, 4]:
                results = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
                    futures = {
                        executor.submit(simulate_game, game_id, ships, seq, full_weights_path, algo): game_id
                        for game_id, seq, ships in test_games_data
                    }
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            results.append(future.result())
                        except Exception as e:
                            pass # Silently ignore
                            
                if results:
                    avg_shots = np.mean(results)
                    med_shots = np.median(results)
                    max_shots = np.max(results)
                    print(f"  {algo_names[algo]:<25} -> Avg: {avg_shots:.1f} | Med: {med_shots:.1f} | Max: {max_shots}")
                else:
                    print(f"  {algo_names[algo]:<25} -> Failed to simulate.")

if __name__ == "__main__":
    run_benchmark()

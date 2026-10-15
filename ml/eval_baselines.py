import os
import sys
import time
import concurrent.futures
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import CognitiveBattleshipDataset, get_grouped_train_test_split
from bots.mycroft import MycroftBot
from bots.sherlock import SherlockBot
from core.game_engine import BattleshipGame

def simulate_baseline_game(game_id, ships_list, bot_name):
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

    if bot_name == 'sherlock':
        bot = SherlockBot()
    else:
        bot = MycroftBot()
        
    while not game.is_game_over() and game.shot_number < 100:
        target_data = bot.get_best_move(game.tracking_board, game.move_history)
        if target_data is None:
            # Fallback to random if solver breaks
            valid = []
            for r in range(10):
                for c in range(10):
                    if game.tracking_board[r][c] == 0:
                        valid.append((r, c))
            if valid:
                target = valid[0]
            else:
                break
        else:
            target = target_data[0]
            
        x, y = target
        if x < 0 or x >= 10 or y < 0 or y >= 10 or game.tracking_board[x][y] != 0:
            break
            
        game.fire_shot(x, y)
                
    return game.shot_number

def run_baselines():
    print("Loading Baseline Benchmark Data...")
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Battleship_Export_20260729.csv')
    df_raw = pd.read_csv(csv_path)
    
    target_players = ["luki", "grandma"]
    
    for tp in target_players:
        print(f"\n======================================")
        print(f"BASELINE TESTING: {tp.upper()}'S FLEET")
        print(f"======================================")
        
        dataset = CognitiveBattleshipDataset(csv_file_path=csv_path, is_offense_model=True, target_player=tp, only_wins=False)
        _, val_dataset = get_grouped_train_test_split(dataset, test_ratio=0.2, seed=42)
        
        test_hashes = set()
        for i in range(len(val_dataset)):
            _, label = val_dataset[i]
            test_hashes.add(hash(label.numpy().tobytes()))
            
        test_games_data = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if hash(label.numpy().tobytes()) in test_hashes:
                ch_spatial = label.numpy()
                ships = []
                visited = np.zeros((10, 10), dtype=bool)
                ship_id = 1
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
                                if len(coords) > 5:
                                    ships.append({"id": ship_id, "coords": coords[:3]})
                                    ship_id += 1
                                    ships.append({"id": ship_id, "coords": coords[3:]})
                                    ship_id += 1
                                else:
                                    ships.append({"id": ship_id, "coords": coords})
                                    ship_id += 1
                test_games_data.append((idx, ships))
                
        print(f"Extracted {len(test_games_data)} test games for {tp.upper()}.")
        max_cores = max(1, (os.cpu_count() or 4) - 1)
        
        for bot_name in ['sherlock', 'mycroft']:
            results = []
                test_games_data.append({"id": idx, "ships": ships, "seq": dataset[idx][0].numpy()})
                
        print(f"Found {len(test_games_data)} distinct validation games for {tp}.")
        if len(test_games_data) == 0: continue
        
        # 1. Base NN Model (From Scratch)
        base_weights = f"moriarty_{tp}_all_offense_best.pth"
        base_shots = []
        if os.path.exists(base_weights):
            for g in test_games_data:
                shots = simulate_game(g["id"], g["ships"], g["seq"], base_weights, algo_level=4)
                base_shots.append(shots)
        
        # 2. Transfer Learning NN Model
        transfer_weights = f"moriarty_{tp}_transfer_offense_best.pth"
        transfer_shots = []
        if os.path.exists(transfer_weights):
            for g in test_games_data:
                shots = simulate_game(g["id"], g["ships"], g["seq"], transfer_weights, algo_level=4)
                transfer_shots.append(shots)
                
        # 3. SherlockBot
        sherlock_shots = []
        for g in test_games_data:
            shots = simulate_baseline_game(g["id"], g["ships"], "sherlock")
            sherlock_shots.append(shots)
            
        print(f"\n[RESULTS FOR {tp.upper()}]")
        print(f"SherlockBot Average:   {np.mean(sherlock_shots):.1f} shots")
        if base_shots: print(f"NN (From Scratch):     {np.mean(base_shots):.1f} shots")
        if transfer_shots: print(f"NN (Transfer Learn):   {np.mean(transfer_shots):.1f} shots")

if __name__ == "__main__":
    run_baselines()

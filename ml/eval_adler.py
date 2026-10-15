import pandas as pd
import numpy as np
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import get_grouped_train_test_split
from core.game_engine import BattleshipGame

def simulate_adler_game(ships_list, prior):
    game = BattleshipGame()
    
    for ship in ships_list:
        ship_id = ship['id']
        length = len(ship['coords'])
        game.ship_pegs[ship_id] = []
        game.ship_health[ship_id] = length
        for (x, y) in ship['coords']:
            game.truth_board[x][y] = ship_id
            game.ship_pegs[ship_id].append((x, y))

    ship_sizes = [5, 4, 3, 3, 2]
    
    while not game.is_game_over() and game.shot_number < 100:
        # Standard density mapping
        density = np.zeros((10, 10))
        for size in ship_sizes:
            # Horizontal
            for r in range(10):
                for c in range(10 - size + 1):
                    valid = True
                    for i in range(size):
                        if game.tracking_board[r][c+i] != 0:
                            valid = False
                            break
                    if valid:
                        for i in range(size):
                            density[r][c+i] += 1
            # Vertical
            for c in range(10):
                for r in range(10 - size + 1):
                    valid = True
                    for i in range(size):
                        if game.tracking_board[r+i][c] != 0:
                            valid = False
                            break
                    if valid:
                        for i in range(size):
                            density[r+i][c] += 1
                            
        # Multiply by Adler's Prior
        adler_heatmap = density * prior
        
        # Mask out already fired shots
        for r in range(10):
            for c in range(10):
                if game.tracking_board[r][c] != 0:
                    adler_heatmap[r][c] = -1
                    
        max_val = np.max(adler_heatmap)
        if max_val <= 0:
            # Fallback to random
            valid_spots = []
            for r in range(10):
                for c in range(10):
                    if game.tracking_board[r][c] == 0:
                        valid_spots.append((r, c))
            if valid_spots:
                target = random.choice(valid_spots)
            else:
                break
        else:
            candidates = np.argwhere(adler_heatmap == max_val)
            target = candidates[random.randint(0, len(candidates)-1)]
            
        x, y = target
        game.fire_shot(x, y)
        
    return game.shot_number

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Battleship_Export_20260729.csv')
    df_raw = pd.read_csv(csv_path)
    
    target_player = "luki"
    player_df = df_raw[(df_raw['Player'] == target_player) | (df_raw['Opponent'] == target_player)].copy()
    
    unique_games = player_df['GameID'].unique()
    import random
    random.seed(42)
    games_list = list(unique_games)
    random.shuffle(games_list)
    split_idx = int(len(games_list) * 0.8)
    train_games = games_list[:split_idx]
    test_games = games_list[split_idx:]
    
    print(f"Calculating Bayesian Prior for {target_player} from {len(train_games)} training games...")
    prior = np.ones((10, 10)) * 0.1 # Base prior
    for game_id in train_games:
        game_rows = player_df[(player_df['GameID'] == game_id) & (player_df['Grid'] == 'Defense') & (player_df['Player'] == target_player)]
        for _, row in game_rows.iterrows():
            if not pd.isna(row['X']) and not pd.isna(row['Y']):
                prior[int(row['X'])][int(row['Y'])] += 1.0
                
    # Normalize prior
    prior = prior / np.max(prior)
    
    print(f"Running AdlerBot evaluation on {len(test_games)} test games...")
    shots = []
    
    for game_id in test_games:
        game_rows = player_df[(player_df['GameID'] == game_id) & (player_df['Grid'] == 'Defense') & (player_df['Player'] == target_player)]
        ships = {}
        for _, row in game_rows.iterrows():
            if pd.isna(row['X']) or pd.isna(row['Y']): continue
            x, y = int(row['X']), int(row['Y'])
            ship_name = row['ShotResult']
            if ship_name not in ships:
                # Mock an ID
                ships[ship_name] = len(ships) + 1
                
        ships_list = []
        for ship_name, s_id in ships.items():
            coords = []
            for _, row in game_rows[game_rows['ShotResult'] == ship_name].iterrows():
                coords.append((int(row['X']), int(row['Y'])))
            ships_list.append({"id": s_id, "coords": coords})
            
        if not ships_list:
            continue
            
        random.seed(str(game_id))
        s = simulate_adler_game(ships_list, prior)
        shots.append(s)
        
    print(f"--- AdlerBot (Bayesian Prior) ---")
    print(f"Avg Shots: {np.mean(shots):.1f}")
    print(f"Med Shots: {np.median(shots):.1f}")
    print(f"Min Shots: {np.min(shots)}")
    print(f"Max Shots: {np.max(shots)}")

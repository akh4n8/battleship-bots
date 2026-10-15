import time
import csv
import os
import concurrent.futures
from datetime import datetime, timezone
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.game_engine import BattleshipGame
from bots.sherlock import SherlockBot

def simulate_single_game(game_id):
    game = BattleshipGame()
    bot = SherlockBot()
    game.place_ships_randomly()
    
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    ships = game.ship_pegs
    
    while not game.is_game_over():
        target, strategy, dead_sizes, diagnostics = bot.get_best_move(
            game.tracking_board, 
            game.move_history
        )
        if target:
            x, y = target[0], target[1]
            game.fire_shot(x, y)
        else:
            break
            
    rows = []
    
    ship_names = {1: 'Carrier', 2: 'Battleship', 3: 'Cruiser', 4: 'Submarine', 5: 'Destroyer'}
    
    for ship_id, coords in ships.items():
        ship_name = ship_names.get(ship_id, 'SHIP')
        for (x, y) in coords:
            rows.append([
                game_id, 'WIN', 'Bot', 'synthetic_sherlock', 'MoriartyBot',
                timestamp, 0, 0, 'Defense', x, y, f"{chr(65+y)}{x+1}", ship_name
            ])
        
    for idx, move in enumerate(game.move_history):
        x = move['x']
        y = move['y']
        res = move['result']
        turn = (idx // 2) + 1
        
        rows.append([
            game_id, 'WIN', 'Bot', 'synthetic_sherlock', 'MoriartyBot',
            timestamp, turn, idx + 1, 'Offense', x, y, f"{chr(65+y)}{x+1}", res
        ])
        
    return rows

def generate_synthetic_data():
    num_games = 100000
    num_cores = max(1, (os.cpu_count() or 4) - 1)
    output_file = 'data/Battleship_Synthetic_100k.csv'
    
    print("\n===========================================")
    print(" 🚀 INITIATING SYNTHETIC DATA GENERATION 🚀")
    print("===========================================")
    print(f"Target: {num_games} Total Games")
    print(f"Engaging {num_cores} CPU Cores")
    print("===========================================\n")
    
    start_time = time.time()
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['GameID', 'MatchOutcome', 'GameMode', 'Player', 'Opponent', 
                         'Timestamp', 'TurnNumber', 'ShotNumber', 'Grid', 'X', 'Y', 'Coordinate', 'ShotResult'])
                         
        completed = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {executor.submit(simulate_single_game, i + 1): i for i in range(num_games)}
            
            for future in concurrent.futures.as_completed(futures):
                rows = future.result()
                if rows:
                    writer.writerows(rows)
                
                completed += 1
                if completed % 10000 == 0:
                    percent = int((completed / num_games) * 100)
                    print(f"Progress: {percent}% ({completed}/{num_games})")
                    
    end_time = time.time()
    print(f"\nSynthetic data generation complete in {end_time - start_time:.2f} seconds.")
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    generate_synthetic_data()

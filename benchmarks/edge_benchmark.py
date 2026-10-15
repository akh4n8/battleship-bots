import random
import time
import csv
import concurrent.futures
from game_engine import BattleshipGame
from mycroft import MycroftBot

def place_ships_on_edges(game):
    """Forces all ships to be placed strictly on the perimeter."""
    for ship_id, length in enumerate(game.fleet_sizes, start=1):
        placed = False
        while not placed:
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                is_horizontal = True
                x = random.randint(0, game.size - length)
                y = 0
            elif edge == 'bottom':
                is_horizontal = True
                x = random.randint(0, game.size - length)
                y = game.size - 1
            elif edge == 'left':
                is_horizontal = False
                x = 0
                y = random.randint(0, game.size - length)
            else: # right
                is_horizontal = False
                x = game.size - 1
                y = random.randint(0, game.size - length)
            
            placed = game.place_ship_manual(x, y, length, is_horizontal, ship_id)

def simulate_edge_game(game_args):
    """Runs a single game. game_args is a tuple of (game_id, safe_mode_active)"""
    game_id, safe_mode_active = game_args
    game = BattleshipGame()
    
    bot = MycroftBot()
    bot.safe_mode = safe_mode_active # Override the bot's mode
    
    place_ships_on_edges(game)
    
    while not game.is_game_over():
        target, strategy, dead_sizes, diagnostics = bot.get_best_move(
            game.tracking_board, 
            game.move_history
        )
        
        if target:
            game.fire_shot(target[0], target[1])
        else:
            return None
            
    return game.shot_number

def run_comparison(num_games=1000):
    print(f"Launching Edge-Placement Benchmark: {num_games} Games per Mode")
    print("Forcing all ships to the perimeter...\n")
    
    results_safe = []
    results_maverick = []
    
    # 1. Test Safe Mycroft
    print("Running Safe Mycroft (Parity Enforced)...")
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args = [(i, True) for i in range(num_games)]
        for shots in executor.map(simulate_edge_game, args):
            if shots: results_safe.append(shots)
    safe_time = time.time() - start_time
    
    # 2. Test Maverick Mycroft
    print("Running Maverick Mycroft (Unrestricted MCMC)...")
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args = [(i, False) for i in range(num_games)]
        for shots in executor.map(simulate_edge_game, args):
            if shots: results_maverick.append(shots)
    maverick_time = time.time() - start_time

    # 3. Output Analysis
    print("\n==================================")
    print("      EDGE-TRAP RESULTS           ")
    print("==================================")
    print(f"Safe Mycroft Avg Shots:     {sum(results_safe)/len(results_safe):.2f} (Time: {safe_time:.2f}s)")
    print(f"Maverick Mycroft Avg Shots: {sum(results_maverick)/len(results_maverick):.2f} (Time: {maverick_time:.2f}s)")
    print("==================================\n")

    # 4. Data Export for Statistical Modeling
    filename = f"mycroft_edge_comparison_{int(time.time())}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Game_ID", "Mode", "Total_Shots"])
        for i, shots in enumerate(results_safe):
            writer.writerow([i, "Safe", shots])
        for i, shots in enumerate(results_maverick):
            writer.writerow([i, "Maverick", shots])
            
    print(f"Exported raw data to {filename}")

if __name__ == "__main__":
    run_comparison(1000)
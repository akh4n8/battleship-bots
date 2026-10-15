import time
import csv
import json
import os
import argparse
import concurrent.futures
from core.game_engine import BattleshipGame

# --- 1. ISOLATE THE SINGLE GAME LOGIC ---
def simulate_single_game(game_id, bot_choice):
    """Runs a single game. Bot choice is passed as a string to remain picklable for multiprocessing."""
    game = BattleshipGame()
    
    # Dynamically instantiate the correct bot based on the string argument
    if bot_choice.lower() in ['neural', 'moriarty']:
        from bots.neural_bot import NeuralBot
        bot = NeuralBot(parity_offset=game_id % 2)
    elif bot_choice.lower() == 'neural_luki':
        from bots.neural_bot import NeuralBot
        weights = os.path.join(os.path.dirname(__file__), 'data', 'moriarty_luki_best.pth')
        bot = NeuralBot(weights_path=weights, parity_offset=game_id % 2)
    elif bot_choice.lower() in ['neural_rl', 'rl']:
        from bots.neural_bot import NeuralBot
        weights = os.path.join(os.path.dirname(__file__), 'data', 'moriarty_rl_best.pth')
        bot = NeuralBot(weights_path=weights, parity_offset=game_id % 2)
    elif bot_choice.lower() == 'mycroft':
        from bots.mycroft import MycroftBot
        bot = MycroftBot()
    elif bot_choice.lower() == 'entropy':
        from bots.entropy_bot import EntropyBot
        bot = EntropyBot()
    elif bot_choice.lower() in ['zdd', 'exact', 'population']:
        from bots.zdd_bot import ZDDBot
        bot = ZDDBot()
    elif bot_choice.lower() == 'sherrinford':
        from bots.sherrinford_bot import SherrinfordBot
        bot = SherrinfordBot(parity_offset=game_id % 2)
    elif bot_choice.lower() in ['sherrinford_parity', 'parity']:
        from bots.sherrinford_parity import SherrinfordParityBot
        bot = SherrinfordParityBot(parity_offset=game_id % 2)
    elif bot_choice.lower() in ['sherrinford_deduction', 'deduction']:
        from bots.sherrinford_deduction import SherrinfordDeductionBot
        bot = SherrinfordDeductionBot(parity_offset=game_id % 2)
    elif bot_choice.lower() in ['sherrinford_apex', 'apex']:
        from bots.sherrinford_apex import SherrinfordApexBot
        bot = SherrinfordApexBot(parity_offset=game_id % 2)
    elif bot_choice.lower() in ['porlock', 'porlock_bot']:
        from bots.porlock_bot import PorlockBot
        bot = PorlockBot(parity_offset=game_id % 2)
    else:
        from bots.sherlock import SherlockBot
        bot = SherlockBot()
    
    game.place_ships_randomly()
    
    while not game.is_game_over():
        target, strategy, dead_sizes, diagnostics = bot.get_best_move(
            game.tracking_board, 
            game.move_history
        )
        
        if target:
            game.fire_shot(target[0], target[1])
        else:
            return None, None # Error case
            
    shot_count = game.shot_number
    outlier_data = None
    
    # Catch anything over 60 shots
    if shot_count > 60:
        outlier_data = {
            "game_id": game_id,
            "shots": shot_count,
            "ships": game.ship_pegs, 
            "history": [(m["x"], m["y"], m["result"]) for m in game.move_history]
        }
        
    return shot_count, outlier_data

# --- 2. THE MULTIPROCESSING MANAGER ---
def run_benchmark_parallel(bot_choice, num_games=10000, requested_cores=None):
    # Detect available CPU cores
    max_cores = os.cpu_count() or 4
    b = bot_choice.lower()
    default_cores = 4 if b in ['neural', 'moriarty'] else max(1, max_cores - 1)
    num_cores = requested_cores if requested_cores else default_cores
    if b in ['neural', 'moriarty']:
        bot_name = "NeuralBot (Deep Sherlock)"
    elif b == 'mycroft':
        bot_name = "Mycroft (Monte Carlo)"
    elif b == 'entropy':
        bot_name = "EntropyBot"
    elif b == 'zdd':
        bot_name = "ZDDBot (Exact Population)"
    elif b == 'sherrinford':
        bot_name = "SherrinfordBot (Pure Exact Bayesian)"
    else:
        bot_name = "Sherlock (CSP)"
    
    print(f"\nBooting {bot_name} Simulation Engine...")
    print(f"Target: {num_games} randomized games.")
    print(f"Engaging {num_cores} CPU cores...\n")
    
    start_time = time.time()
    results = []
    outlier_games = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Pass the bot_choice string into the worker function
        futures = {executor.submit(simulate_single_game, i + 1, bot_choice): i for i in range(num_games)}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            shot_count, outlier_data = future.result()
            
            if shot_count is not None:
                results.append(shot_count)
            if outlier_data is not None:
                outlier_games.append(outlier_data)
                
            completed += 1
            if completed % max(1, (num_games // 20)) == 0:
                percent = int((completed / num_games) * 100)
                elapsed_now = time.time() - start_time
                print(f"Simulation Progress: {percent}% ({completed}/{num_games} games) [{elapsed_now:.1f}s]", flush=True)

    end_time = time.time()
    
    # --- Statistical Analysis ---
    if not results:
        print("No results gathered. Simulation failed.")
        return

    avg_shots = sum(results) / num_games
    min_shots = min(results)
    max_shots = max(results)
    
    sorted_results = sorted(results)
    mid = num_games // 2
    median_shots = (sorted_results[mid] + sorted_results[~mid]) / 2
    
    variance = sum((x - avg_shots) ** 2 for x in results) / num_games
    std_dev = variance ** 0.5

    print("\n==================================")
    print("      SIMULATION COMPLETE         ")
    print("==================================")
    print(f"Total Games:    {num_games}")
    print(f"Average Shots:  {avg_shots:.2f}")
    print(f"Median Shots:   {median_shots:.1f}")
    print(f"Std Deviation:  {std_dev:.2f}")
    print(f"Variance:       {variance:.2f}")
    print(f"Best Game:      {min_shots} shots")
    print(f"Worst Game:     {max_shots} shots")
    print(f"Time Elapsed:   {end_time - start_time:.2f} seconds")
    print("==================================\n")
    
    # --- CSV Exporter ---
    if outlier_games:
        prefix = "mycroft" if bot_choice.lower() == 'mycroft' else "sherlock"
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"{prefix}_outliers_{int(time.time())}.csv")
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Game_ID", "Total_Shots", "Ship_Layout", "Shot_Sequence"])
                
                for og in outlier_games:
                    writer.writerow([
                        og["game_id"],
                        og["shots"],
                        json.dumps(og["ships"]),
                        json.dumps(og["history"])
                    ])
            print(f"[!] Exported {len(outlier_games)} outlier games (>60 shots) to {filename}\n")
        except Exception as e:
            print(f"[!] Note: Could not export outlier csv: {e}\n")
    
    # --- Terminal Histogram ---
    print("Shot Distribution (Bell Curve):")
    bins = range((min_shots // 5) * 5, ((max_shots // 5) * 5) + 10, 5)
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1] - 1
        count = sum(1 for x in results if low <= x <= high)
        
        max_count = max(sum(1 for x in results if b <= x < b+5) for b in bins[:-1])
        bar_length = int((count / max_count) * 50) if max_count > 0 else 0
        bar = "█" * bar_length
        
        print(f"{low:2d} - {high:2d} shots: | {bar} ({count})")

if __name__ == "__main__":
    # 1. Setup CLI
    parser = argparse.ArgumentParser(description="Run Battleship Bot Benchmarks.")
    parser.add_argument(
        "--bot", 
        type=str, 
        choices=["sherlock", "mycroft", "entropy", "neural", "moriarty", "neural_luki", "neural_rl", "rl", "zdd", "sherrinford", "sherrinford_parity", "sherrinford_deduction", "sherrinford_apex"], 
        default="neural",
        help="Which bot to benchmark (sherlock, mycroft, entropy, neural, zdd, sherrinford, sherrinford_parity, sherrinford_deduction, sherrinford_apex)"
    )
    parser.add_argument(
        "--games", 
        type=int, 
        default=10000, 
        help="Number of games to simulate"
    )
    parser.add_argument(
        "--cores", 
        type=int, 
        default=None, 
        help="Number of CPU cores to use (defaults to max-1)"
    )
    
    args = parser.parse_args()
    
    run_benchmark_parallel(
        bot_choice=args.bot,
        num_games=args.games,
        requested_cores=args.cores
    )
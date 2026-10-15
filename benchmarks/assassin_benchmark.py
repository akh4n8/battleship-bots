import time
import random
import concurrent.futures
from game_engine import BattleshipGame, CELL_UNKNOWN
from mycroft import MycroftBot

class AssassinMycroftBot(MycroftBot):
    """
    Maverick Mycroft wrapped with Kotlin's Phase 1: The Assassin.
    If there are active hits, it uses Sherlock's true geometric vector tracer.
    """
    def __init__(self):
        # Force Maverick Mode
        super().__init__(safe_mode=False) 

    def get_best_move(self, board, move_history):
        # 1. Ask Sherlock for active hits AND claimed hits (Needed for the tracer)
        try:
            ui_active, _, ui_claimed, _ = self.sherlock._analyze_board_state(board, move_history)
        except Exception:
            ui_active = []
            ui_claimed = []
            
        # 2. PHASE 1: THE TRUE ASSASSIN OVERRIDE
        if ui_active:
            # Delegate directly to Sherlock's precise vector math!
            target = self.sherlock._execute_linear_kill(board, ui_active, set(ui_claimed))
            if target:
                return target, "Assassin: True Vector Tracer", [], {}
                    
        # 3. If no active hits (Phase 2-6), proceed to standard Maverick MCMC
        return super().get_best_move(board, move_history)

def simulate_assassin_game(game_args):
    game_id, use_assassin = game_args
    game = BattleshipGame()
    game.place_ships_randomly()
    
    # STRICT ISOLATION: Both bots operate in Maverick Mode (safe_mode=False)
    if use_assassin:
        bot = AssassinMycroftBot() 
    else:
        bot = MycroftBot(safe_mode=False)
    
    while not game.is_game_over():
        target, _, _, _ = bot.get_best_move(game.tracking_board, game.move_history)
        if target:
            game.fire_shot(target[0], target[1])
        else:
            return None
    return game.shot_number

def run_assassin_comparison(num_games=2500):
    print(f"Launching Maverick Benchmark: {num_games} Games per bot\n")
    
    # 1. Maverick (Pure MCMC)
    print("Running Pure Maverick (100% MCMC Probabilistic Cleanup)...")
    start_time = time.time()
    results_pure = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args = [(i, False) for i in range(num_games)]
        for shots in executor.map(simulate_assassin_game, args):
            if shots: results_pure.append(shots)
    pure_time = time.time() - start_time
    
    # 2. Maverick + Assassin
    print("Running Maverick + Assassin (Kotlin Phase 1 Override)...")
    start_time = time.time()
    results_assassin = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args = [(i, True) for i in range(num_games)]
        for shots in executor.map(simulate_assassin_game, args):
            if shots: results_assassin.append(shots)
    assassin_time = time.time() - start_time

    # Output
    print("\n=============================================")
    print("      CLEANUP PHASE RESULTS (MAVERICK)       ")
    print("=============================================")
    print(f"Pure Maverick Avg Shots:     {sum(results_pure)/len(results_pure):.2f} (Time: {pure_time:.2f}s)")
    print(f"Maverick + Assassin Avg:     {sum(results_assassin)/len(results_assassin):.2f} (Time: {assassin_time:.2f}s)")
    print("=============================================\n")

if __name__ == "__main__":
    # Adjust this number based on how many CPU cores you want to melt today.
    run_assassin_comparison(12500)
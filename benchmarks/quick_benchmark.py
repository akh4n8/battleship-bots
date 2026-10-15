import time
import csv
import json
import os
import concurrent.futures
from game_engine import BattleshipGame

def simulate_single_game(game_id, personality):
    game = BattleshipGame()
    from mycroft import MycroftBot
    
    # Toggle personality based on argument
    if personality == 'Maverick':
        bot = MycroftBot(safe_mode=False)
    else:
        bot = MycroftBot(safe_mode=True)
        
    game.place_ships_randomly()
    
    while not game.is_game_over():
        target, strategy, dead_sizes, diagnostics = bot.get_best_move(
            game.tracking_board, 
            game.move_history
        )
        if target:
            game.fire_shot(target[0], target[1])
        else:
            return None, None 
            
    shot_count = game.shot_number
    outlier_data = None
    
    # Save anything over 60 shots for forensic review
    if shot_count > 60:
        outlier_data = {
            "game_id": game_id,
            "shots": shot_count,
            "ships": game.ship_pegs, 
            "history": [(m["x"], m["y"], m["result"]) for m in game.move_history]
        }
        
    return shot_count, outlier_data

def run_quick_sim():
    num_games = 10000
    personalities = ['Maverick', 'Safe']
    num_cores = max(1, (os.cpu_count() or 4) - 1)
    
    print("\n===========================================")
    print(" 🚀 INITIATING RAPID BATTLESHIP SIM 🚀")
    print("===========================================")
    print(f"Target: {num_games * 2} Total Games")
    print(f"Engaging {num_cores} CPU Cores")
    print("===========================================\n")
    
    report_lines = [
        "==========================================",
        "      RAPID SIMULATION REPORT             ",
        "==========================================",
        ""
    ]
    
    total_start_time = time.time()
    total_completed_all = 0
    
    for personality in personalities:
        print(f"\nBooting {personality} Personality...")
        start_time = time.time()
        results = []
        outlier_games = []
        completed = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {executor.submit(simulate_single_game, i + 1, personality): i for i in range(num_games)}
            
            try:
                for future in concurrent.futures.as_completed(futures):
                    shot_count, outlier_data = future.result()
                    
                    if shot_count is not None:
                        results.append(shot_count)
                    if outlier_data is not None:
                        outlier_games.append(outlier_data)
                        
                    completed += 1
                    
                    if completed % max(1, (num_games // 10)) == 0:
                        percent = int((completed / num_games) * 100)
                        print(f"[{personality}] Progress: {percent}% ({completed}/{num_games})")
                        
            except KeyboardInterrupt:
                print(f"\n[!] EMERGENCY STOP DETECTED! Canceling pending games...")
                executor.shutdown(wait=False, cancel_futures=True)
                print(f"[*] Force-saving the {completed} games we managed to finish for {personality}...")
                
        total_completed_all += completed
        end_time = time.time()
        
        # Calculate Stats (Only if we completed at least 1 game)
        if results:
            avg_shots = sum(results) / completed
            min_shots = min(results)
            max_shots = max(results)
            sorted_results = sorted(results)
            median_shots = (sorted_results[completed // 2] + sorted_results[~(completed // 2)]) / 2
            
            # Save Outliers
            if outlier_games:
                filename = f"quick_{personality.lower()}_outliers.csv"
                with open(filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Game_ID", "Total_Shots", "Ship_Layout", "Shot_Sequence"])
                    for og in outlier_games:
                        writer.writerow([og["game_id"], og["shots"], json.dumps(og["ships"]), json.dumps(og["history"])])
            
            # Append to Report
            report_lines.extend([
                f"--- {personality.upper()} MYCROFT ---",
                f"Games Played:   {completed}",
                f"Average Shots:  {avg_shots:.3f}",
                f"Median Shots:   {median_shots:.1f}",
                f"Best Game:      {min_shots}",
                f"Worst Game:     {max_shots}",
                f"Outliers (>60): {len(outlier_games)}",
                f"Time Taken:     {(end_time - start_time):.2f} Seconds",
                ""
            ])
        else:
            report_lines.append(f"--- {personality.upper()} MYCROFT ---\nNo games completed.\n")

    report_lines.insert(3, f"Total Games Run: {total_completed_all}")
    
    total_time_mins = (time.time() - total_start_time) / 60
    report_lines.append(f"Total Compute Time: {total_time_mins:.2f} Minutes")
    report_lines.append("==========================================")
    
    # Write Final Report to Disk
    with open("QUICK_REPORT.txt", "w") as f:
        f.write("\n".join(report_lines))
        
    print("\n✅ RAPID SIMULATION COMPLETE!")
    print("✅ Results saved to QUICK_REPORT.txt")

if __name__ == "__main__":
    run_quick_sim()
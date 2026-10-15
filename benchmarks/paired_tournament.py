import os
import sys
import time
import math
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_single_paired_game(game_id):
    """
    Simulates a single board with 4 bots playing the exact same board seed.
    Returns:
        dict with shot counts for each bot
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    
    from core.game_engine import BattleshipGame
    from bots.sherlock import SherlockBot
    from bots.mycroft import MycroftBot
    from bots.neural_bot import NeuralBot
    from bots.entropy_bot import EntropyBot
    from bots.sherrinford_bot import SherrinfordBot

    # Generate the shared ground-truth board
    master_game = BattleshipGame()
    master_game.place_ships_randomly()

    parity = game_id % 2

    # Instantiate bots
    bots = {
        'Sherlock': SherlockBot(),
        'Mycroft': MycroftBot(safe_mode=True),
        'Neural': NeuralBot(device=torch.device('cpu'), parity_offset=parity),
        'Entropy': EntropyBot(),
        'Sherrinford': SherrinfordBot(parity_offset=parity)
    }

    # Lock parity offset
    bots['Sherlock'].parity_offset = parity
    bots['Mycroft'].parity_offset = parity
    bots['Neural'].parity_offset = parity
    bots['Entropy'].parity_offset = parity
    bots['Entropy'].mycroft.parity_offset = parity
    bots['Sherrinford'].parity_offset = parity

    results = {}
    for name, bot in bots.items():
        sim = BattleshipGame()
        sim.truth_board = [r[:] for r in master_game.truth_board]
        sim.ship_pegs = {k: v[:] for k, v in master_game.ship_pegs.items()}
        sim.ship_health = master_game.ship_health.copy()

        while not sim.is_game_over() and sim.shot_number < 100:
            target, _, _, _ = bot.get_best_move(sim.tracking_board, sim.move_history)
            if not target:
                break
            sim.fire_shot(target[0], target[1])

        results[name] = sim.shot_number

    return game_id, results

def calculate_stats(arr):
    n = len(arr)
    mean = sum(arr) / n
    variance = sum((x - mean) ** 2 for x in arr) / (n - 1) if n > 1 else 0
    std = math.sqrt(variance)
    sorted_arr = sorted(arr)
    median = sorted_arr[n // 2] if n % 2 != 0 else (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0
    sem = std / math.sqrt(n) if n > 0 else 0
    ci_lower = mean - 1.96 * sem
    ci_upper = mean + 1.96 * sem
    outliers = sum(1 for x in arr if x > 60)
    return {
        'n': n,
        'mean': mean,
        'median': median,
        'std': std,
        'variance': variance,
        'sem': sem,
        'ci95': (ci_lower, ci_upper),
        'outliers': outliers,
        'min': min(arr),
        'max': max(arr)
    }

def calculate_paired_comparison(arr1, arr2):
    n = len(arr1)
    diffs = [a - b for a, b in zip(arr1, arr2)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0
    std_diff = math.sqrt(var_diff)
    se_diff = std_diff / math.sqrt(n) if n > 0 else 0

    t_stat = mean_diff / se_diff if se_diff > 0 else 0.0

    # Two-tailed p-value via normal CDF / erf
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))

    # Head-to-Head Win / Loss / Tie
    wins1 = sum(1 for d in diffs if d < 0)  # Lower shots is better!
    wins2 = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)

    # Cohen's dz
    cohens_d = abs(mean_diff) / std_diff if std_diff > 0 else 0.0

    ci95_diff = (mean_diff - 1.96 * se_diff, mean_diff + 1.96 * se_diff)

    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'se_diff': se_diff,
        'ci95': ci95_diff,
        't_stat': t_stat,
        'p_val': p_val,
        'wins1': wins1,
        'wins2': wins2,
        'ties': ties,
        'cohens_d': cohens_d
    }

def main():
    parser = argparse.ArgumentParser(description="4-Way Paired Tournament Benchmark")
    parser.add_argument("--games", type=int, default=500, help="Number of paired games (default: 500)")
    parser.add_argument("--workers", type=int, default=16, help="Worker CPU processes (default: 16)")
    args = parser.parse_args()

    num_games = args.games
    workers = min(args.workers, os.cpu_count() or 4)

    print("=" * 72)
    print(" ⚔️  5-WAY PAIRED BATTLESHIP TOURNAMENT BENCHMARK  ⚔️")
    print(f"    Bots Tested   : Sherlock vs Mycroft vs Neural vs Entropy vs Sherrinford")
    print(f"    Total Boards  : {num_games:,} Identical Paired Fleet Layouts")
    print(f"    CPU Workers   : {workers} Cores")
    print("=" * 72)

    bot_names = ['Sherlock', 'Mycroft', 'Neural', 'Entropy', 'Sherrinford']
    all_results = {name: [] for name in bot_names}

    start_time = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_paired_game, gid): gid for gid in range(num_games)}

        last_print = time.time()
        for fut in as_completed(futures):
            gid, res = fut.result()
            for name in bot_names:
                all_results[name].append(res[name])

            completed += 1
            if time.time() - last_print > 5.0 or completed == num_games:
                last_print = time.time()
                elapsed = time.time() - start_time
                gps = completed / elapsed if elapsed > 0 else 0
                eta = (num_games - completed) / gps if gps > 0 else 0
                print(f"[{completed:4d} / {num_games}] ({elapsed:5.1f}s | {gps:4.1f} bds/s | ETA: {eta:4.1f}s) "
                      f"Current Avg: "
                      + " | ".join(f"{b[:4]}: {sum(all_results[b])/len(all_results[b]):.2f}" for b in bot_names))

    total_time = time.time() - start_time
    print("\n" + "=" * 72)
    print(f" 🏆 TOURNAMENT COMPLETED ({num_games} Paired Games in {total_time:.1f}s)")
    print("=" * 72)

    # 1. Individual Performance Table
    print("\n📊 1. OVERALL BOT PERFORMANCE (Ranked by Mean Shots)")
    print("-" * 72)
    print(f"{'Bot':<12} | {'Mean ± SEM':<16} | {'95% CI':<15} | {'Median':<6} | {'Std':<6} | {'>60':<5}")
    print("-" * 72)

    stats = {name: calculate_stats(all_results[name]) for name in bot_names}
    ranked = sorted(bot_names, key=lambda n: stats[n]['mean'])

    for rank, name in enumerate(ranked, 1):
        st = stats[name]
        mean_str = f"{st['mean']:.2f} ± {st['sem']:.2f}"
        ci_str = f"[{st['ci95'][0]:.2f}, {st['ci95'][1]:.2f}]"
        print(f"#{rank} {name:<9} | {mean_str:<16} | {ci_str:<15} | {st['median']:<6.1f} | {st['std']:<6.2f} | {st['outliers']:<5}")

    print("-" * 72)

    # 2. Pairwise Head-to-Head & Statistical Significance Matrix
    print("\n🔬 2. PAIRWISE STATISTICAL SIGNIFICANCE (Paired t-test on Same Boards)")
    print("-" * 72)
    print(f"{'Comparison (A vs B)':<24} | {'Diff (A - B)':<12} | {'t-stat':<7} | {'p-value':<9} | {'Significance':<12}")
    print("-" * 72)

    pairs = [
        ('Sherrinford', 'Sherlock'),
        ('Sherrinford', 'Mycroft'),
        ('Sherrinford', 'Neural'),
        ('Sherrinford', 'Entropy'),
        ('Sherlock', 'Neural'),
        ('Sherlock', 'Mycroft'),
        ('Sherlock', 'Entropy'),
        ('Mycroft', 'Neural'),
        ('Mycroft', 'Entropy'),
        ('Neural', 'Entropy'),
    ]

    for b1, b2 in pairs:
        cmp = calculate_paired_comparison(all_results[b1], all_results[b2])
        diff_str = f"{cmp['mean_diff']:+.2f} shots"
        t_str = f"{cmp['t_stat']:+.2f}"
        p_val = cmp['p_val']
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"

        if p_val < 0.001:
            sig = "✅ Yes (p < 0.001) ***"
        elif p_val < 0.01:
            sig = "✅ Yes (p < 0.01) **"
        elif p_val < 0.05:
            sig = "✅ Yes (p < 0.05) *"
        else:
            sig = "❌ No (ns)"

        comp_label = f"{b1} vs {b2}"
        print(f"{comp_label:<24} | {diff_str:<12} | {t_str:<7} | {p_str:<9} | {sig:<12}")

    print("-" * 72)

    # 3. Head-to-Head Win Rates
    print("\n🥊 3. HEAD-TO-HEAD WIN MATRIX (% of Games Bot A Beat Bot B on Same Board)")
    print("-" * 72)
    for b1, b2 in pairs:
        cmp = calculate_paired_comparison(all_results[b1], all_results[b2])
        w1_pct = (cmp['wins1'] / num_games) * 100
        w2_pct = (cmp['wins2'] / num_games) * 100
        tie_pct = (cmp['ties'] / num_games) * 100
        print(f"  • {b1} vs {b2}: {b1} won {w1_pct:.1f}% | {b2} won {w2_pct:.1f}% | Tied {tie_pct:.1f}%")

    print("-" * 72)

    # Save to /tmp
    import json
    save_path = "/tmp/paired_tournament_results.json"
    try:
        with open(save_path, "w") as f:
            json.dump({
                "num_games": num_games,
                "stats": stats,
                "all_results": all_results
            }, f, indent=2)
        print(f"\n💾 Saved full tournament dataset to {save_path}")
    except Exception as e:
        print(f"\n[!] Note: could not write json to {save_path}: {e}")

if __name__ == "__main__":
    main()

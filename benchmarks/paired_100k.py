import os
import sys
import time
import math
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.game_engine import BattleshipGame
from bots.sherlock import SherlockBot
from bots.sherrinford_bot import SherrinfordBot

def run_chunk(start_idx, count):
    """
    Executes a chunk of paired games on identical boards.
    Returns:
        s_shots: list of shot counts for Sherlock
        sh_shots: list of shot counts for Sherrinford
    """
    s_shots = []
    sh_shots = []

    for i in range(start_idx, start_idx + count):
        rng = random.Random(i * 10007)
        master = BattleshipGame()
        for ship_id, length in enumerate(master.fleet_sizes, start=1):
            placed = False
            while not placed:
                is_horiz = rng.choice([True, False])
                x = rng.randint(0, 9)
                y = rng.randint(0, 9)
                placed = master.place_ship_manual(x, y, length, is_horiz, ship_id)

        parity = i % 2

        # 1. Sherlock
        s_bot = SherlockBot()
        s_bot.parity_offset = parity
        sim_s = BattleshipGame()
        sim_s.truth_board = [r[:] for r in master.truth_board]
        sim_s.ship_pegs = {k: v[:] for k, v in master.ship_pegs.items()}
        sim_s.ship_health = master.ship_health.copy()
        while not sim_s.is_game_over() and sim_s.shot_number < 100:
            t, _, _, _ = s_bot.get_best_move(sim_s.tracking_board, sim_s.move_history)
            sim_s.fire_shot(*t)
        s_shots.append(sim_s.shot_number)

        # 2. Sherrinford
        sh_bot = SherrinfordBot(parity_offset=parity)
        sim_sh = BattleshipGame()
        sim_sh.truth_board = [r[:] for r in master.truth_board]
        sim_sh.ship_pegs = {k: v[:] for k, v in master.ship_pegs.items()}
        sim_sh.ship_health = master.ship_health.copy()
        while not sim_sh.is_game_over() and sim_sh.shot_number < 100:
            t, _, _, _ = sh_bot.get_best_move(sim_sh.tracking_board, sim_sh.move_history)
            sim_sh.fire_shot(*t)
        sh_shots.append(sim_sh.shot_number)

    return s_shots, sh_shots

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
        'variance': variance,
        'std': std,
        'median': median,
        'sem': sem,
        'ci95': (ci_lower, ci_upper),
        'min': min(arr),
        'max': max(arr),
        'outliers': outliers,
    }

def print_ascii_histogram(title, shots, bin_size=5):
    min_val = (min(shots) // bin_size) * bin_size
    max_val = ((max(shots) // bin_size) + 1) * bin_size
    bins = list(range(min_val, max_val + bin_size, bin_size))
    
    counts = []
    labels = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1] - 1
        c = sum(1 for x in shots if low <= x <= high)
        counts.append(c)
        labels.append(f"{low:2d} - {high:2d}")
        
    max_c = max(counts) if counts else 1
    print(f"\n{title}")
    print("-" * 55)
    for lbl, c in zip(labels, counts):
        bar_len = int((c / max_c) * 35) if max_c > 0 else 0
        pct = (c / len(shots)) * 100
        print(f"{lbl} shots: | {'█' * bar_len:<35} {c:6,d} ({pct:5.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="100,000 Paired Tournament: Sherlock vs Sherrinford")
    parser.add_argument("--games", type=int, default=100000, help="Total paired games (default: 100,000)")
    parser.add_argument("--cores", type=int, default=16, help="CPU cores to engage (default: 16)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Games per chunk worker (default: 1,000)")
    args = parser.parse_args()

    num_games = args.games
    cores = min(args.cores, os.cpu_count() or 4)
    chunk_size = args.chunk_size

    print("=" * 76)
    print(f" ⚔️  MEGA-SCALE PAIRED TOURNAMENT: SHERRINFORD VS SHERLOCK  ⚔️")
    print(f"    Total Games   : {num_games:,} Identical Paired Fleet Layouts")
    print(f"    Total Matches : {num_games * 2:,} Complete Simulated Games")
    print(f"    CPU Cores     : {cores} Cores Engaged")
    print(f"    Chunk Size    : {chunk_size:,} Games / Worker Task")
    print("=" * 76)

    chunks = []
    for start in range(0, num_games, chunk_size):
        count = min(chunk_size, num_games - start)
        chunks.append((start, count))

    all_s_shots = []
    all_sh_shots = []

    start_time = time.time()
    completed_games = 0

    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(run_chunk, s, c): (s, c) for s, c in chunks}

        last_print = time.time()
        for fut in as_completed(futures):
            s_res, sh_res = fut.result()
            all_s_shots.extend(s_res)
            all_sh_shots.extend(sh_res)
            completed_games += len(s_res)

            if time.time() - last_print > 3.0 or completed_games == num_games:
                last_print = time.time()
                elapsed = time.time() - start_time
                gps = completed_games / elapsed if elapsed > 0 else 0
                eta = (num_games - completed_games) / gps if gps > 0 else 0
                s_curr_avg = sum(all_s_shots) / len(all_s_shots)
                sh_curr_avg = sum(all_sh_shots) / len(all_sh_shots)
                pct = (completed_games / num_games) * 100
                diff = sh_curr_avg - s_curr_avg
                print(f"[{completed_games:6,d} / {num_games:,}] ({pct:5.1f}%) | {elapsed:5.1f}s | {gps:5.1f} games/s | ETA: {eta:4.1f}s | "
                      f"Sherlock: {s_curr_avg:.2f} | Sherrinford: {sh_curr_avg:.2f} (Diff: {diff:+.3f})")

    total_time = time.time() - start_time
    print("\n" + "=" * 76)
    print(f" 🏆 {num_games:,}-GAME TOURNAMENT COMPLETED in {total_time:.2f} seconds ({num_games / total_time:.1f} paired games/sec)")
    print("=" * 76)

    # 1. Statistics
    st_s = calculate_stats(all_s_shots)
    st_sh = calculate_stats(all_sh_shots)

    print(f"\n📊 1. OVERALL ENGINE BENCHMARK (N = {num_games:,})")
    print("-" * 76)
    print(f"{'Bot':<14} | {'Mean ± SEM':<18} | {'95% CI':<17} | {'Median':<6} | {'Std':<6} | {'>60':<7}")
    print("-" * 76)
    for name, st in [('Sherlock (CSP)', st_s), ('Sherrinford (Bayes)', st_sh)]:
        mean_str = f"{st['mean']:.3f} ± {st['sem']:.3f}"
        ci_str = f"[{st['ci95'][0]:.3f}, {st['ci95'][1]:.3f}]"
        out_pct = (st['outliers'] / num_games) * 100
        out_str = f"{st['outliers']:,} ({out_pct:.2f}%)"
        print(f"{name:<14} | {mean_str:<18} | {ci_str:<17} | {st['median']:<6.1f} | {st['std']:<6.2f} | {out_str:<7}")
    print("-" * 76)

    # 2. Paired Statistics
    diffs = [sh - s for sh, s in zip(all_sh_shots, all_s_shots)]
    mean_diff = sum(diffs) / len(diffs)
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
    std_diff = math.sqrt(var_diff)
    sem_diff = std_diff / math.sqrt(len(diffs))
    t_stat = mean_diff / sem_diff
    # z-approximation for N = 100,000
    p_val = math.erfc(abs(t_stat) / math.sqrt(2))

    print("\n🔬 2. PAIRED STATISTICAL INFERENCE (Same 100,000 Fleet Layouts)")
    print("-" * 76)
    print(f"  • Mean Difference (Sherrinford - Sherlock) : {mean_diff:+.4f} shots")
    print(f"  • Standard Error of the Difference (SEM)   : ±{sem_diff:.4f} shots")
    print(f"  • 95% Confidence Interval for Difference   : [{mean_diff - 1.96 * sem_diff:+.4f}, {mean_diff + 1.96 * sem_diff:+.4f}] shots")
    print(f"  • Paired t-Statistic                       : t = {t_stat:+.3f}")
    p_str = f"{p_val:.6e}" if p_val < 0.0001 else f"{p_val:.4f}"
    print(f"  • Two-Tailed p-value                       : p = {p_str}")
    print(f"  • Statistical Significance                 : {'✅ STATISTICALLY SIGNIFICANT (p < 0.0001) ***' if p_val < 0.0001 else '❌ Not significant'}")

    # 3. Head-to-Head Win Record
    sh_wins = sum(1 for d in diffs if d < 0)
    s_wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)

    print("\n🥊 3. HEAD-TO-HEAD MATCH RECORD")
    print("-" * 76)
    print(f"  • Sherrinford Won : {sh_wins:6,d} games ({sh_wins / num_games * 100:5.2f}%)")
    print(f"  • Sherlock Won    : {s_wins:6,d} games ({s_wins / num_games * 100:5.2f}%)")
    print(f"  • Exact Ties      : {ties:6,d} games ({ties / num_games * 100:5.2f}%)")
    print(f"  • Net Win Margin  : {sh_wins - s_wins:+,d} games in favor of Sherrinford")

    # 4. Histograms
    print_ascii_histogram("📊 SHERRINFORD SHOT DISTRIBUTION", all_sh_shots)
    print_ascii_histogram("📊 SHERLOCK SHOT DISTRIBUTION", all_s_shots)

if __name__ == '__main__':
    main()

import os
import sys
import time
import math
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.game_engine import BattleshipGame
from bots.sherlock import SherlockBot
from bots.sherrinford_bot import SherrinfordBot
from bots.sherrinford_parity import SherrinfordParityBot
from bots.sherrinford_deduction import SherrinfordDeductionBot
from bots.sherrinford_apex import SherrinfordApexBot
from bots.porlock_bot import PorlockBot

BOT_CLASSES = {
    'Sherlock': SherlockBot,
    'Sherrinford': SherrinfordBot,
    'Parity': SherrinfordParityBot,
    'Deduction': SherrinfordDeductionBot,
    'Apex': SherrinfordApexBot,
    'Porlock': PorlockBot,
}

def run_chunk(start_idx, count, bot_keys):
    results = {k: [] for k in bot_keys}

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

        for k in bot_keys:
            cls = BOT_CLASSES[k]
            if k == 'Sherlock':
                bot = cls()
            else:
                bot = cls(parity_offset=parity)
            bot.parity_offset = parity

            sim = BattleshipGame()
            sim.truth_board = [r[:] for r in master.truth_board]
            sim.ship_pegs = {sk: v[:] for sk, v in master.ship_pegs.items()}
            sim.ship_health = master.ship_health.copy()
            while not sim.is_game_over() and sim.shot_number < 100:
                t, _, _, _ = bot.get_best_move(sim.tracking_board, sim.move_history)
                sim.fire_shot(*t)
            results[k].append(sim.shot_number)

    return results

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

def main():
    parser = argparse.ArgumentParser(description="Multi-Engine Battleship Tournament")
    parser.add_argument("--games", type=int, default=5000, help="Total paired games (default: 5,000)")
    parser.add_argument("--cores", type=int, default=16, help="CPU cores (default: 16)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Games per chunk (default: 1,000)")
    parser.add_argument("--bots", nargs="+", default=['Sherlock', 'Apex', 'Porlock'], help="Bots to test (options: Sherlock, Sherrinford, Parity, Deduction, Apex, Porlock)")
    args = parser.parse_args()

    num_games = args.games
    cores = min(args.cores, os.cpu_count() or 4)
    chunk_size = args.chunk_size
    bot_keys = args.bots

    print("=" * 76)
    print(" ⚔️  MEGA-SCALE PAIRED TOURNAMENT: SHERLOCK VS APEX VS PORLOCK  ⚔️")
    print(f"    Engines Tested: {', '.join(bot_keys)}")
    print(f"    Total Boards  : {num_games:,} Identical Paired Fleet Layouts")
    print(f"    Total Games   : {num_games * len(bot_keys):,} Complete Simulated Matches")
    print(f"    CPU Cores     : {cores} Cores Engaged")
    print("=" * 76)

    chunks = []
    for start in range(0, num_games, chunk_size):
        count = min(chunk_size, num_games - start)
        chunks.append((start, count))

    all_results = {k: [] for k in bot_keys}
    start_time = time.time()
    completed_games = 0

    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(run_chunk, s, c, bot_keys): (s, c) for s, c in chunks}

        last_print = time.time()
        for fut in as_completed(futures):
            res = fut.result()
            for k in bot_keys:
                all_results[k].extend(res[k])
            completed_games += len(res[bot_keys[0]])

            if time.time() - last_print > 3.0 or completed_games == num_games:
                last_print = time.time()
                elapsed = time.time() - start_time
                gps = completed_games / elapsed if elapsed > 0 else 0
                eta = (num_games - completed_games) / gps if gps > 0 else 0
                pct = (completed_games / num_games) * 100
                avgs = " | ".join(f"{k}: {sum(all_results[k])/len(all_results[k]):.2f}" for k in bot_keys)
                print(f"[{completed_games:6,d} / {num_games:,}] ({pct:5.1f}%) | {elapsed:5.1f}s | {gps:5.1f} bds/s | ETA: {eta:4.1f}s | {avgs}")

    total_time = time.time() - start_time
    print("\n" + "=" * 76)
    print(f" 🏆 TOURNAMENT COMPLETED in {total_time:.2f}s ({num_games / total_time:.1f} paired boards/sec)")
    print("=" * 76)

    # 1. Performance Table
    print("\n📊 1. OVERALL ENGINE BENCHMARK (Ranked by Mean Shots)")
    print("-" * 76)
    print(f"{'Engine':<16} | {'Mean ± SEM':<18} | {'95% CI':<17} | {'Median':<6} | {'Std':<6} | {'>60':<7}")
    print("-" * 76)

    stats = {k: calculate_stats(all_results[k]) for k in bot_keys}
    ranked = sorted(bot_keys, key=lambda k: stats[k]['mean'])

    for rank, k in enumerate(ranked, 1):
        st = stats[k]
        mean_str = f"{st['mean']:.3f} ± {st['sem']:.3f}"
        ci_str = f"[{st['ci95'][0]:.3f}, {st['ci95'][1]:.3f}]"
        out_pct = (st['outliers'] / num_games) * 100
        out_str = f"{st['outliers']:,} ({out_pct:.2f}%)"
        print(f"#{rank} {k:<13} | {mean_str:<18} | {ci_str:<17} | {st['median']:<6.1f} | {st['std']:<6.2f} | {out_str:<7}")
    print("-" * 76)

    # 2. Pairwise Differences vs Sherlock
    if 'Sherlock' in bot_keys:
        print("\n🔬 2. PAIRED COMPARISON VS SHERLOCK (Baseline CSP)")
        print("-" * 76)
        print(f"{'Engine vs Sherlock':<26} | {'Diff':<12} | {'t-stat':<7} | {'p-value':<9} | {'Significance':<12}")
        print("-" * 76)

        s_arr = all_results['Sherlock']
        for k in [r for r in ranked if r != 'Sherlock']:
            k_arr = all_results[k]
            diffs = [sh - s for sh, s in zip(k_arr, s_arr)]
            mean_diff = sum(diffs) / len(diffs)
            var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
            sem_diff = math.sqrt(var_diff) / math.sqrt(len(diffs))
            t_stat = mean_diff / sem_diff
            p_val = math.erfc(abs(t_stat) / math.sqrt(2))
            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
            if p_val < 0.001: sig = "✅ Yes (p < 0.001) ***"
            elif p_val < 0.01: sig = "✅ Yes (p < 0.01) **"
            elif p_val < 0.05: sig = "✅ Yes (p < 0.05) *"
            else: sig = "❌ No (ns)"
            print(f"{k + ' vs Sherlock':<26} | {mean_diff:+.3f} shots | {t_stat:+6.2f} | {p_str:<9} | {sig}")

    # 3. Porlock vs Apex Showdown
    if 'Porlock' in bot_keys and 'Apex' in bot_keys:
        print("\n⚔️  3. THE TITAN SHOWDOWN: PORLOCK VS SHERRINFORD-APEX")
        print("-" * 76)
        p_arr = all_results['Porlock']
        a_arr = all_results['Apex']
        pa_diffs = [p - a for p, a in zip(p_arr, a_arr)]
        mean_pa = sum(pa_diffs) / len(pa_diffs)
        var_pa = sum((d - mean_pa) ** 2 for d in pa_diffs) / (len(pa_diffs) - 1)
        sem_pa = math.sqrt(var_pa) / math.sqrt(len(pa_diffs))
        t_pa = mean_pa / sem_pa
        p_val_pa = math.erfc(abs(t_pa) / math.sqrt(2))
        p_str_pa = f"{p_val_pa:.4f}" if p_val_pa >= 0.0001 else "< 0.0001"
        p_wins = sum(1 for d in pa_diffs if d < 0)
        a_wins = sum(1 for d in pa_diffs if d > 0)
        pa_ties = sum(1 for d in pa_diffs if d == 0)
        print(f"Porlock vs Apex Difference : {mean_pa:+.3f} shots (t = {t_pa:+.2f}, p = {p_str_pa})")
        print(f"Porlock Head-to-Head Wins : {p_wins:,} ({p_wins/num_games*100:.2f}%)")
        print(f"Apex Head-to-Head Wins    : {a_wins:,} ({a_wins/num_games*100:.2f}%)")
        print(f"Tied Matches              : {pa_ties:,} ({pa_ties/num_games*100:.2f}%)")

    # 4. Head-to-Head Win Matrix vs Sherlock
    if 'Sherlock' in bot_keys:
        print("\n🥊 4. DIRECT HEAD-TO-HEAD VS SHERLOCK (Winning % on Identical Boards)")
        print("-" * 76)
        s_arr = all_results['Sherlock']
        for k in [r for r in ranked if r != 'Sherlock']:
            k_arr = all_results[k]
            diffs = [sh - s for sh, s in zip(k_arr, s_arr)]
            wins = sum(1 for d in diffs if d < 0)
            losses = sum(1 for d in diffs if d > 0)
            ties = sum(1 for d in diffs if d == 0)
            print(f"  • {k:<12}: Won {wins:6,d} ({wins/num_games*100:5.2f}%) | Lost {losses:6,d} ({losses/num_games*100:5.2f}%) | Tied {ties:5,d} ({ties/num_games*100:4.2f}%)")

if __name__ == '__main__':
    main()

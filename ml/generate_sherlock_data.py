import os
import sys
import time
import argparse
import random
import numpy as np
import concurrent.futures

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.game_engine import BattleshipGame, CELL_UNKNOWN, CELL_MISS, CELL_HIT, CELL_SUNK
from bots.sherlock import SherlockBot

def simulate_single_game(game_id, snapshot_count=3):
    """
    Simulates a single game with SherlockBot and extracts mid-game training snapshots.
    Returns:
        samples: list of tuples (input_channels [4, 10, 10], occupancy_target [10, 10], next_shot_target int)
    """
    game = BattleshipGame()
    game.place_ships_randomly()
    bot = SherlockBot()

    # Ground truth fleet occupancy (17 pegs)
    truth_occupancy = np.zeros((10, 10), dtype=np.float32)
    for sid, coords in game.ship_pegs.items():
        for x, y in coords:
            truth_occupancy[x, y] = 1.0

    history_snapshots = []
    parity_offset = game_id % 2

    # Run the full game
    while not game.is_game_over() and game.shot_number < 100:
        target, _, _, _ = bot.get_best_move(game.tracking_board, game.move_history)
        if not target:
            break

        tx, ty = target
        next_shot_idx = tx * 10 + ty

        # Extract snapshot before firing
        board_snapshot = [row[:] for row in game.tracking_board]
        history_snapshots.append((board_snapshot, next_shot_idx))

        game.fire_shot(tx, ty)

    if not history_snapshots:
        return []

    # Pick snapshot_count evenly distributed turns across the hunt/target lifecycle
    total_shots = len(history_snapshots)
    if total_shots <= snapshot_count:
        selected_indices = list(range(total_shots))
    else:
        # e.g., 20%, 50%, 80% through the game
        step = total_shots / (snapshot_count + 1)
        selected_indices = [int(step * (i + 1)) for i in range(snapshot_count)]

    game_samples = []
    for idx in selected_indices:
        board_state, next_shot_idx = history_snapshots[idx]

        # 4 Channels: Misses, Active Hits, Sunk Pegs, Checkerboard Parity
        ch_misses = np.zeros((10, 10), dtype=np.float32)
        ch_hits = np.zeros((10, 10), dtype=np.float32)
        ch_sunk = np.zeros((10, 10), dtype=np.float32)
        ch_parity = np.zeros((10, 10), dtype=np.float32)

        for r in range(10):
            for c in range(10):
                val = board_state[r][c]
                if val == CELL_MISS:
                    ch_misses[r, c] = 1.0
                elif val == CELL_HIT:
                    ch_hits[r, c] = 1.0
                elif val == CELL_SUNK:
                    ch_sunk[r, c] = 1.0

                if (r + c) % 2 == parity_offset:
                    ch_parity[r, c] = 1.0

        channels = np.stack([ch_misses, ch_hits, ch_sunk, ch_parity], axis=0) # [4, 10, 10]
        game_samples.append((channels, truth_occupancy, next_shot_idx))

    return game_samples

def generate_shard(start_game_id, num_games, shard_index, output_dir, snapshot_count):
    """Simulates a batch of games and writes a single compressed .npz shard."""
    inputs_list = []
    occupancy_list = []
    policy_list = []

    for i in range(num_games):
        gid = start_game_id + i
        samples = simulate_single_game(gid, snapshot_count=snapshot_count)
        for inp, occ, pol in samples:
            inputs_list.append(inp)
            occupancy_list.append(occ)
            policy_list.append(pol)

    shard_path = os.path.join(output_dir, f"shard_{shard_index:03d}.npz")
    np.savez_compressed(
        shard_path,
        inputs=np.array(inputs_list, dtype=np.float32),
        occupancy_targets=np.array(occupancy_list, dtype=np.float32),
        policy_targets=np.array(policy_list, dtype=np.int64)
    )
    return shard_path, len(inputs_list)

def main():
    parser = argparse.ArgumentParser(description="High-Speed Parallel Generator for 1M Sherlock Games")
    parser.add_argument("--games", type=int, default=1000000, help="Total number of games to simulate (default: 1,000,000)")
    parser.add_argument("--shard-size", type=int, default=50000, help="Games per .npz shard file (default: 50,000)")
    parser.add_argument("--snapshots", type=int, default=3, help="Snapshots extracted per game (default: 3)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes (default: CPU cores - 2)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save shards")
    args = parser.parse_args()

    max_workers = args.workers if args.workers else max(1, (os.cpu_count() or 4) - 2)
    output_dir = args.output_dir if args.output_dir else os.path.join(PROJECT_ROOT, "data", "sherlock_1m_dataset")
    os.makedirs(output_dir, exist_ok=True)

    total_games = args.games
    shard_size = args.shard_size
    num_shards = (total_games + shard_size - 1) // shard_size

    print("=" * 65)
    print(f"🚀 BATTLESHIP 1M SHERLOCK DATA GENERATOR")
    print(f"   Target Games       : {total_games:,}")
    print(f"   Games Per Shard    : {shard_size:,} ({num_shards} total shards)")
    print(f"   Snapshots / Game   : {args.snapshots} (~{total_games * args.snapshots:,} training states)")
    print(f"   Worker Processes   : {max_workers}")
    print(f"   Output Directory   : {output_dir}")
    print("=" * 65)

    start_time = time.time()
    completed_games = 0
    total_samples = 0

    # Shard task definitions
    shard_tasks = []
    for shard_idx in range(num_shards):
        start_gid = shard_idx * shard_size + 1
        current_shard_games = min(shard_size, total_games - (shard_idx * shard_size))
        shard_tasks.append((start_gid, current_shard_games, shard_idx))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_shard, sgid, count, sidx, output_dir, args.snapshots): (sidx, count)
            for sgid, count, sidx in shard_tasks
        }

        for future in concurrent.futures.as_completed(futures):
            sidx, count = futures[future]
            try:
                shard_path, samples_count = future.result()
                completed_games += count
                total_samples += samples_count

                elapsed = time.time() - start_time
                gps = completed_games / elapsed if elapsed > 0 else 0
                remaining_games = total_games - completed_games
                eta_sec = remaining_games / gps if gps > 0 else 0

                size_mb = os.path.getsize(shard_path) / (1024 * 1024)
                print(f"[SHARD {sidx+1:02d}/{num_shards:02d}] Saved {os.path.basename(shard_path)} "
                      f"({size_mb:.1f} MB, {samples_count:,} samples) | "
                      f"Speed: {gps:.0f} games/s | Progress: {(completed_games/total_games)*100:.1f}% | "
                      f"ETA: {eta_sec/60:.1f}m")
            except Exception as e:
                print(f"❌ Error generating shard {sidx}: {e}")

    total_elapsed = time.time() - start_time
    print("=" * 65)
    print(f"✅ GENERATION COMPLETE!")
    print(f"   Total Games      : {completed_games:,}")
    print(f"   Total Samples    : {total_samples:,}")
    print(f"   Elapsed Time     : {total_elapsed/60:.2f} minutes ({completed_games/total_elapsed:.1f} games/sec)")
    print(f"   Dataset Location : {output_dir}")
    print("=" * 65)

if __name__ == "__main__":
    main()

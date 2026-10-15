import os
import sys
import csv
import random
import argparse
from collections import defaultdict
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.game_engine import BattleshipGame, CELL_UNKNOWN
from bots.sherlock import SherlockBot
from bots.mycroft import MycroftBot
from bots.neural_bot import NeuralBot

SHIP_NAME_MAP = {
    'Carrier': (1, 5),
    'Battleship': (2, 4),
    'Cruiser': (3, 3),
    'Submarine': (4, 3),
    'Destroyer': (5, 2),
    'SHIP': (1, 5)
}

def extract_holdout_games(csv_path, target_player='luki', test_split=0.2, seed=42):
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    player_games = defaultdict(list)
    for r in rows:
        if r['Player'].lower() == target_player.lower() or r['Opponent'].lower() == target_player.lower():
            player_games[r['GameID']].append(r)

    game_ids = list(player_games.keys())
    game_ids.sort()
    random.seed(seed)
    random.shuffle(game_ids)

    split_idx = int(len(game_ids) * (1 - test_split))
    test_gids = game_ids[split_idx:]

    holdout_fleets = []
    for gid in test_gids:
        g_rows = player_games[gid]
        ship_groups = defaultdict(list)
        
        for r in g_rows:
            is_ship = r['ShotResult'] in SHIP_NAME_MAP
            is_target_ship = (r['Player'].lower() == target_player.lower() and r['Grid'] == 'Defense') or \
                             (r['Opponent'].lower() == target_player.lower() and r['Grid'] == 'Offense' and is_ship)
            if is_ship and is_target_ship:
                ship_name = r['ShotResult']
                pos = (int(r['X']), int(r['Y']))
                if pos not in ship_groups[ship_name]:
                    ship_groups[ship_name].append(pos)

        total_pegs = sum(len(pegs) for pegs in ship_groups.values())
        if total_pegs == 17:
            holdout_fleets.append((gid, ship_groups))

    return holdout_fleets

def create_game_with_fleet(ship_groups):
    game = BattleshipGame()
    for ship_name, pegs in ship_groups.items():
        ship_id, _ = SHIP_NAME_MAP.get(ship_name, (1, len(pegs)))
        game.ship_pegs[ship_id] = list(pegs)
        game.ship_health[ship_id] = len(pegs)
        for x, y in pegs:
            game.truth_board[x][y] = ship_id
    return game

def simulate_bot_on_fleet(bot, ship_groups, parity_offset=0):
    if hasattr(bot, 'parity_offset'):
        bot.parity_offset = parity_offset
    if hasattr(bot, 'sherlock_assassin') and hasattr(bot.sherlock_assassin, 'parity_offset'):
        bot.sherlock_assassin.parity_offset = parity_offset

    game = create_game_with_fleet(ship_groups)
    first_hit_turn = None
    first_shot_hit = False

    while not game.is_game_over():
        target, strategy, dead_sizes, diagnostics = bot.get_best_move(
            game.tracking_board,
            game.move_history
        )
        if not target:
            break
        
        result = game.fire_shot(target[0], target[1])
        if game.shot_number == 1 and result in ['HIT', 'SUNK']:
            first_shot_hit = True
        if first_hit_turn is None and result in ['HIT', 'SUNK']:
            first_hit_turn = game.shot_number

    return {
        'shots': game.shot_number,
        'first_hit': first_hit_turn or game.shot_number,
        'first_shot_hit': first_shot_hit
    }

def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate Bots on Human Holdout Matches")
    parser.add_argument("--include-mycroft", action="store_true", help="Include slow MCMC MycroftBot")
    parser.add_argument("--player", type=str, default="luki")
    args = parser.parse_args()

    csv_path = os.path.join(DATA_DIR, 'Battleship_Export_20260729.csv')
    print("=" * 70)
    print(f"🔍 EVALUATING BOTS AGAINST {args.player.upper()}'S HOLDOUT FLEETS")
    print("=" * 70)

    holdout_fleets = extract_holdout_games(csv_path, target_player=args.player)
    print(f"Extracted {len(holdout_fleets)} valid holdout test games for '{args.player}'.\n")

    base_weights = os.path.join(DATA_DIR, 'sherlock_nn_best.pth')
    luki_weights = os.path.join(DATA_DIR, f'moriarty_{args.player.lower()}_best.pth')

    bots = {
        'Sherlock (Parity)': SherlockBot(),
        'Neural Foundation (1M)': NeuralBot(weights_path=base_weights),
        'Neural Moriarty (Fine-Tuned)': NeuralBot(weights_path=luki_weights)
    }

    if args.include_mycroft:
        bots['Mycroft (MCMC)'] = MycroftBot()

    results = defaultdict(list)

    for gid, fleet in holdout_fleets:
        parity_offset = int(gid) % 2 if gid.isdigit() else 0
        for name, bot in bots.items():
            metrics = simulate_bot_on_fleet(bot, fleet, parity_offset=parity_offset)
            results[name].append(metrics)

    print(f"{'Bot Name':<30} | {'Avg Shots':<10} | {'Median':<8} | {'Best/Worst':<12} | {'1st Hit Turn':<13} | {'1st Shot Hit %':<14}")
    print("-" * 97)

    for name in bots.keys():
        res = results[name]
        shots = [m['shots'] for m in res]
        first_hits = [m['first_hit'] for m in res]
        first_hits_pct = (sum(1 for m in res if m['first_shot_hit']) / len(res)) * 100

        avg_shots = np.mean(shots)
        med_shots = np.median(shots)
        best_shots = min(shots)
        worst_shots = max(shots)
        avg_first_hit = np.mean(first_hits)

        print(f"{name:<28} | {avg_shots:<10.2f} | {med_shots:<8.1f} | {best_shots:2d} / {worst_shots:2d}{'':<5} | {avg_first_hit:<13.2f} | {first_hits_pct:<14.1f}%")

    print("=" * 95)

if __name__ == "__main__":
    run_evaluation()

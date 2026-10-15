import os
import sys
import csv
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.train_sherlock_nn import SherlockNeuralModel

SHIP_NAMES = {'Carrier', 'Battleship', 'Cruiser', 'Submarine', 'Destroyer', 'SHIP'}

class PersonalizedFineTuneDataset(Dataset):
    def __init__(self, csv_path, target_player='luki', anchor_shard_path=None, anchor_ratio=0.2, snapshots_per_game=3, is_train=True, test_split=0.2, seed=42):
        self.samples = []
        random.seed(seed)

        # 1. Parse CSV and isolate target player games
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))

        player_games = defaultdict(list)
        for r in rows:
            if r['Player'].lower() == target_player.lower() or r['Opponent'].lower() == target_player.lower():
                player_games[r['GameID']].append(r)

        game_ids = list(player_games.keys())
        game_ids.sort()
        random.shuffle(game_ids)

        split_idx = int(len(game_ids) * (1 - test_split))
        selected_gids = game_ids[:split_idx] if is_train else game_ids[split_idx:]

        print(f"[{'TRAIN' if is_train else 'TEST'}] Extracting samples from {len(selected_gids)} matches for '{target_player}'...")

        # 2. Extract board snapshots for selected games
        for gid in selected_gids:
            g_rows = player_games[gid]
            ships = set()
            incoming_shots = []

            for r in g_rows:
                is_ship = r['ShotResult'] in SHIP_NAMES
                is_target_ship = (r['Player'].lower() == target_player.lower() and r['Grid'] == 'Defense') or \
                                 (r['Opponent'].lower() == target_player.lower() and r['Grid'] == 'Offense' and is_ship)

                if is_ship and is_target_ship:
                    ships.add((int(r['X']), int(r['Y'])))

                is_incoming = (r['Player'].lower() == target_player.lower() and r['Grid'] == 'Defense' and not is_ship) or \
                              (r['Opponent'].lower() == target_player.lower() and r['Grid'] == 'Offense' and not is_ship)
                if is_incoming:
                    incoming_shots.append((int(r['X']), int(r['Y']), r['ShotResult']))

            if len(ships) < 17:
                # Incomplete fleet recording, skip
                continue

            truth_occupancy = np.zeros((10, 10), dtype=np.float32)
            for x, y in ships:
                truth_occupancy[x, y] = 1.0

            parity_offset = int(gid) % 2 if gid.isdigit() else 0

            # Determine snapshot cutoffs
            cutoffs = [0] # Always include Turn 0 (pure hunt phase prior)
            total_shots = len(incoming_shots)
            if total_shots > 0:
                for frac in [0.25, 0.50, 0.75]:
                    c = int(total_shots * frac)
                    if c not in cutoffs and c < total_shots:
                        cutoffs.append(c)

            for shot_idx in cutoffs:
                ch_misses = np.zeros((10, 10), dtype=np.float32)
                ch_hits = np.zeros((10, 10), dtype=np.float32)
                ch_sunk = np.zeros((10, 10), dtype=np.float32)
                ch_parity = np.zeros((10, 10), dtype=np.float32)

                for r in range(10):
                    for c in range(10):
                        if (r + c) % 2 == parity_offset:
                            ch_parity[r, c] = 1.0

                for i in range(shot_idx):
                    sx, sy, sres = incoming_shots[i]
                    if sres == 'MISS':
                        ch_misses[sx, sy] = 1.0
                    elif sres == 'HIT':
                        ch_hits[sx, sy] = 1.0
                    elif sres == 'SUNK':
                        ch_sunk[sx, sy] = 1.0

                input_tensor = np.stack([ch_misses, ch_hits, ch_sunk, ch_parity], axis=0) # [4, 10, 10]

                # Policy target: pick an unrevealed ship peg ON PARITY
                unrevealed_on_parity = [pos for pos in ships if ch_hits[pos[0], pos[1]] == 0 and ch_sunk[pos[0], pos[1]] == 0 and (pos[0] + pos[1]) % 2 == parity_offset]
                all_unrevealed = [pos for pos in ships if ch_hits[pos[0], pos[1]] == 0 and ch_sunk[pos[0], pos[1]] == 0]

                if shot_idx == 0 and is_train and unrevealed_on_parity:
                    # On Turn 0 (pure hunt phase), add samples for each on-parity ship peg so the network learns the full defensive layout
                    for pos in unrevealed_on_parity:
                        self.samples.append((input_tensor, truth_occupancy, pos[0] * 10 + pos[1]))
                else:
                    if unrevealed_on_parity:
                        chosen = random.choice(unrevealed_on_parity)
                        next_target_idx = chosen[0] * 10 + chosen[1]
                    elif all_unrevealed:
                        chosen = random.choice(all_unrevealed)
                        next_target_idx = chosen[0] * 10 + chosen[1]
                    elif incoming_shots and shot_idx < len(incoming_shots):
                        next_target_idx = incoming_shots[shot_idx][0] * 10 + incoming_shots[shot_idx][1]
                    else:
                        next_target_idx = 0
                    self.samples.append((input_tensor, truth_occupancy, next_target_idx))

        num_player_samples = len(self.samples)
        print(f"  -> Generated {num_player_samples} personalized snapshots.")

        # 3. Add Anchor Buffer from Sherlock foundation data (for train set only)
        if is_train and anchor_shard_path and os.path.exists(anchor_shard_path) and anchor_ratio > 0:
            anchor_data = np.load(anchor_shard_path)
            anchor_inputs = anchor_data['inputs']
            anchor_occs = anchor_data['occupancy_targets']
            anchor_pols = anchor_data['policy_targets']

            num_anchors = int(num_player_samples * (anchor_ratio / (1 - anchor_ratio)))
            num_anchors = min(num_anchors, len(anchor_inputs))

            indices = random.sample(range(len(anchor_inputs)), num_anchors)
            for idx in indices:
                self.samples.append((anchor_inputs[idx], anchor_occs[idx], int(anchor_pols[idx])))

            print(f"  -> Injected {num_anchors} anchor foundation states ({anchor_ratio*100:.0f}% buffer).")

        random.shuffle(self.samples)
        print(f"  -> Total dataset size: {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, occ, pol = self.samples[idx]
        return (
            torch.from_numpy(inp),
            torch.from_numpy(occ),
            torch.tensor(pol, dtype=torch.long)
        )

def fine_tune():
    parser = argparse.ArgumentParser(description="Fine-tune Sherlock Foundation Model against a Specific Player")
    parser.add_argument("--player", type=str, default="luki", help="Target player name")
    parser.add_argument("--csv", type=str, default=os.path.join(DATA_DIR, "Battleship_Export_20260729.csv"))
    parser.add_argument("--base-weights", type=str, default=os.path.join(DATA_DIR, "sherlock_nn_best.pth"))
    parser.add_argument("--anchor-shard", type=str, default=os.path.join(DATA_DIR, "sherlock_1m_dataset", "shard_000.npz"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Gentle fine-tuning learning rate")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(f"🎯 FINE-TUNING FOUNDATION MODEL AGAINST: {args.player.upper()}")
    print(f"   Base Foundation Weights : {args.base_weights}")
    print(f"   Device                  : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Epochs                  : {args.epochs}")
    print(f"   Batch Size              : {args.batch_size}")
    print(f"   Fine-Tuning LR          : {args.lr}")
    print("=" * 65)

    if not os.path.exists(args.base_weights):
        raise FileNotFoundError(f"Base weights {args.base_weights} not found! Train Sherlock foundation model first.")

    train_dataset = PersonalizedFineTuneDataset(
        csv_path=args.csv,
        target_player=args.player,
        anchor_shard_path=args.anchor_shard,
        anchor_ratio=0.2,
        is_train=True
    )

    test_dataset = PersonalizedFineTuneDataset(
        csv_path=args.csv,
        target_player=args.player,
        anchor_shard_path=None,
        anchor_ratio=0.0,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = SherlockNeuralModel(in_channels=4).to(device)
    model.load_state_dict(torch.load(args.base_weights, map_location=device))
    print(f"✅ Loaded base foundation weights from {args.base_weights}")

    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    output_weights_path = os.path.join(DATA_DIR, f"moriarty_{args.player.lower()}_best.pth")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_x, batch_occ, batch_pol in train_loader:
            batch_x = batch_x.to(device)
            batch_occ_flat = batch_occ.view(batch_occ.size(0), -1).to(device)
            batch_pol = batch_pol.to(device)

            optimizer.zero_grad()
            occ_logits, pol_logits = model(batch_x)

            loss_occ = bce_loss_fn(occ_logits, batch_occ_flat)
            loss_pol = ce_loss_fn(pol_logits, batch_pol)
            loss = loss_occ + (0.5 * loss_pol)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_samples += batch_x.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for val_x, val_occ, val_pol in test_loader:
                val_x = val_x.to(device)
                val_occ_flat = val_occ.view(val_occ.size(0), -1).to(device)
                val_pol = val_pol.to(device)

                occ_logits, pol_logits = model(val_x)
                loss_occ = bce_loss_fn(occ_logits, val_occ_flat)
                loss_pol = ce_loss_fn(pol_logits, val_pol)
                loss = loss_occ + (0.5 * loss_pol)

                val_loss += loss.item() * val_x.size(0)
                val_samples += val_x.size(0)

        avg_train = train_loss / train_samples if train_samples > 0 else 0
        avg_val = val_loss / val_samples if val_samples > 0 else 0

        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_weights_path)
            print(f"  -> ⭐ Saved new best personalized model to {output_weights_path}")

    print("\n✅ Fine-tuning complete!")

    # ONNX export
    onnx_path = os.path.join(DATA_DIR, f"moriarty_{args.player.lower()}.onnx")
    print(f"Exporting ONNX model to {onnx_path}...")
    model.load_state_dict(torch.load(output_weights_path, map_location='cpu'))
    model.to('cpu')
    model.eval()
    dummy_input = torch.randn(1, 4, 10, 10)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['board_input'],
            output_names=['occupancy_logits', 'policy_logits']
        )
        print(f"✅ Successfully exported personalized ONNX to {onnx_path}")
    except Exception as e:
        print(f"Warning during ONNX export: {e}")

if __name__ == "__main__":
    fine_tune()

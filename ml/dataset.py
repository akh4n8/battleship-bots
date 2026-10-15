import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# CONSTANTS & HYPERPARAMETERS
# ==========================================
SHIP_WEIGHTS = {
    'Carrier': 0.4,
    'Battleship': 0.6,
    'Cruiser': 0.8,
    'Submarine': 0.8,
    'Destroyer': 1.0,
    'SHIP': 0.7
}

TRAUMA_DECAY_GAMMA = 0.85
SEQUENCE_LENGTH = 5

# ==========================================
# DATASET CLASS
# ==========================================
class CognitiveBattleshipDataset(Dataset):
    def __init__(self, csv_file_path, is_offense_model=False, target_player=None, only_wins=False):
        self.is_offense_model = is_offense_model
        self.target_player = target_player
        self.only_wins = only_wins
        self.df_raw = pd.read_csv(csv_file_path)
        
        if 'Timestamp' in self.df_raw.columns:
            self.df_raw['Timestamp'] = pd.to_datetime(self.df_raw['Timestamp'])
            self.df_raw = self.df_raw.sort_values('Timestamp')
            
        self.samples, self.labels = self._build_tensor_sequences()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    def _build_tensor_sequences(self):
        samples = []
        labels = []
        
        all_players = set(self.df_raw['Player'].dropna().unique()) | set(self.df_raw['Opponent'].dropna().unique())
        
        if self.target_player:
            humans = [self.target_player]
            win_text = " (WINS ONLY)" if self.only_wins else ""
            print(f"Personalized Mode: Compiling training data exclusively for '{self.target_player}'{win_text}...")
        else:
            humans = [p for p in all_players if "Bot" not in p and p not in ["PassAndPlay", "Companion"]]
            print(f"Discovered {len(humans)} human profiles. Compiling universal training data...")

        for player_name in humans:
            player_df = self.df_raw[(self.df_raw['Player'] == player_name) | (self.df_raw['Opponent'] == player_name)].copy()
            
            # ELITE FILTERING
            if self.only_wins:
                # Find games where player_name won
                won_games = []
                for gid in player_df['GameID'].unique():
                    g = player_df[player_df['GameID'] == gid]
                    if len(g) > 0:
                        first_row = g.iloc[0]
                        if first_row['MatchOutcome'] == 'WIN' and first_row['Player'] == player_name:
                            won_games.append(gid)
                        elif first_row['MatchOutcome'] == 'LOSS' and first_row['Opponent'] == player_name:
                            won_games.append(gid)
                player_df = player_df[player_df['GameID'].isin(won_games)]

            unique_games = player_df['GameID'].unique()
            
            # CRITICAL PERFORMANCE HOTFIX: 
            # 100,000 games creates 1.2GB of python object overhead. Cap at 20k for pre-training.
            if len(unique_games) > 20000:
                unique_games = unique_games[:20000]
            
            if len(unique_games) < SEQUENCE_LENGTH + 1:
                continue 
                
            processed_games = []
            base_trauma_misses = np.zeros((10, 10), dtype=np.float32)
            base_trauma_hits = np.zeros((10, 10), dtype=np.float32)
            
            # Use groupby to avoid O(N^2) dataframe slicing
            grouped = player_df.groupby('GameID', sort=False)
            
            for game_id in unique_games:
                try:
                    df_game = grouped.get_group(game_id)
                except KeyError:
                    continue
                channels, base_trauma_misses, base_trauma_hits = self._process_single_game(df_game, base_trauma_misses, base_trauma_hits, player_name)
                
                processed_games.append({
                    'game_id': game_id,
                    'channels': channels,
                    'df_game': df_game
                })

            cumulative_priors = []
            current_prior = np.zeros((10, 10), dtype=np.float32)
            for game in processed_games:
                current_prior = current_prior + game['channels'][6]
                cumulative_priors.append(current_prior.copy())
                
            for i in range(len(processed_games) - SEQUENCE_LENGTH):
                history_slice = processed_games[i : i + SEQUENCE_LENGTH]
                target_idx = i + SEQUENCE_LENGTH
                target_game = processed_games[target_idx]
                
                if self.is_offense_model:
                    df_target = target_game['df_game']
                    max_turns = df_target['TurnNumber'].max()
                    if pd.isna(max_turns):
                        continue
                    cutoffs = [int(max_turns * 0.25), int(max_turns * 0.50), int(max_turns * 0.75)]
                    
                    for turn_cutoff in cutoffs:
                        turn_cutoff = max(1, turn_cutoff)
                        partial_df = df_target[df_target['TurnNumber'] <= turn_cutoff].copy()
                        
                        if len(partial_df) == 0:
                            continue
                            
                        live_channels, _, _ = self._process_single_game(partial_df, target_game['channels'][0], target_game['channels'][1], player_name)
                        
                        bayesian_prior = cumulative_priors[target_idx - 1].copy()
                        max_prior = np.max(bayesian_prior)
                        if max_prior > 0:
                            bayesian_prior /= max_prior
                        
                        seq_list = []
                        for game in history_slice:
                            c = game['channels']
                            # Offense 5 Channels: Spatial(6), TraumaMiss(0), TraumaHit(1), BotLiveMiss(7), BotLiveHit(8)
                            ch_5 = np.stack([c[6], c[0], c[1], c[7], c[8]])
                            seq_list.append(ch_5)
                        
                        lc = live_channels
                        live_5 = np.stack([np.zeros((10, 10), dtype=np.float32), lc[0], lc[1], lc[7], lc[8]])
                        seq_list.append(live_5)
                        
                        seq_tensor = np.stack(seq_list)
                        samples.append(torch.tensor(seq_tensor, dtype=torch.float32))
                        labels.append(torch.tensor(target_game['channels'][6], dtype=torch.float32))
                else:
                    seq_list = []
                    for game in history_slice:
                        c = game['channels']
                        # Defense 7 Channels: TraumaMiss(0), TraumaHit(1), HumanMiss(2), HumanHit(3), Clustering(4), Chronology(5), Spatial(6)
                        ch_7 = np.stack([c[0], c[1], c[2], c[3], c[4], c[5], c[6]])
                        seq_list.append(ch_7)
                    
                    seq_tensor = np.stack(seq_list)
                    samples.append(torch.tensor(seq_tensor, dtype=torch.float32))
                    
                    target_c = target_game['channels']
                    target_label = np.clip(target_c[2] + target_c[3], 0.0, 1.0)
                    labels.append(torch.tensor(target_label, dtype=torch.float32))

        if len(samples) == 0:
            raise ValueError(f"No players found with enough games to train!")
            
        return samples, labels

    def _process_single_game(self, df_game, base_trauma_misses, base_trauma_hits, player_name):
        player_is_p1 = (df_game.iloc[0]['Player'] == player_name)
        
        ch_trauma_misses = base_trauma_misses.copy()
        ch_trauma_hits = base_trauma_hits.copy()
        ch_human_misses = np.zeros((10, 10), dtype=np.float32)
        ch_human_hits = np.zeros((10, 10), dtype=np.float32)
        ch_clustering = np.zeros((10, 10), dtype=np.float32)
        ch_chronology = np.zeros((10, 10), dtype=np.float32)
        ch_spatial = np.zeros((10, 10), dtype=np.float32)

        if player_is_p1:
            human_shots = df_game[(df_game['Grid'] == 'Offense') & (~df_game['ShotResult'].isin(SHIP_WEIGHTS.keys()))].copy()
            bot_shots = df_game[(df_game['Grid'] == 'Defense') & (~df_game['ShotResult'].isin(SHIP_WEIGHTS.keys()))].copy()
            human_ships = df_game[(df_game['Grid'] == 'Defense') & (df_game['ShotResult'].isin(SHIP_WEIGHTS.keys()))]
        else:
            human_shots = df_game[(df_game['Grid'] == 'Defense') & (~df_game['ShotResult'].isin(SHIP_WEIGHTS.keys()))].copy()
            bot_shots = df_game[(df_game['Grid'] == 'Offense') & (~df_game['ShotResult'].isin(SHIP_WEIGHTS.keys()))].copy()
            human_ships = df_game[(df_game['Grid'] == 'Offense') & (df_game['ShotResult'].isin(SHIP_WEIGHTS.keys()))]

        human_shots = human_shots.sort_values('ShotNumber')
        total_shots = len(human_shots)
        hit_queue = []

        for idx, (_, row) in enumerate(human_shots.iterrows()):
            x, y = int(row['X']), int(row['Y'])
            if 0 <= x <= 9 and 0 <= y <= 9:
                if total_shots > 1:
                    ch_chronology[x, y] = idx / (total_shots - 1)
                
                is_hit = row['ShotResult'] in ['HIT', 'SUNK']
                if is_hit:
                    ch_human_hits[x, y] = 1.0
                    ch_clustering[x, y] = 1.0
                    hit_queue.append((x, y))
                else:
                    ch_human_misses[x, y] = 1.0
                    for hx, hy in hit_queue:
                        if abs(hx - x) <= 1 and abs(hy - y) <= 1:
                            ch_clustering[x, y] = 0.5
                            break
        
        for _, row in human_ships.iterrows():
            x, y = int(row['X']), int(row['Y'])
            if 0 <= x <= 9 and 0 <= y <= 9:
                ch_spatial[x, y] = SHIP_WEIGHTS.get(row['ShotResult'], 0.7)

        bot_shots_misses = np.zeros((10, 10), dtype=np.float32)
        bot_shots_hits = np.zeros((10, 10), dtype=np.float32)
        for _, row in bot_shots.iterrows():
            x, y = int(row['X']), int(row['Y'])
            if 0 <= x <= 9 and 0 <= y <= 9:
                is_hit = row['ShotResult'] in ['HIT', 'SUNK']
                if is_hit:
                    ship_row = human_ships[(human_ships['X'] == x) & (human_ships['Y'] == y)]
                    weight = SHIP_WEIGHTS.get(ship_row.iloc[0]['ShotResult'], 0.7) if not ship_row.empty else 0.7
                    bot_shots_hits[x, y] += weight
                else:
                    bot_shots_misses[x, y] += 1.0
        
        next_base_trauma_misses = (ch_trauma_misses * TRAUMA_DECAY_GAMMA) + bot_shots_misses
        next_base_trauma_hits = (ch_trauma_hits * TRAUMA_DECAY_GAMMA) + bot_shots_hits

        channels = np.stack([
            ch_trauma_misses, ch_trauma_hits, ch_human_misses, ch_human_hits, 
            ch_clustering, ch_chronology, ch_spatial, bot_shots_misses, bot_shots_hits
        ])
        return channels, next_base_trauma_misses, next_base_trauma_hits

def get_grouped_train_test_split(dataset, test_ratio=0.2, seed=42):
    import random
    from torch.utils.data import Subset
    random.seed(seed)
    
    unique_hashes = list(set(hash(label.numpy().tobytes()) for _, label in dataset))
    unique_hashes.sort()
    random.shuffle(unique_hashes)
    
    test_size = int(len(unique_hashes) * test_ratio)
    test_hashes = set(unique_hashes[:test_size])
    
    train_indices = []
    test_indices = []
    for i in range(len(dataset)):
        label_hash = hash(dataset.labels[i].numpy().tobytes())
        if label_hash in test_hashes:
            test_indices.append(i)
        else:
            train_indices.append(i)
            
    return Subset(dataset, train_indices), Subset(dataset, test_indices)

# ==========================================
# TEST EXECUTOR
# ==========================================
if __name__ == "__main__":
    # Example usage:
    # Replace 'battleship_history.csv' with your actual exported SQLite CSV
    try:
        dataset = CognitiveBattleshipDataset(
            csv_file_path='../data/Battleship_Export_20260729.csv', 
            is_offense_model=False 
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test loading one batch
        sample_batch, label_batch = next(iter(dataloader))
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Input Sequence Tensor Shape: {sample_batch.shape}  -> [Batch, SeqLen, Channels, Height, Width]")
        print(f"Target Label Tensor Shape: {label_batch.shape}  -> [Batch, Height, Width]")
        
    except FileNotFoundError:
        print("Waiting for battleship_history.csv to be provided in the directory...")
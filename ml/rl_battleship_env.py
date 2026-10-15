import torch
import numpy as np
import random

CELL_UNKNOWN = 0
CELL_MISS = -1
CELL_HIT = 1
CELL_SUNK = 2

FLEET_SIZES = [5, 4, 3, 3, 2] # 17 total pegs

class DecomposedBattleshipEnv:
    """
    Decomposed Hunt-Target Battleship Environment for PPO Reinforcement Learning.
    
    Key Features:
    1. The RL Agent ONLY controls the Hunt Phase on active checkerboard parity.
    2. Whenever a Hunt shot lands a Hit, an integrated Deterministic Linear Kill 
       solver immediately resolves the wounded ship to completion.
    3. Smooth, continuous linear reward signal (Zero cliff-edges):
       Reward = -shots_taken + SunkBonus + (50 - total_shots when cleared).
    4. Action Masking strictly enforces legal on-parity hunting tiles.
    """
    def __init__(self, num_envs=256, device='cuda'):
        self.num_envs = num_envs
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.grid_size = 10
        self.total_pegs = 17

        # CPU representations for fast board truth tracking
        self.truth_boards = np.zeros((num_envs, 10, 10), dtype=np.int32)
        self.ship_id_maps = np.zeros((num_envs, 10, 10), dtype=np.int32)
        self.tracking_boards = np.zeros((num_envs, 10, 10), dtype=np.int32) # 0: unknown, -1: miss, 1: hit, 2: sunk
        self.ship_pegs = [{} for _ in range(num_envs)] # dict {ship_id: [(r,c), ...]}
        self.ship_health = np.zeros((num_envs, 6), dtype=np.int32)

        # PyTorch State Tensors [N, 4, 10, 10]
        # Channels: 0: Misses, 1: Hits (active), 2: Sunk, 3: Parity
        self.states = torch.zeros((num_envs, 4, 10, 10), dtype=torch.float32, device=self.device)
        self.action_masks = torch.ones((num_envs, 100), dtype=torch.float32, device=self.device)

        self.pegs_remaining = np.full(num_envs, 17, dtype=np.int32)
        self.shots_taken = np.zeros(num_envs, dtype=np.int32)
        self.parity_offsets = np.zeros(num_envs, dtype=np.int32)
        self.game_counter = 0

        for i in range(num_envs):
            self._reset_env(i)

    def _reset_env(self, env_idx):
        self.game_counter += 1
        truth = np.zeros((10, 10), dtype=np.int32)
        s_map = np.zeros((10, 10), dtype=np.int32)
        pegs_dict = {}
        health = np.zeros(6, dtype=np.int32)

        p_offset = self.game_counter % 2
        self.parity_offsets[env_idx] = p_offset

        # Place 5 ships randomly
        for s_id, length in enumerate(FLEET_SIZES, start=1):
            placed = False
            while not placed:
                is_horiz = random.random() < 0.5
                if is_horiz:
                    r = random.randint(0, 9)
                    c = random.randint(0, 10 - length)
                    if truth[r, c:c+length].sum() == 0:
                        truth[r, c:c+length] = 1
                        s_map[r, c:c+length] = s_id
                        pegs_dict[s_id] = [(r, c+i) for i in range(length)]
                        health[s_id] = length
                        placed = True
                else:
                    r = random.randint(0, 10 - length)
                    c = random.randint(0, 9)
                    if truth[r:r+length, c].sum() == 0:
                        truth[r:r+length, c] = 1
                        s_map[r:r+length, c] = s_id
                        pegs_dict[s_id] = [(r+i, c) for i in range(length)]
                        health[s_id] = length
                        placed = True

        self.truth_boards[env_idx] = truth
        self.ship_id_maps[env_idx] = s_map
        self.tracking_boards[env_idx].fill(CELL_UNKNOWN)
        self.ship_pegs[env_idx] = pegs_dict
        self.ship_health[env_idx] = health
        self.pegs_remaining[env_idx] = 17
        self.shots_taken[env_idx] = 0

        # Tensor state update
        self.states[env_idx].zero_()
        grid_r = torch.arange(10, device=self.device).view(-1, 1).expand(10, 10)
        grid_c = torch.arange(10, device=self.device).view(1, -1).expand(10, 10)
        self.states[env_idx, 3] = ((grid_r + grid_c) % 2 == p_offset).float()

        self._update_action_mask(env_idx)

    def _update_action_mask(self, env_idx):
        tb = self.tracking_boards[env_idx]
        p_offset = self.parity_offsets[env_idx]

        mask = np.zeros(100, dtype=np.float32)
        on_parity_open = 0

        for r in range(10):
            for c in range(10):
                if tb[r, c] == CELL_UNKNOWN:
                    if (r + c) % 2 == p_offset:
                        mask[r * 10 + c] = 1.0
                        on_parity_open += 1

        # If on-parity exhausted, enable all remaining unshot cells
        if on_parity_open == 0:
            for r in range(10):
                for c in range(10):
                    if tb[r, c] == CELL_UNKNOWN:
                        mask[r * 10 + c] = 1.0

        self.action_masks[env_idx] = torch.from_numpy(mask).to(self.device)

    def _execute_linear_kill(self, env_idx, initial_hit_r, initial_hit_c):
        """
        Executes deterministic Linear Kill for a wounded ship until sunk.
        Returns:
            shots_taken (int): total shots consumed in this assassination sequence.
        """
        tb = self.tracking_boards[env_idx]
        truth = self.truth_boards[env_idx]
        s_map = self.ship_id_maps[env_idx]

        shots_consumed = 0
        active_hits = [(initial_hit_r, initial_hit_c)]

        target_ship_id = s_map[initial_hit_r, initial_hit_c]

        while self.ship_health[env_idx, target_ship_id] > 0:
            target = None

            if len(active_hits) == 1:
                # Test orthogonal unknown neighbors
                r, c = active_hits[0]
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                valid = [p for p in neighbors if 0 <= p[0] < 10 and 0 <= p[1] < 10 and tb[p[0], p[1]] == CELL_UNKNOWN]
                if valid:
                    target = valid[0]
            else:
                # Inline extension
                rows = [p[0] for p in active_hits]
                cols = [p[1] for p in active_hits]

                if len(set(rows)) == 1:
                    # Horizontal line
                    r = rows[0]
                    c_min, c_max = min(cols), max(cols)
                    if c_min - 1 >= 0 and tb[r, c_min - 1] == CELL_UNKNOWN:
                        target = (r, c_min - 1)
                    elif c_max + 1 < 10 and tb[r, c_max + 1] == CELL_UNKNOWN:
                        target = (r, c_max + 1)
                else:
                    # Vertical line
                    c = cols[0]
                    r_min, r_max = min(rows), max(rows)
                    if r_min - 1 >= 0 and tb[r_min - 1, c] == CELL_UNKNOWN:
                        target = (r_min - 1, c)
                    elif r_max + 1 < 10 and tb[r_max + 1, c] == CELL_UNKNOWN:
                        target = (r_max + 1, c)

            if target is None:
                # Edge case: fallback to any unknown neighbor
                for hr, hc in active_hits:
                    neighbors = [(hr-1, hc), (hr+1, hc), (hr, hc-1), (hr, hc+1)]
                    valid = [p for p in neighbors if 0 <= p[0] < 10 and 0 <= p[1] < 10 and tb[p[0], p[1]] == CELL_UNKNOWN]
                    if valid:
                        target = valid[0]
                        break
                if target is None:
                    break # Failsafe

            tr, tc = target
            shots_consumed += 1

            if truth[tr, tc] == 1:
                # Hit!
                tb[tr, tc] = CELL_HIT
                hit_s_id = s_map[tr, tc]
                self.ship_health[env_idx, hit_s_id] -= 1
                self.pegs_remaining[env_idx] -= 1
                active_hits.append((tr, tc))
            else:
                # Miss
                tb[tr, tc] = CELL_MISS

        # Once ship is sunk, convert all its pegs to CELL_SUNK
        if self.ship_health[env_idx, target_ship_id] == 0:
            for pr, pc in self.ship_pegs[env_idx][target_ship_id]:
                tb[pr, pc] = CELL_SUNK

        return shots_consumed

    def step(self, hunt_actions):
        """
        Executes Hunt Phase actions chosen by the RL agent.
        hunt_actions: [N] tensor of ints 0..99
        Returns:
            states: [N, 4, 10, 10]
            rewards: [N]
            dones: [N]
            completed_games: list of completed game stats
        """
        actions = hunt_actions.cpu().numpy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        completed_games = []

        for i in range(self.num_envs):
            act = actions[i]
            r = act // 10
            c = act % 10

            tb = self.tracking_boards[i]
            truth = self.truth_boards[i]

            # 1. Evaluate Hunt Shot
            if truth[r, c] == 0:
                # Hunt Miss
                tb[r, c] = CELL_MISS
                self.shots_taken[i] += 1
                
                # Dense Information-Gain Shaping: openness bonus
                open_neighbors = sum(1 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-2,0), (2,0), (0,-2), (0,2)]
                                     if 0 <= r+dr < 10 and 0 <= c+dc < 10 and tb[r+dr, c+dc] == CELL_UNKNOWN)
                open_bonus = (open_neighbors / 8.0) * 0.1
                rewards[i] = -1.0 + open_bonus
            else:
                # Hunt Hit! Trigger Linear Kill assassin
                tb[r, c] = CELL_HIT
                s_id = self.ship_id_maps[i, r, c]
                self.ship_health[i, s_id] -= 1
                self.pegs_remaining[i] -= 1
                self.shots_taken[i] += 1

                kill_shots = self._execute_linear_kill(i, r, c)
                self.shots_taken[i] += kill_shots

                total_shots_for_kill = 1 + kill_shots
                # Reward: cost of shots + Sunk bonus
                rewards[i] = -float(total_shots_for_kill) + 2.0

            # 2. Check Game Completion
            if self.pegs_remaining[i] == 0:
                dones[i] = True
                final_shots = int(self.shots_taken[i])
                
                # Smooth continuous terminal reward (Zero cliff edges!)
                # Every shot saved below 50 is linearly rewarded by +1.0
                rewards[i] += float(50 - final_shots)

                completed_games.append({
                    'env_idx': i,
                    'total_shots': final_shots
                })
                self._reset_env(i)
            else:
                self._update_action_mask(i)

            # Update PyTorch state tensor for env i
            self.states[i, 0] = torch.from_numpy((tb == CELL_MISS).astype(np.float32)).to(self.device)
            self.states[i, 1] = torch.from_numpy((tb == CELL_HIT).astype(np.float32)).to(self.device)
            self.states[i, 2] = torch.from_numpy((tb == CELL_SUNK).astype(np.float32)).to(self.device)

        return self.states.clone(), torch.from_numpy(rewards).to(self.device), torch.from_numpy(dones).to(self.device), completed_games

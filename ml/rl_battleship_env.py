import os
import sys
import random
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.game_engine import BattleshipGame, CELL_UNKNOWN, CELL_MISS, CELL_HIT, CELL_SUNK
from bots.sherlock import SherlockBot

FLEET_SIZES = [5, 4, 3, 3, 2]  # 17 total pegs

def compute_sherlock_baseline(truth_board, parity_offset):
    """
    Computes deterministic baseline shots for SherlockBot on this exact hidden board layout.
    Serves as the Control Variate baseline for Paired-Board Regret Minimization.
    """
    game = BattleshipGame()
    for r in range(10):
        for c in range(10):
            val = int(truth_board[r, c])
            game.truth_board[r][c] = val
            if val > 0:
                if val not in game.ship_pegs:
                    game.ship_pegs[val] = []
                    game.ship_health[val] = 0
                game.ship_pegs[val].append((r, c))
                game.ship_health[val] += 1

    bot = SherlockBot()
    bot.parity_offset = parity_offset
    while not game.is_game_over():
        target, _, _, _ = bot.get_best_move(game.tracking_board, game.move_history)
        game.fire_shot(*target)
    return game.shot_number

class DecomposedBattleshipEnv:
    """
    Decomposed Hunt-Target Battleship Environment for PPO Reinforcement Learning.
    
    Key Features:
    1. The RL Agent ONLY controls the Hunt Phase on active checkerboard parity.
    2. Strict Fog of War Compliance:
       - Only the lethal shot is marked CELL_SUNK; previous hits remain CELL_HIT.
       - Ship lengths/identities are never exposed to the agent observation tensor.
    3. Deduction-Driven Assassin (Sherlock Exact Density / Linear Kill):
       - Resolves all active hits immediately until no wounded ships remain.
    4. Hypothesis Pruning / Information-Gain Reward:
       - Misses are rewarded proportional to candidate placement hypotheses eliminated.
    5. Zero Carrier Penalty:
       - Kill sequence only penalizes missed shots incurred: -(kill_misses) + 2.0 * sunk_count.
    6. Paired-Board Regret Minimization:
       - Terminal reward compares agent shots directly against Sherlock baseline on the exact same board:
         Regret = baseline_shots - agent_shots.
    """
    def __init__(self, num_envs=32, device='cpu'):
        self.num_envs = num_envs
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.grid_size = 10
        self.total_pegs = 17

        # Precompute placement hypotheses bitmasks for fast information-gain calculation
        self._init_placement_hypotheses()

        # Shared deterministic deduction engine for linear kill resolution
        self.sherlock_assassin = SherlockBot()

        # CPU representations for fast board truth tracking
        self.truth_boards = np.zeros((num_envs, 10, 10), dtype=np.int32)
        self.ship_id_maps = np.zeros((num_envs, 10, 10), dtype=np.int32)
        self.tracking_boards = np.zeros((num_envs, 10, 10), dtype=np.int32)
        self.ship_pegs = [{} for _ in range(num_envs)]
        self.ship_health = np.zeros((num_envs, 6), dtype=np.int32)
        self.misses_masks = [0 for _ in range(num_envs)]
        self.move_histories = [[] for _ in range(num_envs)]
        self.baseline_shots = np.zeros(num_envs, dtype=np.int32)

        # PyTorch State Tensors [N, 4, 10, 10]
        # Channels: 0: Misses, 1: Hits, 2: Sunk, 3: Parity
        self.states = torch.zeros((num_envs, 4, 10, 10), dtype=torch.float32, device=self.device)
        self.action_masks = torch.ones((num_envs, 100), dtype=torch.float32, device=self.device)

        self.pegs_remaining = np.full(num_envs, 17, dtype=np.int32)
        self.shots_taken = np.zeros(num_envs, dtype=np.int32)
        self.parity_offsets = np.zeros(num_envs, dtype=np.int32)
        self.game_counter = 0

        for i in range(num_envs):
            self._reset_env(i)

    def _init_placement_hypotheses(self):
        """Precomputes bitmasks for all 760 valid ship placements and cell lookup table."""
        self.all_placements = []
        for length in FLEET_SIZES:
            for r in range(10):
                for c in range(10 - length + 1):
                    coords = [(r, c + k) for k in range(length)]
                    mask = sum(1 << (cr * 10 + cc) for cr, cc in coords)
                    self.all_placements.append((mask, coords))
            for r in range(10 - length + 1):
                for c in range(10):
                    coords = [(r + k, c) for k in range(length)]
                    mask = sum(1 << (cr * 10 + cc) for cr, cc in coords)
                    self.all_placements.append((mask, coords))

        self.cell_placements = [[] for _ in range(100)]
        for idx, (mask, coords) in enumerate(self.all_placements):
            for r, c in coords:
                self.cell_placements[r * 10 + c].append(idx)

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
        self.misses_masks[env_idx] = 0
        self.move_histories[env_idx] = []

        # Paired Control Variate: Compute Sherlock baseline shots on this exact board
        self.baseline_shots[env_idx] = compute_sherlock_baseline(s_map, p_offset)

        # PyTorch state tensor initialization
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

    def _execute_linear_kill(self, env_idx):
        """
        Executes deduction-driven assassination using Sherlock's Exact Target Density / Linear Kill
        until all active wounded ships are sunk.
        Returns:
            kill_shots (int): total shots fired during this assassination sequence.
            kill_misses (int): total misses fired during this assassination sequence.
            sunk_count (int): additional ships sunk during this assassination sequence.
        """
        tb = self.tracking_boards[env_idx]
        truth = self.truth_boards[env_idx]
        s_map = self.ship_id_maps[env_idx]
        history = self.move_histories[env_idx]

        kill_shots = 0
        kill_misses = 0
        sunk_count = 0

        while self.pegs_remaining[env_idx] > 0:
            solutions = self.sherlock_assassin._analyze_board_state_all(tb, history)
            has_active = any(sol[0] for sol in solutions) if solutions else False
            if not has_active:
                break

            target, _, _, _ = self.sherlock_assassin.get_best_move(tb, history)
            if target is None or tb[target[0]][target[1]] != CELL_UNKNOWN:
                # Failsafe: find any unknown neighbor of active hits
                target = None
                for r in range(10):
                    for c in range(10):
                        if tb[r, c] == CELL_HIT:
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < 10 and 0 <= nc < 10 and tb[nr, nc] == CELL_UNKNOWN:
                                    target = (nr, nc)
                                    break
                        if target:
                            break
                    if target:
                        break
                if target is None:
                    break

            tr, tc = int(target[0]), int(target[1])
            kill_shots += 1
            self.shots_taken[env_idx] += 1

            if truth[tr, tc] > 0:
                # Hit!
                hit_s_id = s_map[tr, tc]
                self.ship_health[env_idx, hit_s_id] -= 1
                self.pegs_remaining[env_idx] -= 1
                is_sunk = (self.ship_health[env_idx, hit_s_id] == 0)

                if is_sunk:
                    tb[tr, tc] = CELL_SUNK
                    sunk_count += 1
                    res_str = "SUNK"
                else:
                    tb[tr, tc] = CELL_HIT
                    res_str = "HIT"

                history.append({
                    "shotNumber": int(self.shots_taken[env_idx]),
                    "x": tr,
                    "y": tc,
                    "result": res_str,
                    "isSunk": is_sunk
                })
            else:
                # Miss
                tb[tr, tc] = CELL_MISS
                kill_misses += 1
                self.misses_masks[env_idx] |= (1 << int(tr * 10 + tc))
                history.append({
                    "shotNumber": int(self.shots_taken[env_idx]),
                    "x": tr,
                    "y": tc,
                    "result": "MISS",
                    "isSunk": False
                })

        return kill_shots, kill_misses, sunk_count

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
            act = int(actions[i])
            r = act // 10
            c = act % 10

            tb = self.tracking_boards[i]
            truth = self.truth_boards[i]

            if truth[r, c] == 0:
                # 1. Hunt Miss
                prev_mask = self.misses_masks[i]
                tb[r, c] = CELL_MISS
                self.misses_masks[i] |= (1 << int(act))
                self.shots_taken[i] += 1
                self.move_histories[i].append({
                    "shotNumber": int(self.shots_taken[i]),
                    "x": r, "y": c,
                    "result": "MISS", "isSunk": False
                })

                # Information-Gain: count how many surviving candidate placements were eliminated
                eliminated = 0
                for p_idx in self.cell_placements[act]:
                    p_mask, _ = self.all_placements[p_idx]
                    if (p_mask & prev_mask) == 0:
                        eliminated += 1

                info_gain = min(eliminated / 45.0, 1.0)
                open_neighbors = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
                                     if 0 <= r+dr < 10 and 0 <= c+dc < 10 and tb[r+dr, c+dc] == CELL_UNKNOWN)
                open_bonus = (open_neighbors / 8.0) * 0.05
                rewards[i] = -1.0 + 0.15 * info_gain + open_bonus
            else:
                # 2. Hunt Hit!
                s_id = self.ship_id_maps[i, r, c]
                self.ship_health[i, s_id] -= 1
                self.pegs_remaining[i] -= 1
                self.shots_taken[i] += 1
                is_sunk = (self.ship_health[i, s_id] == 0)

                if is_sunk:
                    tb[r, c] = CELL_SUNK
                    initial_sunk = 1
                    res_str = "SUNK"
                else:
                    tb[r, c] = CELL_HIT
                    initial_sunk = 0
                    res_str = "HIT"

                self.move_histories[i].append({
                    "shotNumber": int(self.shots_taken[i]),
                    "x": r, "y": c,
                    "result": res_str, "isSunk": is_sunk
                })

                # Execute assassin until all active hits are resolved
                kill_shots, kill_misses, sunk_count = self._execute_linear_kill(i)
                total_sunk = initial_sunk + sunk_count

                # Carrier Penalty Removed: Only penalize misses incurred; reward finding and sinking ships
                rewards[i] = -float(kill_misses) + 2.0 * float(total_sunk)

            # 3. Check Game Completion
            if self.pegs_remaining[i] == 0:
                dones[i] = True
                final_shots = int(self.shots_taken[i])
                baseline = int(self.baseline_shots[i])

                # Paired-board regret: positive if agent beat Sherlock baseline
                regret = float(baseline - final_shots)
                tail_penalty = -2.0 * float(max(0, final_shots - 60))
                rewards[i] += regret + tail_penalty

                completed_games.append({
                    'env_idx': i,
                    'total_shots': final_shots,
                    'baseline_shots': baseline,
                    'regret': regret
                })
                self._reset_env(i)
            else:
                self._update_action_mask(i)

            # Update PyTorch state tensor for env i
            self.states[i, 0] = torch.from_numpy((tb == CELL_MISS).astype(np.float32)).to(self.device)
            self.states[i, 1] = torch.from_numpy((tb == CELL_HIT).astype(np.float32)).to(self.device)
            self.states[i, 2] = torch.from_numpy((tb == CELL_SUNK).astype(np.float32)).to(self.device)

        return self.states.clone(), torch.from_numpy(rewards).to(self.device), torch.from_numpy(dones).to(self.device), completed_games

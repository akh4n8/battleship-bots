import os
import sys
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.game_engine import CELL_UNKNOWN, CELL_MISS, CELL_HIT, CELL_SUNK
from bots.sherlock import SherlockBot

class NeuralBot:
    """
    NeuralBot uses the SherlockNeuralModel trained on 1,000,000 Sherlock games.
    Features:
    - Live 10x10 real-time occupancy & policy heatmap generation for Pygame.
    - Integrated Assassin Mode (DeductionEngine LinearKill) for guaranteed sinking of wounded ships.
    - Neural Hunt targeting with parity-guided selection when hunting open water.
    """
    _cached_models = {}

    def __init__(self, weights_path=None, use_linear_kill=True, parity_offset=0, device=None):
        self.use_linear_kill = use_linear_kill
        self.parity_offset = parity_offset
        self.sherlock_assassin = SherlockBot()
        self.sherlock_assassin.parity_offset = self.parity_offset
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        default_weights = os.path.join(DATA_DIR, "sherlock_nn_best.pth")
        self.weights_path = weights_path if weights_path else default_weights

        self.model = None
        self._load_model()
        self.last_heatmap = [[0]*10 for _ in range(10)]

    def _load_model(self):
        cache_key = (self.weights_path, str(self.device))
        if cache_key in NeuralBot._cached_models:
            self.model = NeuralBot._cached_models[cache_key]
            return

        try:
            from ml.train_sherlock_nn import SherlockNeuralModel
            model = SherlockNeuralModel(in_channels=4).to(self.device)
            if os.path.exists(self.weights_path):
                model.load_state_dict(torch.load(self.weights_path, map_location=self.device), strict=False)
                model.eval()
                print(f"[NeuralBot] Successfully loaded model from {self.weights_path} ({self.device})")
            else:
                print(f"[NeuralBot] Weights file {self.weights_path} not found. Running with random initialization until trained.")
                model.eval()
            NeuralBot._cached_models[cache_key] = model
            self.model = model
        except Exception as e:
            print(f"[NeuralBot] Warning: could not initialize model: {e}")
            self.model = None

    def _build_input_tensor(self, board, move_history):
        ch_misses = np.zeros((10, 10), dtype=np.float32)
        ch_hits = np.zeros((10, 10), dtype=np.float32)
        ch_sunk = np.zeros((10, 10), dtype=np.float32)
        ch_parity = np.zeros((10, 10), dtype=np.float32)

        for r in range(10):
            for c in range(10):
                val = board[r][c]
                if val == CELL_MISS:
                    ch_misses[r, c] = 1.0
                elif val == CELL_HIT:
                    ch_hits[r, c] = 1.0
                elif val == CELL_SUNK:
                    ch_sunk[r, c] = 1.0

                if (r + c) % 2 == self.parity_offset:
                    ch_parity[r, c] = 1.0

        tensor = np.stack([ch_misses, ch_hits, ch_sunk, ch_parity], axis=0) # [4, 10, 10]
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device) # [1, 4, 10, 10]

    def get_heatmap(self, board, dead_sizes=None):
        """Returns 10x10 integer heatmap (0..100) for visual rendering in Pygame."""
        if self.model is None:
            return [[0]*10 for _ in range(10)]

        with torch.no_grad():
            x = self._build_input_tensor(board, [])
            if hasattr(self.model, 'occupancy_head') and hasattr(self.model, 'res_layers') and hasattr(self.model, 'stem'):
                features = self.model.res_layers(self.model.stem(x))
                occ_logits = self.model.occupancy_head(features).squeeze(1).view(x.size(0), 100)
                probs = torch.sigmoid(occ_logits)[0].cpu().numpy().reshape(10, 10)
            else:
                out = self.model(x)
                logits = out[0] if isinstance(out, tuple) else out
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().reshape(10, 10)

        heatmap = [[0]*10 for _ in range(10)]
        for r in range(10):
            for c in range(10):
                if board[r][c] == CELL_UNKNOWN:
                    heatmap[r][c] = int(probs[r, c] * 100)
                else:
                    heatmap[r][c] = 0

        self.last_heatmap = heatmap
        return heatmap

    def get_best_move(self, board, move_history):
        # 1. ASSASSIN / TARGET RESOLUTION PHASE
        # Use CSP analysis to determine if there are genuinely active (un-sunk) hits
        solutions = self.sherlock_assassin._analyze_board_state_all(board, move_history)
        has_active = any(sol[0] for sol in solutions) if solutions else False
        if self.use_linear_kill and has_active:
            target, strat, ui_sizes, diagnostics = self.sherlock_assassin.get_best_move(board, move_history)
            if target and board[target[0]][target[1]] == CELL_UNKNOWN:
                heat = self.get_heatmap(board)
                heat[target[0]][target[1]] = 100
                self.last_heatmap = heat
                return target, f"NeuralBot: Assassin ({strat})", ui_sizes, diagnostics

        # 2. NEURAL HUNTING PHASE
        if self.model is None:
            # Fallback to Sherlock if model not loaded
            return self.sherlock_assassin.get_best_move(board, move_history)

        with torch.no_grad():
            x = self._build_input_tensor(board, move_history)
            out = self.model(x)
            if isinstance(out, tuple):
                if out[0].shape == out[1].shape:
                    policy_logits = out[1]
                elif out[0].shape[-1] == 100:
                    policy_logits = out[0]
                else:
                    policy_logits = out[0]
            else:
                policy_logits = out

            # Softmax policy distribution
            policy_probs = torch.softmax(policy_logits, dim=-1)[0].cpu().numpy()

        # Mask out already fired cells
        best_score = -1e9
        best_target = None

        # First pass: try on-parity checkerboard cells
        for r in range(10):
            for c in range(10):
                if board[r][c] == CELL_UNKNOWN and (r + c) % 2 == self.parity_offset:
                    score = policy_probs[r * 10 + c]
                    if score > best_score:
                        best_score = score
                        best_target = (r, c)

        # Second pass: if no valid on-parity cell remains, consider all open cells
        if best_target is None:
            for r in range(10):
                for c in range(10):
                    if board[r][c] == CELL_UNKNOWN:
                        score = policy_probs[r * 10 + c]
                        if score > best_score:
                            best_score = score
                            best_target = (r, c)

        confidence_pct = best_score * 100
        heat = self.get_heatmap(board)
        return best_target, f"NeuralBot: Hunt (Confidence: {confidence_pct:.1f}%)", [], {}

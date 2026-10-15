import math
import random
import re
from bots.mycroft import MycroftBot
from core.game_engine import CELL_UNKNOWN, CELL_HIT

class EntropyBot:
    """
    EntropyBot: Information-Theoretic Bayesian Battleship Bot.
    
    Fixes the classic Twenty-Questions pitfall by combining:
    1. Deterministic Assassin Kill: 100% exploit priority on wounded ships (Linear Kill).
    2. Expected Utility Theory: Weights hit probability heavily (due to Continuous Fire and HP damage).
    3. Shannon Binary Entropy: Acts as an optimal tie-breaker between comparable probabilities.
    4. Spatial Volume Slicing: Favors shots with large open-water coverage over cramped pockets.
    5. Parity Constraint: Preserves checkerboard hunting efficiency.
    """
    def __init__(self):
        self.mycroft = MycroftBot(safe_mode=True)
        self.parity_offset = self.mycroft.parity_offset

    def get_best_move(self, board, move_history):
        # 1. ASSASSIN PRIORITY: Never explore when active wounded hits exist
        has_active_hits = any(board[r][c] == CELL_HIT for r in range(10) for c in range(10))
        if has_active_hits:
            target, strat, dead_sizes, diag = self.mycroft.sherlock.get_best_move(board, move_history)
            if target and board[target[0]][target[1]] == CELL_UNKNOWN:
                return target, f"EntropyBot: Assassin Kill ({strat})", dead_sizes, diag

        # 2. RUN MCMC POSTERIOR
        fallback_target, log, ui_sizes, diagnostics = self.mycroft.get_best_move(board, move_history)
        heatmap = self.mycroft.last_heatmap

        match = re.search(r'\((\d+) Steps\)', log)
        valid_universes = int(match.group(1)) if match else 0

        max_heat = max([max(row) for row in heatmap]) if heatmap else 0
        if valid_universes == 0 or max_heat == 0:
            return fallback_target, log, ui_sizes, diagnostics

        parity = self.mycroft.parity_offset

        # 3. SCORE CANDIDATES VIA EXPECTED UTILITY & SHANNON ENTROPY
        def evaluate_cell(r, c):
            raw_p = heatmap[r][c] / valid_universes
            
            # Shannon binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)
            p = max(1e-5, min(1.0 - 1e-5, raw_p))
            entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

            # Spatial openness (receptive field): count unrevealed neighbors within radius 2
            open_count = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == CELL_UNKNOWN:
                    open_count += 1
            open_factor = open_count / 8.0

            # Utility Formula:
            # - 2.5 * raw_p: Primary exploit (Continuous Fire + HP reduction)
            # - 0.35 * entropy: Information-gain tie breaker
            # - 0.15 * open_factor: Ocean bisecting preference
            return (2.5 * raw_p) + (0.35 * entropy) + (0.15 * open_factor)

        best_score = -1e9
        best_moves = []

        # Pass 1: Strict Checkerboard Parity
        for r in range(10):
            for c in range(10):
                if board[r][c] == CELL_UNKNOWN and (r + c) % 2 == parity:
                    score = evaluate_cell(r, c)
                    if score > best_score:
                        best_score = score
                        best_moves = [(r, c)]
                    elif math.isclose(score, best_score, rel_tol=1e-5):
                        best_moves.append((r, c))

        # Pass 2: If on-parity exhausted, consider all remaining unknown cells
        if not best_moves:
            for r in range(10):
                for c in range(10):
                    if board[r][c] == CELL_UNKNOWN:
                        score = evaluate_cell(r, c)
                        if score > best_score:
                            best_score = score
                            best_moves = [(r, c)]
                        elif math.isclose(score, best_score, rel_tol=1e-5):
                            best_moves.append((r, c))

        if best_moves:
            chosen = random.choice(best_moves)
            p_hit = (heatmap[chosen[0]][chosen[1]] / valid_universes) * 100.0
            return chosen, f"EntropyBot: Info-Utility (Score={best_score:.2f}, P={p_hit:.1f}%)", ui_sizes, diagnostics

        return fallback_target, log, ui_sizes, diagnostics

    def get_heatmap(self, board, dead_sizes=None):
        return self.mycroft.get_heatmap(board, dead_sizes)

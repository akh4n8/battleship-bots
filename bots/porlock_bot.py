import math
import random
from collections import defaultdict
from core.game_engine import CELL_UNKNOWN, CELL_MISS, CELL_HIT, CELL_SUNK

class PorlockBot:
    """
    PorlockBot: Exact Information-Theoretic Bayesian Engine.
    
    The Lore:
    Fred Porlock is the pseudonymous cipher-writer and informant embedded inside
    Moriarty's syndicate (The Valley of Fear). Where Sherlock relies on deductive
    line-tracing, Mycroft on stochastic MCMC sampling, and Sherrinford on pure density fields,
    Porlock operates at the intersection of Cryptography and Information Theory.
    
    Mathematical Foundations:
    1. Multi-Hypothesis Tied CSP Deduction:
       Solves the Fog of War blind-sink problem by analyzing all tied minimum-leftover
       deduction hypotheses, locking confirmed corpses as permanent obstacles.
    2. Pure Exact Bayesian Joint Targeting:
       Resolves wounded hits, touching ships, and complex multi-ship clusters via exact
       2-ship disjoint joint cover (P1, P2) across all tied hypotheses. Zero heuristics.
    3. Shannon Binary Information Entropy:
       Evaluates candidate hunt cells using Claude Shannon's binary entropy function:
           H(p) = -p * log2(p) - (1 - p) * log2(1 - p)
       Acting as an optimal tie-breaker between comparable probability densities.
    4. Spatial Volume Slicing (Ocean Bisecting):
       Measures unrevealed cells in a radius-2 receptive field to favor shots that cleanly
       bisect large open waters over cramped isolated pockets.
    5. Expected Utility Optimization:
       Combines hit probability, information entropy, and spatial openness:
           Utility(r, c) = 2.5 * p + 0.35 * H(p) + 0.15 * Openness(r, c)
    6. Dynamic Stride Parity:
       - Stride 2 (Checkerboard: (r+c)%2) while the 2-peg Patrol Boat is alive.
       - Stride 3 (Diagonal: (r+2c)%3) once the 2-ship is confirmed dead across ALL tied hypotheses.
    7. Deterministic Reproducibility:
       Parity derived from game_id % 2; tie-breaking derived from (parity * 10000) + len(moves).
    """
    FLEET_SIZES = [5, 4, 3, 3, 2]

    EMPTY_BOARD_EXACT_HEATMAP = [
        [ 9, 13, 17, 19, 21, 21, 19, 17, 13,  9],
        [13, 18, 22, 26, 28, 28, 26, 22, 18, 13],
        [17, 22, 28, 33, 35, 35, 33, 28, 22, 17],
        [19, 26, 33, 38, 41, 41, 38, 33, 26, 19],
        [21, 28, 35, 41, 45, 45, 41, 35, 28, 21],
        [21, 28, 35, 41, 45, 45, 41, 35, 28, 21],
        [19, 26, 33, 38, 41, 41, 38, 33, 26, 19],
        [17, 22, 28, 33, 35, 35, 33, 28, 22, 17],
        [13, 18, 22, 26, 28, 28, 26, 22, 18, 13],
        [ 9, 13, 17, 19, 21, 21, 19, 17, 13,  9]
    ]

    def __init__(self, parity_offset=0, w_p=2.5, w_entropy=0.35, w_open=0.15):
        self.parity_offset = parity_offset
        self.w_p = w_p
        self.w_entropy = w_entropy
        self.w_open = w_open
        from bots.sherlock import SherlockBot
        self.sherlock = SherlockBot()
        self.sherlock.parity_offset = self.parity_offset
        self.last_heatmap = [[0] * 10 for _ in range(10)]
        self._all_placements_cache = None
        self._init_placements()

    def _init_placements(self):
        if self._all_placements_cache is not None:
            return

        placements_by_size = defaultdict(list)
        for size in set(self.FLEET_SIZES):
            plist = []
            for r in range(10):
                for c in range(10 - size + 1):
                    mask = sum(1 << (r * 10 + c + k) for k in range(size))
                    coords = tuple((r, c + k) for k in range(size))
                    plist.append((size, mask, coords))
            for r in range(10 - size + 1):
                for c in range(10):
                    mask = sum(1 << ((r + k) * 10 + c) for k in range(size))
                    coords = tuple((r + k, c) for k in range(size))
                    plist.append((size, mask, coords))
            placements_by_size[size] = plist

        self._all_placements_cache = placements_by_size

    def _evaluate_cell_utility(self, board, r, c, raw_p):
        p = max(1e-5, min(1.0 - 1e-5, raw_p))
        entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

        # Spatial openness: count unrevealed neighbors in radius-2 star
        open_count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == CELL_UNKNOWN:
                open_count += 1
        open_factor = open_count / 8.0

        return (self.w_p * raw_p) + (self.w_entropy * entropy) + (self.w_open * open_factor)

    def _compute_active_hit_density(self, board, active_hits, claimed_hits, dead_sizes):
        available_fleet = self.FLEET_SIZES.copy()
        for s in dead_sizes:
            if s in available_fleet: available_fleet.remove(s)

        forbidden_mask = 0
        for r in range(10):
            for c in range(10):
                if board[r][c] == CELL_MISS: forbidden_mask |= (1 << (r * 10 + c))
        for r, c in claimed_hits:
            forbidden_mask |= (1 << (r * 10 + c))

        hits_mask = sum(1 << (r * 10 + c) for r, c in active_hits)

        single_covers = []
        for s in available_fleet:
            for s_len, mask, coords in self._all_placements_cache[s]:
                if (mask & forbidden_mask) == 0 and (mask & hits_mask) == hits_mask:
                    if any(board[r][c] == CELL_UNKNOWN for r, c in coords):
                        single_covers.append(coords)

        if single_covers:
            counts = defaultdict(int)
            for coords in single_covers:
                for r, c in coords:
                    if board[r][c] == CELL_UNKNOWN: counts[(r, c)] += 1
            return counts

        cand_p = []
        for s_idx, s in enumerate(available_fleet):
            for s_len, mask, coords in self._all_placements_cache[s]:
                if (mask & forbidden_mask) == 0:
                    hit_overlap = mask & hits_mask
                    if hit_overlap > 0 and any(board[r][c] == CELL_UNKNOWN for r, c in coords):
                        cand_p.append((s_idx, s_len, mask, coords, hit_overlap))

        pair_covers = []
        n_cand = len(cand_p)
        for i in range(n_cand):
            s_idx1, _, m1, c1, h1 = cand_p[i]
            for j in range(i + 1, n_cand):
                s_idx2, _, m2, c2, h2 = cand_p[j]
                if s_idx1 != s_idx2 and (m1 & m2) == 0:
                    if (h1 | h2) == hits_mask: pair_covers.append((c1, c2))

        if pair_covers:
            counts = defaultdict(int)
            for c1, c2 in pair_covers:
                for r, c in list(c1) + list(c2):
                    if board[r][c] == CELL_UNKNOWN: counts[(r, c)] += 1
            return counts

        counts = defaultdict(int)
        for s_idx, _, m, coords, h in cand_p:
            weight = bin(h).count('1') ** 2
            for r, c in coords:
                if board[r][c] == CELL_UNKNOWN: counts[(r, c)] += weight
        return counts

    def get_best_move(self, board, move_history):
        # 1. Multi-Hypothesis Tied CSP Deduction
        try:
            solutions = self.sherlock._analyze_board_state_all(board, move_history)
        except Exception:
            solutions = [([], [], [], [])]

        primary_sol = solutions[0]
        active_hits, dead_ships, claimed_hits, dead_sizes = primary_sol

        diagnostics = {
            "active": list(active_hits),
            "claimed": list(claimed_hits),
            "dead_ships": dead_ships,
            "dead_sizes": dead_sizes,
            "solutions_count": len(solutions)
        }

        has_active = any(sol[0] for sol in solutions)
        is_two_ship_dead = all(2 in sol[3] for sol in solutions)

        # 2. Target Mode: Pure Exact Multi-Hypothesis Bayesian Joint Targeting
        if has_active:
            total_counts = defaultdict(int)
            for s_active, s_dead_ships, s_claimed, s_sizes in solutions:
                if s_active:
                    counts = self._compute_active_hit_density(board, s_active, s_claimed, s_sizes)
                    for k, v in counts.items():
                        total_counts[k] += v

            self.last_heatmap = [[0] * 10 for _ in range(10)]
            max_c = max(total_counts.values()) if total_counts else 0
            if max_c > 0:
                for (r, c), cnt in total_counts.items():
                    if board[r][c] == CELL_UNKNOWN:
                        self.last_heatmap[r][c] = int((cnt / max_c) * 100)

            best_targets = [k for k, v in total_counts.items() if v == max_c and board[k[0]][k[1]] == CELL_UNKNOWN]
            if best_targets:
                seed = (self.parity_offset * 10000) + len(move_history)
                rng = random.Random(seed)
                return rng.choice(best_targets), "PorlockBot: Exact Joint Target", dead_sizes, diagnostics

        # 3. Hunt Mode: Exact Information-Theoretic Utility Optimization
        combined_counts = [[0] * 10 for _ in range(10)]

        for s_active, s_dead_ships, s_claimed, s_sizes in solutions:
            available_fleet = self.FLEET_SIZES.copy()
            for s in s_sizes:
                if s in available_fleet: available_fleet.remove(s)

            forbidden_cells = set()
            for r in range(10):
                for c in range(10):
                    if board[r][c] == CELL_MISS: forbidden_cells.add((r, c))
            forbidden_cells.update(s_claimed)
            forbidden_mask = sum(1 << (r * 10 + c) for r, c in forbidden_cells)

            for size in available_fleet:
                for s_len, mask, coords in self._all_placements_cache[size]:
                    if (mask & forbidden_mask) == 0:
                        for r, c in coords:
                            combined_counts[r][c] += 1

        p_map = [[0.0] * 10 for _ in range(10)]
        max_cnt = max(max(row) for row in combined_counts) if combined_counts else 0
        if max_cnt > 0:
            for r in range(10):
                for c in range(10):
                    if board[r][c] == CELL_UNKNOWN:
                        p_map[r][c] = combined_counts[r][c] / max_cnt
                        self.last_heatmap[r][c] = int(p_map[r][c] * 100)

        best_score = -1e9
        best_targets = []

        if is_two_ship_dead:
            # Dynamic Stride 3 diagonal sweep: (r + 2*c) % 3 == parity_offset % 3
            target_p = self.parity_offset % 3
            for r in range(10):
                for c in range(10):
                    if board[r][c] == CELL_UNKNOWN and (r + 2 * c) % 3 == target_p:
                        score = self._evaluate_cell_utility(board, r, c, p_map[r][c])
                        if score > best_score:
                            best_score = score
                            best_targets = [(r, c)]
                        elif math.isclose(score, best_score, rel_tol=1e-5):
                            best_targets.append((r, c))
        else:
            # Stride 2 checkerboard sweep: (r + c) % 2 == parity_offset % 2
            target_p = self.parity_offset % 2
            for r in range(10):
                for c in range(10):
                    if board[r][c] == CELL_UNKNOWN and (r + c) % 2 == target_p:
                        score = self._evaluate_cell_utility(board, r, c, p_map[r][c])
                        if score > best_score:
                            best_score = score
                            best_targets = [(r, c)]
                        elif math.isclose(score, best_score, rel_tol=1e-5):
                            best_targets.append((r, c))

        if not best_targets:
            for r in range(10):
                for c in range(10):
                    if board[r][c] == CELL_UNKNOWN:
                        score = self._evaluate_cell_utility(board, r, c, p_map[r][c])
                        if score > best_score:
                            best_score = score
                            best_targets = [(r, c)]
                        elif math.isclose(score, best_score, rel_tol=1e-5):
                            best_targets.append((r, c))

        if not best_targets:
            open_cells = [(r, c) for r in range(10) for c in range(10) if board[r][c] == CELL_UNKNOWN]
            chosen = open_cells[0] if open_cells else (0, 0)
            return chosen, "PorlockBot: Failsafe Fallback", dead_sizes, diagnostics

        seed = (self.parity_offset * 10000) + len(move_history)
        rng = random.Random(seed)
        chosen = rng.choice(best_targets)
        stride_str = "Stride-3" if is_two_ship_dead else "Stride-2"
        return chosen, f"PorlockBot: Info-Utility ({stride_str}, Score={best_score:.3f})", dead_sizes, diagnostics

    def get_heatmap(self, board, dead_sizes=None):
        return self.last_heatmap

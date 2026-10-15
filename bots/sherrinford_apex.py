import math
import random
import time
from collections import defaultdict
from core.game_engine import CELL_UNKNOWN, CELL_MISS, CELL_HIT, CELL_SUNK

class SherrinfordApexBot:
    """
    SherrinfordApexBot: The Pinnacle Engine.
    
    Synthesizes the three greatest mathematical principles of the Battleship project:
    1. Deduction-Locked Memory:
       Uses Sherlock's recursive CSP deduction engine (_analyze_board_state) to lock
       in confirmed sunken corpses (claimed_hits) and definitively separate them from active hits.
       Sunk ships are frozen as permanent obstacles; ghost-ship chasing is eliminated.
    2. Pure Exact Bayesian Joint Cover:
       Active wounded hits and touching / L-shaped ships are resolved via pure 2-ship
       joint cover (P1, P2). ZERO Linear Assassin line-tracing heuristics.
    3. Dynamic Stride Parity:
       - Stride 2 (Checkerboard: (r+c)%2) while the 2-peg Patrol Boat is alive.
       - Stride 3 (Diagonal: (r+2c)%3) once the 2-ship is confirmed sunk, sweeping the
         board in 34 shots instead of 50 shots with 0% chance of missing a surviving ship.
    4. Deterministic parity and tie-breaking compliant with project protocol (gameId % 2).
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

    def __init__(self, parity_offset=0):
        self.parity_offset = parity_offset
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

    def _compute_open_water_density(self, board, available_fleet, forbidden_cells):
        forbidden_mask = sum(1 << (r * 10 + c) for r, c in forbidden_cells)
        counts = [[0] * 10 for _ in range(10)]
        total = 0

        unique_fleet = list(set(available_fleet))
        for size in unique_fleet:
            for s_len, mask, coords in self._all_placements_cache[size]:
                if (mask & forbidden_mask) == 0:
                    total += 1
                    for r, c in coords: counts[r][c] += 1

        heatmap = [[0] * 10 for _ in range(10)]
        max_val = max(max(row) for row in counts) if total > 0 else 0
        if max_val > 0:
            for r in range(10):
                for c in range(10):
                    if board[r][c] == CELL_UNKNOWN:
                        heatmap[r][c] = int((counts[r][c] / max_val) * 100)
        return heatmap

    def get_best_move(self, board, move_history):
        # 1. Deduction-Locked Multi-Hypothesis Memory
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

        # 2. Pure Bayesian Joint Cover Targeting
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

            phase_name = "Sherrinford-Apex: Exact Multi-Hypothesis Bayesian Density"
            use_hunt_parity = False
        else:
            if len(move_history) == 0:
                self.last_heatmap = [row[:] for row in self.EMPTY_BOARD_EXACT_HEATMAP]
            else:
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
                    unique_fleet = list(set(available_fleet))
                    for size in unique_fleet:
                        for s_len, mask, coords in self._all_placements_cache[size]:
                            if (mask & forbidden_mask) == 0:
                                for r, c in coords:
                                    combined_counts[r][c] += 1

                self.last_heatmap = [[0] * 10 for _ in range(10)]
                max_val = max(max(row) for row in combined_counts)
                if max_val > 0:
                    for r in range(10):
                        for c in range(10):
                            if board[r][c] == CELL_UNKNOWN:
                                self.last_heatmap[r][c] = int((combined_counts[r][c] / max_val) * 100)

            stride_str = "Stride-3" if is_two_ship_dead else "Stride-2"
            phase_name = f"Sherrinford-Apex: Exact Multi-Hunt ({stride_str})"
            use_hunt_parity = True

        max_prob = -1
        best_targets = []

        if use_hunt_parity:
            if is_two_ship_dead:
                # Dynamic Stride 3 diagonal sweep: (r + 2*c) % 3 == parity_offset % 3
                target_p = self.parity_offset % 3
                for r in range(10):
                    for c in range(10):
                        if board[r][c] == CELL_UNKNOWN and (r + 2 * c) % 3 == target_p:
                            val = self.last_heatmap[r][c]
                            if val > max_prob:
                                max_prob = val; best_targets = [(r, c)]
                            elif val == max_prob and val > 0:
                                best_targets.append((r, c))
            else:
                # Stride 2 checkerboard sweep: (r + c) % 2 == parity_offset % 2
                target_p = self.parity_offset % 2
                for r in range(10):
                    for c in range(10):
                        if board[r][c] == CELL_UNKNOWN and (r + c) % 2 == target_p:
                            val = self.last_heatmap[r][c]
                            if val > max_prob:
                                max_prob = val; best_targets = [(r, c)]
                            elif val == max_prob and val > 0:
                                best_targets.append((r, c))

        # Fallback to all unrevealed cells if on-parity exhausted or in targeting mode
        if not best_targets:
            max_prob = -1
            for r in range(10):
                for c in range(10):
                    if board[r][c] == CELL_UNKNOWN:
                        val = self.last_heatmap[r][c]
                        if val > max_prob:
                            max_prob = val; best_targets = [(r, c)]
                        elif val == max_prob and val > 0:
                            best_targets.append((r, c))

        if best_targets:
            seed = (self.parity_offset * 10000) + len(move_history)
            rng = random.Random(seed)
            target = rng.choice(best_targets)
            return target, f"{phase_name} ({max_prob}%)", dead_sizes, diagnostics

        open_cells = [(r, c) for r in range(10) for c in range(10) if board[r][c] == CELL_UNKNOWN]
        target = open_cells[0] if open_cells else (0, 0)
        return target, "Sherrinford-Apex: Failsafe Fallback", dead_sizes, diagnostics

    def get_heatmap(self, board, dead_sizes=None):
        return self.last_heatmap

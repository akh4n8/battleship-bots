import math
import random
import time
from collections import defaultdict
from core.game_engine import CELL_UNKNOWN, CELL_MISS, CELL_HIT, CELL_SUNK

class SherrinfordParityBot:
    """
    SherrinfordParityBot: Sherrinford with Dynamic Stride Parity.
    
    Architectural Principles:
    1. Multi-Hypothesis Fog of War Marginalization (evaluates all valid sunk combinations).
    2. Pure Exact Bayesian Joint Cover (zero Linear Assassin heuristics).
    3. Dynamic Stride Parity:
       - Stride 2 (Checkerboard: (r+c)%2) while the 2-peg Patrol Boat is alive.
       - Stride 3 (Diagonal Stride: (r+2c)%3) once the 2-ship is dead, reducing full
         board sweep from 50 down to 34 shots with 0% chance of missing a len-3+ ship.
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

    def _find_all_sunk_hypotheses(self, board, move_history):
        sunks = []
        peg_times = {}
        all_hits = set()
        for m in move_history:
            if m['result'] != 'MISS':
                pt = (m['x'], m['y'])
                peg_times[pt] = m['shotNumber']
                all_hits.add(pt)
                if m['result'] == 'SUNK':
                    sunks.append(pt)

        if not sunks:
            return [([], [], all_hits)]

        sunks.sort(key=lambda pt: peg_times[pt])

        def get_placements_for_sunk(sunk_pt, length):
            sr, sc = sunk_pt
            sunk_time = peg_times[sunk_pt]
            plist = []
            for c_start in range(sc - length + 1, sc + 1):
                if 0 <= c_start and c_start + length <= 10:
                    coords = tuple((sr, c_start + i) for i in range(length))
                    valid = True
                    max_t = -1
                    lethal = None
                    for pt in coords:
                        if pt not in peg_times:
                            valid = False; break
                        t = peg_times[pt]
                        if t > max_t: max_t = t; lethal = pt
                    if valid and lethal == sunk_pt: plist.append(coords)
            for r_start in range(sr - length + 1, sr + 1):
                if 0 <= r_start and r_start + length <= 10:
                    coords = tuple((r_start + i, sc) for i in range(length))
                    valid = True
                    max_t = -1
                    lethal = None
                    for pt in coords:
                        if pt not in peg_times:
                            valid = False; break
                        t = peg_times[pt]
                        if t > max_t: max_t = t; lethal = pt
                    if valid and lethal == sunk_pt: plist.append(coords)
            return plist

        solutions = []

        def search(sunk_idx, available_fleet, current_ships, current_sizes, claimed_pegs):
            if sunk_idx == len(sunks):
                leftovers = all_hits - claimed_pegs
                if leftovers and available_fleet:
                    min_s = min(available_fleet)
                    valid_leftovers = True
                    for hr, hc in leftovers:
                        h_open = sum(1 for c in range(max(0, hc - min_s + 1), min(10, hc + min_s)) 
                                     if board[hr][c] in [CELL_UNKNOWN, CELL_HIT] and (hr, c) not in claimed_pegs)
                        v_open = sum(1 for r in range(max(0, hr - min_s + 1), min(10, hr + min_s)) 
                                     if board[r][hc] in [CELL_UNKNOWN, CELL_HIT] and (r, hc) not in claimed_pegs)
                        if h_open < min_s and v_open < min_s:
                            valid_leftovers = False; break
                    if not valid_leftovers: return
                solutions.append((list(current_ships), list(current_sizes), leftovers))
                return

            sunk_pt = sunks[sunk_idx]
            unique_sizes = sorted(set(available_fleet), reverse=True)
            for size in unique_sizes:
                placements = get_placements_for_sunk(sunk_pt, size)
                for p in placements:
                    p_set = set(p)
                    if not p_set.intersection(claimed_pegs):
                        new_fleet = available_fleet.copy()
                        new_fleet.remove(size)
                        search(sunk_idx + 1, new_fleet, current_ships + [p], current_sizes + [size], claimed_pegs | p_set)

        search(0, self.FLEET_SIZES, [], [], set())
        return solutions

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
        hypotheses = self._find_all_sunk_hypotheses(board, move_history)
        if not hypotheses:
            all_hits = set((r, c) for r in range(10) for c in range(10) if board[r][c] in [CELL_HIT, CELL_SUNK])
            hypotheses = [([], [], all_hits)]

        min_leftovers = min(len(h[2]) for h in hypotheses)
        admissible = [h for h in hypotheses if len(h[2]) == min_leftovers]

        total_target_counts = defaultdict(float)
        has_active_target = False
        primary_diag = admissible[0]

        diagnostics = {
            "active": list(primary_diag[2]),
            "claimed": [p for s in primary_diag[0] for p in s],
            "dead_ships": primary_diag[0],
            "dead_sizes": primary_diag[1],
            "hypotheses_count": len(hypotheses)
        }

        for dead_ships, dead_sizes, active_hits in admissible:
            claimed_pegs = set(peg for ship in dead_ships for peg in ship)
            if active_hits:
                has_active_target = True
                counts = self._compute_active_hit_density(board, active_hits, claimed_pegs, dead_sizes)
                for coord, val in counts.items():
                    total_target_counts[coord] += val

        # Check if the 2-ship is confirmed dead across all admissible hypotheses
        is_two_ship_dead = all(2 in h[1] for h in admissible)

        if has_active_target and total_target_counts:
            self.last_heatmap = [[0] * 10 for _ in range(10)]
            max_c = max(total_target_counts.values())
            if max_c > 0:
                for (r, c), cnt in total_target_counts.items():
                    if board[r][c] == CELL_UNKNOWN:
                        self.last_heatmap[r][c] = int((cnt / max_c) * 100)
            phase_name = "Sherrinford-Parity: Exact Bayesian Target Density"
            use_hunt_parity = False
        else:
            if len(move_history) == 0:
                self.last_heatmap = [row[:] for row in self.EMPTY_BOARD_EXACT_HEATMAP]
            else:
                dead_sizes = primary_diag[1]
                available_fleet = self.FLEET_SIZES.copy()
                for s in dead_sizes:
                    if s in available_fleet: available_fleet.remove(s)

                forbidden_cells = set()
                for r in range(10):
                    for c in range(10):
                        if board[r][c] == CELL_MISS: forbidden_cells.add((r, c))
                forbidden_cells.update(primary_diag[2])
                for s in primary_diag[0]: forbidden_cells.update(s)

                self.last_heatmap = self._compute_open_water_density(board, available_fleet, forbidden_cells)

            stride_str = "Stride-3" if is_two_ship_dead else "Stride-2"
            phase_name = f"Sherrinford-Parity: Exact Hunt ({stride_str})"
            use_hunt_parity = True

        max_prob = -1
        best_targets = []

        if use_hunt_parity:
            # Dynamic Parity Strategy
            if is_two_ship_dead:
                # Stride 3 diagonal sweep: (r + 2*c) % 3 == parity_offset % 3
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
            return target, f"{phase_name} ({max_prob}%)", primary_diag[1], diagnostics

        open_cells = [(r, c) for r in range(10) for c in range(10) if board[r][c] == CELL_UNKNOWN]
        target = open_cells[0] if open_cells else (0, 0)
        return target, "Sherrinford-Parity: Failsafe Fallback", primary_diag[1], diagnostics

    def get_heatmap(self, board, dead_sizes=None):
        return self.last_heatmap

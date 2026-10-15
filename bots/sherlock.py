import random
from collections import defaultdict
from core.game_engine import CELL_UNKNOWN, CELL_MISS, CELL_HIT, CELL_SUNK

class SherlockBot:
    def __init__(self):
        self.original_fleet = [5, 4, 3, 3, 2]
        self.parity_offset = random.choice([0, 1])
        self._all_placements_cache = None
        self.last_heatmap = None
        self._init_placements()

    def _init_placements(self):
        placements_by_size = defaultdict(list)
        for size in set(self.original_fleet):
            for r in range(10):
                for c in range(10 - size + 1):
                    mask = sum(1 << (r * 10 + c + k) for k in range(size))
                    coords = tuple((r, c + k) for k in range(size))
                    placements_by_size[size].append((size, mask, coords))
            for r in range(10 - size + 1):
                for c in range(10):
                    mask = sum(1 << ((r + k) * 10 + c) for k in range(size))
                    coords = tuple((r + k, c) for k in range(size))
                    placements_by_size[size].append((size, mask, coords))
        self._all_placements_cache = placements_by_size

    def _compute_active_hit_density(self, board, active_hits, claimed_hits, dead_sizes):
        available_fleet = self.original_fleet.copy()
        for s in dead_sizes:
            if s in available_fleet:
                available_fleet.remove(s)

        forbidden_mask = 0
        for r in range(10):
            for c in range(10):
                if board[r][c] == CELL_MISS:
                    forbidden_mask |= (1 << (r * 10 + c))
        for r, c in claimed_hits:
            forbidden_mask |= (1 << (r * 10 + c))

        hits_mask = sum(1 << (r * 10 + c) for r, c in active_hits)

        # 1. Single-ship exact covers
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
                    if board[r][c] == CELL_UNKNOWN:
                        counts[(r, c)] += 1
            return counts

        # 2. Joint 2-ship exact covers
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
                    if (h1 | h2) == hits_mask:
                        pair_covers.append((c1, c2))

        if pair_covers:
            counts = defaultdict(int)
            for c1, c2 in pair_covers:
                for r, c in list(c1) + list(c2):
                    if board[r][c] == CELL_UNKNOWN:
                        counts[(r, c)] += 1
            return counts

        # 3. Partial overlap fallback
        counts = defaultdict(int)
        for s_idx, _, m, coords, h in cand_p:
            weight = bin(h).count('1') ** 2
            for r, c in coords:
                if board[r][c] == CELL_UNKNOWN:
                    counts[(r, c)] += weight
        return counts

    def get_best_move(self, board, move_history):
        solutions = self._analyze_board_state_all(board, move_history)
        active_hits, dead_ships, claimed_hits, dead_sizes = solutions[0]
        has_active = any(sol[0] for sol in solutions)

        diagnostics = {
            "active": active_hits,
            "suspended": [], 
            "claimed": claimed_hits,
            "dead_ships": dead_ships,
            "dead_sizes": dead_sizes,
            "deficit": False 
        }

        if has_active:
            total_counts = defaultdict(int)
            for s_active, s_dead_ships, s_claimed, s_sizes in solutions:
                if s_active:
                    counts = self._compute_active_hit_density(board, s_active, s_claimed, s_sizes)
                    for k, v in counts.items():
                        total_counts[k] += v

            max_c = max(total_counts.values()) if total_counts else 0
            best_targets = [k for k, v in total_counts.items() if v == max_c and board[k[0]][k[1]] == CELL_UNKNOWN]

            # Cache heatmap for UI visualization
            self.last_heatmap = [[0] * 10 for _ in range(10)]
            if max_c > 0:
                for (r, c), val in total_counts.items():
                    if board[r][c] == CELL_UNKNOWN:
                        self.last_heatmap[r][c] = int((val / max_c) * 100)

            if best_targets:
                seed = (self.parity_offset * 10000) + len(move_history)
                rng = random.Random(seed)
                target = rng.choice(best_targets)
                return target, "Exact Target Density", dead_sizes, diagnostics

            # Fallback to linear kill if no combinatorial targets
            if active_hits:
                target = self._execute_linear_kill(board, active_hits, claimed_hits)
                if target:
                    return target, "Linear Kill (Fallback)", dead_sizes, diagnostics

        is_two_ship_dead = all(2 in sol[3] for sol in solutions) if solutions else False
        stride_str = "Stride-3" if is_two_ship_dead else "Stride-2"
        target = self._execute_gravitational_hunt(board, dead_sizes, dead_ships, all_solutions=solutions, move_history=move_history)
        return target, f"Gravitational Hunt ({stride_str})", dead_sizes, diagnostics

    # ==========================================
    # CHRONOLOGICAL CONSTRAINT SOLVER
    # ==========================================

    def _analyze_board_state_all(self, board, move_history):
        size = len(board)
        all_hits = set()
        sunks = []
        peg_times = {(m["x"], m["y"]): m["shotNumber"] for m in move_history if m["result"] != "MISS"}

        for r in range(size):
            for c in range(size):
                if board[r][c] == CELL_HIT: all_hits.add((r, c))
                if board[r][c] == CELL_SUNK: sunks.append((r, c))

        if not sunks:
            return [(list(all_hits), [], [], [])]

        sunks.sort(key=lambda s: peg_times.get(s, float('inf')))

        best_score, solutions = self._solve_board_all(board, sunks, peg_times, self.original_fleet.copy(), [], [], all_hits)
        
        if not solutions:
            dead_ships = [[s] for s in sunks]
            claimed_hits = set(sunks)
            dead_sizes = [1] * len(sunks)
            active_hits = list(all_hits - claimed_hits)
            return [(active_hits, dead_ships, list(claimed_hits), dead_sizes)]

        parsed = []
        for dead_ships, dead_sizes, score in solutions:
            claimed_hits = set(peg for ship in dead_ships for peg in ship)
            active_hits = list(all_hits - claimed_hits)
            parsed.append((active_hits, dead_ships, list(claimed_hits), dead_sizes))
        return parsed

    def _analyze_board_state(self, board, move_history):
        solutions = self._analyze_board_state_all(board, move_history)
        active_hits, dead_ships, claimed_hits, dead_sizes = solutions[0]
        return active_hits, dead_ships, claimed_hits, dead_sizes

    def _solve_board_all(self, board, sunks, peg_times, available_fleet, current_dead_ships, current_dead_sizes, all_hits):
        if not sunks:
            claimed_set = set(peg for ship in current_dead_ships for peg in ship)
            leftovers = all_hits - claimed_set
            
            if available_fleet and leftovers:
                min_ship_size = min(available_fleet)
                size = len(board)
                
                for hit in leftovers:
                    r, c = hit
                    
                    left = c
                    while left > 0 and board[r][left-1] in [CELL_UNKNOWN, CELL_HIT] and (r, left-1) not in claimed_set:
                        left -= 1
                    right = c
                    while right < size - 1 and board[r][right+1] in [CELL_UNKNOWN, CELL_HIT] and (r, right+1) not in claimed_set:
                        right += 1
                    horiz_len = right - left + 1
                    horiz_unknowns = sum(1 for i in range(left, right + 1) if board[r][i] == CELL_UNKNOWN)
                    
                    up = r
                    while up > 0 and board[up-1][c] in [CELL_UNKNOWN, CELL_HIT] and (up-1, c) not in claimed_set:
                        up -= 1
                    down = r
                    while down < size - 1 and board[down+1][c] in [CELL_UNKNOWN, CELL_HIT] and (down+1, c) not in claimed_set:
                        down += 1
                    vert_len = down - up + 1
                    vert_unknowns = sum(1 for i in range(up, down + 1) if board[i][c] == CELL_UNKNOWN)
                    
                    valid_horiz = horiz_len >= min_ship_size and horiz_unknowns > 0
                    valid_vert = vert_len >= min_ship_size and vert_unknowns > 0
                    
                    if not (valid_horiz or valid_vert):
                        return float('inf'), []

            return len(leftovers), [(current_dead_ships, current_dead_sizes, len(leftovers))]

        sunk = sunks[0]
        sunk_time = peg_times.get(sunk, float('inf'))

        unique_sizes = list(set(available_fleet))
        unique_sizes.sort(reverse=True)
        
        current_claimed_set = set(peg for ship in current_dead_ships for peg in ship)

        best_score = float('inf')
        best_solutions = []

        for size in unique_sizes:
            placements = self._get_valid_placements(board, sunk, sunk_time, size, peg_times)
            
            for placement in placements:
                placement_set = set(placement)
                
                if not placement_set.intersection(current_claimed_set):
                    new_dead_ships = current_dead_ships + [placement]
                    new_fleet = available_fleet.copy()
                    new_fleet.remove(size)
                    new_dead_sizes = current_dead_sizes + [size]

                    child_score, child_solutions = self._solve_board_all(
                        board, sunks[1:], peg_times, new_fleet, new_dead_ships, new_dead_sizes, all_hits
                    )
                    
                    if child_solutions:
                        if child_score < best_score:
                            best_score = child_score
                            best_solutions = list(child_solutions)
                        elif child_score == best_score:
                            best_solutions.extend(child_solutions)
                                
        deduped = []
        seen = set()
        for ships, sizes, score in best_solutions:
            key = (tuple(sorted(tuple(sorted(s)) for s in ships)), tuple(sorted(sizes)))
            if key not in seen:
                seen.add(key)
                deduped.append((ships, sizes, score))

        return best_score, deduped

    def _solve_board(self, board, sunks, peg_times, available_fleet, current_dead_ships, current_dead_sizes, all_hits):
        best_score, solutions = self._solve_board_all(board, sunks, peg_times, available_fleet, current_dead_ships, current_dead_sizes, all_hits)
        return solutions[0] if solutions else None 

    def _get_valid_placements(self, board, sunk, sunk_time, size, peg_times):
        placements = []
        board_size = len(board)
        sr, sc = sunk
        
        for c_start in range(sc - size + 1, sc + 1):
            valid = True
            peg_list = []
            for i in range(size):
                r, c = sr, c_start + i
                if 0 <= r < board_size and 0 <= c < board_size:
                    if (r, c) == sunk:
                        peg_list.append((r, c))
                    elif (r, c) in peg_times and peg_times[(r, c)] < sunk_time:
                        peg_list.append((r, c))
                    else:
                        valid = False; break
                else:
                    valid = False; break
            if valid: placements.append(peg_list)
                
        for r_start in range(sr - size + 1, sr + 1):
            valid = True
            peg_list = []
            for i in range(size):
                r, c = r_start + i, sc
                if 0 <= r < board_size and 0 <= c < board_size:
                    if (r, c) == sunk:
                        peg_list.append((r, c))
                    elif (r, c) in peg_times and peg_times[(r, c)] < sunk_time:
                        peg_list.append((r, c))
                    else:
                        valid = False; break
                else:
                    valid = False; break
            if valid: placements.append(peg_list)
                
        return placements

    # ==========================================
    # LINEAR KILL TRACER
    # ==========================================

    def _execute_linear_kill(self, board, active_hits, claimed_hits):
        pruned_hits = [h for h in active_hits if h not in claimed_hits]
        unvisited = set(pruned_hits)
        size = len(board)

        while unvisited:
            start = next(iter(unvisited))
            cluster = set()
            queue = [start]

            while queue:
                current = queue.pop(0)
                if current not in cluster:
                    cluster.add(current)
                    unvisited.discard(current)
                    r, c = current
                    neighbors = [(r, c-1), (r, c+1), (r-1, c), (r+1, c)]
                    for n in neighbors:
                        if n in pruned_hits and n not in cluster:
                            queue.append(n)

            rows = set(r for r, c in cluster)
            cols = set(c for r, c in cluster)

            if len(cluster) == 1:
                valid_targets = self._get_unknown_neighbors(board, cluster.pop())
                if valid_targets: return random.choice(valid_targets)
                
            elif len(rows) == 1: 
                r = list(rows)[0]
                endpoints = []
                
                c = min(cols)
                # THE TRUE FIX: Only traverse the verified active cluster!
                while (r, c) in cluster: c -= 1
                if c >= 0 and board[r][c] == CELL_UNKNOWN: endpoints.append((r, c))
                
                c = max(cols)
                # THE TRUE FIX: Only traverse the verified active cluster!
                while (r, c) in cluster: c += 1
                if c < size and board[r][c] == CELL_UNKNOWN: endpoints.append((r, c))
                
                if endpoints: return random.choice(endpoints)
                
            elif len(cols) == 1: 
                c = list(cols)[0]
                endpoints = []
                
                r = min(rows)
                # THE TRUE FIX: Only traverse the verified active cluster!
                while (r, c) in cluster: r -= 1
                if r >= 0 and board[r][c] == CELL_UNKNOWN: endpoints.append((r, c))
                
                r = max(rows)
                # THE TRUE FIX: Only traverse the verified active cluster!
                while (r, c) in cluster: r += 1
                if r < size and board[r][c] == CELL_UNKNOWN: endpoints.append((r, c))
                
                if endpoints: return random.choice(endpoints)
                
            else:
                valid_endpoints = []
                for r, c in cluster:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if (r + dr, c + dc) in cluster:
                            curr_r, curr_c = r, c
                            # This internal loop only traverses the active cluster, 
                            # so it is naturally safe from jumping corpses.
                            while (curr_r, curr_c) in cluster:
                                curr_r += dr
                                curr_c += dc
                            
                            if 0 <= curr_r < size and 0 <= curr_c < size:
                                if board[curr_r][curr_c] == CELL_UNKNOWN:
                                    valid_endpoints.append((curr_r, curr_c))
                
                if valid_endpoints:
                    return random.choice(valid_endpoints)

            for hit in cluster:
                valid_targets = self._get_unknown_neighbors(board, hit)
                if valid_targets: return valid_targets[0]
                
        return None

    def _get_unknown_neighbors(self, board, peg):
        r, c = peg
        neighbors = [(r, c-1), (r, c+1), (r-1, c), (r+1, c)]
        size = len(board)
        return [(nr, nc) for nr, nc in neighbors if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == CELL_UNKNOWN]

    # ==========================================
    # GRAVITATIONAL HUNT MAP & SWEEPER
    # ==========================================

    def _get_perpendicular_ends(self, ship_pegs, board):
        """Calculates the 4 corner 'armpit' squares perpendicular to the extreme ends of a ship."""
        targets = []
        if len(ship_pegs) < 2: 
            return targets
            
        sorted_pegs = sorted(list(ship_pegs))
        size = len(board)
        
        r1, c1 = sorted_pegs[0]  # First end
        r2, c2 = sorted_pegs[-1] # Opposite end
        
        if r1 == r2: # Horizontal ship
            potentials = [(r1-1, c1), (r1+1, c1), (r2-1, c2), (r2+1, c2)]
        else: # Vertical ship
            potentials = [(r1, c1-1), (r1, c1+1), (r2, c2-1), (r2, c2+1)]
            
        for r, c in potentials:
            if 0 <= r < size and 0 <= c < size and board[r][c] == CELL_UNKNOWN:
                targets.append((r, c))
                
        return targets

    def _execute_gravitational_hunt(self, board, dead_sizes, dead_ships, all_solutions=None, move_history=None):
        size = len(board)
        rng = None
        if move_history is not None:
            seed = (self.parity_offset * 10000) + len(move_history)
            rng = random.Random(seed)
        
        # Check if 2-ship is confirmed dead across all tied solutions
        is_two_ship_dead = False
        if all_solutions:
            is_two_ship_dead = all(2 in sol[3] for sol in all_solutions)
        elif dead_sizes:
            is_two_ship_dead = (2 in dead_sizes)

        # --- PHASE 1: Primary Parity Hunt ---
        if all_solutions and len(all_solutions) > 1:
            heat_map = [[0.0] * size for _ in range(size)]
            for sol in all_solutions:
                hm = self.compute_hunt_heatmap(board, sol[3], use_parity=True, is_two_ship_dead=is_two_ship_dead)
                for r in range(size):
                    for c in range(size):
                        heat_map[r][c] += hm[r][c] / len(all_solutions)
        else:
            heat_map = self.compute_hunt_heatmap(board, dead_sizes, use_parity=True, is_two_ship_dead=is_two_ship_dead)

        self.last_heatmap = [[int(heat_map[r][c]) for c in range(size)] for r in range(size)]

        max_heat = -1.0
        best_moves = []
        
        for r in range(size):
            for c in range(size):
                if board[r][c] == CELL_UNKNOWN:
                    heat = heat_map[r][c]
                    if heat > max_heat:
                        max_heat = heat
                        best_moves = [(r, c)]
                    elif abs(heat - max_heat) < 1e-6:
                        best_moves.append((r, c))
                        
        if max_heat > 0 and best_moves: 
            return rng.choice(best_moves) if rng else random.choice(best_moves)

        # --- PHASE 2: 4-Tier Blindspot Sweeper (The Corner Hunter) ---
        hunt_dead_ships = []
        if all_solutions and len(all_solutions) > 1:
            for sol in all_solutions:
                hunt_dead_ships.extend(sol[1])
        else:
            hunt_dead_ships = dead_ships
        
        # Priority 2.1: Perpendicular corners (ends) of the 3-ships
        three_ship_ends = []
        for ship in hunt_dead_ships:
            if len(ship) == 3:
                three_ship_ends.extend(self._get_perpendicular_ends(ship, board))
        if three_ship_ends: 
            return rng.choice(three_ship_ends) if rng else random.choice(three_ship_ends)
            
        # Priority 2.2: Perpendicular corners (ends) of all other corpses
        other_ship_ends = []
        for ship in hunt_dead_ships:
            if len(ship) != 3:
                other_ship_ends.extend(self._get_perpendicular_ends(ship, board))
        if other_ship_ends: 
            return rng.choice(other_ship_ends) if rng else random.choice(other_ship_ends)

        # Priority 2.3: General perimeter (flat edges) of the 3-ships
        three_perimeter = []
        for ship in hunt_dead_ships:
            if len(ship) == 3:
                for r, c in ship:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == CELL_UNKNOWN:
                            three_perimeter.append((nr, nc))
        if three_perimeter: 
            return rng.choice(three_perimeter) if rng else random.choice(three_perimeter)

        # Priority 2.4: General perimeter (flat edges) of all other corpses
        other_perimeter = []
        for ship in hunt_dead_ships:
            if len(ship) != 3:
                for r, c in ship:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == CELL_UNKNOWN:
                            other_perimeter.append((nr, nc))
        if other_perimeter: 
            return rng.choice(other_perimeter) if rng else random.choice(other_perimeter)

        # --- PHASE 3: Emergency Non-Parity Hunt ---
        heat_map_em = self.compute_hunt_heatmap(board, [5, 4, 3, 3], use_parity=False) 
        max_heat_em = -1
        best_moves_em = []
        
        for r in range(size):
            for c in range(size):
                if board[r][c] == CELL_UNKNOWN:
                    heat = heat_map_em[r][c]
                    if heat > max_heat_em:
                        max_heat_em = heat
                        best_moves_em = [(r, c)]
                    elif heat == max_heat_em:
                        best_moves_em.append((r, c))
                        
        if max_heat_em > 0 and best_moves_em: 
            return rng.choice(best_moves_em) if rng else random.choice(best_moves_em)
        
        # --- PHASE 4: Absolute Fallback ---
        open_water = [(r, c) for r in range(size) for c in range(size) if board[r][c] == CELL_UNKNOWN]
        if open_water:
            return rng.choice(open_water) if rng else random.choice(open_water)
            
        return (0, 0)

    def compute_hunt_heatmap(self, board, dead_sizes, use_parity=True, is_two_ship_dead=False):
        size = len(board)
        heat_map = [[0 for _ in range(size)] for _ in range(size)]
        
        living_fleet = self.original_fleet.copy()
        for ds in dead_sizes:
            size_to_remove = 5 if ds > 5 else ds
            if size_to_remove in living_fleet:
                living_fleet.remove(size_to_remove)
                
        if not living_fleet: living_fleet.append(2)
        unique_ships = list(set(living_fleet))

        for ship_size in unique_ships:
            for r in range(size):
                for c in range(size - ship_size + 1):
                    self._evaluate_hunt_placement(board, heat_map, r, c, ship_size, True, use_parity, is_two_ship_dead)
            for r in range(size - ship_size + 1):
                for c in range(size):
                    self._evaluate_hunt_placement(board, heat_map, r, c, ship_size, False, use_parity, is_two_ship_dead)

        return heat_map

    def get_heatmap(self, board, dead_sizes=None, use_parity=True):
        if use_parity is False:
            return self.compute_hunt_heatmap(board, dead_sizes or [], use_parity=False)
        if self.last_heatmap is not None:
            return self.last_heatmap
        is_two_ship_dead = (2 in dead_sizes) if dead_sizes else False
        return self.compute_hunt_heatmap(board, dead_sizes or [], use_parity=use_parity, is_two_ship_dead=is_two_ship_dead)

    def _evaluate_hunt_placement(self, board, heat_map, start_r, start_c, ship_size, is_horizontal, use_parity, is_two_ship_dead=False):
        for i in range(ship_size):
            r = start_r if is_horizontal else start_r + i
            c = start_c + i if is_horizontal else start_c
            if board[r][c] in [CELL_MISS, CELL_SUNK, CELL_HIT]: return

        for i in range(ship_size):
            r = start_r if is_horizontal else start_r + i
            c = start_c + i if is_horizontal else start_c
            if board[r][c] == CELL_UNKNOWN:
                if use_parity:
                    if is_two_ship_dead:
                        target_p = self.parity_offset % 3
                        heat_map[r][c] += 1 if (r + 2 * c) % 3 == target_p else 0
                    else:
                        heat_map[r][c] += 0 if (r + c) % 2 == self.parity_offset else 1
                else:
                    heat_map[r][c] += 1
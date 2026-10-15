import time
import random
from core.game_engine import CELL_UNKNOWN, CELL_MISS, CELL_HIT, CELL_SUNK

class MycroftBot:
    def __init__(self, safe_mode=True): # <--- ADD PARAMETER HERE
        self.fleet_sizes = [5, 4, 3, 3, 2]
        self.safe_mode = safe_mode        # <--- SAVE IT HERE
        self.parity_offset = random.choice([0, 1])
        
        from bots.sherlock import SherlockBot
        self.sherlock = SherlockBot() 
        self.sherlock.parity_offset = self.parity_offset 
        
        self.bit_to_idx = {1 << i: i for i in range(100)}
        self.last_heatmap = [[0 for _ in range(10)] for _ in range(10)]

    def get_best_move(self, board, move_history):
        # 1. Visual Diagnostics ONLY (Passed to UI)
        try:
            ui_active, ui_dead, ui_claimed, ui_sizes = self.sherlock._analyze_board_state(board, move_history)
        except Exception:
            ui_active, ui_dead, ui_claimed, ui_sizes = [], [], [], []
            
        diagnostics = {
            "active": ui_active, "suspended": [], 
            "claimed": ui_claimed, "dead_ships": ui_dead, "deficit": False
        }

        # 2. Extract Absolute Board State & CHRONOLOGY
        miss_mask = 0
        all_hits_mask = 0
        sunk_mask = 0
        num_sunks = 0
        
        # Array lookup is drastically faster than dictionary lookup in MCMC loops
        peg_times_arr = [-1] * 100
        for m in move_history:
            if m["result"] != "MISS":
                idx = m["x"] * 10 + m["y"]
                peg_times_arr[idx] = m["shotNumber"]
        
        for r in range(10):
            for c in range(10):
                idx = r * 10 + c
                cell = board[r][c]
                if cell == CELL_MISS:
                    miss_mask |= (1 << idx)
                elif cell == CELL_HIT:
                    all_hits_mask |= (1 << idx)
                elif cell == CELL_SUNK:
                    all_hits_mask |= (1 << idx)
                    sunk_mask |= (1 << idx)
                    num_sunks += 1

        # 3. Precompute all valid 1D placement masks
        placements_by_size = {}
        for size in set(self.fleet_sizes):
            valid_placements = []
            for r in range(10):
                for c in range(10):
                    # Horizontal
                    if c + size <= 10:
                        p_mask = 0
                        valid = True
                        for i in range(size):
                            idx = r * 10 + (c + i)
                            if miss_mask & (1 << idx): valid = False; break
                            p_mask |= (1 << idx)
                        if valid: valid_placements.append(p_mask)
                    # Vertical
                    if r + size <= 10:
                        p_mask = 0
                        valid = True
                        for i in range(size):
                            idx = (r + i) * 10 + c
                            if miss_mask & (1 << idx): valid = False; break
                            p_mask |= (1 << idx)
                        if valid: valid_placements.append(p_mask)
            placements_by_size[size] = valid_placements
            
        placements_for_ships = [placements_by_size[s] for s in self.fleet_sizes]

        # 4. Phase 1: MCMC Chain Initialization (Exact Cover Hit-Solver)
        covering_placements = {size: {} for size in set(self.fleet_sizes)}
        for size in set(self.fleet_sizes):
            for i in range(100):
                bit = 1 << i
                if bit & all_hits_mask:
                    covering_placements[size][bit] = [p for p in placements_by_size[size] if (p & bit)]
                    
        open_water_placements = {size: [p for p in placements_by_size[size] if (p & all_hits_mask) == 0] for size in set(self.fleet_sizes)}

        def is_state_valid(state_masks):
            occupied = state_masks[0] | state_masks[1] | state_masks[2] | state_masks[3] | state_masks[4]
            if (occupied & all_hits_mask) != all_hits_mask: return False
            
            fully_hit_count = 0
            fully_hit_sunk_mask = 0
            
            for m in state_masks:
                if (m & all_hits_mask) == m:
                    fully_hit_count += 1
                    max_time = -1
                    lethal_mask = 0
                    temp_m = m
                    while temp_m:
                        b = temp_m & -temp_m
                        idx = self.bit_to_idx[b]
                        t = peg_times_arr[idx]
                        if t > max_time:
                            max_time = t
                            lethal_mask = b
                        temp_m ^= b
                        
                    if (lethal_mask & sunk_mask) == 0: return False
                    fully_hit_sunk_mask |= lethal_mask
                    
            if fully_hit_count != num_sunks: return False
            if fully_hit_sunk_mask != sunk_mask: return False
            return True

        dfs_start = time.time()

        def place_remaining(placed_mask, occ, st):
            final_state = [0] * 5
            for i, p in st: final_state[i] = p
            unplaced = [i for i in range(5) if not (placed_mask & (1 << i))]
            
            def recurse(idx_ptr, current_occ):
                if idx_ptr == len(unplaced): return True
                i = unplaced[idx_ptr]
                for p in open_water_placements[self.fleet_sizes[i]]:
                    if (current_occ & p) == 0:
                        final_state[i] = p
                        if recurse(idx_ptr + 1, current_occ | p): return True
                return False
                
            if recurse(0, occ): return final_state
            return None

        # DFS to find exactly ONE valid seed to start the Markov Chain
        def solve_hits(placed_mask, occ, st, current_sunk_mask, fully_hit_count):
            if time.time() - dfs_start > 0.4: return None
                
            uncovered_hits = all_hits_mask & ~occ
            
            if uncovered_hits == 0:
                if fully_hit_count == num_sunks and current_sunk_mask == sunk_mask:
                    return place_remaining(placed_mask, occ, st)
                return None
                
            target_hit = uncovered_hits & -uncovered_hits
            
            for i, size in enumerate(self.fleet_sizes):
                if not (placed_mask & (1 << i)):
                    for p in covering_placements[size].get(target_hit, []):
                        if (occ & p) == 0:
                            new_fully_hit_count = fully_hit_count
                            new_sunk_mask = current_sunk_mask
                            
                            if ((p & all_hits_mask) == p):
                                new_fully_hit_count += 1
                                max_time, lethal_mask = -1, 0
                                temp_m = p
                                while temp_m:
                                    b = temp_m & -temp_m
                                    t = peg_times_arr[self.bit_to_idx[b]]
                                    if t > max_time: max_time, lethal_mask = t, b
                                    temp_m ^= b
                                    
                                if (lethal_mask & sunk_mask) == 0: continue 
                                new_sunk_mask |= lethal_mask
                                
                            res = solve_hits(placed_mask | (1 << i), occ | p, st + [(i, p)], new_sunk_mask, new_fully_hit_count)
                            if res: return res
            return None

        start_time = time.time()
        current_state = solve_hits(0, 0, [], 0, 0)
        
        # 5. Phase 2: Unrestrained Markov Chain Monte Carlo Loop
        raw_heatmap = [0] * 100
        valid_universes = 0
        
        if current_state:
            target_sims = 100_000
            max_time = 0.5 
            
            for _ in range(target_sims):
                if time.time() - start_time > max_time: break
                
                # THE FIX: Allow 2-Ship jumps 20% of the time to escape local minima!
                if random.random() < 0.2:
                    indices_to_mutate = random.sample(range(5), 2)
                else:
                    indices_to_mutate = [random.randrange(5)]
                
                old_placements = [current_state[i] for i in indices_to_mutate]
                
                other_occupied = 0
                for i in range(5):
                    if i not in indices_to_mutate:
                        other_occupied |= current_state[i]
                        
                valid_proposal = True
                needs_full_check = False
                proposed_occ = other_occupied
                
                for i, ship_idx in enumerate(indices_to_mutate):
                    new_p = random.choice(placements_for_ships[ship_idx])
                    if proposed_occ & new_p:
                        valid_proposal = False
                        break
                    
                    # Optimization: Only run heavy chronological checks if hits are involved
                    if (new_p & all_hits_mask) or (old_placements[i] & all_hits_mask):
                        needs_full_check = True
                        
                    proposed_occ |= new_p
                    current_state[ship_idx] = new_p
                
                # Verify Metropolis-Hastings Acceptance
                if valid_proposal:
                    if needs_full_check and not is_state_valid(current_state):
                        valid_proposal = False
                        
                if not valid_proposal:
                    # Reject: Revert to previous valid state
                    for i, ship_idx in enumerate(indices_to_mutate):
                        current_state[ship_idx] = old_placements[i]
                
                # Record Heat
                for p in current_state:
                    temp_mask = p
                    while temp_mask:
                        bit = temp_mask & -temp_mask
                        raw_heatmap[self.bit_to_idx[bit]] += 1
                        temp_mask ^= bit
                        
                valid_universes += 1

        # 6. Heatmap Assembly & Targeting (PARITY RESTORED FOR HUNTING)
        max_heat = -1
        best_moves = []
        self.last_heatmap = [[0 for _ in range(10)] for _ in range(10)]

        # Detect active hits: If all_hits_mask != fully_hit_mask of the current state
        has_active_hits = False
        if current_state:
            fully_hit_mask = 0
            for m in current_state:
                if (m & all_hits_mask) == m: fully_hit_mask |= m
            if (all_hits_mask & ~fully_hit_mask) > 0:
                has_active_hits = True

        for r in range(10):
            for c in range(10):
                if board[r][c] == CELL_UNKNOWN:
                    idx = r * 10 + c
                    heat = raw_heatmap[idx]
                    
                    # THE FIX: Only apply parity if Safe Mode is ON
                    if not has_active_hits and self.safe_mode and (r + c) % 2 != self.parity_offset:
                        heat = 0
                        
                    self.last_heatmap[r][c] = heat

                    if heat > max_heat:
                        max_heat = heat
                        best_moves = [(r, c)]
                    elif heat == max_heat and heat > 0:
                        best_moves.append((r, c))

        if max_heat > 0 and best_moves:
            target = random.choice(best_moves)
            log = f"Pure MCMC ({valid_universes} Steps)"
        else:
            open_water = [(r, c) for r in range(10) for c in range(10) if board[r][c] == CELL_UNKNOWN]
            target = random.choice(open_water) if open_water else (0, 0)
            log = "Random Fallback (MCMC Exhaustion)"

        return target, log, ui_sizes, diagnostics

    def get_heatmap(self, board, dead_sizes):
        return self.last_heatmap
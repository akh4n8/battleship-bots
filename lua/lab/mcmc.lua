-- lua/lab/mcmc.lua
-- Research Lab Engine: MCMCProbabilityBot (Metropolis-Hastings Markov Chain Monte Carlo)
-- Formerly known as MycroftBot in the Python codebase (~45.79 average shots).
-- Approximates the #P-complete Battleship posterior distribution by running a 100,000-step
-- Metropolis-Hastings random walk over valid 5-ship full-board state configurations.

local bit = require("bit")
local BaseBot = require("lua.bots.base_bot")
local GameEngine = require("lua.game_engine")
local Deduction = require("lua.deduction")
local Combinatorics = require("lua.combinatorics")
local Bitboard = require("lua.bitboard")

local MCMCProbabilityBot = BaseBot:extend("MCMCProbabilityBot")

--- Constructor
-- @param parity_offset number (0 or 1, default 0)
-- @param num_samples number: Number of Metropolis-Hastings steps per turn (default 100000)
-- @param safe_mode boolean: Enforce checkerboard parity during Hunt mode (default true)
function MCMCProbabilityBot:init(parity_offset, num_samples, safe_mode)
    self.parity_offset = parity_offset or 0
    self.num_samples = num_samples or 100000
    if safe_mode == nil then
        self.safe_mode = true
    else
        self.safe_mode = safe_mode
    end
    self.deduction = Deduction.new()
    self.combinatorics = Combinatorics.new()
end

--- Fast deterministic LCG PRNG for reproducible Markov Chain proposals
local function make_rng(seed)
    local state = (seed or 1) % 2147483647
    if state <= 0 then state = state + 2147483646 end
    return function(n)
        state = (state * 48271) % 2147483647
        if n then
            return (state % n) + 1
        end
        return state / 2147483647
    end
end

--- Core Decision Engine: Exact-Cover Seed DFS + 100,000-Step Metropolis-Hastings MCMC
-- @param board table: 10x10 tracking matrix
-- @param move_history table: Array of move records
-- @return target table: { x = r, y = c }
-- @return log string: Strategy rationale
-- @return dead_sizes table: Confirmed dead sizes (for UI telemetry)
-- @return diagnostics table: Telemetry table for Love2D UI
function MCMCProbabilityBot:get_best_move(board, move_history)
    -- 1. Visual Diagnostics for UI Telemetry
    local solutions = self.deduction:analyze(board, move_history)
    local primary_sol = solutions[1]
    local ui_sizes = primary_sol.dead_sizes

    local diagnostics = {
        active = primary_sol.active_hits,
        claimed = primary_sol.claimed_pegs,
        dead_ships = primary_sol.dead_ships,
        dead_sizes = ui_sizes,
        solutions_count = #solutions
    }

    -- 2. Extract Board Bitboards & Chronological Shot Timestamps
    local peg_times = {}
    for i = 0, 99 do peg_times[i] = -1 end

    for _, m in ipairs(move_history) do
        if m.result ~= "MISS" then
            local idx = m.x * 10 + m.y
            peg_times[idx] = m.shotNumber
        end
    end

    local miss_bb = Bitboard.empty()
    local all_hits_bb = Bitboard.empty()
    local sunk_bit_map = {}
    local hit_indices = {}
    local num_sunks = 0

    for r = 0, 9 do
        for c = 0, 9 do
            local idx = r * 10 + c
            local cell = board[r][c]
            if cell == GameEngine.CELL_MISS then
                miss_bb:set(r, c)
            elseif cell == GameEngine.CELL_HIT then
                all_hits_bb:set(r, c)
                table.insert(hit_indices, idx)
            elseif cell == GameEngine.CELL_SUNK then
                all_hits_bb:set(r, c)
                table.insert(hit_indices, idx)
                sunk_bit_map[idx] = bit.lshift(1, num_sunks)
                num_sunks = num_sunks + 1
            end
        end
    end

    local all_hits_lo = all_hits_bb.lo
    local all_hits_hi = all_hits_bb.hi
    local target_sunk_bits = bit.lshift(1, num_sunks) - 1

    -- 3. Precompute Turn-Specific Valid Placement Objects
    -- Any placement whose cells are 100% hit is immediately checked against chronological Fog-of-War rules:
    -- Its latest-hit cell MUST be a CELL_SUNK cell, otherwise it is pruned before MCMC begins.
    local placements_by_size = {}
    local open_water_by_size = {}
    local covering_by_size = {}
    local all_valid_entries = {}

    local unique_sizes = { 5, 4, 3, 2 }
    for _, size in ipairs(unique_sizes) do
        local valid_list = {}
        local open_list = {}
        local cover_map = {}
        for _, h_idx in ipairs(hit_indices) do
            cover_map[h_idx] = {}
        end

        local raw_list = self.combinatorics:get_placements(size)
        for _, p in ipairs(raw_list) do
            if not p.mask:intersects(miss_bb) then
                local is_fully_hit = all_hits_bb:contains_all(p.mask)
                local sunk_bit = 0
                local keep = true

                if is_fully_hit then
                    local max_time = -1
                    local lethal_idx = -1
                    for i = 1, size do
                        local pt = p.coords[i]
                        local idx = pt.r * 10 + pt.c
                        local t = peg_times[idx]
                        if t > max_time then
                            max_time = t
                            lethal_idx = idx
                        end
                    end
                    local sbit = sunk_bit_map[lethal_idx]
                    if not sbit then
                        keep = false -- Contradicts Fog-of-War: ship is fully hit without a lethal SUNK marker
                    else
                        sunk_bit = sbit
                    end
                end

                if keep then
                    local hit_lo = bit.band(p.mask.lo, all_hits_lo)
                    local hit_hi = bit.band(p.mask.hi, all_hits_hi)
                    local touches_hits = (hit_lo ~= 0ULL) or (hit_hi ~= 0ULL)

                    local entry = {
                        lo = p.mask.lo,
                        hi = p.mask.hi,
                        hit_lo = hit_lo,
                        hit_hi = hit_hi,
                        touches_hits = touches_hits,
                        is_fully_hit = is_fully_hit,
                        sunk_bit = sunk_bit,
                        coords = p.coords,
                        visit_count = 0
                    }

                    table.insert(valid_list, entry)
                    table.insert(all_valid_entries, entry)

                    if not touches_hits then
                        table.insert(open_list, entry)
                    else
                        for i = 1, size do
                            local pt = p.coords[i]
                            local idx = pt.r * 10 + pt.c
                            if cover_map[idx] then
                                table.insert(cover_map[idx], entry)
                            end
                        end
                    end
                end
            end
        end

        placements_by_size[size] = valid_list
        open_water_by_size[size] = open_list
        covering_by_size[size] = cover_map
    end

    local placements_for_ships = {}
    for i = 1, 5 do
        placements_for_ships[i] = placements_by_size[self.fleet_sizes[i]]
    end

    -- 4. Phase 1: Exact-Cover DFS to Find Initial Markov Chain Seed State
    local seed_state = { nil, nil, nil, nil, nil }

    local function place_remaining(placed_mask, occ_lo, occ_hi)
        local unplaced = {}
        for i = 1, 5 do
            if bit.band(placed_mask, bit.lshift(1, i - 1)) == 0 then
                table.insert(unplaced, i)
            end
        end

        local num_unplaced = #unplaced
        local function recurse_open(ptr, cur_lo, cur_hi)
            if ptr > num_unplaced then
                return true
            end
            local ship_idx = unplaced[ptr]
            local olist = open_water_by_size[self.fleet_sizes[ship_idx]]
            for i = 1, #olist do
                local entry = olist[i]
                if bit.band(cur_lo, entry.lo) == 0ULL and bit.band(cur_hi, entry.hi) == 0ULL then
                    seed_state[ship_idx] = entry
                    if recurse_open(ptr + 1, bit.bor(cur_lo, entry.lo), bit.bor(cur_hi, entry.hi)) then
                        return true
                    end
                    seed_state[ship_idx] = nil
                end
            end
            return false
        end

        return recurse_open(1, occ_lo, occ_hi)
    end

    local function solve_hits(placed_mask, occ_lo, occ_hi, cur_sunk_bits)
        -- Find first uncovered hit cell
        local target_hit_idx = nil
        for i = 1, #hit_indices do
            local h_idx = hit_indices[i]
            local is_covered
            if h_idx < 64 then
                is_covered = (bit.band(occ_lo, bit.lshift(1ULL, h_idx)) ~= 0ULL)
            else
                is_covered = (bit.band(occ_hi, bit.lshift(1ULL, h_idx - 64)) ~= 0ULL)
            end
            if not is_covered then
                target_hit_idx = h_idx
                break
            end
        end

        if not target_hit_idx then
            if cur_sunk_bits == target_sunk_bits then
                return place_remaining(placed_mask, occ_lo, occ_hi)
            end
            return false
        end

        for ship_idx = 1, 5 do
            local ship_bit = bit.lshift(1, ship_idx - 1)
            if bit.band(placed_mask, ship_bit) == 0 then
                -- Symmetry breaking: if ship 3 (size 3) is unplaced, skip ship 4 (also size 3)
                local skip = (ship_idx == 4 and bit.band(placed_mask, bit.lshift(1, 2)) == 0)
                if not skip then
                    local size = self.fleet_sizes[ship_idx]
                    local candidates = covering_by_size[size][target_hit_idx]
                    if candidates then
                        for i = 1, #candidates do
                            local entry = candidates[i]
                            if bit.band(occ_lo, entry.lo) == 0ULL and bit.band(occ_hi, entry.hi) == 0ULL then
                                seed_state[ship_idx] = entry
                                if solve_hits(
                                    bit.bor(placed_mask, ship_bit),
                                    bit.bor(occ_lo, entry.lo),
                                    bit.bor(occ_hi, entry.hi),
                                    bit.bor(cur_sunk_bits, entry.sunk_bit)
                                ) then
                                    return true
                                end
                                seed_state[ship_idx] = nil
                            end
                        end
                    end
                end
            end
        end

        return false
    end

    local found_seed = solve_hits(0, 0ULL, 0ULL, 0)

    -- 5. Phase 2: 100,000-Step Metropolis-Hastings Markov Chain Walk
    local valid_universes = 0
    local rng_state = bit.bor((self.parity_offset + 1) * 100003 + #move_history * 1009, 1)

    local function next_rand(n)
        rng_state = bit.bxor(rng_state, bit.lshift(rng_state, 13))
        rng_state = bit.bxor(rng_state, bit.rshift(rng_state, 17))
        rng_state = bit.bxor(rng_state, bit.lshift(rng_state, 5))
        -- Mask to 30-bit positive integer
        local pos = bit.band(rng_state, 0x3fffffff)
        return (pos % n) + 1
    end

    if found_seed then
        local s1, s2, s3, s4, s5 = seed_state[1], seed_state[2], seed_state[3], seed_state[4], seed_state[5]
        local p1_list = placements_for_ships[1]
        local p2_list = placements_for_ships[2]
        local p3_list = placements_for_ships[3]
        local p4_list = placements_for_ships[4]
        local p5_list = placements_for_ships[5]
        local n1, n2, n3, n4, n5 = #p1_list, #p2_list, #p3_list, #p4_list, #p5_list

        if n1 > 0 and n2 > 0 and n3 > 0 and n4 > 0 and n5 > 0 then
            local state = { s1, s2, s3, s4, s5 }
            local lists = { p1_list, p2_list, p3_list, p4_list, p5_list }
            local counts = { n1, n2, n3, n4, n5 }
            local target_sims = self.num_samples

            -- Since all 5 ships in state are disjoint, their union can be updated in O(1) via XOR!
            local cur_occ_lo = bit.bor(s1.lo, s2.lo, s3.lo, s4.lo, s5.lo)
            local cur_occ_hi = bit.bor(s1.hi, s2.hi, s3.hi, s4.hi, s5.hi)
            local cur_sunk = bit.bor(s1.sunk_bit, s2.sunk_bit, s3.sunk_bit, s4.sunk_bit, s5.sunk_bit)

            for _ = 1, target_sims do
                -- Advance inline xorshift32
                rng_state = bit.bxor(rng_state, bit.lshift(rng_state, 13))
                rng_state = bit.bxor(rng_state, bit.rshift(rng_state, 17))
                rng_state = bit.bxor(rng_state, bit.lshift(rng_state, 5))
                local rbits = bit.band(rng_state, 0x3fffffff)

                -- 80% single-ship mutation, 20% two-ship jump to escape local minima
                if (rbits % 5) ~= 0 then
                    local idx1 = (bit.rshift(rbits, 3) % 5) + 1
                    local old_p = state[idx1]

                    rng_state = bit.bxor(rng_state, bit.lshift(rng_state, 13))
                    rng_state = bit.bxor(rng_state, bit.rshift(rng_state, 17))
                    rng_state = bit.bxor(rng_state, bit.lshift(rng_state, 5))
                    local pick = (bit.band(rng_state, 0x3fffffff) % counts[idx1]) + 1
                    local new_p = lists[idx1][pick]

                    local other_lo = bit.bxor(cur_occ_lo, old_p.lo)
                    local other_hi = bit.bxor(cur_occ_hi, old_p.hi)

                    if bit.band(other_lo, new_p.lo) == 0ULL and bit.band(other_hi, new_p.hi) == 0ULL then
                        local tot_lo = bit.bor(other_lo, new_p.lo)
                        local tot_hi = bit.bor(other_hi, new_p.hi)
                        local other_sunk = bit.bxor(cur_sunk, old_p.sunk_bit)
                        local tot_sunk = bit.bor(other_sunk, new_p.sunk_bit)

                        local accept = true
                        if old_p.touches_hits or new_p.touches_hits then
                            if bit.band(tot_lo, all_hits_lo) ~= all_hits_lo or
                               bit.band(tot_hi, all_hits_hi) ~= all_hits_hi or
                               tot_sunk ~= target_sunk_bits then
                                accept = false
                            end
                        end
                        if accept then
                            state[idx1] = new_p
                            cur_occ_lo = tot_lo
                            cur_occ_hi = tot_hi
                            cur_sunk = tot_sunk
                        end
                    end
                else
                    -- Two-ship simultaneous jump mutation
                    local idx1 = (bit.rshift(rbits, 3) % 5) + 1
                    local idx2 = (bit.rshift(rbits, 6) % 4) + 1
                    if idx2 >= idx1 then idx2 = idx2 + 1 end

                    local old_p1 = state[idx1]
                    local old_p2 = state[idx2]

                    rng_state = bit.bxor(rng_state, bit.lshift(rng_state, 13))
                    rng_state = bit.bxor(rng_state, bit.rshift(rng_state, 17))
                    rng_state = bit.bxor(rng_state, bit.lshift(rng_state, 5))
                    local r2 = bit.band(rng_state, 0x3fffffff)
                    local new_p1 = lists[idx1][(r2 % counts[idx1]) + 1]
                    local new_p2 = lists[idx2][(bit.rshift(r2, 10) % counts[idx2]) + 1]

                    if bit.band(new_p1.lo, new_p2.lo) == 0ULL and bit.band(new_p1.hi, new_p2.hi) == 0ULL then
                        local other_lo = bit.bxor(cur_occ_lo, bit.bxor(old_p1.lo, old_p2.lo))
                        local other_hi = bit.bxor(cur_occ_hi, bit.bxor(old_p1.hi, old_p2.hi))
                        local pair_lo = bit.bor(new_p1.lo, new_p2.lo)
                        local pair_hi = bit.bor(new_p1.hi, new_p2.hi)

                        if bit.band(other_lo, pair_lo) == 0ULL and bit.band(other_hi, pair_hi) == 0ULL then
                            local tot_lo = bit.bor(other_lo, pair_lo)
                            local tot_hi = bit.bor(other_hi, pair_hi)
                            local other_sunk = bit.bxor(cur_sunk, bit.bxor(old_p1.sunk_bit, old_p2.sunk_bit))
                            local tot_sunk = bit.bor(other_sunk, bit.bor(new_p1.sunk_bit, new_p2.sunk_bit))

                            local accept = true
                            if old_p1.touches_hits or old_p2.touches_hits or new_p1.touches_hits or new_p2.touches_hits then
                                if bit.band(tot_lo, all_hits_lo) ~= all_hits_lo or
                                   bit.band(tot_hi, all_hits_hi) ~= all_hits_hi or
                                   tot_sunk ~= target_sunk_bits then
                                    accept = false
                                end
                            end
                            if accept then
                                state[idx1] = new_p1
                                state[idx2] = new_p2
                                cur_occ_lo = tot_lo
                                cur_occ_hi = tot_hi
                                cur_sunk = tot_sunk
                            end
                        end
                    end
                end

                -- O(1) stationary distribution accumulation per placement
                state[1].visit_count = state[1].visit_count + 1
                state[2].visit_count = state[2].visit_count + 1
                state[3].visit_count = state[3].visit_count + 1
                state[4].visit_count = state[4].visit_count + 1
                state[5].visit_count = state[5].visit_count + 1
            end

            valid_universes = target_sims
            seed_state = state
        end
    end

    -- 6. Project Placement Visit Counts onto 10x10 Grid
    local raw_heatmap = self:create_blank_matrix(0)
    for i = 1, #all_valid_entries do
        local entry = all_valid_entries[i]
        local cnt = entry.visit_count
        if cnt > 0 then
            local coords = entry.coords
            for j = 1, #coords do
                local pt = coords[j]
                raw_heatmap[pt.r][pt.c] = raw_heatmap[pt.r][pt.c] + cnt
            end
        end
    end

    -- Detect active wounded hits from current state
    local has_active_hits = false
    if found_seed then
        local fully_hit_lo, fully_hit_hi = 0ULL, 0ULL
        for i = 1, 5 do
            local si = seed_state[i]
            if si.is_fully_hit then
                fully_hit_lo = bit.bor(fully_hit_lo, si.lo)
                fully_hit_hi = bit.bor(fully_hit_hi, si.hi)
            end
        end
        if fully_hit_lo ~= all_hits_lo or fully_hit_hi ~= all_hits_hi then
            has_active_hits = true
        end
    end
    diagnostics.has_active = has_active_hits
    diagnostics.valid_universes = valid_universes

    -- 7. Heatmap Normalization & Target Selection
    local max_heat = -1
    local best_moves = {}
    local filtered_heatmap = self:create_blank_matrix(0)

    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_UNKNOWN then
                local heat = raw_heatmap[r][c]
                if not has_active_hits and self.safe_mode and ((r + c) % 2 ~= (self.parity_offset % 2)) then
                    heat = 0
                end
                filtered_heatmap[r][c] = heat

                if heat > max_heat then
                    max_heat = heat
                    best_moves = { { x = r, y = c } }
                elseif heat == max_heat and heat > 0 then
                    table.insert(best_moves, { x = r, y = c })
                end
            end
        end
    end

    self.last_heatmap = self:normalize_heatmap(filtered_heatmap)

    local target, log
    if max_heat > 0 and #best_moves > 0 then
        local pick_idx = next_rand(#best_moves)
        target = best_moves[pick_idx]
        log = string.format("Pure MCMC (%d Steps)", valid_universes)
    else
        local open_water = self:get_unrevealed_cells(board)
        if #open_water > 0 then
            target = open_water[next_rand(#open_water)]
        else
            target = { x = 0, y = 0 }
        end
        log = "Random Fallback (MCMC Exhaustion)"
    end

    return target, log, ui_sizes, diagnostics
end

--- Return normalized density heatmap for UI
-- @param board table: 10x10 tracking matrix
-- @param dead_sizes table (optional)
-- @return table: 10x10 normalized integer heatmap (0..100)
function MCMCProbabilityBot:get_heatmap(board, dead_sizes)
    if self.last_heatmap then
        return self.last_heatmap
    end
    self:get_best_move(board, {})
    return self.last_heatmap
end

return MCMCProbabilityBot

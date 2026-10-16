-- lua/bots/adler.lua
-- Tier 2: AdlerBot (Irene Adler — "The Woman" / Empirical Bayesian Human Profiler)
-- Canonical Strategy: Multi-Hypothesis CSP + Empirical Bayesian MAP Estimation (Likelihood * Prior^gamma)
--                     + Inverse-Heat Defensive Fleet Generation (Epsilon-Greedy & Softmax Roulette).
-- When given a human opponent's historical dossier, Adler constructs Laplace-smoothed, exponentially
-- decaying (0.85^t) offensive and defensive prior matrices to exploit human placement and hunting biases.
-- When no human prior is provided (uniform 1.0 fallback), gracefully matches Watson/Sherlock (~45.5 shots).

local BaseBot = require("lua.bots.base_bot")
local GameEngine = require("lua.game_engine")
local Deduction = require("lua.deduction")
local Combinatorics = require("lua.combinatorics")
local Bitboard = require("lua.bitboard")

local AdlerBot = BaseBot:extend("AdlerBot")

--- Constructor
-- @param parity_offset number (0 or 1, default 0)
-- @param offensive_prior table (optional): 0-indexed 10x10 matrix of human ship placement multipliers
-- @param defensive_prior table (optional): 0-indexed 10x10 matrix of human hunt shot multipliers
-- @param prior_exponent number (optional): Exponent gamma for Prior^gamma MAP weighting (default 3.0)
function AdlerBot:init(parity_offset, offensive_prior, defensive_prior, prior_exponent)
    self.parity_offset = parity_offset or 0
    self.prior_exponent = prior_exponent or 3.0
    self.deduction = Deduction.new()
    self.combinatorics = Combinatorics.new()
    self.rng_counter = 0

    self:set_offensive_prior(offensive_prior)
    self:set_defensive_prior(defensive_prior)
end

--- Set or reset the 10x10 offensive prior matrix (where the opponent places ships)
-- @param prior table (optional): 0-indexed 10x10 matrix; defaults to uniform 1.0
function AdlerBot:set_offensive_prior(prior)
    self.offensive_prior = self:create_blank_matrix(1.0)
    self.is_uniform_fallback = true
    self.max_bias = 1.0
    self.min_bias = 1.0

    if prior then
        local max_b = -1e9
        local min_b = 1e9
        for r = 0, 9 do
            if prior[r] then
                for c = 0, 9 do
                    local val = prior[r][c] or 1.0
                    self.offensive_prior[r][c] = val
                    if val > max_b then max_b = val end
                    if val < min_b then min_b = val end
                end
            end
        end
        if max_b > -1e8 then
            self.max_bias = max_b
            self.min_bias = min_b
            if math.abs(max_b - 1.0) > 1e-4 or math.abs(min_b - 1.0) > 1e-4 then
                self.is_uniform_fallback = false
            end
        end
    end
end

--- Set or reset the 10x10 defensive prior matrix (where the opponent fires hunt shots)
-- @param prior table (optional): 0-indexed 10x10 matrix; defaults to uniform 1.0
function AdlerBot:set_defensive_prior(prior)
    self.defensive_prior = self:create_blank_matrix(1.0)
    if prior then
        for r = 0, 9 do
            if prior[r] then
                for c = 0, 9 do
                    self.defensive_prior[r][c] = prior[r][c] or 1.0
                end
            end
        end
    end
end

--- Build an Empirical Bayesian Prior Matrix from historical coordinate observations
-- Uses Laplace smoothing (base 1.0) and exponential recency decay (default 0.85^age),
-- normalized by the board-wide mean so that 1.0 represents average uniform expectation.
-- @param history table: Chronological array of coordinate lists (oldest first, newest last),
--                       where each entry is an array of { x = r, y = c } or { r = r, c = c }.
-- @param decay_factor number (optional): Exponential decay per game step backward (default 0.85)
-- @return table: 0-indexed 10x10 normalized multiplier matrix
function AdlerBot.build_empirical_prior(history, decay_factor)
    decay_factor = decay_factor or 0.85
    local prior = {}
    for r = 0, 9 do
        prior[r] = {}
        for c = 0, 9 do
            prior[r][c] = 1.0 -- Base 1.0 Laplace smoothing
        end
    end

    if not history or #history == 0 then
        return prior
    end

    local current_weight = 1.0
    local total_weighted_samples = 0.0

    -- Iterate backward from most recent match to oldest match
    for i = #history, 1, -1 do
        local coords = history[i]
        if coords then
            for _, pt in ipairs(coords) do
                local r = pt.x or pt.r
                local c = pt.y or pt.c
                if r and c and r >= 0 and r <= 9 and c >= 0 and c <= 9 then
                    prior[r][c] = prior[r][c] + current_weight
                end
            end
            total_weighted_samples = total_weighted_samples + current_weight
            current_weight = current_weight * decay_factor
        end
    end

    if total_weighted_samples > 0 then
        local sum = 0.0
        for r = 0, 9 do
            for c = 0, 9 do
                sum = sum + prior[r][c]
            end
        end
        local avg = sum / 100.0
        if avg > 0 then
            for r = 0, 9 do
                for c = 0, 9 do
                    prior[r][c] = prior[r][c] / avg
                end
            end
        end
    end

    return prior
end

--- Merge a general human prior with a bot-specific prior (default 30% general, 70% specific)
-- @param general table: 0-indexed 10x10 matrix
-- @param specific table: 0-indexed 10x10 matrix
-- @param general_weight number (optional, default 0.30)
-- @param specific_weight number (optional, default 0.70)
-- @return table: 0-indexed 10x10 merged multiplier matrix
function AdlerBot.merge_priors(general, specific, general_weight, specific_weight)
    general_weight = general_weight or 0.30
    specific_weight = specific_weight or 0.70

    local has_specific = false
    if specific then
        for r = 0, 9 do
            if specific[r] then
                for c = 0, 9 do
                    local v = specific[r][c]
                    if v and math.abs(v - 1.0) > 1e-4 and math.abs(v) > 1e-4 then
                        has_specific = true
                        break
                    end
                end
            end
            if has_specific then break end
        end
    end

    local merged = {}
    for r = 0, 9 do
        merged[r] = {}
        for c = 0, 9 do
            local g_val = (general and general[r] and general[r][c]) or 1.0
            if has_specific then
                local s_val = (specific and specific[r] and specific[r][c]) or 1.0
                merged[r][c] = (g_val * general_weight) + (s_val * specific_weight)
            else
                merged[r][c] = g_val
            end
        end
    end
    return merged
end

--- Helper: Compute available surviving ship sizes
local function get_available_fleet(fleet_sizes, dead_sizes)
    local dead_map = {}
    if dead_sizes then
        for _, sz in ipairs(dead_sizes) do
            dead_map[sz] = (dead_map[sz] or 0) + 1
        end
    end

    local available = {}
    for _, sz in ipairs(fleet_sizes) do
        if (dead_map[sz] or 0) > 0 then
            dead_map[sz] = dead_map[sz] - 1
        else
            table.insert(available, sz)
        end
    end
    return available
end

--- Core Decision Engine: Multi-Hypothesis CSP + Empirical Bayesian MAP (Likelihood * Prior^gamma)
-- @param board table: 10x10 tracking matrix
-- @param move_history table: Array of move records
-- @return target table: { x = r, y = c }
-- @return log string: Strategy rationale
-- @return dead_sizes table: Confirmed dead sizes
-- @return diagnostics table: Telemetry table for Love2D UI
function AdlerBot:get_best_move(board, move_history)
    -- 1. Multi-Hypothesis Tied CSP Deduction
    local solutions = self.deduction:analyze(board, move_history)
    local primary_sol = solutions[1]
    local dead_sizes = primary_sol.dead_sizes
    local has_active = Deduction.has_active_hits(solutions)
    local is_2_dead = Deduction.is_size_dead(solutions, 2)

    local misses_bb = Bitboard.empty()
    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_MISS then
                misses_bb:set(r, c)
            end
        end
    end

    local diagnostics = {
        active = primary_sol.active_hits,
        claimed = primary_sol.claimed_pegs,
        dead_ships = primary_sol.dead_ships,
        dead_sizes = dead_sizes,
        solutions_count = #solutions,
        is_two_ship_dead = is_2_dead,
        has_active = has_active,
        is_uniform_fallback = self.is_uniform_fallback,
        max_bias = self.max_bias,
        min_bias = self.min_bias
    }

    local gamma = self.prior_exponent

    -- 2. TARGET MODE: Prior-Weighted Bayesian Joint Target Density (P1, P2)
    if has_active then
        local total_target_scores = {}
        local max_target_score = -1.0

        for _, sol in ipairs(solutions) do
            if #sol.active_hits > 0 then
                local forbidden_bb = misses_bb:copy()
                for _, pt in ipairs(sol.claimed_pegs) do
                    forbidden_bb:set(pt.r, pt.c)
                end

                local active_hits_bb = Bitboard.from_coords(sol.active_hits)
                local available_fleet = get_available_fleet(self.fleet_sizes, sol.dead_sizes)

                local counts, _ = self.combinatorics:compute_target_density(
                    board, available_fleet, forbidden_bb, active_hits_bb
                )

                for idx, cnt in pairs(counts) do
                    local r = math.floor(idx / 10)
                    local c = idx % 10
                    local bias = self.offensive_prior[r][c] ^ gamma
                    local weighted = (total_target_scores[idx] or 0.0) + (cnt * bias)
                    total_target_scores[idx] = weighted
                    if weighted > max_target_score then
                        max_target_score = weighted
                    end
                end
            end
        end

        local target_heatmap = self:create_blank_matrix(0)
        if max_target_score > 0 then
            for idx, score in pairs(total_target_scores) do
                local r = math.floor(idx / 10)
                local c = idx % 10
                target_heatmap[r][c] = math.floor((score / max_target_score) * 100)
            end
        end
        self.last_heatmap = target_heatmap

        local best_targets = {}
        for idx, score in pairs(total_target_scores) do
            if math.abs(score - max_target_score) <= 1e-6 then
                local r = math.floor(idx / 10)
                local c = idx % 10
                if board[r][c] == GameEngine.CELL_UNKNOWN then
                    table.insert(best_targets, { x = r, y = c })
                end
            end
        end

        if #best_targets > 0 then
            local pick_idx = ((self.parity_offset * 10007 + #move_history) % #best_targets) + 1
            local target = best_targets[pick_idx]
            local log = string.format("Adler: MAP Target (Peak=%.2f, Bias=[%.2fx..%.2fx])",
                max_target_score, self.min_bias, self.max_bias)
            return target, log, dead_sizes, diagnostics
        end
    end

    -- 3. HUNT MODE: Empirical Bayesian MAP Hunt (Combinatorial Likelihood * Prior^gamma)
    local combined_counts = self:create_blank_matrix(0)
    for _, sol in ipairs(solutions) do
        local forbidden_bb = misses_bb:copy()
        for _, pt in ipairs(sol.claimed_pegs) do
            forbidden_bb:set(pt.r, pt.c)
        end

        local available_fleet = get_available_fleet(self.fleet_sizes, sol.dead_sizes)
        local density, _ = self.combinatorics:compute_hunt_density(available_fleet, forbidden_bb)

        for r = 0, 9 do
            for c = 0, 9 do
                if board[r][c] == GameEngine.CELL_UNKNOWN then
                    combined_counts[r][c] = combined_counts[r][c] + density[r][c]
                end
            end
        end
    end

    local map_scores = self:create_blank_matrix(0.0)
    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_UNKNOWN then
                local base_likelihood = combined_counts[r][c]
                local human_bias = self.offensive_prior[r][c] ^ gamma
                map_scores[r][c] = base_likelihood * human_bias
            end
        end
    end

    self.last_heatmap = self:normalize_heatmap(map_scores)

    local stride = is_2_dead and 3 or 2
    local best_score = -1.0
    local best_candidates = {}

    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_UNKNOWN then
                if self:is_cell_on_stride(r, c, stride, self.parity_offset) then
                    local score = map_scores[r][c]
                    if score > best_score + 1e-6 then
                        best_score = score
                        best_candidates = { { x = r, y = c } }
                    elseif math.abs(score - best_score) <= 1e-6 and score > 0 then
                        table.insert(best_candidates, { x = r, y = c })
                    end
                end
            end
        end
    end

    -- Fallback to any unrevealed cell if stride covering net is exhausted
    if best_score <= 0 or #best_candidates == 0 then
        for r = 0, 9 do
            for c = 0, 9 do
                if board[r][c] == GameEngine.CELL_UNKNOWN then
                    local score = map_scores[r][c]
                    if score > best_score + 1e-6 then
                        best_score = score
                        best_candidates = { { x = r, y = c } }
                    elseif math.abs(score - best_score) <= 1e-6 then
                        table.insert(best_candidates, { x = r, y = c })
                    end
                end
            end
        end
    end

    local target
    if #best_candidates > 0 then
        local pick_idx = ((self.parity_offset * 10007 + #move_history) % #best_candidates) + 1
        target = best_candidates[pick_idx]
    else
        local unrevealed = self:get_unrevealed_cells(board)
        target = unrevealed[1] or { x = 0, y = 0 }
    end

    local mode_tag = self.is_uniform_fallback and "Uniform" or string.format("Prior^%.1f", gamma)
    local log = string.format("Adler: Bayesian MAP Hunt (%s, Stride-%d, Peak=%.1f)",
        mode_tag, stride, best_score)
    return target, log, dead_sizes, diagnostics
end

--- Generate an Anti-Human Phantom Fleet avoiding the opponent's hot hunting zones
-- Supports two defensive strategies from TacticalEngine.kt:
--   1. "softmax_roulette" (default): Inverse-heat Boltzmann roulette wheel (weight = exp(-temp * heat))
--   2. "epsilon_greedy": 70% minimum-heat cold-zone placement, 30% uniform random placement
-- @param defensive_prior table (optional): 0-indexed 10x10 matrix (defaults to self.defensive_prior)
-- @param mode string (optional): "softmax_roulette" (default) or "epsilon_greedy"
-- @param temperature number (optional): Softmax inverse-heat temperature (default 1.0)
-- @param seed number (optional): Deterministic seed for reproducible fleet generation
-- @return table: Array of 5 ship records { id=i, size=s, r=r, c=c, is_horizontal=b, coords={...}, heat=h }
function AdlerBot:generate_defensive_fleet(defensive_prior, mode, temperature, seed)
    local prior = defensive_prior or self.defensive_prior
    mode = mode or "softmax_roulette"
    temperature = temperature or 1.0

    self.rng_counter = self.rng_counter + 1
    local rng_state = (seed or (self.parity_offset * 100003 + self.rng_counter * 7919)) % 2147483647
    if rng_state <= 0 then rng_state = rng_state + 2147483646 end

    local function rand_float()
        rng_state = (rng_state * 48271) % 2147483647
        return rng_state / 2147483647.0
    end

    local function rand_int(n)
        rng_state = (rng_state * 48271) % 2147483647
        return (rng_state % n) + 1
    end

    local occupied_bb = Bitboard.empty()
    local fleet = {}

    for ship_id, size in ipairs(self.fleet_sizes) do
        local plist = self.combinatorics:get_placements(size)
        local valid_candidates = {}

        for _, p in ipairs(plist) do
            if not p.mask:intersects(occupied_bb) then
                local heat = 0.0
                for _, pt in ipairs(p.coords) do
                    heat = heat + ((prior and prior[pt.r] and prior[pt.r][pt.c]) or 1.0)
                end
                table.insert(valid_candidates, {
                    placement = p,
                    heat = heat
                })
            end
        end

        if #valid_candidates > 0 then
            local chosen_cand
            if mode == "epsilon_greedy" then
                if rand_float() < 0.30 then
                    chosen_cand = valid_candidates[rand_int(#valid_candidates)]
                else
                    local min_heat = 1e9
                    local coldest = {}
                    for _, cand in ipairs(valid_candidates) do
                        if cand.heat < min_heat - 1e-6 then
                            min_heat = cand.heat
                            coldest = { cand }
                        elseif math.abs(cand.heat - min_heat) <= 1e-6 then
                            table.insert(coldest, cand)
                        end
                    end
                    chosen_cand = coldest[rand_int(#coldest)]
                end
            else
                -- Softmax Roulette: weight = exp(-temperature * (heat - min_heat))
                local min_heat = 1e9
                for _, cand in ipairs(valid_candidates) do
                    if cand.heat < min_heat then min_heat = cand.heat end
                end

                local weights = {}
                local total_weight = 0.0
                for i, cand in ipairs(valid_candidates) do
                    local w = math.exp(-temperature * (cand.heat - min_heat))
                    weights[i] = w
                    total_weight = total_weight + w
                end

                local pick = rand_float() * total_weight
                chosen_cand = valid_candidates[#valid_candidates]
                for i, w in ipairs(weights) do
                    pick = pick - w
                    if pick <= 0 then
                        chosen_cand = valid_candidates[i]
                        break
                    end
                end
            end

            local p = chosen_cand.placement
            occupied_bb = occupied_bb:bor(p.mask)
            local first_pt = p.coords[1]
            local second_pt = p.coords[2]
            local is_horiz = (second_pt.c > first_pt.c)

            table.insert(fleet, {
                id = ship_id,
                size = size,
                r = first_pt.r,
                c = first_pt.c,
                is_horizontal = is_horiz,
                coords = p.coords,
                heat = chosen_cand.heat
            })
        end
    end

    return fleet
end

--- Convenience helper: Place Adler's anti-human defensive fleet directly onto a GameEngine instance
-- @param game table: GameEngine instance
-- @param defensive_prior table (optional)
-- @param mode string (optional): "softmax_roulette" or "epsilon_greedy"
-- @param temperature number (optional)
-- @param seed number (optional)
-- @return table: The generated fleet array
function AdlerBot:place_defensive_fleet(game, defensive_prior, mode, temperature, seed)
    local fleet = self:generate_defensive_fleet(defensive_prior, mode, temperature, seed)
    for _, ship in ipairs(fleet) do
        local ok = game:place_ship_manual(ship.r, ship.c, ship.size, ship.is_horizontal, ship.id)
        if not ok then
            error(string.format("AdlerBot: Failed to place defensive ship %d at (%d,%d)", ship.id, ship.r, ship.c))
        end
    end
    return fleet
end

--- Return normalized density heatmap for UI
-- @param board table: 10x10 tracking matrix
-- @param dead_sizes table (optional)
-- @return table: 10x10 normalized integer heatmap (0..100)
function AdlerBot:get_heatmap(board, dead_sizes)
    if self.last_heatmap then
        return self.last_heatmap
    end
    self:get_best_move(board, {})
    return self.last_heatmap
end

return AdlerBot

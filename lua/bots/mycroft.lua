-- lua/bots/mycroft.lua
-- Rank 4: MycroftBot (The Champion — Shannon Entropy & Expected Utility Optimization)
-- Formerly known as PorlockBot (~45.13 average shots).
-- Combines exact Bayesian multi-hypothesis CSP deduction with Claude Shannon's binary information entropy
-- and radius-2 ocean openness star pattern to solve Battleship with optimal information gain.

local BaseBot = require("lua.bots.base_bot")
local GameEngine = require("lua.game_engine")
local Deduction = require("lua.deduction")
local Combinatorics = require("lua.combinatorics")
local Bitboard = require("lua.bitboard")

local MycroftBot = BaseBot:extend("MycroftBot")

local INV_LN2 = 1.0 / math.log(2.0)

local STAR_OFFSETS = {
    { -1,  0 }, {  1,  0 }, {  0, -1 }, {  0,  1 },
    { -2,  0 }, {  2,  0 }, {  0, -2 }, {  0,  2 }
}

--- Constructor
-- @param parity_offset number (0 or 1, default 0)
-- @param w_p number: Weight for raw hit probability (default 2.5)
-- @param w_entropy number: Weight for Shannon information entropy (default 0.35)
-- @param w_open number: Weight for radius-2 ocean openness (default 0.15)
function MycroftBot:init(parity_offset, w_p, w_entropy, w_open)
    self.parity_offset = parity_offset or 0
    self.w_p = w_p or 2.5
    self.w_entropy = w_entropy or 0.35
    self.w_open = w_open or 0.15
    self.deduction = Deduction.new()
    self.combinatorics = Combinatorics.new()
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

--- Evaluate Expected Information-Theoretic Utility for a cell
-- Utility(r, c) = w_p * p + w_entropy * H(p) + w_open * Openness(r, c)
-- where H(p) is Shannon binary entropy: -p*log2(p) - (1-p)*log2(1-p)
-- and Openness(r, c) is the unrevealed ratio in an 8-neighbor radius-2 star pattern.
-- @param board table: 10x10 tracking matrix
-- @param r number (0..9)
-- @param c number (0..9)
-- @param raw_p number: Normalized hit probability in [0, 1]
-- @return number: Composite utility score
function MycroftBot:evaluate_cell_utility(board, r, c, raw_p)
    local p = raw_p or 0
    if p < 1e-5 then
        p = 1e-5
    elseif p > 1.0 - 1e-5 then
        p = 1.0 - 1e-5
    end

    -- Shannon binary information entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)
    local entropy = -(p * (math.log(p) * INV_LN2) + (1.0 - p) * (math.log(1.0 - p) * INV_LN2))

    -- Spatial ocean openness: count unrevealed cells in radius-2 star pattern
    local open_count = 0
    for i = 1, 8 do
        local off = STAR_OFFSETS[i]
        local nr = r + off[1]
        local nc = c + off[2]
        if nr >= 0 and nr <= 9 and nc >= 0 and nc <= 9 and board[nr][nc] == GameEngine.CELL_UNKNOWN then
            open_count = open_count + 1
        end
    end
    local open_factor = open_count / 8.0

    return (self.w_p * (raw_p or 0)) + (self.w_entropy * entropy) + (self.w_open * open_factor)
end

--- Core Decision Engine: Multi-hypothesis tied CSP + pure joint covers + Shannon Expected Utility Hunt
-- @param board table: 10x10 tracking matrix
-- @param move_history table: Array of move records
-- @return target table: { x = r, y = c }
-- @return log string: Strategy rationale
-- @return dead_sizes table: Confirmed dead sizes
-- @return diagnostics table: Telemetry table for Love2D UI
function MycroftBot:get_best_move(board, move_history)
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
        has_active = has_active
    }

    -- 2. TARGET MODE: Pure Exact Multi-Hypothesis Bayesian Joint Targeting (P1, P2)
    if has_active then
        local total_target_counts = {}
        local max_target_count = 0

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
                    local cur = (total_target_counts[idx] or 0) + cnt
                    total_target_counts[idx] = cur
                    if cur > max_target_count then
                        max_target_count = cur
                    end
                end
            end
        end

        local target_heatmap = self:create_blank_matrix(0)
        if max_target_count > 0 then
            for idx, cnt in pairs(total_target_counts) do
                local r = math.floor(idx / 10)
                local c = idx % 10
                target_heatmap[r][c] = math.floor((cnt / max_target_count) * 100)
            end
        end
        self.last_heatmap = target_heatmap

        local best_targets = {}
        for idx, cnt in pairs(total_target_counts) do
            if cnt == max_target_count then
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
            local log = string.format("Mycroft: Pure Bayesian Joint Cover (P1, P2 peak=%d, ties=%d)",
                max_target_count, #best_targets)
            return target, log, dead_sizes, diagnostics
        end
    end

    -- 3. HUNT MODE: Expected Information-Theoretic Utility Optimization + Dynamic Stride Parity
    local combined_counts = self:create_blank_matrix(0)
    local max_hunt_val = 0

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
                    local d = combined_counts[r][c] + density[r][c]
                    combined_counts[r][c] = d
                    if d > max_hunt_val then
                        max_hunt_val = d
                    end
                end
            end
        end
    end

    local p_map = self:create_blank_matrix(0)
    local heatmap = self:create_blank_matrix(0)
    if max_hunt_val > 0 then
        for r = 0, 9 do
            for c = 0, 9 do
                if board[r][c] == GameEngine.CELL_UNKNOWN then
                    local p = combined_counts[r][c] / max_hunt_val
                    p_map[r][c] = p
                    heatmap[r][c] = math.floor(p * 100)
                end
            end
        end
    end
    self.last_heatmap = heatmap

    local stride = is_2_dead and 3 or 2
    local best_score = -1e9
    local best_candidates = {}

    -- Evaluate candidate cells on the active dynamic stride covering net
    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_UNKNOWN then
                if self:is_cell_on_stride(r, c, stride, self.parity_offset) then
                    local score = self:evaluate_cell_utility(board, r, c, p_map[r][c])
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

    -- Fallback to any unrevealed cell if stride net is exhausted
    if #best_candidates == 0 then
        for r = 0, 9 do
            for c = 0, 9 do
                if board[r][c] == GameEngine.CELL_UNKNOWN then
                    local score = self:evaluate_cell_utility(board, r, c, p_map[r][c])
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

    local log = string.format("Mycroft: Info-Utility Hunt (Stride-%d, Score=%.3f, ties=%d)",
        stride, best_score, #best_candidates)
    return target, log, dead_sizes, diagnostics
end

--- Return normalized density heatmap for UI
-- @param board table: 10x10 tracking matrix
-- @param dead_sizes table (optional)
-- @return table: 10x10 normalized integer heatmap (0..100)
function MycroftBot:get_heatmap(board, dead_sizes)
    if self.last_heatmap then
        return self.last_heatmap
    end
    self:get_best_move(board, {})
    return self.last_heatmap
end

return MycroftBot

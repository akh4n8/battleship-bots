-- lua/bots/watson.lua
-- Rank 2: WatsonBot (Dr. John H. Watson — The Consulting Partner)
-- Canonical Strategy: Multi-Hypothesis Chronological CSP Deduction + Dynamic Stride Parity + Exact Target Density
-- Formerly known as SherlockBot in the Python laboratory (~45.33 average shots).
-- Achieves an extraordinary 11.2-shot leap over Lestrade by eliminating blind-sink guessing and heuristic line-tracing.

local BaseBot = require("lua.bots.base_bot")
local GameEngine = require("lua.game_engine")
local Deduction = require("lua.deduction")
local Combinatorics = require("lua.combinatorics")
local Bitboard = require("lua.bitboard")

local WatsonBot = BaseBot:extend("WatsonBot")

--- Constructor
-- @param parity_offset number (optional: 0 or 1, default 0)
function WatsonBot:init(parity_offset)
    self.parity_offset = parity_offset or 0
    self.deduction = Deduction.new()
    self.combinatorics = Combinatorics.new()
end

--- Helper: Compute available surviving ship sizes given confirmed dead sizes
-- @param dead_sizes table: array of dead ship lengths
-- @return table: array of remaining ship lengths
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

--- Core Decision Engine: Multi-hypothesis CSP deduction + exact targeting + dynamic parity hunt
-- @param board table: 10x10 tracking matrix (0=UNKNOWN, -1=MISS, 1=HIT, 2=SUNK)
-- @param move_history table: Array of move records
-- @return target table: { x = r, y = c }
-- @return log string: Strategy rationale
-- @return dead_sizes table: Confirmed dead sizes from primary hypothesis
-- @return diagnostics table: Telemetry table for Love2D UI
function WatsonBot:get_best_move(board, move_history)
    -- 1. Multi-Hypothesis Chronological CSP Deduction
    local solutions = self.deduction:analyze(board, move_history)
    local primary_sol = solutions[1]
    local dead_sizes = primary_sol.dead_sizes
    local has_active = Deduction.has_active_hits(solutions)
    local is_2_dead = Deduction.is_size_dead(solutions, 2)

    -- Extract misses bitboard from tracking board
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

    -- 2. TARGET MODE: Pure Exact Multi-Hypothesis Bayesian Joint Targeting
    if has_active then
        local total_target_counts = {}
        local max_target_count = 0

        for _, sol in ipairs(solutions) do
            if #sol.active_hits > 0 then
                -- Forbidden = misses + corpses claimed in this specific hypothesis
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

        -- Build diagnostic heatmap for target mode
        local target_heatmap = self:create_blank_matrix(0)
        if max_target_count > 0 then
            for idx, cnt in pairs(total_target_counts) do
                local r = math.floor(idx / 10)
                local c = idx % 10
                target_heatmap[r][c] = math.floor((cnt / max_target_count) * 100)
            end
        end
        self.last_heatmap = target_heatmap

        -- Gather all candidate cells that achieve the peak target density
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
            -- Deterministic tie-breaking derived from parity and move count
            local pick_idx = ((self.parity_offset * 10007 + #move_history) % #best_targets) + 1
            local target = best_targets[pick_idx]
            local log = string.format("Watson: Exact Target Density (peak=%d, ties=%d)", max_target_count, #best_targets)
            return target, log, dead_sizes, diagnostics
        end
    end

    -- 3. HUNT MODE: Dynamic Stride Parity Placement Density
    local stride = is_2_dead and 3 or 2
    local total_hunt_density = self:create_blank_matrix(0)
    local max_hunt_d = 0

    for _, sol in ipairs(solutions) do
        local forbidden_bb = misses_bb:copy()
        for _, pt in ipairs(sol.claimed_pegs) do
            forbidden_bb:set(pt.r, pt.c)
        end

        local available_fleet = get_available_fleet(self.fleet_sizes, sol.dead_sizes)

        -- Distinct fleet lengths to prevent combinatorial distortion under parity
        local seen = {}
        local unique_fleet = {}
        for _, sz in ipairs(available_fleet) do
            if not seen[sz] then
                seen[sz] = true
                table.insert(unique_fleet, sz)
            end
        end

        local density, _ = self.combinatorics:compute_hunt_density(unique_fleet, forbidden_bb)

        for r = 0, 9 do
            for c = 0, 9 do
                if board[r][c] == GameEngine.CELL_UNKNOWN then
                    local d = total_hunt_density[r][c] + density[r][c]
                    total_hunt_density[r][c] = d
                    if d > max_hunt_d then max_hunt_d = d end
                end
            end
        end
    end

    self.last_heatmap = self:normalize_heatmap(total_hunt_density)

    -- Select candidates matching active dynamic stride covering net
    local best_hunt_score = -1
    local best_hunt_candidates = {}

    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_UNKNOWN then
                local on_stride = self:is_cell_on_stride(r, c, stride, self.parity_offset)
                if on_stride then
                    local d = total_hunt_density[r][c]
                    if d > best_hunt_score then
                        best_hunt_score = d
                        best_hunt_candidates = { { x = r, y = c } }
                    elseif d == best_hunt_score and d >= 0 then
                        table.insert(best_hunt_candidates, { x = r, y = c })
                    end
                end
            end
        end
    end

    -- Fallback to any unrevealed cell if stride net is exhausted
    if best_hunt_score <= 0 or #best_hunt_candidates == 0 then
        for r = 0, 9 do
            for c = 0, 9 do
                if board[r][c] == GameEngine.CELL_UNKNOWN then
                    local d = total_hunt_density[r][c]
                    if d > best_hunt_score then
                        best_hunt_score = d
                        best_hunt_candidates = { { x = r, y = c } }
                    elseif d == best_hunt_score then
                        table.insert(best_hunt_candidates, { x = r, y = c })
                    end
                end
            end
        end
    end

    local target
    if #best_hunt_candidates > 0 then
        local pick_idx = ((self.parity_offset * 10007 + #move_history) % #best_hunt_candidates) + 1
        target = best_hunt_candidates[pick_idx]
    else
        local unrevealed = self:get_unrevealed_cells(board)
        target = unrevealed[1] or { x = 0, y = 0 }
    end

    local log = string.format("Watson: Dynamic Parity Hunt (Stride-%d, peak=%d)", stride, best_hunt_score)
    return target, log, dead_sizes, diagnostics
end

--- Return normalized density heatmap for UI
-- @param board table: 10x10 tracking matrix
-- @param dead_sizes table (optional)
-- @return table: 10x10 normalized integer heatmap (0..100)
function WatsonBot:get_heatmap(board, dead_sizes)
    if self.last_heatmap then
        return self.last_heatmap
    end
    self:get_best_move(board, {})
    return self.last_heatmap
end

return WatsonBot

-- lua/bots/lestrade.lua
-- Rank 1: LestradeBot (Inspector G. Lestrade — Scotland Yard Baseline)
-- Canonical Strategy: Pure Marginal Placement Density (Faithful port of the Android app's WatsonBot)
-- Average Performance: ~55.6 shots, ~28% outliers (>60 shots).
-- Zero CSP deduction, zero hard assassin override: a pure unified geometric density field.

local BaseBot = require("lua.bots.base_bot")
local GameEngine = require("lua.game_engine")

local LestradeBot = BaseBot:extend("LestradeBot")

--- Constructor
-- @param parity_offset number (optional, unused by pure Lestrade)
function LestradeBot:init(parity_offset)
    -- Lestrade operates as a pure geometric density estimator without parity constraints
end

--- Core Decision Engine: Faithful port of Android TacticalEngine.kt's WatsonBot
-- @param board table: 10x10 tracking matrix (0=UNKNOWN, -1=MISS, 1=HIT, 2=SUNK)
-- @param move_history table: Array of move records
-- @return target table: { x = r, y = c }
-- @return log string: Strategy rationale
-- @return dead_sizes table: Empty (Lestrade cannot deduce specific dead ships)
-- @return diagnostics table: Telemetry table for UI
function LestradeBot:get_best_move(board, move_history)
    local density = self:create_blank_matrix(0)
    local total_hits = 0

    -- Count total hits currently visible on the board
    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_HIT or board[r][c] == GameEngine.CELL_SUNK then
                total_hits = total_hits + 1
            end
        end
    end

    -- Compute marginal density across all distinct ship lengths (5, 4, 3, 2)
    -- Matching exact Android logic: for (size in fleetSizes.distinct())
    for _, size in ipairs({ 5, 4, 3, 2 }) do
        -- Horizontal placements
        for r = 0, 9 do
            for c = 0, 10 - size do
                local can_fit = true
                local overlap_hits = 0

                for i = 0, size - 1 do
                    local cell = board[r][c + i]
                    if cell == GameEngine.CELL_MISS then
                        can_fit = false
                        break
                    elseif cell == GameEngine.CELL_HIT or cell == GameEngine.CELL_SUNK then
                        overlap_hits = overlap_hits + 1
                    end
                end

                if can_fit then
                    -- EXACT APP FORMULA: val weight = if (overlapHits > 0) 1 + overlapHits else 1
                    local weight = (overlap_hits > 0) and (1 + overlap_hits) or 1
                    for i = 0, size - 1 do
                        if board[r][c + i] == GameEngine.CELL_UNKNOWN then
                            density[r][c + i] = density[r][c + i] + weight
                        end
                    end
                end
            end
        end

        -- Vertical placements
        for c = 0, 9 do
            for r = 0, 10 - size do
                local can_fit = true
                local overlap_hits = 0

                for i = 0, size - 1 do
                    local cell = board[r + i][c]
                    if cell == GameEngine.CELL_MISS then
                        can_fit = false
                        break
                    elseif cell == GameEngine.CELL_HIT or cell == GameEngine.CELL_SUNK then
                        overlap_hits = overlap_hits + 1
                    end
                end

                if can_fit then
                    -- EXACT APP FORMULA: val weight = if (overlapHits > 0) 1 + overlapHits else 1
                    local weight = (overlap_hits > 0) and (1 + overlap_hits) or 1
                    for i = 0, size - 1 do
                        if board[r + i][c] == GameEngine.CELL_UNKNOWN then
                            density[r + i][c] = density[r + i][c] + weight
                        end
                    end
                end
            end
        end
    end

    -- Save normalized heatmap for UI rendering
    self.last_heatmap = self:normalize_heatmap(density)

    -- Select the unrevealed cell with the highest raw density
    local max_val = -1
    local best_candidates = {}

    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_UNKNOWN then
                local d = density[r][c]
                if d > max_val then
                    max_val = d
                    best_candidates = { { x = r, y = c } }
                elseif d == max_val and d >= 0 then
                    table.insert(best_candidates, { x = r, y = c })
                end
            end
        end
    end

    -- Deterministic tie-breaking if multiple cells have identical maximum density
    local target
    if #best_candidates > 0 then
        local idx = (#move_history % #best_candidates) + 1
        target = best_candidates[idx]
    else
        local unrevealed = self:get_unrevealed_cells(board)
        target = unrevealed[1] or { x = 0, y = 0 }
    end

    local log = string.format("Lestrade: Geometric Density (peak=%d, ties=%d)", max_val, #best_candidates)

    self.last_diagnostics = {
        max_density = max_val,
        candidates_count = #best_candidates,
        total_hits = total_hits
    }

    return target, log, {}, self.last_diagnostics
end

--- Return normalized density heatmap for UI
-- @param board table: 10x10 tracking matrix
-- @param dead_sizes table (unused by Lestrade)
-- @return table: 10x10 normalized integer heatmap (0..100)
function LestradeBot:get_heatmap(board, dead_sizes)
    if self.last_heatmap then
        return self.last_heatmap
    end
    self:get_best_move(board, {})
    return self.last_heatmap
end

return LestradeBot

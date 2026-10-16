-- lua/bots/base_bot.lua
-- Abstract Base Class and Standard Protocol Contract for Battleship Bots
-- Provides prototypical inheritance, abstract method enforcement, and telemetry helpers.

local BaseBot = {}
BaseBot.__index = BaseBot

-- Canonical fleet definition: Carrier 5, Battleship 4, Cruiser 3, Submarine 3, Destroyer 2
BaseBot.FLEET_SIZES = { 5, 4, 3, 3, 2 }

--- Base constructor
-- @param name string: Bot display name
-- @param parity_offset number (optional: 0 or 1, default 0)
function BaseBot.new(name, parity_offset)
    local self = setmetatable({}, BaseBot)
    self.name = name or "BaseBot"
    self.parity_offset = parity_offset or 0
    self.fleet_sizes = { 5, 4, 3, 3, 2 }
    self.last_heatmap = nil
    self.last_diagnostics = {}
    return self
end

--- Reset bot state before a new game
-- @param game_id number (optional: game index used to derive deterministic parity)
-- @param parity_offset number (optional: explicit parity 0 or 1)
function BaseBot:reset(game_id, parity_offset)
    if parity_offset ~= nil then
        self.parity_offset = parity_offset
    elseif game_id ~= nil then
        self.parity_offset = game_id % 2
    else
        self.parity_offset = 0
    end

    self.last_heatmap = nil
    self.last_diagnostics = {}

    -- Optional subclass hook
    if self.on_reset then
        self:on_reset(game_id)
    end
end

--- [ABSTRACT] Select the best move given the current board state and ledger
-- Subclasses MUST override this method.
-- @param board table: 10x10 tracking matrix (0=UNKNOWN, -1=MISS, 1=HIT, 2=SUNK)
-- @param move_history table: Array of move records { {shotNumber, x, y, result, isSunk, shipId}, ... }
-- @return target table: { x = row, y = col } (0..9)
-- @return log string: Human-readable rationale description
-- @return dead_sizes table: Array of confirmed dead ship lengths { 5, 3 }
-- @return diagnostics table: Telemetry table { active={}, claimed={}, solutions_count=N }
function BaseBot:get_best_move(board, move_history)
    error(string.format("[%s] Abstract method 'get_best_move' must be implemented by subclass", self.name), 2)
end

--- [ABSTRACT / HOOK] Generate a 10x10 heatmap for UI diagnostics
-- Default implementation returns self.last_heatmap or a zeroed 10x10 matrix.
-- Subclasses may override for custom diagnostic visualization.
-- @param board table: 10x10 tracking matrix
-- @param dead_sizes table: Array of confirmed eliminated ship sizes
-- @return table: 10x10 integer matrix (0..100) representing probability density
function BaseBot:get_heatmap(board, dead_sizes)
    if self.last_heatmap then
        return self.last_heatmap
    end
    return self:create_blank_matrix(0)
end

--- OOP Subclass Factory Helper
-- Enables clean prototypical inheritance:
--   local MyBot = BaseBot:extend("MyBot")
--   function MyBot:init(...) ... end
--   function MyBot:get_best_move(...) ... end
-- @param subclass_name string: The name of the new bot class
-- @return table: The newly created subclass table
function BaseBot:extend(subclass_name)
    local Subclass = {}
    Subclass.__index = Subclass
    setmetatable(Subclass, { __index = self })

    Subclass.new = function(...)
        local instance = setmetatable(BaseBot.new(subclass_name), Subclass)
        if instance.init then
            instance:init(...)
        end
        return instance
    end

    return Subclass
end

-- ============================================================================
-- SHARED UTILITIES FOR SUBCLASSES
-- ============================================================================

--- Create a 10x10 matrix initialized to a default value
-- @param default_val any (default 0)
-- @return table: 10x10 matrix
function BaseBot:create_blank_matrix(default_val)
    local val = (default_val ~= nil) and default_val or 0
    local mat = {}
    for r = 0, 9 do
        mat[r] = {}
        for c = 0, 9 do
            mat[r][c] = val
        end
    end
    return mat
end

--- Normalize a 10x10 raw count matrix into 0..100 integer range for UI heatmaps
-- @param raw_matrix table: 10x10 table of numbers
-- @return table: 10x10 table with values in [0..100]
function BaseBot:normalize_heatmap(raw_matrix)
    local max_val = 0
    for r = 0, 9 do
        for c = 0, 9 do
            local v = raw_matrix[r][c] or 0
            if v > max_val then
                max_val = v
            end
        end
    end

    local normalized = self:create_blank_matrix(0)
    if max_val > 0 then
        for r = 0, 9 do
            for c = 0, 9 do
                local v = raw_matrix[r][c] or 0
                normalized[r][c] = math.floor((v / max_val) * 100)
            end
        end
    end

    return normalized
end

--- Determine the minimum surviving ship length (Dynamic Stride: 2 -> 3 -> 4 -> 5)
-- @param dead_sizes table: Array of confirmed sunk ship lengths, e.g. {2}, {2, 3}, {2, 3, 3}
-- @return number: Minimum surviving length (2, 3, 4, or 5)
function BaseBot:get_dynamic_stride(dead_sizes)
    local surviving = { 5, 4, 3, 3, 2 }
    if dead_sizes and #dead_sizes > 0 then
        for _, s in ipairs(dead_sizes) do
            for idx, live_s in ipairs(surviving) do
                if live_s == s then
                    table.remove(surviving, idx)
                    break
                end
            end
        end
    end

    local min_len = 5
    for _, s in ipairs(surviving) do
        if s < min_len then
            min_len = s
        end
    end
    return min_len
end

--- Check if coordinate (r, c) lies on the optimal covering net for a given stride
-- Mathematically proven covering nets that guarantee hitting 100% of ships >= stride:
--   Stride 2: (r + 1*c) % 2 == offset % 2
--   Stride 3: (r + 2*c) % 3 == offset % 3  (gcd(2, 3) = 1)
--   Stride 4: (r + 3*c) % 4 == offset % 4  (gcd(3, 4) = 1)
--   Stride 5: (r + 2*c) % 5 == offset % 5  (gcd(2, 5) = 1)
-- @param r number (0..9)
-- @param c number (0..9)
-- @param stride number (2, 3, 4, or 5)
-- @param parity_offset number (optional: defaults to self.parity_offset)
-- @return boolean
function BaseBot:is_cell_on_stride(r, c, stride, parity_offset)
    local s = stride or 2
    local offset = parity_offset or self.parity_offset or 0

    if s == 2 then
        return (r + c) % 2 == (offset % 2)
    elseif s == 3 then
        return (r + 2 * c) % 3 == (offset % 3)
    elseif s == 4 then
        return (r + 3 * c) % 4 == (offset % 4)
    elseif s >= 5 then
        return (r + 2 * c) % 5 == (offset % 5)
    end
    return true
end

--- Extract all currently unrevealed cells from the tracking board
-- @param board table: 10x10 tracking matrix
-- @return table: Array of { x = r, y = c } coordinates
function BaseBot:get_unrevealed_cells(board)
    local unrevealed = {}
    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == 0 then -- CELL_UNKNOWN == 0
                table.insert(unrevealed, { x = r, y = c })
            end
        end
    end
    return unrevealed
end

--- Check if (r, c) is a valid coordinate on the 10x10 grid
-- @param r number
-- @param c number
-- @return boolean
function BaseBot:is_valid_cell(r, c)
    return r >= 0 and r < 10 and c >= 0 and c < 10
end

return BaseBot

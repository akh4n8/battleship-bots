-- lua/game_engine.lua
-- High-Performance Battleship Physics Engine with Strict Fog of War and Bitboard State Tracking
-- Standard 10x10 Grid, 5-Ship Fleet (5, 4, 3, 3, 2), Continuous Fire mechanics.

local Bitboard = require("lua.bitboard")

local GameEngine = {}
GameEngine.__index = GameEngine

-- 1. CELL CONSTANTS
GameEngine.CELL_UNKNOWN = 0
GameEngine.CELL_MISS    = -1
GameEngine.CELL_HIT     = 1
GameEngine.CELL_SUNK    = 2

-- Fleet specifications: { size, name, id }
GameEngine.FLEET_SPECS = {
    { size = 5, name = "Carrier",    id = 1 },
    { size = 4, name = "Battleship", id = 2 },
    { size = 3, name = "Cruiser",    id = 3 },
    { size = 3, name = "Submarine",  id = 4 },
    { size = 2, name = "Destroyer",  id = 5 },
}

GameEngine.FLEET_SIZES = { 5, 4, 3, 3, 2 }
GameEngine.TOTAL_PEGS  = 17 -- 5 + 4 + 3 + 3 + 2

--- Constructor: Initializes a clean 10x10 game state
function GameEngine.new()
    local self = setmetatable({}, GameEngine)
    self.size = 10
    self.total_pegs = GameEngine.TOTAL_PEGS
    self.pegs_hit = 0
    self.shot_number = 0

    -- Truth board: 10x10 grid containing ship_id (1..5) or 0 (water)
    self.truth_board = {}
    -- Tracking board: 10x10 grid containing CELL_UNKNOWN, CELL_MISS, CELL_HIT, CELL_SUNK
    self.tracking_board = {}

    for r = 0, 9 do
        self.truth_board[r] = {}
        self.tracking_board[r] = {}
        for c = 0, 9 do
            self.truth_board[r][c] = 0
            self.tracking_board[r][c] = GameEngine.CELL_UNKNOWN
        end
    end

    -- Fleet tracking
    self.ship_pegs = {}   -- { [ship_id] = { {r, c}, ... } }
    self.ship_health = {} -- { [ship_id] = remaining_hp }
    self.ship_sizes = {}  -- { [ship_id] = total_length }

    for _, spec in ipairs(GameEngine.FLEET_SPECS) do
        self.ship_pegs[spec.id] = {}
        self.ship_health[spec.id] = spec.size
        self.ship_sizes[spec.id] = spec.size
    end

    -- Move history ledger for forensics, replays, and undo
    self.move_history = {}

    -- Bitboard representations for instant O(1) state queries
    self.miss_bb     = Bitboard.empty()
    self.hit_bb      = Bitboard.empty() -- hit pegs (excluding lethal sunk)
    self.sunk_bb     = Bitboard.empty() -- lethal sunk pegs
    self.all_hits_bb = Bitboard.empty() -- hits | sunks

    return self
end

--- Check if a ship can legally be placed at (r, c)
-- @param r number (0..9)
-- @param c number (0..9)
-- @param length number
-- @param is_horizontal boolean
-- @return boolean
function GameEngine:can_place(r, c, length, is_horizontal)
    for i = 0, length - 1 do
        local cr = is_horizontal and r or (r + i)
        local cc = is_horizontal and (c + i) or c

        if cr < 0 or cr >= self.size or cc < 0 or cc >= self.size then
            return false
        end
        if self.truth_board[cr][cc] ~= 0 then
            return false
        end
    end
    return true
end

--- Place a ship manually onto the truth board
-- @param r number (0..9)
-- @param c number (0..9)
-- @param length number
-- @param is_horizontal boolean
-- @param ship_id number (1..5)
-- @return boolean
function GameEngine:place_ship_manual(r, c, length, is_horizontal, ship_id)
    if not self:can_place(r, c, length, is_horizontal) then
        return false
    end

    self.ship_pegs[ship_id] = {}
    self.ship_health[ship_id] = length
    self.ship_sizes[ship_id] = length

    for i = 0, length - 1 do
        local cr = is_horizontal and r or (r + i)
        local cc = is_horizontal and (c + i) or c
        self.truth_board[cr][cc] = ship_id
        table.insert(self.ship_pegs[ship_id], { r = cr, c = cc })
    end

    return true
end

--- Randomly place all 5 fleet ships using Lua's math.random or custom RNG
-- @param rng_func function (optional: function(min, max) returns int)
function GameEngine:place_ships_randomly(rng_func)
    local rand_int = rng_func or function(min, max)
        return math.random(min, max)
    end
    local rand_bool = function()
        return rand_int(0, 1) == 1
    end

    for _, spec in ipairs(GameEngine.FLEET_SPECS) do
        local placed = false
        while not placed do
            local is_horizontal = rand_bool()
            local r = rand_int(0, self.size - 1)
            local c = rand_int(0, self.size - 1)
            placed = self:place_ship_manual(r, c, spec.size, is_horizontal, spec.id)
        end
    end
end

--- Fire a shot at coordinate (r, c)
-- Strict Fog of War compliance:
--   - Water -> CELL_MISS ("MISS")
--   - Non-lethal Hit -> CELL_HIT ("HIT")
--   - Lethal Hit -> CELL_SUNK ("SUNK"). Previous hits on this ship REMAIN CELL_HIT!
-- @param r number (0..9)
-- @param c number (0..9)
-- @return string: "MISS" | "HIT" | "SUNK" | "INVALID"
function GameEngine:fire_shot(r, c)
    if r < 0 or r >= self.size or c < 0 or c >= self.size then
        return "INVALID"
    end
    if self.tracking_board[r][c] ~= GameEngine.CELL_UNKNOWN then
        return "INVALID"
    end

    self.shot_number = self.shot_number + 1
    local truth_cell = self.truth_board[r][c]
    local is_sunk = false
    local result_str = ""
    local hit_ship_id = nil

    if truth_cell == 0 then
        -- Miss
        self.tracking_board[r][c] = GameEngine.CELL_MISS
        self.miss_bb:set(r, c)
        result_str = "MISS"
    else
        -- Hit
        hit_ship_id = truth_cell
        self.ship_health[hit_ship_id] = self.ship_health[hit_ship_id] - 1
        self.pegs_hit = self.pegs_hit + 1
        self.all_hits_bb:set(r, c)

        if self.ship_health[hit_ship_id] == 0 then
            -- Lethal sunk shot: only this specific shot receives CELL_SUNK
            self.tracking_board[r][c] = GameEngine.CELL_SUNK
            self.sunk_bb:set(r, c)
            is_sunk = true
            result_str = "SUNK"
        else
            -- Non-lethal hit
            self.tracking_board[r][c] = GameEngine.CELL_HIT
            self.hit_bb:set(r, c)
            result_str = "HIT"
        end
    end

    -- Record in move history ledger
    table.insert(self.move_history, {
        shotNumber = self.shot_number,
        x = r,
        y = c,
        result = result_str,
        isSunk = is_sunk,
        shipId = hit_ship_id
    })

    return result_str
end

--- Undo the last shot in the ledger (full rollback)
-- @return boolean: true if undone, false if ledger empty
function GameEngine:undo_last_shot()
    if #self.move_history == 0 then
        return false
    end

    local last_move = table.remove(self.move_history)
    local r = last_move.x
    local c = last_move.y
    local result = last_move.result
    local ship_id = last_move.shipId

    self.shot_number = self.shot_number - 1
    self.tracking_board[r][c] = GameEngine.CELL_UNKNOWN

    if result == "MISS" then
        self.miss_bb:clear(r, c)
    elseif result == "HIT" then
        self.hit_bb:clear(r, c)
        self.all_hits_bb:clear(r, c)
        self.ship_health[ship_id] = self.ship_health[ship_id] + 1
        self.pegs_hit = self.pegs_hit - 1
    elseif result == "SUNK" then
        self.sunk_bb:clear(r, c)
        self.all_hits_bb:clear(r, c)
        self.ship_health[ship_id] = self.ship_health[ship_id] + 1
        self.pegs_hit = self.pegs_hit - 1
    end

    return true
end

--- Check if game is over (all 17 pegs destroyed)
-- @return boolean
function GameEngine:is_game_over()
    return self.pegs_hit == self.total_pegs
end

--- Get a deep copy of the tracking board matrix
-- @return table: 10x10 matrix
function GameEngine:get_tracking_board_copy()
    local copy = {}
    for r = 0, 9 do
        copy[r] = {}
        for c = 0, 9 do
            copy[r][c] = self.tracking_board[r][c]
        end
    end
    return copy
end

--- Return an ASCII visual representation of tracking vs truth boards
-- @param reveal_truth boolean (optional, default false)
-- @return string
function GameEngine:to_ascii_string(reveal_truth)
    local lines = { "    TRACKING BOARD                 TRUTH BOARD", "    0 1 2 3 4 5 6 7 8 9            0 1 2 3 4 5 6 7 8 9" }
    for r = 0, 9 do
        local track_chars = {}
        local truth_chars = {}
        for c = 0, 9 do
            local tv = self.tracking_board[r][c]
            local tc = "."
            if tv == GameEngine.CELL_MISS then tc = "o"
            elseif tv == GameEngine.CELL_HIT then tc = "x"
            elseif tv == GameEngine.CELL_SUNK then tc = "X"
            end
            track_chars[c + 1] = tc

            local true_val = self.truth_board[r][c]
            truth_chars[c + 1] = (true_val > 0) and tostring(true_val) or "."
        end
        local track_str = table.concat(track_chars, " ")
        local truth_str = reveal_truth and table.concat(truth_chars, " ") or "Hidden (Fog of War)"
        lines[#lines + 1] = string.format("%d | %s |        %d | %s |", r, track_str, r, truth_str)
    end
    return table.concat(lines, "\n")
end

return GameEngine

-- lua/combinatorics.lua
-- Combinatorics, Placement Library & Exact Joint Target Solver (P1, P2)
-- Precomputes 760 valid fleet ship placements as 100-bit bitboards.
-- Solves touching ships, L-shapes, T-shapes, and complex hit clusters with exact disjoint joint covers.

local Bitboard = require("lua.bitboard")
local GameEngine = require("lua.game_engine")

local Combinatorics = {}
Combinatorics.__index = Combinatorics

Combinatorics.FLEET_SIZES = { 5, 4, 3, 3, 2 }

--- Constructor: Precomputes all 760 possible straight-line ship placements
function Combinatorics.new()
    local self = setmetatable({}, Combinatorics)
    self.placements_by_size = {}
    self:init_placements()
    return self
end

--- Precompute the 760 valid fleet ship placements on a 10x10 board
function Combinatorics:init_placements()
    for _, size in ipairs({ 5, 4, 3, 2 }) do
        if not self.placements_by_size[size] then
            local list = {}

            -- Horizontal placements
            for r = 0, 9 do
                for c = 0, 10 - size do
                    local bb = Bitboard.empty()
                    local coords = {}
                    for i = 0, size - 1 do
                        bb:set(r, c + i)
                        table.insert(coords, { r = r, c = c + i })
                    end
                    table.insert(list, { size = size, mask = bb, coords = coords })
                end
            end

            -- Vertical placements
            for c = 0, 9 do
                for r = 0, 10 - size do
                    local bb = Bitboard.empty()
                    local coords = {}
                    for i = 0, size - 1 do
                        bb:set(r + i, c)
                        table.insert(coords, { r = r + i, c = c })
                    end
                    table.insert(list, { size = size, mask = bb, coords = coords })
                end
            end

            self.placements_by_size[size] = list
        end
    end
end

--- Return all precomputed placements for a given ship size
-- @param size number (2, 3, 4, 5)
-- @return table of { size=N, mask=bitboard_t, coords={...} }
function Combinatorics:get_placements(size)
    return self.placements_by_size[size] or {}
end

--- Compute Hunt Mode density: how many valid placements of surviving ships cross each cell
-- @param available_fleet table: array of remaining ship sizes, e.g. {5, 4, 3, 3}
-- @param forbidden_bb bitboard_t: misses + confirmed dead ship corpses
-- @return table: 10x10 matrix of raw placement counts
-- @return number: maximum density value found
function Combinatorics:compute_hunt_density(available_fleet, forbidden_bb)
    local density = {}
    for r = 0, 9 do
        density[r] = {}
        for c = 0, 9 do density[r][c] = 0 end
    end
    local max_d = 0

    for _, size in ipairs(available_fleet) do
        local plist = self.placements_by_size[size]
        if plist then
            for _, p in ipairs(plist) do
                if not p.mask:intersects(forbidden_bb) then
                    for _, pt in ipairs(p.coords) do
                        local d = density[pt.r][pt.c] + 1
                        density[pt.r][pt.c] = d
                        if d > max_d then max_d = d end
                    end
                end
            end
        end
    end

    return density, max_d
end

--- Compute Target Mode density (The Touching Ships Solver):
-- Evaluates exact single-ship (P1) and disjoint 2-ship (P1, P2) covers across active hits.
-- @param board table: 10x10 tracking matrix
-- @param available_fleet table: array of surviving ship sizes
-- @param forbidden_bb bitboard_t: misses + confirmed dead ship corpses
-- @param active_hits_bb bitboard_t: all currently active wounded hits
-- @return table: map [r*10 + c] = count of how often each unknown cell completes a valid ship cover
-- @return number: maximum count found
function Combinatorics:compute_target_density(board, available_fleet, forbidden_bb, active_hits_bb)
    local counts = {}
    local max_count = 0

    local function add_count(r, c, weight)
        if board[r][c] == 0 then -- CELL_UNKNOWN == 0
            local idx = r * 10 + c
            local cur = (counts[idx] or 0) + (weight or 1)
            counts[idx] = cur
            if cur > max_count then max_count = cur end
        end
    end

    -- ------------------------------------------------------------------------
    -- PHASE 1: Single-Ship Exact Covers (P1)
    -- ------------------------------------------------------------------------
    -- Check if any single surviving ship can explain ALL active hits
    local single_covers = {}
    for _, size in ipairs(available_fleet) do
        local plist = self.placements_by_size[size]
        if plist then
            for _, p in ipairs(plist) do
                -- Placement must not hit misses/corpses AND must contain all active hits
                if not p.mask:intersects(forbidden_bb) and p.mask:contains_all(active_hits_bb) then
                    -- Must cover at least one unknown cell
                    local has_unknown = false
                    for _, pt in ipairs(p.coords) do
                        if board[pt.r][pt.c] == 0 then
                            has_unknown = true
                            break
                        end
                    end
                    if has_unknown then
                        table.insert(single_covers, p)
                    end
                end
            end
        end
    end

    if #single_covers > 0 then
        for _, p in ipairs(single_covers) do
            for _, pt in ipairs(p.coords) do
                add_count(pt.r, pt.c, 1)
            end
        end
        return counts, max_count
    end

    -- ------------------------------------------------------------------------
    -- PHASE 2: Disjoint 2-Ship Joint Covers (P1, P2)
    -- ------------------------------------------------------------------------
    -- No single ship can explain all hits (e.g. touching ships forming L-shapes or T-shapes).
    -- Filter candidate placements that touch at least 1 hit and have unknown cells.
    local candidates = {}
    for ship_idx, size in ipairs(available_fleet) do
        local plist = self.placements_by_size[size]
        if plist then
            for _, p in ipairs(plist) do
                if not p.mask:intersects(forbidden_bb) and p.mask:intersects(active_hits_bb) then
                    local has_unknown = false
                    for _, pt in ipairs(p.coords) do
                        if board[pt.r][pt.c] == 0 then
                            has_unknown = true
                            break
                        end
                    end
                    if has_unknown then
                        table.insert(candidates, {
                            ship_idx = ship_idx,
                            size = size,
                            mask = p.mask,
                            coords = p.coords,
                            hit_overlap = p.mask:band(active_hits_bb)
                        })
                    end
                end
            end
        end
    end

    local pair_covers = {}
    local num_cand = #candidates

    for i = 1, num_cand do
        local c1 = candidates[i]
        for j = i + 1, num_cand do
            local c2 = candidates[j]
            -- Must be two different ship instances from the available fleet
            if c1.ship_idx ~= c2.ship_idx then
                -- Must NOT collide with each other
                if not c1.mask:intersects(c2.mask) then
                    -- Together, their union must cover 100% of the active hits!
                    local union_hits = c1.hit_overlap:bor(c2.hit_overlap)
                    if union_hits:contains_all(active_hits_bb) then
                        table.insert(pair_covers, { c1, c2 })
                    end
                end
            end
        end
    end

    if #pair_covers > 0 then
        for _, pair in ipairs(pair_covers) do
            local c1, c2 = pair[1], pair[2]
            for _, pt in ipairs(c1.coords) do add_count(pt.r, pt.c, 1) end
            for _, pt in ipairs(c2.coords) do add_count(pt.r, pt.c, 1) end
        end
        return counts, max_count
    end

    -- ------------------------------------------------------------------------
    -- PHASE 3: Weighted Fallback (3+ Touching Ships Cluster)
    -- ------------------------------------------------------------------------
    for _, c in ipairs(candidates) do
        local hit_count = c.hit_overlap:popcnt()
        local weight = hit_count * hit_count -- quadratic reward for explaining more hits
        for _, pt in ipairs(c.coords) do
            add_count(pt.r, pt.c, weight)
        end
    end

    return counts, max_count
end

return Combinatorics

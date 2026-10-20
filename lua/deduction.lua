-- lua/deduction.lua
-- The Shared Brain: Chronological Multi-Hypothesis Constraint Satisfaction Problem (CSP) Deduction Engine
-- Solves the Fog of War blind-sink problem by isolating confirmed dead corpses from active wounded hits.
-- Employs 100-bit bitboards for ultra-fast candidate overlap checks and leftover hit calculations.

local Bitboard = require("lua.bitboard")
local GameEngine = require("lua.game_engine")

local Deduction = {}
Deduction.__index = Deduction

Deduction.FLEET_SIZES = { 5, 4, 3, 3, 2 }

--- Constructor
function Deduction.new(fleet_sizes)
    local self = setmetatable({}, Deduction)
    self.fleet_sizes = fleet_sizes or Deduction.FLEET_SIZES
    return self
end

--- Find all valid placements of a ship of 'size' that pass through 'sunk_coord'
-- Placements must:
--   1. Be entirely horizontal or entirely vertical.
--   2. Only consist of coordinates that were hit at or before 'sunk_time'.
--   3. Not overlap any coordinate already in 'claimed_bb'.
-- @param board table: 10x10 tracking matrix
-- @param sr number: sunk row (0..9)
-- @param sc number: sunk col (0..9)
-- @param sunk_time number: shotNumber when the lethal shot occurred
-- @param size number: candidate ship length
-- @param peg_times table: flat array of shot times [idx] = shotNumber
-- @param claimed_bb bitboard_t: bitboard of pegs already claimed by other dead ships
-- @return table of placements: { { coords = {{r, c}, ...}, bb = bitboard_t }, ... }
function Deduction:get_valid_placements(board, sr, sc, sunk_time, size, peg_times, claimed_bb)
    local placements = {}

    -- 1. Horizontal placements passing through (sr, sc)
    for c_start = sc - size + 1, sc do
        if c_start >= 0 and (c_start + size - 1) < 10 then
            local valid = true
            local coords = {}
            local bb = Bitboard.empty()

            for i = 0, size - 1 do
                local r = sr
                local c = c_start + i
                local idx = r * 10 + c
                local t = peg_times[idx] or -1

                if claimed_bb:is_set(r, c) then
                    valid = false
                    break
                end

                if (r == sr and c == sc) then
                    table.insert(coords, { r = r, c = c })
                    bb:set(r, c)
                elseif t > 0 and t < sunk_time then
                    table.insert(coords, { r = r, c = c })
                    bb:set(r, c)
                else
                    valid = false
                    break
                end
            end

            if valid then
                table.insert(placements, { coords = coords, bb = bb })
            end
        end
    end

    -- 2. Vertical placements passing through (sr, sc)
    for r_start = sr - size + 1, sr do
        if r_start >= 0 and (r_start + size - 1) < 10 then
            local valid = true
            local coords = {}
            local bb = Bitboard.empty()

            for i = 0, size - 1 do
                local r = r_start + i
                local c = sc
                local idx = r * 10 + c
                local t = peg_times[idx] or -1

                if claimed_bb:is_set(r, c) then
                    valid = false
                    break
                end

                if (r == sr and c == sc) then
                    table.insert(coords, { r = r, c = c })
                    bb:set(r, c)
                elseif t > 0 and t < sunk_time then
                    table.insert(coords, { r = r, c = c })
                    bb:set(r, c)
                else
                    valid = false
                    break
                end
            end

            if valid then
                table.insert(placements, { coords = coords, bb = bb })
            end
        end
    end

    return placements
end

--- Recursive CSP solver: explores tree of possible ship identities for each sunken peg
-- @param board table: 10x10 tracking matrix
-- @param sunks table: list of {r, c} coordinates where CELL_SUNK occurred
-- @param sunk_idx number: current index in sunks list (1..#sunks)
-- @param peg_times table: flat array of shot times
-- @param available_fleet table: array of remaining ship sizes
-- @param current_dead_ships table: list of deduced dead ship coordinate lists
-- @param current_dead_sizes table: list of deduced dead ship sizes
-- @param all_hits_bb bitboard_t: all known hit pegs
-- @param claimed_bb bitboard_t: pegs claimed by current dead ships
-- @return best_score number: minimum leftover score found
-- @return solutions table: list of valid solution tuples
function Deduction:solve_all(board, sunks, sunk_idx, peg_times, available_fleet, current_dead_ships, current_dead_sizes, all_hits_bb, claimed_bb)
    -- Base Case: All sunken pegs have been assigned dead ships
    if sunk_idx > #sunks then
        local leftovers_bb = all_hits_bb:band(claimed_bb:bnot())
        local num_leftovers = leftovers_bb:popcnt()

        -- Feasibility check: Can each leftover hit still belong to a valid surviving ship?
        if #available_fleet > 0 and num_leftovers > 0 then
            local min_ship_size = 5
            for _, s in ipairs(available_fleet) do
                if s < min_ship_size then min_ship_size = s end
            end

            local leftover_coords = leftovers_bb:to_coords()
            for _, hit in ipairs(leftover_coords) do
                local r, c = hit.r, hit.c

                -- Check horizontal clearance
                local left = c
                while left > 0 and (board[r][left - 1] == 0 or board[r][left - 1] == 1) and not claimed_bb:is_set(r, left - 1) do
                    left = left - 1
                end
                local right = c
                while right < 9 and (board[r][right + 1] == 0 or board[r][right + 1] == 1) and not claimed_bb:is_set(r, right + 1) do
                    right = right + 1
                end
                local horiz_len = right - left + 1

                local horiz_unknowns = 0
                for col = left, right do
                    if board[r][col] == 0 then horiz_unknowns = horiz_unknowns + 1 end
                end

                -- Check vertical clearance
                local up = r
                while up > 0 and (board[up - 1][c] == 0 or board[up - 1][c] == 1) and not claimed_bb:is_set(up - 1, c) do
                    up = up - 1
                end
                local down = r
                while down < 9 and (board[down + 1][c] == 0 or board[down + 1][c] == 1) and not claimed_bb:is_set(down + 1, c) do
                    down = down + 1
                end
                local vert_len = down - up + 1

                local vert_unknowns = 0
                for row = up, down do
                    if board[row][c] == 0 then vert_unknowns = vert_unknowns + 1 end
                end

                local valid_horiz = (horiz_len >= min_ship_size) and (horiz_unknowns > 0)
                local valid_vert  = (vert_len >= min_ship_size) and (vert_unknowns > 0)

                -- If a leftover hit is completely trapped in dead water, this hypothesis is physically impossible
                if not (valid_horiz or valid_vert) then
                    return math.huge, {}
                end
            end
        end

        -- Clone dead ships and dead sizes
        local dead_ships_copy = {}
        for _, s in ipairs(current_dead_ships) do
            local scopy = {}
            for _, pt in ipairs(s) do table.insert(scopy, { r = pt.r, c = pt.c }) end
            table.insert(dead_ships_copy, scopy)
        end

        local dead_sizes_copy = {}
        for _, sz in ipairs(current_dead_sizes) do table.insert(dead_sizes_copy, sz) end

        local sol_record = {
            dead_ships = dead_ships_copy,
            dead_sizes = dead_sizes_copy,
            claimed_bb = claimed_bb:copy(),
            leftovers_bb = leftovers_bb:copy(),
            score = num_leftovers
        }
        return num_leftovers, { sol_record }
    end

    -- Recursive Step: Process sunk_idx
    local sunk = sunks[sunk_idx]
    local sunk_time = peg_times[sunk.r * 10 + sunk.c] or math.huge

    -- Get unique sizes present in available_fleet, sorted descending
    local unique_sizes = {}
    local seen_sizes = {}
    for _, sz in ipairs(available_fleet) do
        if not seen_sizes[sz] then
            seen_sizes[sz] = true
            table.insert(unique_sizes, sz)
        end
    end
    table.sort(unique_sizes, function(a, b) return a > b end)

    local best_score = math.huge
    local best_solutions = {}

    for _, size in ipairs(unique_sizes) do
        local candidate_placements = self:get_valid_placements(board, sunk.r, sunk.c, sunk_time, size, peg_times, claimed_bb)

        for _, placement in ipairs(candidate_placements) do
            -- Ensure candidate does not overlap currently claimed corpses
            if not claimed_bb:intersects(placement.bb) then
                local next_claimed = claimed_bb:bor(placement.bb)
                table.insert(current_dead_ships, placement.coords)
                table.insert(current_dead_sizes, size)

                -- Remove one instance of 'size' from available fleet
                local next_fleet = {}
                local removed = false
                for _, sz in ipairs(available_fleet) do
                    if sz == size and not removed then
                        removed = true
                    else
                        table.insert(next_fleet, sz)
                    end
                end

                local child_score, child_solutions = self:solve_all(
                    board, sunks, sunk_idx + 1, peg_times, next_fleet,
                    current_dead_ships, current_dead_sizes, all_hits_bb, next_claimed
                )

                if #child_solutions > 0 then
                    if child_score < best_score then
                        best_score = child_score
                        best_solutions = {}
                        for _, sol in ipairs(child_solutions) do table.insert(best_solutions, sol) end
                    elseif child_score == best_score then
                        for _, sol in ipairs(child_solutions) do table.insert(best_solutions, sol) end
                    end
                end

                table.remove(current_dead_ships)
                table.remove(current_dead_sizes)
            end
        end
    end

    return best_score, best_solutions
end

--- Main Entry Point: Analyze board state and return all tied minimum-leftover deduction hypotheses
-- @param board table: 10x10 tracking matrix
-- @param move_history table: Array of move records
-- @return table of solution tables:
--   {
--       active_hits  = { {r, c}, ... },
--       dead_ships   = { { {r, c}, ... }, ... },
--       claimed_pegs = { {r, c}, ... },
--       dead_sizes   = { 2, 3, ... },
--       score        = N
--   }
function Deduction:analyze(board, move_history)
    local all_hits_bb = Bitboard.empty()
    local sunks = {}
    local peg_times = {}

    -- Fast 1D peg times array [idx] = shotNumber
    if move_history then
        for _, m in ipairs(move_history) do
            if m.result ~= "MISS" then
                local idx = m.x * 10 + m.y
                peg_times[idx] = m.shotNumber
            end
        end
    end

    for r = 0, 9 do
        for c = 0, 9 do
            local val = board[r][c]
            if val == GameEngine.CELL_HIT or val == GameEngine.CELL_SUNK then
                all_hits_bb:set(r, c)
            end
            if val == GameEngine.CELL_SUNK then
                table.insert(sunks, { r = r, c = c })
            end
        end
    end

    -- If no ships have been sunk yet, all hits are 100% active wounded hits!
    if #sunks == 0 then
        return {
            {
                active_hits  = all_hits_bb:to_coords(),
                dead_ships   = {},
                claimed_pegs = {},
                dead_sizes   = {},
                score        = all_hits_bb:popcnt()
            }
        }
    end

    -- Chronological sort: earlier sunks are resolved first
    table.sort(sunks, function(a, b)
        local ta = peg_times[a.r * 10 + a.c] or 0
        local tb = peg_times[b.r * 10 + b.c] or 0
        return ta < tb
    end)

    local initial_fleet = {}
    for _, sz in ipairs(self.fleet_sizes) do table.insert(initial_fleet, sz) end

    local empty_claimed = Bitboard.empty()
    local best_score, raw_solutions = self:solve_all(
        board, sunks, 1, peg_times, initial_fleet, {}, {}, all_hits_bb, empty_claimed
    )

    -- Fallback: If no mathematically consistent hypothesis could be found
    if #raw_solutions == 0 then
        local fallback_dead_ships = {}
        local fallback_claimed = Bitboard.empty()
        for _, s in ipairs(sunks) do
            table.insert(fallback_dead_ships, { s })
            fallback_claimed:set(s.r, s.c)
        end
        local fallback_active = all_hits_bb:band(fallback_claimed:bnot()):to_coords()
        return {
            {
                active_hits  = fallback_active,
                dead_ships   = fallback_dead_ships,
                claimed_pegs = fallback_claimed:to_coords(),
                dead_sizes   = {},
                score        = #fallback_active
            }
        }
    end

    -- Deduplicate solutions and format output
    local parsed = {}
    local seen_keys = {}

    for _, sol in ipairs(raw_solutions) do
        -- Build canonical key for deduplication
        table.sort(sol.dead_sizes)
        local key = table.concat(sol.dead_sizes, ",") .. "|" .. sol.claimed_bb:to_grid_string()

        if not seen_keys[key] then
            seen_keys[key] = true
            table.insert(parsed, {
                active_hits  = sol.leftovers_bb:to_coords(),
                dead_ships   = sol.dead_ships,
                claimed_pegs = sol.claimed_bb:to_coords(),
                dead_sizes   = sol.dead_sizes,
                score        = sol.score
            })
        end
    end

    return parsed
end

--- Check if a specific ship size is confirmed dead across ALL tied hypotheses
-- @param solutions table: Array of solution records from analyze()
-- @param size number: Ship size to verify (e.g. 2 for Patrol Boat)
-- @return boolean: true only if size is present in EVERY tied solution
function Deduction.is_size_dead(solutions, size)
    if not solutions or #solutions == 0 then return false end
    for _, sol in ipairs(solutions) do
        local found = false
        for _, sz in ipairs(sol.dead_sizes) do
            if sz == size then
                found = true
                break
            end
        end
        if not found then return false end
    end
    return true
end

--- Check if any tied hypothesis contains active wounded hits
-- @param solutions table: Array of solution records
-- @return boolean: true if active wounded ships require finishing off
function Deduction.has_active_hits(solutions)
    if not solutions or #solutions == 0 then return false end
    for _, sol in ipairs(solutions) do
        if #sol.active_hits > 0 then return true end
    end
    return false
end

--- Build multi-hypothesis consensus diagnostics table
-- Active hits = union across all tied hypotheses
-- Claimed pegs & dead ships = intersection across all tied hypotheses
function Deduction.build_diagnostics(solutions)
    if not solutions or #solutions == 0 then
        return {
            active = {},
            claimed = {},
            dead_ships = {},
            dead_sizes = {},
            solutions_count = 0,
            is_two_ship_dead = false,
            has_active = false
        }
    end

    local primary = solutions[1]
    local has_active = Deduction.has_active_hits(solutions)
    local is_2_dead = Deduction.is_size_dead(solutions, 2)

    if #solutions == 1 then
        return {
            active = primary.active_hits,
            claimed = primary.claimed_pegs,
            dead_ships = primary.dead_ships,
            dead_sizes = primary.dead_sizes,
            solutions_count = 1,
            is_two_ship_dead = is_2_dead,
            has_active = has_active
        }
    end

    local active_bb = Bitboard.empty()
    local consensus_claimed_bb = Bitboard.from_coords(primary.claimed_pegs)

    for _, sol in ipairs(solutions) do
        for _, ah in ipairs(sol.active_hits) do
            active_bb:set(ah.r, ah.c)
        end
        consensus_claimed_bb = consensus_claimed_bb:band(Bitboard.from_coords(sol.claimed_pegs))
    end

    local consensus_dead_ships = {}
    for _, hull in ipairs(primary.dead_ships) do
        local trimmed = {}
        for _, pt in ipairs(hull) do
            if consensus_claimed_bb:is_set(pt.r, pt.c) then
                table.insert(trimmed, { r = pt.r, c = pt.c })
            end
        end
        if #trimmed > 0 then
            table.insert(consensus_dead_ships, trimmed)
        end
    end

    return {
        active = active_bb:to_coords(),
        claimed = consensus_claimed_bb:to_coords(),
        dead_ships = consensus_dead_ships,
        dead_sizes = primary.dead_sizes,
        solutions_count = #solutions,
        is_two_ship_dead = is_2_dead,
        has_active = has_active
    }
end

return Deduction

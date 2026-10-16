-- lua/test/test_game_engine.lua
-- Unit Tests for lua/game_engine.lua (Battleship Game Physics & Fog of War Rules)

local GameEngine = require("lua.game_engine")
local Bitboard = require("lua.bitboard")

local function assert_eq(actual, expected, msg)
    if actual ~= expected then
        error(string.format("ASSERTION FAILED: %s (Expected %s, got %s)",
            tostring(msg), tostring(expected), tostring(actual)), 2)
    end
end

local function assert_true(cond, msg)
    if not cond then
        error(string.format("ASSERTION FAILED: %s (Expected true, got %s)",
            tostring(msg), tostring(cond)), 2)
    end
end

print("=== Running lua/game_engine.lua Test Suite ===")

-- TEST 1: Initial Game Invariants
do
    local game = GameEngine.new()
    assert_eq(game.size, 10, "Grid size == 10")
    assert_eq(game.total_pegs, 17, "Total fleet pegs == 17")
    assert_eq(game.pegs_hit, 0, "Initial pegs hit == 0")
    assert_eq(game.shot_number, 0, "Initial shot number == 0")
    assert_true(not game:is_game_over(), "Game not over initially")
    assert_true(game.miss_bb:is_zero(), "Initial miss bitboard empty")
    assert_true(game.all_hits_bb:is_zero(), "Initial hits bitboard empty")

    for r = 0, 9 do
        for c = 0, 9 do
            assert_eq(game.truth_board[r][c], 0, "Truth board empty")
            assert_eq(game.tracking_board[r][c], GameEngine.CELL_UNKNOWN, "Tracking board unknown")
        end
    end
    print("[PASS] Test 1: Initial state invariants verified.")
end

-- TEST 2: Manual Ship Placement & Collision Bounds
do
    local game = GameEngine.new()

    -- Legal horizontal Carrier at (0, 0)
    assert_true(game:can_place(0, 0, 5, true), "Can place horizontal Carrier at (0,0)")
    assert_true(game:place_ship_manual(0, 0, 5, true, 1), "Place Carrier at (0,0)")

    -- Illegal placement: out of bounds horizontally
    assert_true(not game:can_place(0, 7, 5, true), "Carrier at (0,7) horizontal out of bounds")

    -- Illegal placement: out of bounds vertically
    assert_true(not game:can_place(7, 0, 5, false), "Carrier at (7,0) vertical out of bounds")

    -- Illegal placement: collision with existing Carrier
    assert_true(not game:can_place(0, 2, 3, false), "Vertical Cruiser through (0,2) collides with Carrier")

    -- Legal vertical Battleship at (2, 0)
    assert_true(game:can_place(2, 0, 4, false), "Can place vertical Battleship at (2,0)")
    assert_true(game:place_ship_manual(2, 0, 4, false, 2), "Place Battleship at (2,0)")

    -- Verify truth board
    for c = 0, 4 do
        assert_eq(game.truth_board[0][c], 1, "Carrier peg on truth board")
    end
    for r = 2, 5 do
        assert_eq(game.truth_board[r][0], 2, "Battleship peg on truth board")
    end

    print("[PASS] Test 2: Manual placement and collision checks verified.")
end

-- TEST 3: Random Placement (Complete Fleet of 5 Ships)
do
    local game = GameEngine.new()
    math.randomseed(12345)
    game:place_ships_randomly()

    local occupied_count = 0
    local ship_tallies = { [1] = 0, [2] = 0, [3] = 0, [4] = 0, [5] = 0 }

    for r = 0, 9 do
        for c = 0, 9 do
            local id = game.truth_board[r][c]
            if id > 0 then
                occupied_count = occupied_count + 1
                ship_tallies[id] = ship_tallies[id] + 1
            end
        end
    end

    assert_eq(occupied_count, 17, "Total occupied cells == 17")
    assert_eq(ship_tallies[1], 5, "Carrier has 5 pegs")
    assert_eq(ship_tallies[2], 4, "Battleship has 4 pegs")
    assert_eq(ship_tallies[3], 3, "Cruiser has 3 pegs")
    assert_eq(ship_tallies[4], 3, "Submarine has 3 pegs")
    assert_eq(ship_tallies[5], 2, "Destroyer has 2 pegs")

    print("[PASS] Test 3: Random fleet placement verified (all 17 pegs placed with zero overlaps).")
end

-- TEST 4: Strict Fog of War & Shot Firing Mechanics
do
    local game = GameEngine.new()
    -- Place Destroyer (size 2) at (4, 4) and (4, 5)
    assert_true(game:place_ship_manual(4, 4, 2, true, 5), "Place Destroyer")

    -- 1. Fire Miss
    local res1 = game:fire_shot(0, 0)
    assert_eq(res1, "MISS", "Water shot is MISS")
    assert_eq(game.tracking_board[0][0], GameEngine.CELL_MISS, "Tracking board is CELL_MISS")
    assert_true(game.miss_bb:is_set(0, 0), "miss_bb is set at (0,0)")
    assert_eq(game.shot_number, 1, "Shot number is 1")

    -- 2. Duplicate Shot (Invalid)
    local res_dup = game:fire_shot(0, 0)
    assert_eq(res_dup, "INVALID", "Duplicate shot is INVALID")
    assert_eq(game.shot_number, 1, "Invalid shot does not increment shot number")

    -- 3. Out of Bounds (Invalid)
    assert_eq(game:fire_shot(-1, 0), "INVALID", "Negative row is INVALID")
    assert_eq(game:fire_shot(0, 10), "INVALID", "Row 10 is INVALID")

    -- 4. Fire First Hit on Destroyer
    local res2 = game:fire_shot(4, 4)
    assert_eq(res2, "HIT", "First shot on Destroyer is HIT")
    assert_eq(game.tracking_board[4][4], GameEngine.CELL_HIT, "Tracking cell is CELL_HIT")
    assert_true(game.hit_bb:is_set(4, 4), "hit_bb is set at (4,4)")
    assert_true(game.all_hits_bb:is_set(4, 4), "all_hits_bb is set at (4,4)")
    assert_true(not game.sunk_bb:is_set(4, 4), "sunk_bb NOT set yet")
    assert_eq(game.pegs_hit, 1, "pegs_hit is 1")
    assert_eq(game.ship_health[5], 1, "Destroyer remaining health is 1")

    -- 5. Fire Lethal Shot on Destroyer (Sunk)
    local res3 = game:fire_shot(4, 5)
    assert_eq(res3, "SUNK", "Lethal shot on Destroyer is SUNK")
    assert_eq(game.tracking_board[4][5], GameEngine.CELL_SUNK, "Tracking cell (4,5) is CELL_SUNK")
    assert_true(game.sunk_bb:is_set(4, 5), "sunk_bb is set at (4,5)")
    assert_true(game.all_hits_bb:is_set(4, 5), "all_hits_bb is set at (4,5)")
    assert_eq(game.pegs_hit, 2, "pegs_hit is 2")
    assert_eq(game.ship_health[5], 0, "Destroyer health is 0")

    -- 6. STRICT FOG OF WAR INVARIANT:
    -- The previous hit at (4,4) MUST REMAIN CELL_HIT! It must NOT be converted to CELL_SUNK!
    assert_eq(game.tracking_board[4][4], GameEngine.CELL_HIT,
        "STRICT FOG OF WAR: Earlier hit at (4,4) must remain CELL_HIT!")

    print("[PASS] Test 4: Strict Fog of War, Hit/Miss/Sunk logic, and bitboard sync verified.")
end

-- TEST 5: Move Ledger and Undo Rollback
do
    local game = GameEngine.new()
    game:place_ship_manual(2, 2, 2, true, 5) -- Destroyer at (2,2), (2,3)

    game:fire_shot(0, 0) -- MISS
    game:fire_shot(2, 2) -- HIT
    game:fire_shot(2, 3) -- SUNK

    assert_eq(game.shot_number, 3, "Shot number == 3")
    assert_eq(game.pegs_hit, 2, "Pegs hit == 2")
    assert_eq(#game.move_history, 3, "Ledger length == 3")

    -- Undo SUNK
    assert_true(game:undo_last_shot(), "Undo SUNK")
    assert_eq(game.shot_number, 2, "Shot number rolled back to 2")
    assert_eq(game.pegs_hit, 1, "Pegs hit rolled back to 1")
    assert_eq(game.ship_health[5], 1, "Destroyer health restored to 1")
    assert_eq(game.tracking_board[2][3], GameEngine.CELL_UNKNOWN, "Cell (2,3) restored to UNKNOWN")
    assert_true(not game.sunk_bb:is_set(2, 3), "sunk_bb cleared for (2,3)")

    -- Undo HIT
    assert_true(game:undo_last_shot(), "Undo HIT")
    assert_eq(game.shot_number, 1, "Shot number rolled back to 1")
    assert_eq(game.pegs_hit, 0, "Pegs hit rolled back to 0")
    assert_eq(game.ship_health[5], 2, "Destroyer health restored to 2")
    assert_eq(game.tracking_board[2][2], GameEngine.CELL_UNKNOWN, "Cell (2,2) restored to UNKNOWN")
    assert_true(not game.all_hits_bb:is_set(2, 2), "all_hits_bb cleared for (2,2)")

    -- Undo MISS
    assert_true(game:undo_last_shot(), "Undo MISS")
    assert_eq(game.shot_number, 0, "Shot number rolled back to 0")
    assert_eq(game.tracking_board[0][0], GameEngine.CELL_UNKNOWN, "Cell (0,0) restored to UNKNOWN")
    assert_true(game.miss_bb:is_zero(), "miss_bb is empty")

    -- Empty undo
    assert_true(not game:undo_last_shot(), "Cannot undo empty ledger")

    print("[PASS] Test 5: Complete ledger rollback and state restoration verified.")
end

-- TEST 6: Full Game Simulation to Completion
do
    local game = GameEngine.new()
    math.randomseed(9999)
    game:place_ships_randomly()

    -- Exhaustively shoot all 100 cells
    local shots_fired = 0
    for r = 0, 9 do
        for c = 0, 9 do
            if not game:is_game_over() then
                local res = game:fire_shot(r, c)
                assert_true(res == "MISS" or res == "HIT" or res == "SUNK", "Valid shot result")
                shots_fired = shots_fired + 1
            end
        end
    end

    assert_true(game:is_game_over(), "Game is over after sinking all ships")
    assert_eq(game.pegs_hit, 17, "All 17 pegs hit")
    assert_eq(game.all_hits_bb:popcnt(), 17, "all_hits_bb has exactly 17 bits set")
    assert_eq(game.sunk_bb:popcnt(), 5, "Exactly 5 lethal SUNK pegs recorded (1 per ship)")
    assert_eq(game.hit_bb:popcnt(), 12, "Exactly 12 non-lethal HIT pegs recorded (17 - 5 = 12)")

    print(string.format("[PASS] Test 6: Full game play-through verified (won in %d shots, 5 lethal sunks).", shots_fired))
end

print("\n🎉 ALL GAME ENGINE TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

-- lua/test/test_deduction.lua
-- Unit Tests for lua/deduction.lua (Chronological Multi-Hypothesis CSP Solver)

local Deduction = require("lua.deduction")
local GameEngine = require("lua.game_engine")

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

print("=== Running lua/deduction.lua Test Suite ===")

-- TEST 1: No Sunks (All Active Hits)
do
    local deduction = Deduction.new()
    local game = GameEngine.new()
    game.tracking_board[2][2] = GameEngine.CELL_HIT
    game.tracking_board[2][3] = GameEngine.CELL_HIT

    local move_history = {
        { shotNumber = 1, x = 2, y = 2, result = "HIT" },
        { shotNumber = 2, x = 2, y = 3, result = "HIT" }
    }

    local solutions = deduction:analyze(game.tracking_board, move_history)
    assert_eq(#solutions, 1, "Exactly 1 solution when no sunks")
    assert_eq(#solutions[1].active_hits, 2, "Both hits are active")
    assert_eq(#solutions[1].dead_ships, 0, "No dead ships")
    assert_true(Deduction.has_active_hits(solutions), "has_active_hits is true")
    assert_true(not Deduction.is_size_dead(solutions, 2), "2-ship is not dead")

    print("[PASS] Test 1: No-sunk active hits isolation verified.")
end

-- TEST 2: Clean Single Sunk (Destroyer Sunk)
do
    local deduction = Deduction.new()
    local game = GameEngine.new()
    game.tracking_board[4][4] = GameEngine.CELL_HIT
    game.tracking_board[4][5] = GameEngine.CELL_SUNK

    local move_history = {
        { shotNumber = 1, x = 4, y = 4, result = "HIT" },
        { shotNumber = 2, x = 4, y = 5, result = "SUNK" }
    }

    local solutions = deduction:analyze(game.tracking_board, move_history)
    assert_true(#solutions >= 1, "At least 1 solution found")

    local primary = solutions[1]
    assert_eq(#primary.dead_ships, 1, "Exactly 1 dead ship deduced")
    assert_eq(#primary.dead_ships[1], 2, "Dead ship has length 2")
    assert_eq(#primary.active_hits, 0, "Zero active hits remaining")
    assert_true(Deduction.is_size_dead(solutions, 2), "2-ship is confirmed dead across solutions")
    assert_true(not Deduction.has_active_hits(solutions), "has_active_hits is false")

    print("[PASS] Test 2: Clean single ship deduction verified (Destroyer dead, 0 active hits).")
end

-- TEST 3: Sunk Ship + Active Hits on Surviving Wounded Ship
do
    local deduction = Deduction.new()
    local game = GameEngine.new()

    -- Destroyer sunk at (1,1)-(1,2)
    game.tracking_board[1][1] = GameEngine.CELL_HIT
    game.tracking_board[1][2] = GameEngine.CELL_SUNK

    -- Battleship wounded at (6,6)-(6,7)
    game.tracking_board[6][6] = GameEngine.CELL_HIT
    game.tracking_board[6][7] = GameEngine.CELL_HIT

    local move_history = {
        { shotNumber = 1, x = 1, y = 1, result = "HIT" },
        { shotNumber = 2, x = 1, y = 2, result = "SUNK" },
        { shotNumber = 3, x = 6, y = 6, result = "HIT" },
        { shotNumber = 4, x = 6, y = 7, result = "HIT" }
    }

    local solutions = deduction:analyze(game.tracking_board, move_history)
    assert_true(#solutions >= 1, "Solution found")

    local primary = solutions[1]
    assert_eq(#primary.dead_ships, 1, "Exactly 1 dead ship deduced")
    assert_eq(#primary.active_hits, 2, "Exactly 2 active hits isolated on surviving wounded ship")
    assert_true(Deduction.is_size_dead(solutions, 2), "2-ship is confirmed dead")
    assert_true(Deduction.has_active_hits(solutions), "Active hits properly detected")

    print("[PASS] Test 3: Sunk corpse locking and active hit isolation verified.")
end

-- TEST 4: Chronological Causality Invariant
do
    local deduction = Deduction.new()
    local game = GameEngine.new()

    -- Suppose cell (3,3) was hit at t=1, and sunk at (3,4) at t=2
    -- Later at t=5, we hit (3,5) which happens to be collinear with the dead Destroyer!
    game.tracking_board[3][3] = GameEngine.CELL_HIT
    game.tracking_board[3][4] = GameEngine.CELL_SUNK
    game.tracking_board[3][5] = GameEngine.CELL_HIT

    local move_history = {
        { shotNumber = 1, x = 3, y = 3, result = "HIT" },
        { shotNumber = 2, x = 3, y = 4, result = "SUNK" },
        { shotNumber = 5, x = 3, y = 5, result = "HIT" }
    }

    local solutions = deduction:analyze(game.tracking_board, move_history)
    assert_true(#solutions >= 1, "Solution found")

    -- Cell (3,5) was hit AFTER the sunk shot at t=2.
    -- Therefore, it is CAUSALLY IMPOSSIBLE for (3,5) to be part of the ship sunk at t=2!
    -- It MUST be classified as an active hit!
    local primary = solutions[1]
    local found_3_5_active = false
    for _, pt in ipairs(primary.active_hits) do
        if pt.r == 3 and pt.c == 5 then found_3_5_active = true end
    end
    assert_true(found_3_5_active, "Cell (3,5) fired at t=5 must be classified as active hit, not corpse")

    print("[PASS] Test 4: Chronological causality invariant verified (post-sunk hits cannot be part of earlier corpse).")
end

-- TEST 5: Performance & Speed (Solving 1,000 Complex Board States)
do
    local deduction = Deduction.new()
    local game = GameEngine.new()
    game.tracking_board[0][0] = GameEngine.CELL_HIT
    game.tracking_board[0][1] = GameEngine.CELL_SUNK
    game.tracking_board[5][5] = GameEngine.CELL_HIT
    game.tracking_board[5][6] = GameEngine.CELL_HIT

    local moves = {
        { shotNumber = 1, x = 0, y = 0, result = "HIT" },
        { shotNumber = 2, x = 0, y = 1, result = "SUNK" },
        { shotNumber = 3, x = 5, y = 5, result = "HIT" },
        { shotNumber = 4, x = 5, y = 6, result = "HIT" }
    }

    local t0 = os.clock()
    local iters = 1000
    for i = 1, iters do
        local sols = deduction:analyze(game.tracking_board, moves)
    end
    local elapsed = os.clock() - t0
    local ms_per_solve = (elapsed / iters) * 1000

    print(string.format("[BENCHMARK] 1,000 CSP board deductions: %.4f s (%.3f ms per solve, %.0f solves/sec)",
        elapsed, ms_per_solve, iters / elapsed))
    assert_true(ms_per_solve < 1.0, "CSP solve must take less than 1.0 ms")
end

print("\n🎉 ALL DEDUCTION CSP TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

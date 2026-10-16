-- lua/test/test_combinatorics.lua
-- Unit Tests and Touching-Ships Verification for lua/combinatorics.lua

local Combinatorics = require("lua.combinatorics")
local Bitboard = require("lua.bitboard")
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

print("=== Running lua/combinatorics.lua Test Suite ===")

local comb = Combinatorics.new()

-- TEST 1: Precomputed 760-Placement Library
do
    local p5 = comb:get_placements(5)
    local p4 = comb:get_placements(4)
    local p3 = comb:get_placements(3)
    local p2 = comb:get_placements(2)

    assert_eq(#p5, 120, "Carrier size 5 has 120 placements (60 horiz + 60 vert)")
    assert_eq(#p4, 140, "Battleship size 4 has 140 placements (70 horiz + 70 vert)")
    assert_eq(#p3, 160, "Cruiser size 3 has 160 placements (80 horiz + 80 vert)")
    assert_eq(#p2, 180, "Destroyer size 2 has 180 placements (90 horiz + 90 vert)")

    -- Total fleet slots = 120 + 140 + 160 + 160 + 180 = 760
    local total_slots = #p5 + #p4 + (#p3 * 2) + #p2
    assert_eq(total_slots, 760, "Total fleet placement slots == 760")

    -- Verify popcount of every mask
    for _, p in ipairs(p5) do assert_eq(p.mask:popcnt(), 5, "Size 5 popcount") end
    for _, p in ipairs(p4) do assert_eq(p.mask:popcnt(), 4, "Size 4 popcount") end
    for _, p in ipairs(p3) do assert_eq(p.mask:popcnt(), 3, "Size 3 popcount") end
    for _, p in ipairs(p2) do assert_eq(p.mask:popcnt(), 2, "Size 2 popcount") end

    print("[PASS] Test 1: Precomputed 760-placement library verified (100% popcount accuracy).")
end

-- TEST 2: Hunt Mode Density
do
    local empty_forbidden = Bitboard.empty()
    local density, max_d = comb:compute_hunt_density({ 5, 4, 3, 3, 2 }, empty_forbidden)

    -- Center squares have maximum density on empty board
    assert_true(density[4][4] == max_d or density[5][5] == max_d, "Center cells have peak density")
    -- Corners have minimum density
    assert_true(density[0][0] < (max_d / 2), "Corner density is much lower than center")

    -- Add forbidden miss at center (4, 4)
    local forbidden = Bitboard.from_coord(4, 4)
    local d2, max_d2 = comb:compute_hunt_density({ 5, 4, 3, 3, 2 }, forbidden)
    assert_eq(d2[4][4], 0, "Forbidden miss cell has 0 density")

    print("[PASS] Test 2: Hunt mode density and miss exclusion verified.")
end

-- TEST 3: Target Mode Phase 1 — Single-Ship Exact Cover (P1)
do
    local game = GameEngine.new()
    -- Hits at (3, 3) and (3, 4)
    game.tracking_board[3][3] = GameEngine.CELL_HIT
    game.tracking_board[3][4] = GameEngine.CELL_HIT

    local active_hits_bb = Bitboard.from_coords({{3, 3}, {3, 4}})
    local forbidden_bb = Bitboard.empty()

    -- Available fleet: only Cruiser (size 3)
    local counts, max_c = comb:compute_target_density(game.tracking_board, { 3 }, forbidden_bb, active_hits_bb)

    -- Placements covering (3,3)-(3,4) must be (3,2)-(3,4) and (3,3)-(3,5)
    -- So unknown cells (3, 2) and (3, 5) must receive counts!
    local idx_left = 3 * 10 + 2
    local idx_right = 3 * 10 + 5

    assert_true(counts[idx_left] ~= nil and counts[idx_left] > 0, "Cell (3,2) is a valid candidate")
    assert_true(counts[idx_right] ~= nil and counts[idx_right] > 0, "Cell (3,5) is a valid candidate")

    -- Cells (3,3) and (3,4) are already HIT, so they must have 0 candidate count
    assert_eq(counts[3 * 10 + 3] or 0, 0, "Already-hit cell (3,3) has 0 target count")
    assert_eq(counts[3 * 10 + 4] or 0, 0, "Already-hit cell (3,4) has 0 target count")

    print("[PASS] Test 3: Single-ship exact cover (P1) target density verified.")
end

-- TEST 4: Target Mode Phase 2 — Touching Ships Disjoint Cover (P1, P2)
do
    local game = GameEngine.new()
    -- L-shape hits:
    -- Horizontal Battleship hits at (3, 3), (3, 4), (3, 5)
    -- Vertical Destroyer hit at (4, 4) touching it!
    game.tracking_board[3][3] = GameEngine.CELL_HIT
    game.tracking_board[3][4] = GameEngine.CELL_HIT
    game.tracking_board[3][5] = GameEngine.CELL_HIT
    game.tracking_board[4][4] = GameEngine.CELL_HIT

    local active_hits_bb = Bitboard.from_coords({{3, 3}, {3, 4}, {3, 5}, {4, 4}})
    local forbidden_bb = Bitboard.empty()

    -- Fleet has Battleship (4) and Destroyer (2)
    local counts, max_c = comb:compute_target_density(game.tracking_board, { 4, 2 }, forbidden_bb, active_hits_bb)

    -- No single ship can cover an L-shape!
    -- Disjoint joint cover must find:
    --   P1: Battleship horizontal at (3,2)-(3,5) or (3,3)-(3,6)
    --   P2: Destroyer vertical at (4,4)-(5,4)
    local idx_left_battleship = 3 * 10 + 2
    local idx_right_battleship = 3 * 10 + 6
    local idx_destroyer_tail = 5 * 10 + 4

    assert_true(counts[idx_left_battleship] ~= nil and counts[idx_left_battleship] > 0,
        "L-shape solver: candidate at (3,2) detected")
    assert_true(counts[idx_right_battleship] ~= nil and counts[idx_right_battleship] > 0,
        "L-shape solver: candidate at (3,6) detected")
    assert_true(counts[idx_destroyer_tail] ~= nil and counts[idx_destroyer_tail] > 0,
        "L-shape solver: candidate at (5,4) detected")

    print("[PASS] Test 4: Touching ships L-shape disjoint cover (P1, P2) verified with zero errors!")
end

-- TEST 5: Microbenchmark (10,000 Target Density Solves)
do
    local game = GameEngine.new()
    game.tracking_board[4][4] = GameEngine.CELL_HIT
    game.tracking_board[4][5] = GameEngine.CELL_HIT
    local hits_bb = Bitboard.from_coords({{4, 4}, {4, 5}})
    local forbidden_bb = Bitboard.empty()

    local t0 = os.clock()
    local iters = 10000
    for i = 1, iters do
        local counts, max_c = comb:compute_target_density(game.tracking_board, { 5, 4, 3, 3, 2 }, forbidden_bb, hits_bb)
    end
    local elapsed = os.clock() - t0
    local ms_per_solve = (elapsed / iters) * 1000

    print(string.format("[BENCHMARK] 10,000 target density calculations: %.4f s (%.3f ms per solve, %.0f solves/sec)",
        elapsed, ms_per_solve, iters / elapsed))
    assert_true(ms_per_solve < 0.1, "Target solve must be under 0.1 ms")
end

print("\n🎉 ALL COMBINATORICS TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

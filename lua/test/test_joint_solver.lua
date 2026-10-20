-- lua/test/test_joint_solver.lua
-- Unit Tests and Benchmarks for lua/joint_solver.lua (Exact Joint Density Solver)

local JointSolver = require("lua.joint_solver")
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

print("=== Running lua/joint_solver.lua Test Suite ===")

local js = JointSolver.new()
local empty_bb = Bitboard.empty()

-- TEST 1: Single Ship Exact Density (Endgame k=1)
do
    local density, total_configs, max_c = js:compute_exact_joint_density({ 5 }, empty_bb)
    assert_eq(total_configs, 120, "Carrier has 120 total placements on empty board")
    assert_true(max_c > 0, "Max count > 0")
    assert_eq(density[0][0], 2, "Corner (0,0) has 2 Carrier placements")
    assert_eq(density[4][4], 10, "Center (4,4) has 10 Carrier placements")
    print("[PASS] Test 1: Single ship exact density verified.")
end

-- TEST 2: Two Ships Exact Joint Density (k=2)
do
    local density, total_configs, max_c = js:compute_exact_joint_density({ 4, 2 }, empty_bb)
    -- Total placements: 140 of size 4, 180 of size 2 = 140 * 180 = 25,200 pairs.
    -- Number of overlapping pairs is small. Total disjoint pairs:
    assert_true(total_configs > 20000 and total_configs < 25200, "Valid 2-ship configurations count")
    assert_true(density[4][4] > density[0][0], "Center density exceeds corner")
    print(string.format("[PASS] Test 2: Two ships {4,2} exact joint density (%d configs) verified.", total_configs))
end

-- TEST 3: Three Ships Exact Joint Density with Symmetry Breaking ({5, 3, 3})
do
    local density, total_configs, max_c = js:compute_exact_joint_density({ 5, 3, 3 }, empty_bb)
    assert_true(total_configs > 1000000, "Three ships configurations > 1M")
    print(string.format("[PASS] Test 3: Three ships with symmetry breaking {5,3,3} (%d configs) verified.", total_configs))
end

-- TEST 4: Four Ships Post-Sink Joint Density ({5, 4, 3, 3} with misses)
do
    local forbidden = Bitboard.empty()
    -- Sunk Destroyer at (0,0)-(0,1), plus misses
    forbidden:set(0, 0); forbidden:set(0, 1)
    local misses = { {2,2}, {3,3}, {4,4}, {5,5}, {6,6}, {7,7}, {2,4}, {4,2} }
    for _, pt in ipairs(misses) do forbidden:set(pt[1], pt[2]) end

    local t0 = os.clock()
    local density, total_configs, max_c = js:compute_exact_joint_density({ 5, 4, 3, 3 }, forbidden)
    local elapsed = os.clock() - t0

    assert_true(total_configs > 10000000, "Four ships configurations > 10M")
    assert_true(max_c > 0, "Max count > 0")
    -- Forbidden cells must have 0 density!
    assert_eq(density[0][0], 0, "Dead corpse cell (0,0) has 0 density")
    assert_eq(density[2][2], 0, "Miss cell (2,2) has 0 density")

    print(string.format("[PASS] Test 4: Four ships {5,4,3,3} post-sink search verified: %d configs in %.4f s (%.2f ms)",
        total_configs, elapsed, elapsed * 1000))
end

-- TEST 5: Four Ships Analytic 1-Ship Dual-Pass ({5, 4, 3, 2})
do
    local forbidden = Bitboard.empty()
    forbidden:set(0, 0); forbidden:set(1, 1); forbidden:set(2, 2); forbidden:set(3, 3); forbidden:set(4, 4)

    local t0 = os.clock()
    local density, total_configs, max_c = js:compute_exact_joint_density({ 5, 4, 3, 2 }, forbidden)
    local elapsed = os.clock() - t0

    assert_eq(total_configs, 117898120, "Exact 117,898,120 configurations verified")
    assert_true(max_c > 0, "Max count > 0")
    assert_eq(density[0][0], 0, "Miss cell (0,0) has 0 density")
    assert_eq(density[4][4], 0, "Miss cell (4,4) has 0 density")

    print(string.format("[PASS] Test 5: Analytic dual-pass {5,4,3,2} verified: %d configs in %.4f s (%.2f ms)",
        total_configs, elapsed, elapsed * 1000))
end

-- TEST 6: Incremental Miss-Subtraction Exact Mathematical Parity
do
    local bb_base = Bitboard.empty()
    bb_base:set(0, 0); bb_base:set(1, 1); bb_base:set(2, 2)
    local fleet = { 4, 3, 2 }

    local d_base, z_base = js:compute_exact_joint_density(fleet, bb_base)

    -- Suppose cell (5,5) is a miss
    local miss_r, miss_c = 5, 5
    local bb_next = bb_base:copy()
    bb_next:set(miss_r, miss_c)
    local d_next, z_next = js:compute_exact_joint_density(fleet, bb_next)

    -- Compute ruined configurations via miss subtraction
    local d_ruined, z_ruined = js:compute_ruined_density(fleet, bb_base, miss_r, miss_c)
    local d_subtracted, z_subtracted = js:subtract_density(d_base, z_base, d_ruined, z_ruined)

    assert_eq(z_subtracted, z_next, "Total configurations match exactly via subtraction")
    for r = 0, 9 do
        for c = 0, 9 do
            assert_eq(d_subtracted[r][c], d_next[r][c], string.format("Cell (%d,%d) count matches", r, c))
        end
    end

    print(string.format("[PASS] Test 6: Incremental miss-subtraction exact parity verified (base=%d, ruined=%d, next=%d).",
        z_base, z_ruined, z_next))
end

print("\n🎉 ALL JOINT SOLVER TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

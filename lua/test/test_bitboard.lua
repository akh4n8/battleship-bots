-- lua/test/test_bitboard.lua
-- Unit Tests and Microbenchmarks for lua/bitboard.lua

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

print("=== Running lua/bitboard.lua Test Suite ===")

-- TEST 1: Empty Bitboard
do
    local bb = Bitboard.empty()
    assert_true(bb:is_zero(), "Empty bitboard is_zero()")
    assert_eq(bb:popcnt(), 0, "Empty bitboard popcnt() == 0")
    for r = 0, 9 do
        for c = 0, 9 do
            assert_true(not bb:is_set(r, c), string.format("Cell (%d,%d) should be 0", r, c))
        end
    end
    print("[PASS] Test 1: Empty bitboard invariants verified.")
end

-- TEST 2: Boundary Bits (Index 0, 63, 64, 99)
do
    local bb = Bitboard.empty()
    -- (0, 0) => idx 0 (lo bit 0)
    bb:set(0, 0)
    assert_true(bb:is_set(0, 0), "Set (0,0)")
    assert_eq(bb:popcnt(), 1, "Popcnt after setting (0,0)")

    -- (6, 3) => idx 6*10 + 3 = 63 (highest bit in lo)
    bb:set(6, 3)
    assert_true(bb:is_set(6, 3), "Set (6,3) [bit 63]")
    assert_eq(bb:popcnt(), 2, "Popcnt after setting (6,3)")

    -- (6, 4) => idx 6*10 + 4 = 64 (lowest bit in hi)
    bb:set(6, 4)
    assert_true(bb:is_set(6, 4), "Set (6,4) [bit 64]")
    assert_eq(bb:popcnt(), 3, "Popcnt after setting (6,4)")

    -- (9, 9) => idx 9*10 + 9 = 99 (highest bit in hi, bit 35)
    bb:set(9, 9)
    assert_true(bb:is_set(9, 9), "Set (9,9) [bit 99]")
    assert_eq(bb:popcnt(), 4, "Popcnt after setting (9,9)")

    -- Test clearing
    bb:clear(6, 3)
    assert_true(not bb:is_set(6, 3), "Cleared (6,3)")
    assert_eq(bb:popcnt(), 3, "Popcnt after clearing (6,3)")

    bb:clear(6, 4)
    assert_true(not bb:is_set(6, 4), "Cleared (6,4)")
    assert_eq(bb:popcnt(), 2, "Popcnt after clearing (6,4)")

    print("[PASS] Test 2: Boundary bit manipulations (idx 0, 63, 64, 99) verified.")
end

-- TEST 3: Full Bitboard
do
    local full = Bitboard.full()
    assert_eq(full:popcnt(), 100, "Full bitboard popcnt() == 100")
    for r = 0, 9 do
        for c = 0, 9 do
            assert_true(full:is_set(r, c), string.format("Full cell (%d,%d) must be set", r, c))
        end
    end

    local not_full = full:bnot()
    assert_true(not_full:is_zero(), "bnot(full) must be empty")
    assert_eq(not_full:popcnt(), 0, "bnot(full) popcnt == 0")
    print("[PASS] Test 3: Full bitboard and bnot() verified.")
end

-- TEST 4: Bitwise Logic (AND, OR, XOR, Intersects, ContainsAll)
do
    local a = Bitboard.from_coords({{0, 0}, {0, 1}, {0, 2}})
    local b = Bitboard.from_coords({{0, 1}, {0, 2}, {0, 3}})

    -- band
    local a_and_b = a:band(b)
    assert_eq(a_and_b:popcnt(), 2, "a & b popcount")
    assert_true(a_and_b:is_set(0, 1), "a & b has (0,1)")
    assert_true(a_and_b:is_set(0, 2), "a & b has (0,2)")
    assert_true(not a_and_b:is_set(0, 0), "a & b does not have (0,0)")

    -- bor
    local a_or_b = a:bor(b)
    assert_eq(a_or_b:popcnt(), 4, "a | b popcount")

    -- bxor
    local a_xor_b = a:bxor(b)
    assert_eq(a_xor_b:popcnt(), 2, "a ^ b popcount")
    assert_true(a_xor_b:is_set(0, 0), "a ^ b has (0,0)")
    assert_true(a_xor_b:is_set(0, 3), "a ^ b has (0,3)")

    -- intersects
    assert_true(a:intersects(b), "a intersects b")
    local c = Bitboard.from_coords({{5, 5}, {5, 6}})
    assert_true(not a:intersects(c), "a does NOT intersect c")

    -- contains_all
    assert_true(a_or_b:contains_all(a), "a | b contains all of a")
    assert_true(a_or_b:contains_all(b), "a | b contains all of b")
    assert_true(not a:contains_all(b), "a does not contain all of b")

    print("[PASS] Test 4: Bitwise operations (AND, OR, XOR, intersects, contains_all) verified.")
end

-- TEST 5: Coordinates and Serialization
do
    local input_coords = {{1, 2}, {3, 4}, {5, 6}, {7, 8}}
    local bb = Bitboard.from_coords(input_coords)
    assert_eq(bb:popcnt(), 4, "from_coords popcnt")

    local output_coords = bb:to_coords()
    assert_eq(#output_coords, 4, "to_coords length")

    for _, pt in ipairs(output_coords) do
        assert_true(bb:is_set(pt.r, pt.c), "Recovered coordinate is set")
    end

    local grid_str = bb:to_grid_string()
    assert_true(#grid_str > 100, "to_grid_string returns formatted grid")
    print("[PASS] Test 5: Coordinates conversion and ASCII string formatting verified.")
end

-- TEST 6: High-Speed Microbenchmark (10,000,000 Operations)
do
    local a = Bitboard.from_coords({{1, 1}, {2, 2}, {3, 3}, {7, 7}, {8, 8}})
    local b = Bitboard.from_coords({{3, 3}, {4, 4}, {7, 7}, {9, 9}})
    local iters = 10000000

    local t0 = os.clock()
    local dummy = 0
    for i = 1, iters do
        if a:intersects(b) then
            dummy = dummy + 1
        end
    end
    local elapsed = os.clock() - t0
    local ns_per_op = (elapsed / iters) * 1e9

    print(string.format("[BENCHMARK] 10,000,000 intersects() checks: %.4f s (%.2f ns/op)", elapsed, ns_per_op))
    assert_true(ns_per_op < 5.0, "Intersects benchmark should be well under 5 ns per op")
end

print("\n🎉 ALL BITBOARD TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

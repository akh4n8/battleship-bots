-- lua/test/test_base_bot.lua
-- Unit Tests for lua/bots/base_bot.lua (Abstract Base Contract & Inheritance)

local BaseBot = require("lua.bots.base_bot")

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

print("=== Running lua/bots/base_bot.lua Test Suite ===")

-- TEST 1: Abstract Method Enforcement
do
    local base = BaseBot.new("TestBase")
    local ok, err = pcall(function()
        base:get_best_move({}, {})
    end)

    assert_true(not ok, "Calling abstract get_best_move on BaseBot must throw error")
    assert_true(string.find(err, "Abstract method 'get_best_move' must be implemented"),
        "Error message should mention abstract method")

    print("[PASS] Test 1: Abstract method enforcement verified.")
end

-- TEST 2: Inheritance via BaseBot:extend()
do
    local MockBot = BaseBot:extend("MockBot")

    function MockBot:init(parity_offset, test_val)
        self.test_val = test_val or 42
    end

    function MockBot:get_best_move(board, move_history)
        local unrevealed = self:get_unrevealed_cells(board)
        local target = unrevealed[1] or { x = 0, y = 0 }
        return target, "Mock: First Available Shot", {}, { active = {} }
    end

    local bot = MockBot.new(1, 999)
    assert_eq(bot.name, "MockBot", "Subclass name preserved")
    assert_eq(bot.test_val, 999, "Subclass init hook executed")

    local dummy_board = bot:create_blank_matrix(0)
    local target, log, dead_sizes, diag = bot:get_best_move(dummy_board, {})

    assert_eq(target.x, 0, "Target x")
    assert_eq(target.y, 0, "Target y")
    assert_eq(log, "Mock: First Available Shot", "Log rationale")
    assert_true(type(dead_sizes) == "table", "dead_sizes is table")
    assert_true(type(diag) == "table", "diagnostics is table")

    print("[PASS] Test 2: Subclass extension, constructor chaining, and method dispatch verified.")
end

-- TEST 3: Deterministic Parity and Reset Protocol
do
    local MockBot = BaseBot:extend("ParityBot")
    function MockBot:get_best_move(b, m) return {x=0,y=0}, "", {}, {} end

    local bot = MockBot.new()

    -- Reset with even game_id
    bot:reset(100)
    assert_eq(bot.parity_offset, 0, "Game 100 has parity 0")

    -- Reset with odd game_id
    bot:reset(101)
    assert_eq(bot.parity_offset, 1, "Game 101 has parity 1")

    -- Reset with explicit parity override
    bot:reset(100, 1)
    assert_eq(bot.parity_offset, 1, "Explicit parity overrides game_id")

    print("[PASS] Test 3: Reset protocol and parity offset derivation verified.")
end

-- TEST 4: Shared Utilities (Heatmap normalization, blank matrix, unrevealed cells)
do
    local base = BaseBot.new()

    -- 1. Blank matrix
    local mat = base:create_blank_matrix(-1)
    for r = 0, 9 do
        for c = 0, 9 do
            assert_eq(mat[r][c], -1, "Blank matrix cell matches default value")
        end
    end

    -- 2. Heatmap normalization
    local raw = base:create_blank_matrix(0)
    raw[3][4] = 50
    raw[5][5] = 25
    local norm = base:normalize_heatmap(raw)
    assert_eq(norm[3][4], 100, "Max value normalized to 100")
    assert_eq(norm[5][5], 50, "Half-max value normalized to 50")
    assert_eq(norm[0][0], 0, "Zero value normalized to 0")

    -- 3. Unrevealed cells extractor
    local board = base:create_blank_matrix(0)
    board[0][0] = -1 -- MISS
    board[0][1] = 1  -- HIT
    local unrevealed = base:get_unrevealed_cells(board)
    assert_eq(#unrevealed, 98, "100 - 2 revealed cells == 98 unrevealed")

    -- 4. Bounds check
    assert_true(base:is_valid_cell(0, 0), "(0,0) is valid")
    assert_true(base:is_valid_cell(9, 9), "(9,9) is valid")
    assert_true(not base:is_valid_cell(-1, 0), "(-1,0) is invalid")
    assert_true(not base:is_valid_cell(0, 10), "(0,10) is invalid")

    print("[PASS] Test 4: Shared mathematical utilities verified.")
end

-- TEST 5: Dynamic Stride Progression (2 -> 3 -> 4 -> 5)
do
    local base = BaseBot.new()

    -- 1. All ships alive -> smallest is 2
    assert_eq(base:get_dynamic_stride({}), 2, "All alive -> stride 2")

    -- 2. 2-ship dead -> smallest is 3
    assert_eq(base:get_dynamic_stride({2}), 3, "2-ship dead -> stride 3")

    -- 3. 2-ship and ONE 3-ship dead -> second 3-ship still alive! Smallest is still 3
    assert_eq(base:get_dynamic_stride({2, 3}), 3, "2 and one 3 dead -> stride 3")

    -- 4. 2-ship and BOTH 3-ships dead -> smallest is Battleship (4)!
    assert_eq(base:get_dynamic_stride({2, 3, 3}), 4, "2 and both 3s dead -> stride 4")

    -- 5. 2-ship, both 3-ships, and 4-ship dead -> smallest is Carrier (5)!
    assert_eq(base:get_dynamic_stride({2, 3, 3, 4}), 5, "Only Carrier alive -> stride 5")

    -- 6. Verify stride covering nets guarantee hitting ships of length S
    for _, S in ipairs({2, 3, 4, 5}) do
        -- Horizontal ships
        for r = 0, 9 do
            for c = 0, 10 - S do
                local hit = false
                for i = 0, S - 1 do
                    if base:is_cell_on_stride(r, c + i, S, 0) then
                        hit = true
                        break
                    end
                end
                assert_true(hit, string.format("Horizontal ship of length %d must hit stride net", S))
            end
        end
        -- Vertical ships
        for c = 0, 9 do
            for r = 0, 10 - S do
                local hit = false
                for i = 0, S - 1 do
                    if base:is_cell_on_stride(r + i, c, S, 0) then
                        hit = true
                        break
                    end
                end
                assert_true(hit, string.format("Vertical ship of length %d must hit stride net", S))
            end
        end
    end

    print("[PASS] Test 5: Dynamic stride progression (2 -> 3 -> 4 -> 5) and covering nets verified.")
end

print("\n🎉 ALL BASE BOT TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

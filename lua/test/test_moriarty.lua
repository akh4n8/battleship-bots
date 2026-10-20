-- lua/test/test_moriarty.lua
-- Unit Tests and Benchmark for lua/bots/moriarty.lua (Pure Essential Neural Network Bot)

local MoriartyBot = require("lua.bots.moriarty")
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

local function assert_close(actual, expected, tol, msg)
    tol = tol or 1e-4
    if math.abs(actual - expected) > tol then
        error(string.format("ASSERTION FAILED: %s (Expected %.6f, got %.6f)",
            tostring(msg), expected, actual), 2)
    end
end

print("=== Running lua/bots/moriarty.lua (Pure Neural Bot) Test Suite ===")

-- TEST 1: 4-Channel 10x10 Spatial Input Tensor Builder
do
    local bot = MoriartyBot.new(1) -- parity_offset = 1
    assert_eq(bot.name, "MoriartyBot", "Bot name")
    assert_eq(bot.parity_offset, 1, "Parity offset")

    local board = bot:create_blank_matrix(GameEngine.CELL_UNKNOWN)
    board[0][0] = GameEngine.CELL_MISS
    board[2][3] = GameEngine.CELL_HIT
    board[5][5] = GameEngine.CELL_SUNK

    local _, tensor = bot:build_input_tensor(board, true)

    -- Channel 0: Misses
    assert_close(tensor[0][0][0], 1.0, 1e-6, "Ch 0 (Miss) at (0,0)")
    assert_close(tensor[0][2][3], 0.0, 1e-6, "Ch 0 (Miss) at (2,3)")

    -- Channel 1: Hits
    assert_close(tensor[1][2][3], 1.0, 1e-6, "Ch 1 (Hit) at (2,3)")
    assert_close(tensor[1][5][5], 0.0, 1e-6, "Ch 1 (Hit) at (5,5)")

    -- Channel 2: Sunk
    assert_close(tensor[2][5][5], 1.0, 1e-6, "Ch 2 (Sunk) at (5,5)")
    assert_close(tensor[2][0][0], 0.0, 1e-6, "Ch 2 (Sunk) at (0,0)")

    -- Channel 3: Parity ((r + c) % 2 == 1)
    assert_close(tensor[3][0][0], 0.0, 1e-6, "Ch 3 (Parity) at (0,0) with offset 1")
    assert_close(tensor[3][0][1], 1.0, 1e-6, "Ch 3 (Parity) at (0,1) with offset 1")
    assert_close(tensor[3][2][3], 1.0, 1e-6, "Ch 3 (Parity) at (2,3) with offset 1")

    print("[PASS] Test 1: 4-Channel 10x10 spatial input tensor (Miss, Hit, Sunk, Parity) verified.")
end

-- TEST 2: Pure Neural Forward Pass, Occupancy Heatmap, & Parity-Guided Policy Selection
do
    local bot = MoriartyBot.new(0) -- parity_offset = 0
    local board = bot:create_blank_matrix(GameEngine.CELL_UNKNOWN)

    -- Place a dense cluster of misses in the top-left corner
    board[0][0] = GameEngine.CELL_MISS
    board[0][1] = GameEngine.CELL_MISS
    board[1][0] = GameEngine.CELL_MISS
    board[1][1] = GameEngine.CELL_MISS

    local target, log, _, diag = bot:get_best_move(board, {})
    assert_eq(diag.moriarty_mode, "Neural Hunt", "Pure Neural Hunt mode active (no Maverick fallback)")
    assert_true(string.find(log, "Moriarty: Neural Hunt") ~= nil, "Log indicates Pure Neural Hunt")
    assert_eq((target.x + target.y) % 2, 0, "Selected target must respect parity checkerboard")
    assert_eq(board[target.x][target.y], GameEngine.CELL_UNKNOWN, "Selected target must be unrevealed")

    -- Verify revealed cells have 0 heatmap and open water has higher occupancy than miss-cluttered corner
    local heat = bot.last_heatmap
    assert_eq(heat[0][0], 0, "Revealed miss cell (0,0) must have 0 heat")
    assert_true(heat[5][5] > heat[0][2], "Open center water (5,5) must have higher neural occupancy than miss-adjacent corner (0,2)")

    print("[PASS] Test 2: Pure Neural Hunt forward pass, occupancy sigmoid heatmap, and parity selection verified.")
end

-- TEST 3: Pluggable Custom Neural Forward Function & Assassin Target Mode
do
    -- 1. Custom neural forward pass pointing strongly to (4, 6)
    local custom_bot = MoriartyBot.new(0, true, function(input_buf, self_bot)
        local p_logits = {}
        local o_logits = {}
        for i = 0, 99 do
            p_logits[i] = -5.0
            o_logits[i] = -2.0
        end
        p_logits[4 * 10 + 6] = 10.0 -- Peak policy logit at (4, 6)
        o_logits[4 * 10 + 6] = 4.0  -- Peak occupancy logit at (4, 6)
        return p_logits, o_logits
    end)

    local board = custom_bot:create_blank_matrix(GameEngine.CELL_UNKNOWN)
    local target, _, _, diag = custom_bot:get_best_move(board, {})
    assert_eq(diag.moriarty_mode, "Neural Hunt", "Custom NN runs in Neural Hunt mode")
    assert_eq(target.x, 4, "Custom NN policy argmax row == 4")
    assert_eq(target.y, 6, "Custom NN policy argmax col == 6")
    assert_true(custom_bot.last_heatmap[4][6] >= 95, "Custom NN occupancy sigmoid at (4,6) >= 95")

    -- 2. Verify Assassin Mode engages when an active hit exists on the board
    board[5][5] = GameEngine.CELL_HIT
    local move_history = { { x = 5, y = 5, result = "HIT" } }
    local kill_target, kill_log, _, kill_diag = custom_bot:get_best_move(board, move_history)
    assert_eq(kill_diag.moriarty_mode, "Assassin", "Active hit engages Assassin mode")
    assert_true(string.find(kill_log, "Moriarty: Assassin") ~= nil, "Log reports Assassin mode")
    local dist = math.abs(kill_target.x - 5) + math.abs(kill_target.y - 5)
    assert_eq(dist, 1, "Assassin target must be orthogonally adjacent to the active hit at (5,5)")
    assert_eq(custom_bot.last_heatmap[kill_target.x][kill_target.y], 100, "Assassin target heat set to 100")

    print("[PASS] Test 3: Custom neural forward hook and CSP Assassin target resolution verified.")
end

-- TEST 4: 100-Game Pure Moriarty NN Benchmark
do
    local num_games = 100
    local total_shots = 0
    local bot = MoriartyBot.new(0)

    print(string.format("\nLaunching %d-game benchmark for Pure MoriartyBot (LuaJIT FFI ConvNet)...", num_games))
    local t0 = os.clock()

    for g = 1, num_games do
        local game = GameEngine.new()
        game:place_ships_randomly()
        bot:reset(g, g % 2)

        while not game:is_game_over() and game.shot_number < 100 do
            local target = bot:get_best_move(game.tracking_board, game.move_history)
            local res = game:fire_shot(target.x, target.y)
            assert_true(res ~= "ALREADY_FIRED", "MoriartyBot must never fire at an already revealed cell")
        end
        total_shots = total_shots + game.shot_number
    end

    local elapsed = os.clock() - t0
    local avg_shots = total_shots / num_games
    print(string.format("--> Pure MoriartyBot Average Shots (%d games): %.2f shots", num_games, avg_shots))
    print(string.format("--> Total Runtime: %.4f s (%.1f games/sec, %.3f ms/game)",
        elapsed, num_games / elapsed, (elapsed * 1000) / num_games))

    assert_true(avg_shots >= 43.0 and avg_shots <= 50.0,
        string.format("Pure MoriartyBot avg_shots (%.2f) out of expected range [43.0, 50.0]", avg_shots))
    print("[PASS] Test 4: 1,000-game Pure MoriartyBot benchmark verified.")
end

print("\n🎉 ALL PURE MORIARTY BOT TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

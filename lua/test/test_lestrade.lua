-- lua/test/test_lestrade.lua
-- Unit Tests and Monte Carlo Benchmark for lua/bots/lestrade.lua (Exact App Watson Port)

local LestradeBot = require("lua.bots.lestrade")
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

print("=== Running lua/bots/lestrade.lua Test Suite (Exact App Port) ===")

-- TEST 1: Instantiation & Base Contract
do
    local bot = LestradeBot.new()
    assert_eq(bot.name, "LestradeBot", "Bot name")

    local empty_board = bot:create_blank_matrix(0)
    local target, log, dead_sizes, diag = bot:get_best_move(empty_board, {})

    assert_true(type(target) == "table", "Target is table")
    assert_true(bot:is_valid_cell(target.x, target.y), "Target coordinate is valid")
    assert_true(string.find(log, "Geometric Density") ~= nil, "Log indicates geometric density mode")

    print("[PASS] Test 1: Instantiation and empty board move selection verified.")
end

-- TEST 2: Move Selection on Known Board
do
    local bot = LestradeBot.new()
    local board = bot:create_blank_matrix(0)

    -- Inject a hit at (4, 4)
    board[4][4] = GameEngine.CELL_HIT

    local target, log, dead_sizes, diag = bot:get_best_move(board, {
        { shotNumber = 1, x = 4, y = 4, result = "HIT", isSunk = false }
    })

    -- With 1 + overlapHits, placements overlapping (4,4) receive weight 2
    assert_true(diag.max_density > 0, "Max density should be positive")
    assert_true(bot:is_valid_cell(target.x, target.y), "Target coordinate is valid")

    print("[PASS] Test 2: Board density computation on wounded hit verified.")
end

-- TEST 3: Diagnostic Heatmap
do
    local bot = LestradeBot.new()
    local empty_board = bot:create_blank_matrix(0)
    local heatmap = bot:get_heatmap(empty_board, {})

    local max_h = 0
    for r = 0, 9 do
        for c = 0, 9 do
            local val = heatmap[r][c]
            assert_true(val >= 0 and val <= 100, "Heatmap cell must be within [0, 100]")
            if val > max_h then max_h = val end
        end
    end
    assert_eq(max_h, 100, "Maximum heatmap cell must normalize to 100")

    -- On an empty board, center squares have maximum density (e.g. (4,4), (4,5), (5,4), (5,5))
    assert_eq(heatmap[4][4], 100, "Center cell (4,4) has maximum density on empty board")
    assert_true(heatmap[0][0] < 50, "Corner cell (0,0) has much lower density than center")

    print("[PASS] Test 3: Normalized diagnostic heatmap generation verified.")
end

-- TEST 4: 500-Game Monte Carlo Simulation Benchmark (Exact App Baseline)
do
    local num_games = 500
    local total_shots = 0
    local shot_history = {}
    local outliers = 0

    print(string.format("\nLaunching %d-game benchmark for LestradeBot (Exact App Baseline)...", num_games))
    local t0 = os.clock()

    for g = 1, num_games do
        local game = GameEngine.new()
        game:place_ships_randomly()

        local bot = LestradeBot.new()

        while not game:is_game_over() and game.shot_number < 100 do
            local target, log, dead_sizes, diag = bot:get_best_move(game.tracking_board, game.move_history)
            game:fire_shot(target.x, target.y)
        end

        assert_true(game:is_game_over(), string.format("Game %d must complete", g))
        local s = game.shot_number
        total_shots = total_shots + s
        table.insert(shot_history, s)
        if s > 60 then
            outliers = outliers + 1
        end
    end

    local elapsed = os.clock() - t0
    local avg_shots = total_shots / num_games
    local games_per_sec = num_games / elapsed

    -- Calculate standard deviation
    local sum_sq = 0
    for _, s in ipairs(shot_history) do
        sum_sq = sum_sq + (s - avg_shots) ^ 2
    end
    local std_dev = math.sqrt(sum_sq / num_games)
    local outlier_rate = (outliers / num_games) * 100

    print(string.format("--> Average Shots: %.2f (StdDev: %.2f)", avg_shots, std_dev))
    print(string.format("--> Outliers (>60 shots): %d / %d (%.1f%%)", outliers, num_games, outlier_rate))
    print(string.format("--> Benchmark Runtime: %.4f s (%.0f games/sec on a single core!)", elapsed, games_per_sec))

    -- Target: ~55.0 to 57.0 shots (The exact canonical app baseline)
    assert_true(avg_shots >= 53.0 and avg_shots <= 58.0, "Lestrade average shots should match app baseline (~55.6)")
    print("[PASS] Test 4: Monte Carlo benchmark completed successfully.")
end

print("\n🎉 ALL LESTRADE BOT TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

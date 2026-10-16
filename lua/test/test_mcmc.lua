-- lua/test/test_mcmc.lua
-- Unit Tests and Performance Benchmark for lua/lab/mcmc.lua (MCMCProbabilityBot)

local MCMCProbabilityBot = require("lua.lab.mcmc")
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

print("=== Running lua/lab/mcmc.lua Test Suite ===")

-- TEST 1: Instantiation & 100,000-Step MCMC Single-Turn Latency Benchmark
do
    local bot = MCMCProbabilityBot.new(0, 100000, true)
    assert_eq(bot.name, "MCMCProbabilityBot", "Bot name")
    assert_eq(bot.num_samples, 100000, "Default 100,000 MCMC samples")

    local empty_board = bot:create_blank_matrix(0)
    local t0 = os.clock()
    local target, log, dead_sizes, diag = bot:get_best_move(empty_board, {})
    local elapsed_ms = (os.clock() - t0) * 1000

    assert_true(bot:is_valid_cell(target.x, target.y), "Target coordinate is valid")
    assert_eq(diag.valid_universes, 100000, "Completed all 100,000 MCMC steps")
    assert_eq((target.x + target.y) % 2, 0, "Target obeys checkerboard parity in safe_mode")
    assert_true(string.find(log, "100000 Steps") ~= nil, "Log reports 100000 steps")

    print(string.format("[PASS] Test 1: 100,000-step MCMC turn completed in %.2f ms (%.0f steps/sec!).",
        elapsed_ms, 100000 / (elapsed_ms / 1000)))
    assert_true(elapsed_ms < 50.0, string.format("100k MCMC turn (%.2f ms) should be under 50 ms", elapsed_ms))
end

-- TEST 2: Chronological Fog-of-War Sunk Locking & Active Hit Targeting
do
    local bot = MCMCProbabilityBot.new(0, 50000, true)
    local board = bot:create_blank_matrix(0)

    -- Sunk Patrol Boat at (0,0)-(0,1), plus active wounded hit at (5,5)
    board[0][0] = GameEngine.CELL_HIT
    board[0][1] = GameEngine.CELL_SUNK
    board[5][5] = GameEngine.CELL_HIT

    local move_history = {
        { shotNumber = 1, x = 0, y = 0, result = "HIT", isSunk = false },
        { shotNumber = 2, x = 0, y = 1, result = "SUNK", isSunk = true },
        { shotNumber = 3, x = 5, y = 5, result = "HIT", isSunk = false }
    }

    local target, log, dead_sizes, diag = bot:get_best_move(board, move_history)
    assert_true(diag.has_active, "MCMC detects active wounded hit at (5,5)")
    assert_eq(diag.valid_universes, 50000, "Completed all 50,000 MCMC steps")

    -- Must target an orthogonal neighbor of (5,5)
    local is_neighbor = (target.x == 4 and target.y == 5) or
                        (target.x == 6 and target.y == 5) or
                        (target.x == 5 and target.y == 4) or
                        (target.x == 5 and target.y == 6)
    assert_true(is_neighbor, string.format("Target (%d,%d) must be orthogonal neighbor of active hit (5,5)", target.x, target.y))

    print("[PASS] Test 2: Chronological Fog-of-War corpse locking and active hit targeting verified.")
end

-- TEST 3: Touching Ships Cluster Resolution (L-Shape)
do
    local bot = MCMCProbabilityBot.new(0, 50000, true)
    local board = bot:create_blank_matrix(0)

    board[3][3] = GameEngine.CELL_HIT
    board[3][4] = GameEngine.CELL_HIT
    board[3][5] = GameEngine.CELL_HIT
    board[4][4] = GameEngine.CELL_HIT

    local move_history = {
        { shotNumber = 1, x = 3, y = 3, result = "HIT", isSunk = false },
        { shotNumber = 2, x = 3, y = 4, result = "HIT", isSunk = false },
        { shotNumber = 3, x = 3, y = 5, result = "HIT", isSunk = false },
        { shotNumber = 4, x = 4, y = 4, result = "HIT", isSunk = false }
    }

    local target, log, dead_sizes, diag = bot:get_best_move(board, move_history)
    assert_true(diag.has_active, "MCMC detects active L-shape hit cluster")
    assert_eq(diag.valid_universes, 50000, "Completed all 50,000 MCMC steps on L-shape cluster")

    -- Target must be adjacent to the cluster
    local dist_ok = false
    local cluster = { {3,3}, {3,4}, {3,5}, {4,4} }
    for _, pt in ipairs(cluster) do
        if math.abs(target.x - pt[1]) + math.abs(target.y - pt[2]) == 1 then
            dist_ok = true
            break
        end
    end
    assert_true(dist_ok, string.format("Target (%d,%d) must be adjacent to L-shape cluster", target.x, target.y))

    print("[PASS] Test 3: Touching ships L-shape cluster resolution via MCMC verified.")
end

-- TEST 4: Full-Game Monte Carlo Simulation Benchmark
do
    local num_games = 50
    local samples_per_turn = 25000
    local total_shots = 0
    local shot_history = {}
    local outliers = 0

    print(string.format("\nLaunching %d-game benchmark for MCMCProbabilityBot (%d steps/turn)...",
        num_games, samples_per_turn))
    local t0 = os.clock()

    for g = 1, num_games do
        local game = GameEngine.new()
        game:place_ships_randomly()

        local bot = MCMCProbabilityBot.new(g % 2, samples_per_turn, true)

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

    local sum_sq = 0
    for _, s in ipairs(shot_history) do
        sum_sq = sum_sq + (s - avg_shots) ^ 2
    end
    local std_dev = math.sqrt(sum_sq / num_games)
    local outlier_rate = (outliers / num_games) * 100

    print(string.format("--> Average Shots: %.2f (StdDev: %.2f)", avg_shots, std_dev))
    print(string.format("--> Outliers (>60 shots): %d / %d (%.2f%%)", outliers, num_games, outlier_rate))
    print(string.format("--> Benchmark Runtime: %.4f s (%.2f games/sec on a single core!)", elapsed, games_per_sec))

    assert_true(avg_shots < 48.0, string.format("MCMC average shots (%.2f) should be under 48.0", avg_shots))
    print("[PASS] Test 4: Full-game MCMC simulation benchmark completed successfully.")
end

print("\n🎉 ALL MCMC PROBABILITY BOT TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

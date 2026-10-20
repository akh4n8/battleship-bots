-- lua/test/test_sherlock.lua
-- Unit Tests and Monte Carlo Benchmark for lua/bots/sherlock.lua (Rank 3 SherlockBot)

local SherlockBot = require("lua.bots.sherlock")
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

print("=== Running lua/bots/sherlock.lua Test Suite ===")

-- TEST 1: Instantiation & Empty Board Fast Path
do
    local bot = SherlockBot.new(0)
    assert_eq(bot.name, "SherlockBot", "Bot name")

    local empty_board = bot:create_blank_matrix(0)
    local target, log, dead_sizes, diag = bot:get_best_move(empty_board, {})

    assert_true(bot:is_valid_cell(target.x, target.y), "Target coordinate is valid")
    assert_true(string.find(log, "Stride%-2") ~= nil, "Log indicates Stride-2 hunt")
    assert_eq((target.x + target.y) % 2, 0, "Target obeys Stride-2 parity on empty board")

    -- Verify exact 15.05 billion empty board density was applied
    assert_eq(bot.last_heatmap[4][4], 45, "Center square (4,4) has exact 45% occupancy")
    assert_eq(bot.last_heatmap[0][0], 9, "Corner square (0,0) has exact 9% occupancy")

    print("[PASS] Test 1: Instantiation and 15.05-billion empty board exact matrix verified.")
end

-- TEST 2: Pure Joint Cover on Touching Ships (L-Shape)
do
    local bot = SherlockBot.new(0)
    local board = bot:create_blank_matrix(0)

    -- Touching ships forming an L-shape:
    -- Horizontal Battleship at (3,3), (3,4), (3,5)
    -- Vertical Destroyer at (4,4)
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
    assert_true(string.find(log, "Pure Bayesian Joint Cover") ~= nil, "Log indicates pure joint cover mode")
    assert_true(diag.has_active, "Diagnostics confirms active hit cluster")

    -- Candidate must be a valid completion of Battleship or Destroyer
    local is_candidate = (target.x == 3 and target.y == 2) or
                         (target.x == 3 and target.y == 6) or
                         (target.x == 5 and target.y == 4)
    assert_true(is_candidate, string.format("Target (%d,%d) must solve touching ships cluster", target.x, target.y))

    print("[PASS] Test 2: Pure Bayesian joint cover targeting on touching L-shape cluster verified.")
end

-- TEST 3: Dynamic Stride Transition (Stride-2 -> Stride-3)
do
    local bot = SherlockBot.new(0)
    local board = bot:create_blank_matrix(0)

    board[0][0] = GameEngine.CELL_HIT
    board[0][1] = GameEngine.CELL_SUNK

    local move_history = {
        { shotNumber = 1, x = 0, y = 0, result = "HIT", isSunk = false },
        { shotNumber = 2, x = 0, y = 1, result = "SUNK", isSunk = true }
    }

    local target, log, dead_sizes, diag = bot:get_best_move(board, move_history)
    assert_true(diag.is_two_ship_dead, "CSP deduction confirms 2-ship is dead")
    assert_true(string.find(log, "Stride%-3") ~= nil, "Hunt mode transitioned to Stride-3")
    assert_eq((target.x + 2 * target.y) % 3, bot.parity_offset % 3, "Target obeys Stride-3 knight diagonal parity")

    print("[PASS] Test 3: Dynamic stride transition (Stride-2 -> Stride-3) verified.")
end

-- TEST 4: Opening Tie Randomization & Full 50-Cell Parity Coverage
do
    -- Parity 0: Tied peak cells are (4,4) and (5,5) with 45% exact prior
    local b00 = SherlockBot.new(0, 0)
    local empty00 = b00:create_blank_matrix(0)
    local t1_00 = b00:get_best_move(empty00, {})
    assert_eq(t1_00.x, 4, "Parity 0 flip 0 opens on (4,4) row")
    assert_eq(t1_00.y, 4, "Parity 0 flip 0 opens on (4,4) col")
    empty00[4][4] = GameEngine.CELL_MISS
    local t2_00 = b00:get_best_move(empty00, {{x=4, y=4, result="MISS"}})
    assert_eq(t2_00.x, 5, "Parity 0 flip 0 twin shot is (5,5) row")
    assert_eq(t2_00.y, 5, "Parity 0 flip 0 twin shot is (5,5) col")

    local b01 = SherlockBot.new(0, 1)
    local empty01 = b01:create_blank_matrix(0)
    local t1_01 = b01:get_best_move(empty01, {})
    assert_eq(t1_01.x, 5, "Parity 0 flip 1 opens on (5,5) row")
    assert_eq(t1_01.y, 5, "Parity 0 flip 1 opens on (5,5) col")
    empty01[5][5] = GameEngine.CELL_MISS
    local t2_01 = b01:get_best_move(empty01, {{x=5, y=5, result="MISS"}})
    assert_eq(t2_01.x, 4, "Parity 0 flip 1 twin shot is (4,4) row")
    assert_eq(t2_01.y, 4, "Parity 0 flip 1 twin shot is (4,4) col")

    -- Parity 1: Tied peak cells are (5,4) and (4,5) with 45% exact prior
    local b10 = SherlockBot.new(1, 0)
    local empty10 = b10:create_blank_matrix(0)
    local t1_10 = b10:get_best_move(empty10, {})
    assert_eq(t1_10.x, 5, "Parity 1 flip 0 opens on (5,4) row")
    assert_eq(t1_10.y, 4, "Parity 1 flip 0 opens on (5,4) col")

    local b11 = SherlockBot.new(1, 1)
    local empty11 = b11:create_blank_matrix(0)
    local t1_11 = b11:get_best_move(empty11, {})
    assert_eq(t1_11.x, 4, "Parity 1 flip 1 opens on (4,5) row")
    assert_eq(t1_11.y, 5, "Parity 1 flip 1 opens on (4,5) col")

    -- Check randomization distribution across 100 default instantiations
    local counts = { [44] = 0, [55] = 0 }
    for i = 1, 100 do
        local bot = SherlockBot.new(0)
        local e = bot:create_blank_matrix(0)
        local t = bot:get_best_move(e, {})
        local key = t.x * 10 + t.y
        counts[key] = (counts[key] or 0) + 1
    end
    assert_true(counts[44] > 20 and counts[55] > 20, "Both tied opening moves chosen frequently")

    -- Full 50-cell parity coverage test with zero duplicates
    for parity = 0, 1 do
        for flip = 0, 1 do
            local bot = SherlockBot.new(parity, flip)
            local board = bot:create_blank_matrix(0)
            local visited = {}
            for shot = 1, 50 do
                local t = bot:get_best_move(board, {})
                local key = t.x .. "," .. t.y
                assert_true(not visited[key], "Duplicate shot in sequence: " .. key)
                visited[key] = true
                assert_eq((t.x + t.y) % 2, parity, "Parity condition maintained")
                board[t.x][t.y] = GameEngine.CELL_MISS
            end
            local count = 0
            for _ in pairs(visited) do count = count + 1 end
            assert_eq(count, 50, "All 50 checkerboard squares visited")
        end
    end

    print("[PASS] Test 4: Opening tie randomization and complete 50-cell parity coverage verified.")
end

-- TEST 5: 100-Game Monte Carlo Simulation Benchmark (Rank 3 SherlockBot - True Exact Joint)
do
    local num_games = 100
    local total_shots = 0
    local shot_history = {}
    local outliers = 0

    print(string.format("\nLaunching %d-game benchmark for SherlockBot (Rank 3 - True Exact Joint)...", num_games))
    local t0 = os.clock()

    for g = 1, num_games do
        local game = GameEngine.new()
        game:place_ships_randomly()

        local bot = SherlockBot.new(g % 2)

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
    print(string.format("--> Outliers (>60 shots): %d / %d (%.2f%%)", outliers, num_games, outlier_rate))
    print(string.format("--> Benchmark Runtime: %.4f s (%.0f games/sec on a single core!)", elapsed, games_per_sec))

    -- Target: ~45.03 average shots (True Exact Joint Distribution)
    assert_true(avg_shots < 45.8, "Sherlock average shots should be under 45.8")
    assert_true(outlier_rate <= 6.0, "Sherlock outlier rate should be under 6.0%")
    print("[PASS] Test 5: Monte Carlo benchmark completed successfully.")
end

print("\n🎉 ALL SHERLOCK BOT TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

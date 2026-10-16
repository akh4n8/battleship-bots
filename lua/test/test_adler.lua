-- lua/test/test_adler.lua
-- Unit Tests and Behavioral Exploitation Benchmark for lua/bots/adler.lua (Tier 2: AdlerBot)

local AdlerBot = require("lua.bots.adler")
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

print("=== Running lua/bots/adler.lua Test Suite ===")

-- TEST 1: Empirical Prior Dossier Builder & Prior Merging
do
    local history = {}
    for g = 1, 10 do
        local pegs = {}
        -- Human habitually places Carrier at top edge (r=0, c=0..4)
        for c = 0, 4 do
            table.insert(pegs, { x = 0, y = c })
        end
        table.insert(history, pegs)
    end

    local prior = AdlerBot.build_empirical_prior(history, 0.85)

    -- Verify board-wide mean is normalized to 1.0
    local sum = 0.0
    for r = 0, 9 do
        for c = 0, 9 do
            sum = sum + prior[r][c]
        end
    end
    assert_close(sum / 100.0, 1.0, 1e-4, "Empirical prior mean must equal 1.0")
    assert_true(prior[0][2] > 3.0, "Habitual top-edge cell must have >3.0x multiplier")
    assert_true(prior[5][5] < 1.0, "Unused center cell must be <1.0x multiplier")

    -- Test merge_priors (30% general, 70% specific)
    local uniform = AdlerBot.build_empirical_prior({})
    local merged = AdlerBot.merge_priors(uniform, prior, 0.30, 0.70)
    local expected_02 = 0.30 * 1.0 + 0.70 * prior[0][2]
    assert_close(merged[0][2], expected_02, 1e-4, "30/70 merged prior value at (0,2)")

    print("[PASS] Test 1: Empirical Bayesian prior builder (Laplace + 0.85^t decay) and 30/70 merger verified.")
end

-- TEST 2: Offensive MAP Skew (Hunt & Target Modes)
do
    local history = {}
    for g = 1, 15 do
        local pegs = {}
        for c = 0, 9 do
            table.insert(pegs, { x = 0, y = c })
        end
        table.insert(history, pegs)
    end
    local edge_prior = AdlerBot.build_empirical_prior(history, 0.85)

    local bot = AdlerBot.new(0, edge_prior, nil, 3.0)
    assert_true(not bot.is_uniform_fallback, "Bot recognizes non-uniform human prior")

    local empty_board = bot:create_blank_matrix(0)
    local target, log, dead_sizes, diag = bot:get_best_move(empty_board, {})

    -- Despite row 0 having lower geometric combinatorics on an empty board, Prior^3 pulls Adler to row 0!
    assert_eq(target.x, 0, "Prior^3 MAP pulls opening hunt shot directly to human's habitual row 0")
    assert_true(string.find(log, "Prior%^3.0") ~= nil, "Log indicates Prior^3.0 MAP mode")

    -- Target mode tie-breaking via prior: single hit at (5,5), bias (5,6) slightly higher
    local custom_prior = bot:create_blank_matrix(1.0)
    custom_prior[5][6] = 2.0
    bot:set_offensive_prior(custom_prior)

    local hit_board = bot:create_blank_matrix(0)
    hit_board[5][5] = GameEngine.CELL_HIT
    local move_history = {
        { shotNumber = 1, x = 5, y = 5, result = "HIT", isSunk = false }
    }
    local t_hit, log_hit = bot:get_best_move(hit_board, move_history)
    assert_eq(t_hit.x, 5, "Target mode selects row 5")
    assert_eq(t_hit.y, 6, "Target mode selects biased neighbor (5,6)")
    assert_true(string.find(log_hit, "MAP Target") ~= nil, "Log indicates MAP Target")

    print("[PASS] Test 2: Offensive Bayesian MAP targeting in both Hunt and Target modes verified.")
end

-- TEST 3: Anti-Human Defensive Fleet Generation (Softmax Roulette & Epsilon-Greedy)
do
    local bot = AdlerBot.new(0)
    -- Construct a defensive prior where human heavily bombs rows 3..6 (heat = 5.0) and ignores rows 0, 1, 8, 9 (heat = 0.2)
    local def_prior = bot:create_blank_matrix(0.2)
    for r = 3, 6 do
        for c = 0, 9 do
            def_prior[r][c] = 5.0
        end
    end

    local total_roulette_heat = 0.0
    local num_trials = 30
    for t = 1, num_trials do
        local game = GameEngine.new()
        local fleet = bot:place_defensive_fleet(game, def_prior, "softmax_roulette", 2.0, t * 101)
        assert_eq(#fleet, 5, "Defensive fleet must have 5 ships")

        local peg_count = 0
        for r = 0, 9 do
            for c = 0, 9 do
                if game.truth_board[r][c] > 0 then
                    peg_count = peg_count + 1
                    total_roulette_heat = total_roulette_heat + def_prior[r][c]
                end
            end
        end
        assert_eq(peg_count, 17, "All 17 pegs placed without collision")
    end

    local avg_fleet_heat = total_roulette_heat / num_trials
    -- Uniform random placement across 40 hot cells (5.0) and 60 cold cells (0.2) has expected heat:
    -- 17 * (0.4 * 5.0 + 0.6 * 0.2) = 17 * 2.12 = 36.04.
    -- Softmax roulette avoidance should keep average fleet heat well under 15.0!
    assert_true(avg_fleet_heat < 15.0, string.format("Softmax roulette fleet heat (%.2f) must evade hot center (<15.0)", avg_fleet_heat))

    -- Also verify epsilon_greedy placement
    local game_eg = GameEngine.new()
    local fleet_eg = bot:place_defensive_fleet(game_eg, def_prior, "epsilon_greedy", 1.0, 999)
    assert_eq(#fleet_eg, 5, "Epsilon-greedy fleet has 5 ships")

    print(string.format("[PASS] Test 3: Anti-human defensive fleet evasion verified (Avg Heat %.2f vs 36.04 random).", avg_fleet_heat))
end

-- TEST 4: 500-Game Uniform Fallback + 500-Game Biased Human Exploitation Benchmark
do
    -- Helper: Place ships with a human edge/corner bias (85% perimeter preference)
    local function place_human_biased_ships(game, seed)
        local rng = seed * 48271 + 1
        local function next_r(n)
            rng = (rng * 48271) % 2147483647
            return (rng % n)
        end

        for ship_id, size in ipairs(GameEngine.FLEET_SIZES) do
            local placed = false
            while not placed do
                local is_horiz = (next_r(2) == 0)
                local r, c
                if next_r(100) < 85 then
                    -- Place along outer perimeter (row 0, row 9, col 0, or col 9)
                    if is_horiz then
                        r = (next_r(2) == 0) and 0 or 9
                        c = next_r(10 - size + 1)
                    else
                        r = next_r(10 - size + 1)
                        c = (next_r(2) == 0) and 0 or 9
                    end
                else
                    r = next_r(10)
                    c = next_r(10)
                end
                placed = game:place_ship_manual(r, c, size, is_horiz, ship_id)
            end
        end
    end

    -- Build empirical prior from 25 training games of this simulated human
    local train_history = {}
    for g = 1, 25 do
        local train_game = GameEngine.new()
        place_human_biased_ships(train_game, g * 7919)
        local pegs = {}
        for r = 0, 9 do
            for c = 0, 9 do
                if train_game.truth_board[r][c] > 0 then
                    table.insert(pegs, { x = r, y = c })
                end
            end
        end
        table.insert(train_history, pegs)
    end
    local human_prior = AdlerBot.build_empirical_prior(train_history, 0.85)

    local num_games = 500
    local uniform_shots = 0
    local exploited_shots = 0

    print(string.format("\nLaunching %d-game benchmark for AdlerBot (Uniform vs Biased Human)...", num_games))
    local t0 = os.clock()

    for g = 1, num_games do
        -- 1. Uniform Fallback against Uniform Random Opponent
        local u_game = GameEngine.new()
        u_game:place_ships_randomly()
        local u_bot = AdlerBot.new(g % 2, nil, nil, 3.0)
        while not u_game:is_game_over() and u_game.shot_number < 100 do
            local target = u_bot:get_best_move(u_game.tracking_board, u_game.move_history)
            u_game:fire_shot(target.x, target.y)
        end
        uniform_shots = uniform_shots + u_game.shot_number

        -- 2. Profiled Adler against Biased Human Opponent
        local h_game = GameEngine.new()
        place_human_biased_ships(h_game, (g + 1000) * 104729)
        local h_bot = AdlerBot.new(g % 2, human_prior, nil, 3.0)
        while not h_game:is_game_over() and h_game.shot_number < 100 do
            local target = h_bot:get_best_move(h_game.tracking_board, h_game.move_history)
            h_game:fire_shot(target.x, target.y)
        end
        exploited_shots = exploited_shots + h_game.shot_number
    end

    local elapsed = os.clock() - t0
    local avg_uniform = uniform_shots / num_games
    local avg_exploited = exploited_shots / num_games

    print(string.format("--> Adler Uniform Fallback (vs Random):  %.2f avg shots", avg_uniform))
    print(string.format("--> Adler MAP Profiler (vs Edge Human): %.2f avg shots (%.2f shot reduction!)",
        avg_exploited, avg_uniform - avg_exploited))
    print(string.format("--> Benchmark Runtime (1,000 total games): %.4f s (%.0f games/sec)",
        elapsed, (num_games * 2) / elapsed))

    assert_true(avg_uniform < 46.2, string.format("Uniform fallback (%.2f) should be < 46.2 shots", avg_uniform))
    assert_true(avg_exploited < 36.0, string.format("Profiled Adler (%.2f) should crush edge-biased human in < 36.0 shots", avg_exploited))
    print("[PASS] Test 4: Uniform fallback and human behavioral exploitation benchmarks verified.")
end

print("\n🎉 ALL ADLER BOT TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

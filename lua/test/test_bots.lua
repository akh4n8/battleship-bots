-- lua/test/test_bots.lua
-- Unified Bot Contract Verification & Multi-Core Paired-Board Tournament Test Suite

local Benchmark = require("lua.benchmark")
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

print("=== Running lua/test/test_bots.lua & lua/benchmark.lua Test Suite ===")

-- TEST 1: Unified BaseBot Contract Across All 7 Bots
do
    local all_keys = { "lestrade", "watson", "sherlock", "mycroft", "adler", "moriarty", "mcmc" }
    local master_fleet = Benchmark.generate_master_fleet(42)
    assert_eq(#master_fleet, 5, "Master fleet must contain 5 ships")

    for _, key in ipairs(all_keys) do
        local bot = Benchmark.create_bot(key, 0, 5000)
        assert_true(bot.name ~= nil and #bot.name > 0, "Bot must have a valid name")

        local game = GameEngine.new()
        for _, sp in ipairs(master_fleet) do
            assert_true(game:place_ship_manual(sp.r, sp.c, sp.size, sp.is_horiz, sp.id), "Master ship placement")
        end

        while not game:is_game_over() and game.shot_number < 100 do
            local target, log, dead_sizes, diag = bot:get_best_move(game.tracking_board, game.move_history)
            assert_true(type(target) == "table", key .. " target must be a table")
            assert_true(bot:is_valid_cell(target.x, target.y), key .. " target must be on board")
            assert_eq(game.tracking_board[target.x][target.y], GameEngine.CELL_UNKNOWN, key .. " must fire at unrevealed cell")
            assert_true(type(log) == "string" and #log > 0, key .. " must return non-empty rationale log")
            assert_true(type(dead_sizes) == "table", key .. " must return dead_sizes table")
            assert_true(type(diag) == "table", key .. " must return diagnostics table")

            local hm = bot:get_heatmap(game.tracking_board, dead_sizes)
            assert_true(type(hm) == "table" and type(hm[0]) == "table", key .. " must return 0-indexed 10x10 heatmap")

            game:fire_shot(target.x, target.y)
        end

        assert_true(game:is_game_over(), string.format("Bot '%s' must complete the game within 100 shots", key))
    end

    print("[PASS] Test 1: All 7 bots (lestrade, watson, sherlock, mycroft, adler, moriarty, mcmc) satisfy full BaseBot contract.")
end

-- TEST 2: Bit-Exact Reproducibility Between 1-Worker and 4-Worker POSIX Fork Execution
do
    local bots = { "lestrade", "watson", "sherlock", "mycroft" }
    local res_single = Benchmark.run_tournament({
        games = 40,
        bots = bots,
        workers = 1,
        quiet = true
    })

    local res_multi = Benchmark.run_tournament({
        games = 40,
        bots = bots,
        workers = 4,
        quiet = true
    })

    for _, key in ipairs(bots) do
        local b1 = res_single.bots[key]
        local b4 = res_multi.bots[key]
        assert_eq(b1.count, b4.count, key .. " game count match")
        assert_eq(b1.avg_shots, b4.avg_shots, key .. " exact average shots match across POSIX fork workers")
        assert_eq(b1.median, b4.median, key .. " exact median match across POSIX fork workers")
        assert_eq(b1.min, b4.min, key .. " exact min match")
        assert_eq(b1.max, b4.max, key .. " exact max match")
        assert_eq(b1.outliers, b4.outliers, key .. " exact outlier count match")
    end

    print("[PASS] Test 2: Single-worker and 4-worker POSIX fork() tournaments produce bit-identical results.")
end

-- TEST 3: Multi-Core 2,000-Board Paired Tournament Across Canonical Hierarchy
do
    local res = Benchmark.run_tournament({
        games = 2000,
        bots = { "lestrade", "watson", "sherlock", "mycroft", "adler" },
        quiet = false
    })

    local mycroft_avg = res.bots["mycroft"].avg_shots
    local sherlock_avg = res.bots["sherlock"].avg_shots
    local watson_avg = res.bots["watson"].avg_shots
    local lestrade_avg = res.bots["lestrade"].avg_shots

    assert_true(mycroft_avg < lestrade_avg - 10.0, "Mycroft must beat Lestrade by >10 shots")
    assert_true(sherlock_avg < lestrade_avg - 10.0, "Sherlock must beat Lestrade by >10 shots")
    assert_true(watson_avg < lestrade_avg - 10.0, "Watson must beat Lestrade by >10 shots")
    assert_true(mycroft_avg <= sherlock_avg + 0.15, "Mycroft should lead or tie at the top of the hierarchy")

    print("[PASS] Test 3: Multi-core 2,000-board paired tournament verified canonical hierarchy.")
end

print("\n🎉 ALL BOT CONTRACT & MULTI-CORE TOURNAMENT TESTS PASSED! ZERO DEFECTS.")

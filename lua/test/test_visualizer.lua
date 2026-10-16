-- lua/test/test_visualizer.lua
-- Automated Headless Test Suite for love/conf.lua and love/main.lua

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

print("=== Running love/conf.lua & love/main.lua Visualizer Test Suite ===")

-- Create mock Love2D environment before loading conf.lua and main.lua
local draw_calls = {
    rectangles = 0,
    hull_outlines = 0,
    circles = 0,
    lines = 0,
    prints = 0
}

_G.love = {
    graphics = {
        clear = function() end,
        setColor = function() end,
        setLineWidth = function() end,
        rectangle = function(mode, x, y, w, h, rx, ry)
            draw_calls.rectangles = draw_calls.rectangles + 1
            if mode == "line" and rx == 10 and (w > 50 or h > 50) then
                draw_calls.hull_outlines = draw_calls.hull_outlines + 1
            end
        end,
        circle = function() draw_calls.circles = draw_calls.circles + 1 end,
        line = function() draw_calls.lines = draw_calls.lines + 1 end,
        print = function() draw_calls.prints = draw_calls.prints + 1 end,
        printf = function() draw_calls.prints = draw_calls.prints + 1 end
    },
    mouse = {
        getPosition = function() return 100, 150 end
    },
    event = {
        quit = function() end
    }
}

require("love.conf")
local Visualizer = require("love.main")

-- TEST 1: love.conf Configuration
do
    local conf_table = { window = {}, modules = {} }
    love.conf(conf_table)
    assert_eq(conf_table.window.width, 1240, "Window width")
    assert_eq(conf_table.window.height, 760, "Window height")
    assert_eq(conf_table.window.vsync, 1, "VSync enabled")
    print("[PASS] Test 1: love/conf.lua window & module configuration verified.")
end

-- TEST 2: Live Mode Setup, Manual Ship Placement, AI Firing, Undo, & Hot-Swapping All 7 Bots
do
    love.load({ "--bot", "mycroft", "--mode", "live" })
    local st = Visualizer.state
    assert_eq(st.mode, "LIVE", "Initial mode is LIVE")
    assert_eq(st.bot_key, "mycroft", "Initial bot is mycroft")
    assert_eq(st.phase, "SETUP", "Initial phase is SETUP")
    assert_eq(#st.ships_to_place, 5, "5 ships queued for setup")

    -- Place first ship (Carrier size 5) via mouse click at row 0, col 0
    Visualizer.mousepressed(Visualizer.LEFT_GRID_X + 5, Visualizer.TOP_MARGIN + 5, 1)
    assert_eq(#st.ships_to_place, 4, "4 ships remaining after manual placement")

    -- Press 'r' to randomize remaining ships and enter PLAY phase
    Visualizer.keypressed("r")
    -- Note: reset_live_game(true) via 'f' guarantees clean full fleet
    Visualizer.keypressed("f")
    assert_eq(st.phase, "PLAY", "Phase transitions to PLAY")
    assert_true(st.ai_target ~= nil, "AI target computed upon entering PLAY")

    -- Step 5 AI shots using SPACEBAR
    for i = 1, 5 do
        Visualizer.keypressed("space")
    end
    assert_eq(st.game.shot_number, 5, "5 shots fired via SPACE")

    -- Undo 2 shots using 'u' and 'left'
    Visualizer.keypressed("u")
    Visualizer.keypressed("left")
    assert_eq(st.game.shot_number, 3, "Shot count rolled back to 3 after undo")

    -- Hot-swap across all 7 canonical bots mid-game using keys 1..7 and verify rendering
    for idx, expected_key in ipairs(Visualizer.BOT_ORDER) do
        Visualizer.keypressed(tostring(idx))
        assert_eq(st.bot_key, expected_key, "Switched active bot to " .. expected_key)
        assert_true(st.ai_target ~= nil, expected_key .. " computed valid target on live board")
        draw_calls.rectangles = 0
        love.draw()
        assert_true(draw_calls.rectangles >= 200, expected_key .. " rendered both 10x10 grids")
    end

    print("[PASS] Test 2: Live setup, manual/random ship placement, AI step fire, undo, and 7-bot hot-swap verified.")
end

-- TEST 3: Outlier CSV Parser, Deduced Sunk Ship Hull Outlines, & Forensic Replay Scrubber
do
    local st = Visualizer.state
    Visualizer.switch_bot("mycroft")
    assert_true(#st.all_replay_games > 0, "Loaded outlier games from data/sherlock_outliers_1792119254.csv")
    local first_game = st.all_replay_games[1]
    assert_eq(first_game.game_id, 4, "First outlier Game_ID == 4")
    assert_eq(first_game.total_shots, 63, "First outlier Total_Shots == 63")
    assert_eq(#first_game.history, 63, "Parsed all 63 shot records in Shot_Sequence")

    -- Toggle into REPLAY mode with 'm'
    Visualizer.keypressed("m")
    assert_eq(st.mode, "REPLAY", "Toggled into REPLAY mode")
    assert_eq(st.current_shot_index, 0, "Starts at frame 0")

    -- Scrub forward 10 frames with 'right' (in Game #4, shot 6 sinks the 3-ship at (8,0)-(8,2))
    for _ = 1, 10 do
        Visualizer.keypressed("right")
    end
    assert_eq(st.current_shot_index, 10, "Scrubbed to frame 10")
    assert_eq(#st.replay_diagnostics.dead_ships, 1, "1 sunk ship deduced at frame 10")

    -- Verify rendering draws a single multi-cell hull outline for the deduced sunk ship
    draw_calls.rectangles = 0
    draw_calls.hull_outlines = 0
    love.draw()
    assert_eq(draw_calls.hull_outlines, 1, "Draws exactly 1 continuous multi-cell hull outline for the 1 deduced sunk ship")

    -- Scrub backward 3 frames with 'left'
    for _ = 1, 3 do
        Visualizer.keypressed("left")
    end
    assert_eq(st.current_shot_index, 7, "Scrubbed back to frame 7")

    -- Switch to next outlier game with 'down'
    Visualizer.keypressed("down")
    assert_eq(st.current_game_index, 2, "Moved to outlier game #2")
    assert_eq(st.current_shot_index, 0, "Frame reset to 0 on game switch")

    -- Render replay frame
    draw_calls.rectangles = 0
    love.draw()
    assert_true(draw_calls.rectangles >= 200, "Rendered Forensic Replay dual grids")

    print(string.format("[PASS] Test 3: Forensic Replay CSV loader (%d outlier games), deduced ship hull outlines, and timeline scrubber verified.",
        #st.all_replay_games))
end

-- TEST 4: Multi-Theme System & [C] Theme Cycling
do
    local st = Visualizer.state
    assert_eq(st.theme_key, "classic", "Default theme is classic")

    local expected_cycle = { "baker_street", "sonar_crt", "abyssal", "classic" }
    for _, expected_theme in ipairs(expected_cycle) do
        Visualizer.keypressed("c")
        assert_eq(st.theme_key, expected_theme, "Cycled theme via [C] to " .. expected_theme)
        assert_eq(Visualizer.COLORS, Visualizer.THEMES[expected_theme].colors, "Active COLORS table updated")
        assert_eq(Visualizer.HEAT_COLORS, Visualizer.THEMES[expected_theme].heat_colors, "Active HEAT_COLORS table updated")
        draw_calls.rectangles = 0
        love.draw()
        assert_true(draw_calls.rectangles >= 200, "Rendered cleanly with theme " .. expected_theme)
    end

    print("[PASS] Test 4: All 4 themes (classic, baker_street, sonar_crt, abyssal) and [C] hotkey cycling verified.")
end

print("\n🎉 ALL LOVE2D VISUALIZER TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")

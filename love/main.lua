-- love/main.lua
-- Battleship AI Research Lab — Interactive 60fps Love2D Visualizer & Forensic Replay Suite
-- Launch from workspace root:
--   love love --bot mycroft --mode live
--   love love --bot sherlock --mode replay

package.path = "./?.lua;./lua/?.lua;../?.lua;../lua/?.lua;" .. package.path

local GameEngine = require("lua.game_engine")
local Deduction = require("lua.deduction")
local LestradeBot = require("lua.bots.lestrade")
local WatsonBot = require("lua.bots.watson")
local SherlockBot = require("lua.bots.sherlock")
local MycroftBot = require("lua.bots.mycroft")
local AdlerBot = require("lua.bots.adler")
local MoriartyBot = require("lua.bots.moriarty")
local MCMCProbabilityBot = require("lua.lab.mcmc")

local Visualizer = {}

-- ==========================================
-- 1. LAYOUT & COLOR PALETTE CONSTANTS
-- ==========================================
Visualizer.CELL_SIZE = 50
Visualizer.MARGIN = 80
Visualizer.TOP_MARGIN = 110
Visualizer.GRID_SIZE = 10

Visualizer.LEFT_GRID_X = Visualizer.MARGIN
Visualizer.RIGHT_GRID_X = (Visualizer.MARGIN * 2) + (Visualizer.CELL_SIZE * Visualizer.GRID_SIZE)

Visualizer.THEME_ORDER = {
    "classic", "baker_street", "sonar_crt", "abyssal"
}

Visualizer.THEMES = {
    classic = {
        label = "Classic Tactical",
        colors = {
            BG           = { 25 / 255,  25 / 255,  30 / 255, 1.0 },
            GRID_LINE    = { 100 / 255, 100 / 255, 100 / 255, 1.0 },
            WATER        = { 10 / 255,  60 / 255, 120 / 255, 1.0 },
            SHIP         = { 60 / 255,  70 / 255,  60 / 255, 1.0 },
            GHOST_VALID  = { 0.0,       1.0,       0.0,       0.47 },
            GHOST_INVAL  = { 1.0,       0.0,       0.0,       0.47 },
            LABEL        = { 180 / 255, 180 / 255, 180 / 255, 1.0 },
            HEADER       = { 225 / 255, 225 / 255, 235 / 255, 1.0 },
            TELEMETRY    = { 255 / 255, 200 / 255, 100 / 255, 1.0 },
            VICTORY      = { 60 / 255,  255 / 255, 120 / 255, 1.0 },
            PEG_MISS     = { 220 / 255, 220 / 255, 220 / 255, 1.0 },
            PEG_HIT      = { 255 / 255,  50 / 255,  50 / 255, 1.0 },
            PEG_SUNK     = { 50 / 255,  150 / 255, 255 / 255, 1.0 },
            DIAG_CLAIMED = { 180 / 255, 190 / 255, 205 / 255, 0.90 },
            DIAG_ACTIVE  = { 0.0,       229 / 255, 1.0,       1.0 },
            TARGET_RET   = { 1.0,       240 / 255, 60 / 255,  1.0 }
        },
        heat_colors = {
            { 10 / 255,  60 / 255, 120 / 255, 1.0 },
            { 50 / 255, 120 / 255, 160 / 255, 1.0 },
            { 80 / 255, 160 / 255, 120 / 255, 1.0 },
            { 180 / 255, 180 / 255, 60 / 255, 1.0 },
            { 220 / 255, 120 / 255, 40 / 255, 1.0 },
            { 255 / 255,  50 / 255, 50 / 255, 1.0 }
        }
    },

    baker_street = {
        label = "221B Gaslight Ledger",
        colors = {
            BG           = { 18 / 255,  14 / 255,  12 / 255, 1.0 },
            GRID_LINE    = { 115 / 255, 92 / 255,  62 / 255, 1.0 },
            WATER        = { 24 / 255,  38 / 255,  44 / 255, 1.0 },
            SHIP         = { 78 / 255,  58 / 255,  40 / 255, 1.0 },
            GHOST_VALID  = { 110 / 255, 210 / 255, 110 / 255, 0.47 },
            GHOST_INVAL  = { 220 / 255, 60 / 255,  50 / 255,  0.47 },
            LABEL        = { 205 / 255, 185 / 255, 148 / 255, 1.0 },
            HEADER       = { 242 / 255, 224 / 255, 186 / 255, 1.0 },
            TELEMETRY    = { 245 / 255, 188 / 255, 82 / 255,  1.0 },
            VICTORY      = { 150 / 255, 235 / 255, 130 / 255, 1.0 },
            PEG_MISS     = { 232 / 255, 218 / 255, 190 / 255, 1.0 },
            PEG_HIT      = { 225 / 255, 58 / 255,  45 / 255,  1.0 },
            PEG_SUNK     = { 245 / 255, 190 / 255, 70 / 255,  1.0 },
            DIAG_CLAIMED = { 235 / 255, 195 / 255, 115 / 255, 0.92 },
            DIAG_ACTIVE  = { 105 / 255, 225 / 255, 210 / 255, 1.0 },
            TARGET_RET   = { 255 / 255, 230 / 255, 120 / 255, 1.0 }
        },
        heat_colors = {
            { 24 / 255,  38 / 255,  44 / 255, 1.0 },
            { 42 / 255,  78 / 255,  76 / 255, 1.0 },
            { 110 / 255, 122 / 255, 64 / 255, 1.0 },
            { 188 / 255, 142 / 255, 52 / 255, 1.0 },
            { 218 / 255, 92 / 255,  42 / 255, 1.0 },
            { 205 / 255, 38 / 255,  38 / 255, 1.0 }
        }
    },

    sonar_crt = {
        label = "Cold War Phosphor Sonar",
        colors = {
            BG           = { 6 / 255,   14 / 255,  10 / 255, 1.0 },
            GRID_LINE    = { 28 / 255,  105 / 255, 62 / 255, 1.0 },
            WATER        = { 8 / 255,   28 / 255,  18 / 255, 1.0 },
            SHIP         = { 24 / 255,  72 / 255,  46 / 255, 1.0 },
            GHOST_VALID  = { 40 / 255,  255 / 255, 120 / 255, 0.45 },
            GHOST_INVAL  = { 255 / 255, 60 / 255,  60 / 255,  0.45 },
            LABEL        = { 110 / 255, 215 / 255, 145 / 255, 1.0 },
            HEADER       = { 150 / 255, 250 / 255, 180 / 255, 1.0 },
            TELEMETRY    = { 190 / 255, 255 / 255, 110 / 255, 1.0 },
            VICTORY      = { 80 / 255,  255 / 255, 150 / 255, 1.0 },
            PEG_MISS     = { 130 / 255, 210 / 255, 160 / 255, 1.0 },
            PEG_HIT      = { 255 / 255, 150 / 255, 30 / 255,  1.0 },
            PEG_SUNK     = { 40 / 255,  245 / 255, 220 / 255, 1.0 },
            DIAG_CLAIMED = { 120 / 255, 255 / 255, 160 / 255, 0.92 },
            DIAG_ACTIVE  = { 255 / 255, 230 / 255, 60 / 255,  1.0 },
            TARGET_RET   = { 255 / 255, 255 / 255, 140 / 255, 1.0 }
        },
        heat_colors = {
            { 8 / 255,   28 / 255,  18 / 255, 1.0 },
            { 16 / 255,  72 / 255,  42 / 255, 1.0 },
            { 36 / 255, 130 / 255,  64 / 255, 1.0 },
            { 95 / 255, 190 / 255,  55 / 255, 1.0 },
            { 205 / 255, 210 / 255, 45 / 255, 1.0 },
            { 255 / 255, 145 / 255, 35 / 255, 1.0 }
        }
    },

    abyssal = {
        label = "Abyssal Bioluminescence",
        colors = {
            BG           = { 10 / 255,  11 / 255,  24 / 255, 1.0 },
            GRID_LINE    = { 68 / 255,  78 / 255, 138 / 255, 1.0 },
            WATER        = { 16 / 255,  22 / 255,  48 / 255, 1.0 },
            SHIP         = { 52 / 255,  44 / 255,  92 / 255, 1.0 },
            GHOST_VALID  = { 0.0,       245 / 255, 190 / 255, 0.45 },
            GHOST_INVAL  = { 255 / 255, 40 / 255,  110 / 255, 0.45 },
            LABEL        = { 170 / 255, 182 / 255, 230 / 255, 1.0 },
            HEADER       = { 225 / 255, 232 / 255, 255 / 255, 1.0 },
            TELEMETRY    = { 110 / 255, 240 / 255, 255 / 255, 1.0 },
            VICTORY      = { 80 / 255,  255 / 255, 200 / 255, 1.0 },
            PEG_MISS     = { 195 / 255, 205 / 255, 240 / 255, 1.0 },
            PEG_HIT      = { 255 / 255, 55 / 255,  145 / 255, 1.0 },
            PEG_SUNK     = { 0.0,       235 / 255, 255 / 255, 1.0 },
            DIAG_CLAIMED = { 190 / 255, 165 / 255, 255 / 255, 0.92 },
            DIAG_ACTIVE  = { 255 / 255, 225 / 255, 75 / 255,  1.0 },
            TARGET_RET   = { 80 / 255,  255 / 255, 220 / 255, 1.0 }
        },
        heat_colors = {
            { 16 / 255,  22 / 255,  48 / 255, 1.0 },
            { 52 / 255,  32 / 255, 108 / 255, 1.0 },
            { 118 / 255, 36 / 255, 138 / 255, 1.0 },
            { 188 / 255, 52 / 255, 122 / 255, 1.0 },
            { 238 / 255, 108 / 255, 68 / 255, 1.0 },
            { 252 / 255, 198 / 255, 58 / 255, 1.0 }
        }
    }
}

Visualizer.COLORS = Visualizer.THEMES.classic.colors
Visualizer.HEAT_COLORS = Visualizer.THEMES.classic.heat_colors

Visualizer.BOT_ORDER = {
    "mycroft", "sherlock", "watson", "lestrade", "adler", "moriarty", "mcmc"
}

Visualizer.BOT_SPECS = {
    mycroft  = { class = MycroftBot,         title = "MycroftBot (Rank 4 Entropy + Joint)" },
    sherlock = { class = SherlockBot,        title = "SherlockBot (Rank 3 Exact Joint)" },
    watson   = { class = WatsonBot,          title = "WatsonBot (Rank 2 CSP + Parity)" },
    lestrade = { class = LestradeBot,        title = "LestradeBot (Rank 1 Marginal)" },
    adler    = { class = AdlerBot,           title = "AdlerBot (Tier 2 Bayesian MAP)" },
    moriarty = { class = MoriartyBot,        title = "Moriarty (Neural ResNet)" },
    mcmc     = { class = MCMCProbabilityBot, title = "MCMCProbabilityBot (Lab 25k)" }
}

-- ==========================================
-- 2. BOT FACTORY & OUTLIER CSV PARSER
-- ==========================================
function Visualizer.create_bot(bot_key, parity_offset)
    local key = string.lower(bot_key or "mycroft")
    if key == "porlock" or key == "entropy" then key = "mycroft" end
    if key == "apex" or key == "sherrinford_apex" then key = "sherlock" end
    if key == "neural" or key == "rl" then key = "moriarty" end

    local spec = Visualizer.BOT_SPECS[key] or Visualizer.BOT_SPECS["mycroft"]
    local p_off = parity_offset or 0
    if key == "mcmc" then
        return spec.class.new(p_off, 25000), key
    end
    return spec.class.new(p_off), key
end

--- Parse outlier CSV records exported in `data/*_outliers_*.csv`
-- Format: Game_ID,Total_Shots,Ship_Layout,Shot_Sequence
function Visualizer.parse_outlier_csv(csv_path)
    local f = io.open(csv_path, "r")
    if not f then
        -- Try relative to parent directory if launched from inside `love/`
        f = io.open("../" .. csv_path, "r")
    end
    if not f then
        return {}
    end

    local games = {}
    local is_header = true
    for line in f:lines() do
        line = line:gsub("\r$", "")
        if is_header then
            is_header = false
        elseif #line > 0 then
            local game_id, total_shots, ship_raw, seq_raw =
                line:match("^(%d+),(%d+),\"(.-)\",\"(.-)\"$")
            if game_id and total_shots and ship_raw and seq_raw then
                local ship_json = ship_raw:gsub('""', '"')
                local seq_json  = seq_raw:gsub('""', '"')

                local ships = {}
                for id_str, coords_str in ship_json:gmatch('"(%d+)":%s*%[(.-)%]%s*%]') do
                    local sid = tonumber(id_str)
                    local pegs = {}
                    for r_str, c_str in (coords_str .. "]"):gmatch("%[(%d+),%s*(%d+)%]") do
                        table.insert(pegs, { r = tonumber(r_str), c = tonumber(c_str) })
                    end
                    ships[sid] = pegs
                end

                local history = {}
                for r_str, c_str, res_str in seq_json:gmatch('%[(%d+),%s*(%d+),%s*"(%u+)"%]') do
                    table.insert(history, {
                        x = tonumber(r_str),
                        y = tonumber(c_str),
                        result = res_str
                    })
                end

                table.insert(games, {
                    game_id = tonumber(game_id),
                    total_shots = tonumber(total_shots),
                    ships = ships,
                    history = history
                })
            end
        end
    end

    f:close()
    return games
end

-- ==========================================
-- 3. VISUALIZER STATE MANAGEMENT
-- ==========================================
function Visualizer.switch_theme(theme_key)
    local key = string.lower(theme_key or "classic")
    if not Visualizer.THEMES[key] then
        key = "classic"
    end
    local theme = Visualizer.THEMES[key]
    Visualizer.COLORS = theme.colors
    Visualizer.HEAT_COLORS = theme.heat_colors
    if Visualizer.state then
        Visualizer.state.theme_key = key
    end
    return key
end

function Visualizer.init_state(opts)
    opts = opts or {}
    local state = {
        mode = string.upper(opts.mode or "LIVE"), -- "LIVE" or "REPLAY"
        bot_key = string.lower(opts.bot or "mycroft"),
        theme_key = string.lower(opts.theme or "classic"),
        csv_path = opts.csv or "data/sherlock_outliers_1792119254.csv",
        deduction = Deduction.new(),
        autoplay = false,
        autoplay_timer = 0.0,
        autoplay_interval = 0.08,
        last_infer_ms = 0.0
    }
    Visualizer.state = state
    Visualizer.switch_theme(state.theme_key)
    Visualizer.reset_live_game()
    Visualizer.load_replay_data(state.csv_path)
    return state
end

function Visualizer.reset_live_game(randomize_immediately)
    local st = Visualizer.state
    st.game = GameEngine.new()
    st.ai_bot, st.bot_key = Visualizer.create_bot(st.bot_key, 0)
    st.phase = "SETUP"
    st.ships_to_place = {}
    for _, sz in ipairs(GameEngine.FLEET_SIZES) do
        table.insert(st.ships_to_place, sz)
    end
    st.current_ship_id = 1
    st.is_horizontal = true
    st.last_shot_count = -1
    st.ai_target = nil
    st.strategy = "Awaiting Fleet Deployment..."
    st.dead_sizes = {}
    st.diagnostics = {}
    st.show_ships = true
    st.autoplay = false

    if randomize_immediately then
        st.game:place_ships_randomly()
        st.ships_to_place = {}
        st.phase = "PLAY"
        Visualizer.refresh_live_ai()
    end
end

function Visualizer.load_preset(preset_id)
    Visualizer.reset_live_game(false)
    local st = Visualizer.state
    local g = st.game
    if preset_id == 1 then
        g:place_ship_manual(3, 1, 2, true, 5)
        g:place_ship_manual(3, 3, 3, true, 3)
        g:place_ship_manual(0, 0, 5, true, 1)
        g:place_ship_manual(5, 8, 4, false, 2)
        g:place_ship_manual(8, 1, 3, true, 4)
    elseif preset_id == 2 then
        g:place_ship_manual(4, 2, 4, true, 2)
        g:place_ship_manual(1, 4, 3, false, 3)
        g:place_ship_manual(0, 0, 5, false, 1)
        g:place_ship_manual(7, 1, 3, true, 4)
        g:place_ship_manual(8, 7, 2, true, 5)
    else
        g:place_ship_manual(3, 2, 5, true, 1)
        g:place_ship_manual(4, 2, 4, true, 2)
        g:place_ship_manual(5, 2, 3, true, 3)
        g:place_ship_manual(6, 2, 3, true, 4)
        g:place_ship_manual(7, 2, 2, true, 5)
    end
    st.ships_to_place = {}
    st.phase = "PLAY"
    st.preset_id = preset_id
    Visualizer.refresh_live_ai()
end

function Visualizer.refresh_live_ai()
    local st = Visualizer.state
    if st.phase == "PLAY" then
        if not st.game:is_game_over() then
            local t0 = os.clock()
            local target, strat, dead_sizes, diag =
                st.ai_bot:get_best_move(st.game.tracking_board, st.game.move_history)
            st.last_infer_ms = (os.clock() - t0) * 1000.0
            st.ai_target = target
            st.strategy = strat or ""
            st.dead_sizes = dead_sizes or {}
            st.diagnostics = diag or {}
            st.last_shot_count = st.game.shot_number
        elseif st.ai_bot and st.ai_bot.deduction then
            local solutions = st.ai_bot.deduction:analyze(st.game.tracking_board, st.game.move_history)
            local primary = solutions and solutions[1]
            if primary then
                st.dead_sizes = primary.dead_sizes or {}
                st.diagnostics = {
                    active = primary.active_hits,
                    claimed = primary.claimed_pegs,
                    dead_ships = primary.dead_ships,
                    dead_sizes = primary.dead_sizes
                }
            end
            st.ai_target = nil
            st.last_shot_count = st.game.shot_number
        end
    end
end

function Visualizer.switch_bot(new_bot_key)
    local st = Visualizer.state
    local current_parity = st.ai_bot and st.ai_bot.parity_offset or 0
    st.ai_bot, st.bot_key = Visualizer.create_bot(new_bot_key, current_parity)
    if st.mode == "LIVE" then
        st.last_shot_count = -1
        Visualizer.refresh_live_ai()
    else
        st.replay_last_calc_shot = -1
        Visualizer.refresh_replay_frame()
    end
end

function Visualizer.load_replay_data(csv_path)
    local st = Visualizer.state
    st.all_replay_games = Visualizer.parse_outlier_csv(csv_path)
    st.min_shots_filter = 0
    st.filtered_games = st.all_replay_games
    st.current_game_index = 1
    st.current_shot_index = 0
    st.replay_last_calc_shot = -1
    st.replay_board = st.ai_bot:create_blank_matrix(GameEngine.CELL_UNKNOWN)
    st.replay_move_history = {}
    st.replay_diagnostics = {}
    st.replay_dead_sizes = {}
    st.replay_target = nil
    st.replay_strategy = ""
    if #st.filtered_games > 0 then
        Visualizer.refresh_replay_frame()
    end
end

function Visualizer.apply_replay_filter(new_min_shots)
    local st = Visualizer.state
    st.min_shots_filter = math.max(0, new_min_shots)
    st.filtered_games = {}
    for _, g in ipairs(st.all_replay_games) do
        if g.total_shots >= st.min_shots_filter then
            table.insert(st.filtered_games, g)
        end
    end
    st.current_game_index = 1
    st.current_shot_index = 0
    st.replay_last_calc_shot = -1
    Visualizer.refresh_replay_frame()
end

function Visualizer.refresh_replay_frame()
    local st = Visualizer.state
    if not st.filtered_games or #st.filtered_games == 0 then
        return
    end

    local game_data = st.filtered_games[st.current_game_index]
    if not game_data then return end

    local board = st.ai_bot:create_blank_matrix(GameEngine.CELL_UNKNOWN)
    local move_history = {}

    for i = 1, math.min(st.current_shot_index, #game_data.history) do
        local h = game_data.history[i]
        if h.result == "MISS" then
            board[h.x][h.y] = GameEngine.CELL_MISS
        elseif h.result == "HIT" then
            board[h.x][h.y] = GameEngine.CELL_HIT
        else
            board[h.x][h.y] = GameEngine.CELL_SUNK
        end
        table.insert(move_history, {
            x = h.x,
            y = h.y,
            result = h.result,
            isSunk = (h.result == "SUNK"),
            shotNumber = i
        })
    end

    if #game_data.history > 0 then
        local first = game_data.history[1]
        st.ai_bot.parity_offset = (first.x + first.y) % 2
    end

    st.replay_board = board
    st.replay_move_history = move_history

    local t0 = os.clock()
    local target, strat, dead_sizes, diag = st.ai_bot:get_best_move(board, move_history)
    st.last_infer_ms = (os.clock() - t0) * 1000.0
    st.replay_target = target
    st.replay_strategy = strat or ""
    st.replay_dead_sizes = dead_sizes or {}
    st.replay_diagnostics = diag or {}
    st.replay_last_calc_shot = st.current_shot_index
end

-- ==========================================
-- 4. INPUT & UPDATE HANDLERS
-- ==========================================
function Visualizer.update(dt)
    local st = Visualizer.state
    if st.mode == "LIVE" and st.phase == "PLAY" then
        if st.game.shot_number ~= st.last_shot_count and not st.game:is_game_over() then
            Visualizer.refresh_live_ai()
        end

        if st.autoplay and not st.game:is_game_over() then
            st.autoplay_timer = st.autoplay_timer + dt
            if st.autoplay_timer >= st.autoplay_interval then
                st.autoplay_timer = 0.0
                if st.ai_target then
                    st.game:fire_shot(st.ai_target.x, st.ai_target.y)
                    Visualizer.refresh_live_ai()
                end
            end
        end
    end
end

function Visualizer.keypressed(key)
    local st = Visualizer.state

    -- Global hotkeys: 1..7 or TAB switches active bot architecture immediately
    local num = tonumber(key)
    if num and num >= 1 and num <= #Visualizer.BOT_ORDER then
        Visualizer.switch_bot(Visualizer.BOT_ORDER[num])
        return
    elseif key == "tab" then
        local cur_idx = 1
        for i, k in ipairs(Visualizer.BOT_ORDER) do
            if k == st.bot_key then cur_idx = i; break end
        end
        local next_idx = (cur_idx % #Visualizer.BOT_ORDER) + 1
        Visualizer.switch_bot(Visualizer.BOT_ORDER[next_idx])
        return
    elseif key == "c" then
        local cur_idx = 1
        for i, k in ipairs(Visualizer.THEME_ORDER) do
            if k == st.theme_key then cur_idx = i; break end
        end
        local next_idx = (cur_idx % #Visualizer.THEME_ORDER) + 1
        Visualizer.switch_theme(Visualizer.THEME_ORDER[next_idx])
        return
    elseif key == "m" then
        st.mode = (st.mode == "LIVE") and "REPLAY" or "LIVE"
        if st.mode == "REPLAY" then
            Visualizer.refresh_replay_frame()
        else
            Visualizer.refresh_live_ai()
        end
        return
    elseif key == "escape" then
        if love and love.event then love.event.quit() end
        return
    end

    if st.mode == "LIVE" then
        if key == "n" then
            Visualizer.reset_live_game(false)
        elseif key == "f" or key == "r" then
            Visualizer.reset_live_game(true)
        elseif key == "f1" then
            Visualizer.load_preset(1)
        elseif key == "f2" then
            Visualizer.load_preset(2)
        elseif key == "f3" then
            Visualizer.load_preset(3)
        elseif key == "p" then
            local next_p = ((st.preset_id or 0) % 3) + 1
            Visualizer.load_preset(next_p)
        elseif key == "t" and st.phase == "SETUP" then
            st.is_horizontal = not st.is_horizontal
        elseif st.phase == "PLAY" then
            if key == "u" or key == "left" then
                st.autoplay = false
                st.game:undo_last_shot()
                st.last_shot_count = -1
                Visualizer.refresh_live_ai()
            elseif key == "h" then
                st.show_ships = not st.show_ships
            elseif key == "space" and not st.game:is_game_over() and st.ai_target then
                st.game:fire_shot(st.ai_target.x, st.ai_target.y)
                Visualizer.refresh_live_ai()
            elseif key == "a" then
                st.autoplay = not st.autoplay
                st.autoplay_timer = 0.0
            end
        end
    elseif st.mode == "REPLAY" then
        if key == "w" then
            Visualizer.apply_replay_filter(st.min_shots_filter + 5)
        elseif key == "s" then
            Visualizer.apply_replay_filter(st.min_shots_filter - 5)
        elseif #st.filtered_games > 0 then
            local game_data = st.filtered_games[st.current_game_index]
            if key == "right" then
                st.current_shot_index = math.min(st.current_shot_index + 1, game_data.total_shots)
                Visualizer.refresh_replay_frame()
            elseif key == "left" then
                st.current_shot_index = math.max(st.current_shot_index - 1, 0)
                Visualizer.refresh_replay_frame()
            elseif key == "up" then
                st.current_game_index = math.max(st.current_game_index - 1, 1)
                st.current_shot_index = 0
                Visualizer.refresh_replay_frame()
            elseif key == "down" then
                st.current_game_index = math.min(st.current_game_index + 1, #st.filtered_games)
                st.current_shot_index = 0
                Visualizer.refresh_replay_frame()
            end
        end
    end
end

function Visualizer.mousepressed(mx, my, button)
    local st = Visualizer.state
    if st.mode ~= "LIVE" then return end

    local col = math.floor((mx - Visualizer.LEFT_GRID_X) / Visualizer.CELL_SIZE)
    local row = math.floor((my - Visualizer.TOP_MARGIN) / Visualizer.CELL_SIZE)
    local on_left_grid = (row >= 0 and row < Visualizer.GRID_SIZE and col >= 0 and col < Visualizer.GRID_SIZE)

    if st.phase == "SETUP" then
        if button == 2 then
            st.is_horizontal = not st.is_horizontal
        elseif button == 1 and on_left_grid and #st.ships_to_place > 0 then
            local sz = st.ships_to_place[1]
            if st.game:place_ship_manual(row, col, sz, st.is_horizontal, st.current_ship_id) then
                table.remove(st.ships_to_place, 1)
                st.current_ship_id = st.current_ship_id + 1
                if #st.ships_to_place == 0 then
                    st.phase = "PLAY"
                    Visualizer.refresh_live_ai()
                end
            end
        end
    elseif st.phase == "PLAY" and button == 1 and on_left_grid and not st.game:is_game_over() then
        st.game:fire_shot(row, col)
        Visualizer.refresh_live_ai()
    end
end

-- ==========================================
-- 5. RENDERING ENGINE (Love2D Graphics)
-- ==========================================
local function has_coord(coord_list, r, c)
    if not coord_list then return false end
    for _, pt in ipairs(coord_list) do
        local pr = pt.r or pt.x or pt[1]
        local pc = pt.c or pt.y or pt[2]
        if pr == r and pc == c then return true end
    end
    return false
end

--- Smoothly interpolate across the active theme's heatmap color stops
local function sample_heat_gradient(cell_heat, max_heat, stops)
    if max_heat <= 0 or cell_heat <= 0 or not stops or #stops < 2 then
        return stops and stops[1] or { 0.04, 0.23, 0.47, 1.0 }
    end
    local ratio = math.max(0.0, math.min(1.0, cell_heat / max_heat))
    -- Map ratio in (0..1] across stops[2..#stops] so any non-zero heat is visibly warm
    local num_active = #stops - 1
    local scaled = 2.0 + ratio * (num_active - 1)
    local idx_lo = math.min(#stops - 1, math.max(2, math.floor(scaled)))
    local idx_hi = math.min(#stops, idx_lo + 1)
    local t = scaled - idx_lo

    local c1 = stops[idx_lo]
    local c2 = stops[idx_hi]
    return {
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
        c1[3] + (c2[3] - c1[3]) * t,
        1.0
    }
end

local function draw_grid_labels(lg, offset_x)
    lg.setColor(Visualizer.COLORS.LABEL)
    for i = 0, Visualizer.GRID_SIZE - 1 do
        local col_label = tostring(i + 1)
        local cx = offset_x + i * Visualizer.CELL_SIZE
        lg.printf(col_label, cx, Visualizer.TOP_MARGIN - 24, Visualizer.CELL_SIZE, "center")

        local row_label = string.char(65 + i)
        local ry = Visualizer.TOP_MARGIN + i * Visualizer.CELL_SIZE + (Visualizer.CELL_SIZE * 0.5) - 8
        lg.printf(row_label, offset_x - 32, ry, 24, "center")
    end
end

local function draw_ships(lg, ship_pegs_map, offset_x)
    if not ship_pegs_map then return end
    for _, pegs in pairs(ship_pegs_map) do
        if pegs and #pegs > 0 then
            local min_r, max_r = 99, -1
            local min_c, max_c = 99, -1
            for _, pt in ipairs(pegs) do
                local r = pt.r or pt[1]
                local c = pt.c or pt[2]
                if r < min_r then min_r = r end
                if r > max_r then max_r = r end
                if c < min_c then min_c = c end
                if c > max_c then max_c = c end
            end
            local x = offset_x + min_c * Visualizer.CELL_SIZE + 4
            local y = Visualizer.TOP_MARGIN + min_r * Visualizer.CELL_SIZE + 4
            local w = (max_c - min_c + 1) * Visualizer.CELL_SIZE - 8
            local h = (max_r - min_r + 1) * Visualizer.CELL_SIZE - 8
            lg.setColor(Visualizer.COLORS.SHIP)
            lg.rectangle("fill", x, y, w, h, 12, 12)
        end
    end
end

local function draw_pegs_and_diagnostics(lg, board, diagnostics, offset_x)
    local cs = Visualizer.CELL_SIZE
    local active_list = diagnostics and diagnostics.active or nil
    local dead_ships = diagnostics and diagnostics.dead_ships or nil

    -- 1. Outline each deduced sunk ship as a continuous rounded ship hull
    if dead_ships then
        lg.setColor(Visualizer.COLORS.DIAG_CLAIMED)
        lg.setLineWidth(3)
        for _, ship_pegs in ipairs(dead_ships) do
            if ship_pegs and #ship_pegs > 0 then
                local min_r, max_r = 99, -1
                local min_c, max_c = 99, -1
                for _, pt in ipairs(ship_pegs) do
                    local r = pt.r or pt.x or pt[1]
                    local c = pt.c or pt.y or pt[2]
                    if r < min_r then min_r = r end
                    if r > max_r then max_r = r end
                    if c < min_c then min_c = c end
                    if c > max_c then max_c = c end
                end
                local sx = offset_x + min_c * cs + 5
                local sy = Visualizer.TOP_MARGIN + min_r * cs + 5
                local sw = (max_c - min_c + 1) * cs - 10
                local sh = (max_r - min_r + 1) * cs - 10
                lg.rectangle("line", sx, sy, sw, sh, 10, 10)
            end
        end
    end

    -- 2. Draw shot pegs (Miss circles, Hit/Sunk X markers) and active CSP hit rings
    for r = 0, Visualizer.GRID_SIZE - 1 do
        for c = 0, Visualizer.GRID_SIZE - 1 do
            local x = offset_x + c * cs
            local y = Visualizer.TOP_MARGIN + r * cs
            local cx = x + cs * 0.5
            local cy = y + cs * 0.5

            local val = board[r][c]
            if val == GameEngine.CELL_MISS then
                lg.setColor(Visualizer.COLORS.PEG_MISS)
                lg.circle("fill", cx, cy, cs / 6)
            elseif val == GameEngine.CELL_HIT then
                lg.setColor(Visualizer.COLORS.PEG_HIT)
                lg.setLineWidth(4)
                local pad = 12
                lg.line(x + pad, y + pad, x + cs - pad, y + cs - pad)
                lg.line(x + cs - pad, y + pad, x + pad, y + cs - pad)
            elseif val == GameEngine.CELL_SUNK then
                lg.setColor(Visualizer.COLORS.PEG_SUNK)
                lg.setLineWidth(6)
                local pad = 10
                lg.line(x + pad, y + pad, x + cs - pad, y + cs - pad)
                lg.line(x + cs - pad, y + pad, x + pad, y + cs - pad)
            end

            if has_coord(active_list, r, c) then
                lg.setColor(Visualizer.COLORS.DIAG_ACTIVE)
                lg.setLineWidth(3)
                lg.rectangle("line", x + 4, y + 4, cs - 8, cs - 8, 4, 4)
                lg.circle("line", cx, cy, cs * 0.25)
            end
        end
    end
    lg.setLineWidth(1)
end

local function draw_heatmap_grid(lg, board, ai_bot, dead_sizes, ai_target, offset_x)
    local cs = Visualizer.CELL_SIZE
    draw_grid_labels(lg, offset_x)

    local spec = Visualizer.BOT_SPECS[Visualizer.state.bot_key]
    local title = (spec and spec.title or ai_bot.name) .. " — Probability Heatmap"
    lg.setColor(Visualizer.COLORS.HEADER)
    lg.print(title, offset_x, Visualizer.TOP_MARGIN - 55)

    local heat_map = ai_bot:get_heatmap(board, dead_sizes)
    local max_heat = 0
    for r = 0, Visualizer.GRID_SIZE - 1 do
        for c = 0, Visualizer.GRID_SIZE - 1 do
            local h = (heat_map and heat_map[r] and heat_map[r][c]) or 0
            if h > max_heat then max_heat = h end
        end
    end

    for r = 0, Visualizer.GRID_SIZE - 1 do
        for c = 0, Visualizer.GRID_SIZE - 1 do
            local x = offset_x + c * cs
            local y = Visualizer.TOP_MARGIN + r * cs
            local cell_heat = (heat_map and heat_map[r] and heat_map[r][c]) or 0

            local fill_color = sample_heat_gradient(cell_heat, max_heat, Visualizer.HEAT_COLORS)
            lg.setColor(fill_color)
            lg.rectangle("fill", x, y, cs, cs)

            if cell_heat > 0 then
                -- Adaptive luminance check so numbers remain crisp on bright yellow/amber heat cells
                local lum = 0.299 * fill_color[1] + 0.587 * fill_color[2] + 0.114 * fill_color[3]
                if lum > 0.68 then
                    lg.setColor(0.08, 0.08, 0.10, 1.0)
                else
                    lg.setColor(1.0, 1.0, 1.0, 1.0)
                end
                lg.printf(tostring(cell_heat), x, y + cs * 0.5 - 7, cs, "center")
            end

            lg.setColor(Visualizer.COLORS.GRID_LINE)
            lg.setLineWidth(1)
            lg.rectangle("line", x, y, cs, cs)
        end
    end

    -- Highlight AI's primary recommended target cell
    if ai_target and board[ai_target.x] and board[ai_target.x][ai_target.y] == GameEngine.CELL_UNKNOWN then
        local tx = offset_x + ai_target.y * cs
        local ty = Visualizer.TOP_MARGIN + ai_target.x * cs
        lg.setColor(Visualizer.COLORS.TARGET_RET)
        lg.setLineWidth(3)
        lg.rectangle("line", tx + 2, ty + 2, cs - 4, cs - 4, 4, 4)
        lg.setLineWidth(1)
    end
end

function Visualizer.draw(lg)
    lg = lg or (love and love.graphics)
    if not lg then return end

    local st = Visualizer.state
    local cs = Visualizer.CELL_SIZE
    lg.clear(Visualizer.COLORS.BG[1], Visualizer.COLORS.BG[2], Visualizer.COLORS.BG[3], 1.0)

    -- Top Bot & Theme Selector Ribbon
    local theme_spec = Visualizer.THEMES[st.theme_key] or Visualizer.THEMES.classic
    lg.setColor(Visualizer.COLORS.TELEMETRY)
    local bot_bar = string.format(
        "Bot: [%s] (1..7/TAB)  |  [C] Theme: %s  |  [M] Mode: %s",
        string.upper(st.bot_key), theme_spec.label, st.mode
    )
    lg.print(bot_bar, Visualizer.LEFT_GRID_X, 18)

    if st.mode == "LIVE" then
        draw_grid_labels(lg, Visualizer.LEFT_GRID_X)
        lg.setColor(Visualizer.COLORS.HEADER)
        lg.print(string.format("Play & Diagnostics Board (Shots: %d / Hits: %d/%d)",
            st.game.shot_number, st.game.pegs_hit, st.game.total_pegs),
            Visualizer.LEFT_GRID_X, Visualizer.TOP_MARGIN - 55)

        for r = 0, Visualizer.GRID_SIZE - 1 do
            for c = 0, Visualizer.GRID_SIZE - 1 do
                local x = Visualizer.LEFT_GRID_X + c * cs
                local y = Visualizer.TOP_MARGIN + r * cs
                lg.setColor(Visualizer.COLORS.WATER)
                lg.rectangle("fill", x, y, cs, cs)
                lg.setColor(Visualizer.COLORS.GRID_LINE)
                lg.rectangle("line", x, y, cs, cs)
            end
        end

        if st.phase == "SETUP" or st.show_ships then
            draw_ships(lg, st.game.ship_pegs, Visualizer.LEFT_GRID_X)
        end

        draw_pegs_and_diagnostics(lg, st.game.tracking_board, st.diagnostics, Visualizer.LEFT_GRID_X)

        if st.phase == "SETUP" then
            local mx, my = 0, 0
            if love and love.mouse then mx, my = love.mouse.getPosition() end
            local hc = math.floor((mx - Visualizer.LEFT_GRID_X) / cs)
            local hr = math.floor((my - Visualizer.TOP_MARGIN) / cs)
            if hr >= 0 and hr < Visualizer.GRID_SIZE and hc >= 0 and hc < Visualizer.GRID_SIZE and #st.ships_to_place > 0 then
                local sz = st.ships_to_place[1]
                local sw = st.is_horizontal and (sz * cs) or cs
                local sh = st.is_horizontal and cs or (sz * cs)
                local valid = st.game:can_place(hr, hc, sz, st.is_horizontal)
                lg.setColor(valid and Visualizer.COLORS.GHOST_VALID or Visualizer.COLORS.GHOST_INVAL)
                lg.rectangle("fill",
                    Visualizer.LEFT_GRID_X + hc * cs + 4,
                    Visualizer.TOP_MARGIN + hr * cs + 4,
                    sw - 8, sh - 8, 12, 12)
            end

            draw_heatmap_grid(lg, st.game.tracking_board, st.ai_bot, st.dead_sizes, nil, Visualizer.RIGHT_GRID_X)
            lg.setColor(Visualizer.COLORS.LABEL)
            lg.print("SETUP — Left Click: Place Ship | Right Click / [T]: Rotate | [R]: Randomize & Start | [F]: Quick Start | [C]: Theme",
                Visualizer.LEFT_GRID_X, Visualizer.TOP_MARGIN + 10 * cs + 28)
        else
            draw_heatmap_grid(lg, st.game.tracking_board, st.ai_bot, st.dead_sizes, st.ai_target, Visualizer.RIGHT_GRID_X)
            if not st.game:is_game_over() then
                lg.setColor(Visualizer.COLORS.TELEMETRY)
                lg.print(string.format("Strategy: %s   (%.2f ms)", st.strategy, st.last_infer_ms),
                    Visualizer.LEFT_GRID_X, Visualizer.TOP_MARGIN + 10 * cs + 22)
                lg.setColor(Visualizer.COLORS.LABEL)
                lg.print("[SPACE] Step Shot  |  [A] Auto-Fire  |  [U / LEFT] Undo  |  [H] Ships  |  [C] Theme  |  [N] Setup  |  [F] Fast Reset",
                    Visualizer.LEFT_GRID_X, Visualizer.TOP_MARGIN + 10 * cs + 48)
            else
                lg.setColor(Visualizer.COLORS.VICTORY)
                lg.print(string.format("VICTORY! Fleet eliminated in %d shots. Press [F] for instant rematch or [N] for manual setup.",
                    st.game.shot_number),
                    Visualizer.LEFT_GRID_X, Visualizer.TOP_MARGIN + 10 * cs + 32)
            end
        end
    else
        -- REPLAY MODE
        if st.filtered_games and #st.filtered_games > 0 then
            local game_data = st.filtered_games[st.current_game_index]
            lg.setColor(Visualizer.COLORS.HEADER)
            lg.print(string.format("[FORENSIC REPLAY] Game #%s (%d / %d)  |  Filter >= %d shots",
                tostring(game_data.game_id), st.current_game_index, #st.filtered_games, st.min_shots_filter),
                Visualizer.LEFT_GRID_X, Visualizer.TOP_MARGIN - 55)

            draw_grid_labels(lg, Visualizer.LEFT_GRID_X)
            for r = 0, Visualizer.GRID_SIZE - 1 do
                for c = 0, Visualizer.GRID_SIZE - 1 do
                    local x = Visualizer.LEFT_GRID_X + c * cs
                    local y = Visualizer.TOP_MARGIN + r * cs
                    lg.setColor(Visualizer.COLORS.WATER)
                    lg.rectangle("fill", x, y, cs, cs)
                    lg.setColor(Visualizer.COLORS.GRID_LINE)
                    lg.rectangle("line", x, y, cs, cs)
                end
            end

            draw_ships(lg, game_data.ships, Visualizer.LEFT_GRID_X)
            draw_pegs_and_diagnostics(lg, st.replay_board, st.replay_diagnostics, Visualizer.LEFT_GRID_X)
            draw_heatmap_grid(lg, st.replay_board, st.ai_bot, st.replay_dead_sizes, st.replay_target, Visualizer.RIGHT_GRID_X)

            lg.setColor(Visualizer.COLORS.TELEMETRY)
            lg.print(string.format("Frame: %d / %d Shots   |   %s   (%.2f ms)",
                st.current_shot_index, game_data.total_shots, st.replay_strategy, st.last_infer_ms),
                Visualizer.LEFT_GRID_X, Visualizer.TOP_MARGIN + 10 * cs + 22)
            lg.setColor(Visualizer.COLORS.LABEL)
            lg.print("LEFT/RIGHT: Scrub Shot Timeline  |  UP/DOWN: Switch Game  |  W/S: Shot Filter  |  [C] Theme  |  1..7: Switch Bot",
                Visualizer.LEFT_GRID_X, Visualizer.TOP_MARGIN + 10 * cs + 48)
        else
            lg.setColor(Visualizer.COLORS.LABEL)
            lg.print(string.format("No outlier games found with >= %d shots. Press [S] to lower filter or [M] for Live Play.",
                st.min_shots_filter),
                Visualizer.LEFT_GRID_X, Visualizer.TOP_MARGIN)
        end
    end
end

-- ==========================================
-- 6. LOVE2D LIFECYCLE HOOKS
-- ==========================================
local function parse_love_args(arg_list)
    local opts = { bot = "mycroft", mode = "live", theme = "classic" }
    if not arg_list then return opts end
    local i = 1
    while i <= #arg_list do
        local a = arg_list[i]
        if a == "--bot" and arg_list[i + 1] then
            opts.bot = arg_list[i + 1]
            i = i + 2
        elseif a == "--mode" and arg_list[i + 1] then
            opts.mode = arg_list[i + 1]
            i = i + 2
        elseif a == "--theme" and arg_list[i + 1] then
            opts.theme = arg_list[i + 1]
            i = i + 2
        elseif a == "--csv" and arg_list[i + 1] then
            opts.csv = arg_list[i + 1]
            i = i + 2
        else
            i = i + 1
        end
    end
    return opts
end

if type(love) == "table" then
    function love.load(args)
        local opts = parse_love_args(args)
        Visualizer.init_state(opts)
    end

    function love.update(dt)
        Visualizer.update(dt)
    end

    function love.draw()
        Visualizer.draw(love.graphics)
    end

    function love.keypressed(key)
        Visualizer.keypressed(key)
    end

    function love.mousepressed(x, y, button)
        Visualizer.mousepressed(x, y, button)
    end
end

return Visualizer

-- lua/benchmark.lua
-- High-Performance Multi-Core Paired-Board Tournament & Benchmark Engine (LuaJIT FFI + POSIX fork)
-- Can be required as a module: local Benchmark = require("lua.benchmark")
-- Or executed directly from CLI:
--   luajit lua/benchmark.lua --games 1000 --bots lestrade watson sherlock mycroft --workers 8

package.path = "./?.lua;./lua/?.lua;" .. package.path

local ffi = require("ffi")
local GameEngine = require("lua.game_engine")
local LestradeBot = require("lua.bots.lestrade")
local WatsonBot = require("lua.bots.watson")
local SherlockBot = require("lua.bots.sherlock")
local MycroftBot = require("lua.bots.mycroft")
local AdlerBot = require("lua.bots.adler")
local MoriartyBot = require("lua.bots.moriarty")
local MCMCProbabilityBot = require("lua.lab.mcmc")

-- POSIX declarations for multi-core fork/pipe parallelism and wall-clock timing
ffi.cdef[[
    typedef int pid_t;
    typedef long ssize_t;
    typedef long time_t;

    struct timespec {
        time_t tv_sec;
        long   tv_nsec;
    };

    int pipe(int pipefd[2]);
    pid_t fork(void);
    ssize_t read(int fd, void *buf, size_t count);
    ssize_t write(int fd, const void *buf, size_t count);
    int close(int fd);
    pid_t waitpid(pid_t pid, int *status, int options);
    void _exit(int status);
    long sysconf(int name);
    int clock_gettime(int clk_id, struct timespec *tp);
]]

local Benchmark = {}

Benchmark.BOT_REGISTRY = {
    lestrade = { class = LestradeBot, label = "LestradeBot (Rank 1)" },
    watson   = { class = WatsonBot,   label = "WatsonBot   (Rank 2)" },
    sherlock = { class = SherlockBot, label = "SherlockBot (Rank 3)" },
    mycroft  = { class = MycroftBot,  label = "MycroftBot  (Rank 4)" },
    adler    = { class = AdlerBot,    label = "AdlerBot    (Tier 2)" },
    moriarty = { class = MoriartyBot, label = "MoriartyBot (Tier 2)" },
    mcmc     = { class = MCMCProbabilityBot, label = "MCMCBot     (Lab)   " }
}

--- High-resolution monotonic wall-clock timer in seconds
function Benchmark.wall_time()
    local ts = ffi.new("struct timespec")
    ffi.C.clock_gettime(1, ts) -- CLOCK_MONOTONIC = 1 on Linux
    return tonumber(ts.tv_sec) + tonumber(ts.tv_nsec) * 1e-9
end

--- Detect available CPU cores on Linux (_SC_NPROCESSORS_ONLN = 84)
function Benchmark.get_cpu_count()
    local n = tonumber(ffi.C.sysconf(84))
    if not n or n < 1 then return 4 end
    return n
end

--- Generate a deterministic master fleet placement from game_id so all bots face the exact same board
-- @param game_id number
-- @return table: Array of 5 ship placement descriptors { id=i, size=s, r=r, c=c, is_horiz=b }
function Benchmark.generate_master_fleet(game_id)
    local rng = (game_id * 10007 + 12345) % 2147483647
    if rng <= 0 then rng = rng + 2147483646 end

    local function next_int(n)
        rng = (rng * 48271) % 2147483647
        return rng % n
    end

    local master = GameEngine.new()
    local fleet_placements = {}

    for ship_id, length in ipairs(GameEngine.FLEET_SIZES) do
        local placed = false
        while not placed do
            local is_horiz = (next_int(2) == 0)
            local r = next_int(10)
            local c = next_int(10)
            if master:place_ship_manual(r, c, length, is_horiz, ship_id) then
                table.insert(fleet_placements, {
                    id = ship_id,
                    size = length,
                    r = r,
                    c = c,
                    is_horiz = is_horiz
                })
                placed = true
            end
        end
    end

    return fleet_placements
end

--- Instantiate a bot by key and parity offset
function Benchmark.create_bot(bot_key, parity_offset, mcmc_samples)
    local key = string.lower(bot_key)
    local entry = Benchmark.BOT_REGISTRY[key]
    if not entry then
        error(string.format("Unknown bot key '%s'", tostring(bot_key)))
    end
    if key == "mcmc" then
        return entry.class.new(parity_offset, mcmc_samples or 25000, true)
    end
    return entry.class.new(parity_offset)
end

--- Run a contiguous chunk of paired games [start_id .. end_id] in the current process
-- @param start_id number
-- @param end_id number
-- @param bot_keys table: Array of bot keys, e.g. {"lestrade", "watson", "sherlock", "mycroft"}
-- @param mcmc_samples number (optional)
-- @return table: Map of bot_key -> { count, sum, sum_sq, min, max, outliers, hist = { [1..100] } }
function Benchmark.run_chunk(start_id, end_id, bot_keys, mcmc_samples)
    local stats = {}
    local bots = {}

    for _, key in ipairs(bot_keys) do
        local hist = {}
        for s = 1, 100 do hist[s] = 0 end
        stats[key] = {
            count = 0,
            sum = 0,
            sum_sq = 0,
            min = 999,
            max = 0,
            outliers = 0,
            hist = hist
        }
        bots[key] = Benchmark.create_bot(key, 0, mcmc_samples)
    end

    for game_id = start_id, end_id do
        local master_fleet = Benchmark.generate_master_fleet(game_id)
        local parity = game_id % 2

        for _, key in ipairs(bot_keys) do
            local bot = bots[key]
            bot:reset(game_id, parity)

            local sim = GameEngine.new()
            for i = 1, #master_fleet do
                local sp = master_fleet[i]
                sim:place_ship_manual(sp.r, sp.c, sp.size, sp.is_horiz, sp.id)
            end

            while not sim:is_game_over() and sim.shot_number < 100 do
                local target = bot:get_best_move(sim.tracking_board, sim.move_history)
                sim:fire_shot(target.x, target.y)
            end

            local s = sim.shot_number
            local st = stats[key]
            st.count = st.count + 1
            st.sum = st.sum + s
            st.sum_sq = st.sum_sq + (s * s)
            if s < st.min then st.min = s end
            if s > st.max then st.max = s end
            if s > 60 then st.outliers = st.outliers + 1 end
            st.hist[s] = (st.hist[s] or 0) + 1
        end
    end

    return stats
end

--- Merge chunk statistics into an accumulator
local function merge_stats(total, chunk)
    for key, ch in pairs(chunk) do
        local tot = total[key]
        if not tot then
            total[key] = ch
        else
            tot.count = tot.count + ch.count
            tot.sum = tot.sum + ch.sum
            tot.sum_sq = tot.sum_sq + ch.sum_sq
            if ch.min < tot.min then tot.min = ch.min end
            if ch.max > tot.max then tot.max = ch.max end
            tot.outliers = tot.outliers + ch.outliers
            for s = 1, 100 do
                tot.hist[s] = tot.hist[s] + ch.hist[s]
            end
        end
    end
end

--- Compute percentile from a 1..100 shot histogram
local function compute_percentile(hist, total_count, p)
    local target_rank = math.ceil(total_count * p)
    if target_rank < 1 then target_rank = 1 end
    local cum = 0
    for s = 1, 100 do
        cum = cum + hist[s]
        if cum >= target_rank then
            return s
        end
    end
    return 100
end

--- Execute a Paired-Board Tournament across multiple POSIX worker processes
-- @param opts table: { games = 1000, bots = {"lestrade", "watson", "sherlock", "mycroft"}, workers = N, quiet = false }
-- @return table: Final results map per bot_key + metadata
function Benchmark.run_tournament(opts)
    opts = opts or {}
    local num_games = opts.games or 500
    local bot_keys = opts.bots or { "lestrade", "watson", "sherlock", "mycroft" }
    local max_cores = Benchmark.get_cpu_count()
    local num_workers = opts.workers or math.max(1, math.min(max_cores - 1, math.floor(num_games / 10)))
    if num_workers < 1 then num_workers = 1 end
    if num_workers > num_games then num_workers = num_games end

    local t0 = Benchmark.wall_time()
    local combined_stats = {}

    if num_workers == 1 then
        combined_stats = Benchmark.run_chunk(1, num_games, bot_keys, opts.mcmc_samples)
    else
        -- Multi-process POSIX fork() + binary C-struct pipe aggregation
        -- Each child sends an array of 106 int64_t values per bot:
        -- [count, sum, sum_sq, min, max, outliers, hist[1]..hist[100]]
        local num_bots = #bot_keys
        local ints_per_bot = 106
        local payload_ints = num_bots * ints_per_bot
        local payload_bytes = payload_ints * 8

        local workers = {}
        local base_chunk = math.floor(num_games / num_workers)
        local remainder = num_games % num_workers
        local cur_start = 1

        for w = 1, num_workers do
            local chunk_size = base_chunk + (w <= remainder and 1 or 0)
            local cur_end = cur_start + chunk_size - 1

            local pipefd = ffi.new("int[2]")
            if ffi.C.pipe(pipefd) ~= 0 then
                error("Failed to create POSIX pipe")
            end

            local pid = ffi.C.fork()
            if pid < 0 then
                error("Failed to fork worker process")
            elseif pid == 0 then
                -- Child worker process
                ffi.C.close(pipefd[0]) -- close read end
                local chunk_stats = Benchmark.run_chunk(cur_start, cur_end, bot_keys, opts.mcmc_samples)
                local buf = ffi.new("int64_t[?]", payload_ints)
                local offset = 0
                for _, key in ipairs(bot_keys) do
                    local st = chunk_stats[key]
                    buf[offset + 0] = st.count
                    buf[offset + 1] = st.sum
                    buf[offset + 2] = st.sum_sq
                    buf[offset + 3] = st.min
                    buf[offset + 4] = st.max
                    buf[offset + 5] = st.outliers
                    for s = 1, 100 do
                        buf[offset + 5 + s] = st.hist[s]
                    end
                    offset = offset + ints_per_bot
                end
                ffi.C.write(pipefd[1], buf, payload_bytes)
                ffi.C.close(pipefd[1])
                ffi.C._exit(0)
            else
                -- Parent coordinator process
                ffi.C.close(pipefd[1]) -- close write end
                table.insert(workers, { pid = pid, read_fd = pipefd[0] })
            end

            cur_start = cur_end + 1
        end

        -- Read binary payloads from all child workers and waitpid
        for _, w in ipairs(workers) do
            local buf = ffi.new("int64_t[?]", payload_ints)
            local bytes_read = 0
            local ptr = ffi.cast("uint8_t*", buf)
            while bytes_read < payload_bytes do
                local n = tonumber(ffi.C.read(w.read_fd, ptr + bytes_read, payload_bytes - bytes_read))
                if n <= 0 then break end
                bytes_read = bytes_read + n
            end
            ffi.C.close(w.read_fd)
            local status = ffi.new("int[1]")
            ffi.C.waitpid(w.pid, status, 0)

            local decoded = {}
            local offset = 0
            for _, key in ipairs(bot_keys) do
                local hist = {}
                for s = 1, 100 do
                    hist[s] = tonumber(buf[offset + 5 + s])
                end
                decoded[key] = {
                    count    = tonumber(buf[offset + 0]),
                    sum      = tonumber(buf[offset + 1]),
                    sum_sq   = tonumber(buf[offset + 2]),
                    min      = tonumber(buf[offset + 3]),
                    max      = tonumber(buf[offset + 4]),
                    outliers = tonumber(buf[offset + 5]),
                    hist     = hist
                }
                offset = offset + ints_per_bot
            end
            merge_stats(combined_stats, decoded)
        end
    end

    local elapsed = Benchmark.wall_time() - t0
    local total_bot_games = num_games * #bot_keys
    local results = {
        elapsed = elapsed,
        num_games = num_games,
        num_workers = num_workers,
        total_bot_games = total_bot_games,
        games_per_sec = total_bot_games / math.max(1e-6, elapsed),
        bots = {},
        leaderboard = {}
    }

    for _, key in ipairs(bot_keys) do
        local st = combined_stats[key]
        local n = st.count
        local mean = st.sum / n
        local variance = math.max(0.0, (st.sum_sq / n) - (mean * mean))
        local std_dev = math.sqrt(variance)
        local ci95 = 1.96 * (std_dev / math.sqrt(n))
        local median = compute_percentile(st.hist, n, 0.50)
        local p90 = compute_percentile(st.hist, n, 0.90)
        local outlier_rate = (st.outliers / n) * 100.0

        local entry = {
            key = key,
            label = Benchmark.BOT_REGISTRY[key].label,
            count = n,
            avg_shots = mean,
            std_dev = std_dev,
            ci95 = ci95,
            median = median,
            p90 = p90,
            min = st.min,
            max = st.max,
            outliers = st.outliers,
            outlier_rate = outlier_rate
        }
        results.bots[key] = entry
        table.insert(results.leaderboard, entry)
    end

    table.sort(results.leaderboard, function(a, b)
        return a.avg_shots < b.avg_shots
    end)

    if not opts.quiet then
        print(string.format("\n=========================================================================================="))
        print(string.format(" 🚢 BATTLESHIP PAIRED-BOARD TOURNAMENT (%d Boards x %d Bots = %d Games | %d Workers)",
            num_games, #bot_keys, total_bot_games, num_workers))
        print(string.format("=========================================================================================="))
        print(string.format(" %-4s | %-22s | %-16s | %-6s | %-6s | %-9s | %-12s",
            "Rank", "Bot Architecture", "Avg Shots (95%CI)", "Median", "P90", "Min / Max", "Outliers >60"))
        print(string.format("------+------------------------+------------------+--------+--------+-----------+-------------"))
        for rank, b in ipairs(results.leaderboard) do
            print(string.format(" #%-3d | %-22s | %6.2f ± %-6.3f | %6d | %6d | %3d / %-3d | %4d (%5.2f%%)",
                rank, b.label, b.avg_shots, b.ci95, b.median, b.p90, b.min, b.max, b.outliers, b.outlier_rate))
        end
        print(string.format("=========================================================================================="))
        print(string.format(" Wall Time: %.3f s | Aggregate Throughput: %.0f games/sec\n",
            results.elapsed, results.games_per_sec))
    end

    return results
end

-- CLI Entry Point when executed directly (`luajit lua/benchmark.lua ...`)
if arg and arg[0] and string.find(arg[0], "benchmark%.lua") then
    local games = 1000
    local workers = nil
    local mcmc_samples = 25000
    local bots = {}

    local i = 1
    while i <= #arg do
        local a = arg[i]
        if a == "--games" or a == "-g" then
            i = i + 1
            games = tonumber(arg[i]) or 1000
        elseif a == "--workers" or a == "-w" then
            i = i + 1
            workers = tonumber(arg[i])
        elseif a == "--mcmc-samples" then
            i = i + 1
            mcmc_samples = tonumber(arg[i]) or 25000
        elseif a == "--bots" or a == "-b" then
            i = i + 1
            while i <= #arg and not string.match(arg[i], "^%-") do
                table.insert(bots, string.lower(arg[i]))
                i = i + 1
            end
            i = i - 1
        end
        i = i + 1
    end

    if #bots == 0 then
        bots = { "lestrade", "watson", "sherlock", "mycroft" }
    end

    Benchmark.run_tournament({
        games = games,
        bots = bots,
        workers = workers,
        mcmc_samples = mcmc_samples,
        quiet = false
    })
end

return Benchmark

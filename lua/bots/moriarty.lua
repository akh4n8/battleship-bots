-- lua/bots/moriarty.lua
-- Tier 2: MoriartyBot (Professor James Moriarty — Neural Network Bot)
-- Canonical Strategy:
--   1. 4-Channel 10x10 Spatial Input Tensor (matching `bots/neural_bot.py`):
--        Channel 0: Misses (1.0 where board[r][c] == CELL_MISS)
--        Channel 1: Active Hits (1.0 where board[r][c] == CELL_HIT)
--        Channel 2: Sunk Ships (1.0 where board[r][c] == CELL_SUNK)
--        Channel 3: Parity Mask (1.0 where (r + c) % 2 == parity_offset)
--   2. Trained Residual ConvNet Forward Pass (`data/moriarty_fused.bin` trained on 1M games):
--        - Stem: Conv2d(4->64, 3x3, pad=1) + BN (fused) + LeakyReLU(0.1)
--        - Backbone: 4 x ResConvBlock(64) with LeakyReLU(0.1)
--        - Occupancy Head: Conv2d(64->32, 3x3) + LeakyReLU(0.1) + Conv2d(32->1, 1x1) -> sigmoid * 100
--        - Policy Head: Conv2d(64->32, 3x3) + LeakyReLU(0.1) -> Linear(3200->256) -> Linear(256->100) -> softmax
--   3. Assassin Mode (use_linear_kill = true default):
--        - Uses CSP Deduction + Exact Combinatorial Target Density to finish wounded ships.
--   4. Pure Neural Hunt Mode (Zero Maverick/Mycroft Fallback):
--        - Selects argmax(policy_probs) over unrevealed parity cells (or all unrevealed cells if exhausted).

local ffi = require("ffi")
local BaseBot = require("lua.bots.base_bot")
local GameEngine = require("lua.game_engine")
local Deduction = require("lua.deduction")
local Combinatorics = require("lua.combinatorics")
local Bitboard = require("lua.bitboard")

local MoriartyBot = BaseBot:extend("MoriartyBot")

-- Number of input channels: [0]=Misses, [1]=Hits, [2]=Sunk, [3]=Parity
MoriartyBot.IN_CHANNELS = 4
MoriartyBot.HIDDEN_CHANNELS = 16

-- Pre-allocate FFI C types for branchless 12x12 padded 3x3 convolutions & 10x10 tensors
local float_tensor_4x100_t  = ffi.typeof("float[400]")
local float_pad_4x144_t     = ffi.typeof("float[576]")
local float_pad_16x144_t    = ffi.typeof("float[2304]")
local float_pad_64x144_t    = ffi.typeof("float[9216]")
local float_tensor_64x100_t = ffi.typeof("float[6400]")
local float_tensor_32x100_t = ffi.typeof("float[3200]")
local float_vec_256_t       = ffi.typeof("float[256]")
local float_vec_100_t       = ffi.typeof("float[100]")

-- ============================================================================
-- 1. TRAINED RESNET WEIGHT LOADER (`data/moriarty_fused.bin`)
-- ============================================================================
local RESNET_TOTAL_FLOATS = 1179909
local _cached_resnet_weights = nil
local _tried_loading_resnet = false

-- Exact float offsets from `ml/export_fused_weights.py`
local OFFSETS = {
    stem_w = 0,       stem_b = 2304,
    res0_w1 = 2368,   res0_b1 = 39232,
    res0_w2 = 39296,  res0_b2 = 76160,
    res1_w1 = 76224,  res1_b1 = 113088,
    res1_w2 = 113152, res1_b2 = 150016,
    res2_w1 = 150080, res2_b1 = 186944,
    res2_w2 = 187008, res2_b2 = 223872,
    res3_w1 = 223936, res3_b1 = 260800,
    res3_w2 = 260864, res3_b2 = 297728,
    occ1_w = 297792,  occ1_b = 316224,
    occ2_w = 316256,  occ2_b = 316288,
    pol1_w = 316289,  pol1_b = 334721,
    fc1_w  = 334753,  fc1_b  = 1153953,
    fc2_w  = 1154209, fc2_b  = 1179809
}

local function load_trained_resnet_weights()
    if _tried_loading_resnet then
        return _cached_resnet_weights
    end
    _tried_loading_resnet = true

    local paths = {
        "data/moriarty_fused.bin",
        "../data/moriarty_fused.bin",
        "/home/khan8/Work/battleship-bots/data/moriarty_fused.bin"
    }

    local f = nil
    for _, p in ipairs(paths) do
        f = io.open(p, "rb")
        if f then break end
    end
    if not f then
        return nil
    end

    local raw = f:read("*a")
    f:close()
    if not raw or #raw < 8 + RESNET_TOTAL_FLOATS * 4 then
        return nil
    end

    local magic = raw:sub(1, 4)
    if magic ~= "MOR1" then
        return nil
    end

    local buf = ffi.new("float[?]", RESNET_TOTAL_FLOATS)
    ffi.copy(buf, ffi.cast("const char*", raw) + 8, RESNET_TOTAL_FLOATS * 4)
    _cached_resnet_weights = buf
    return _cached_resnet_weights
end

-- ============================================================================
-- 2. COMPACT FALLBACK CONVNET WEIGHTS
-- ============================================================================
local CONV1_W_SIZE = MoriartyBot.HIDDEN_CHANNELS * MoriartyBot.IN_CHANNELS * 9
local CONV1_B_SIZE = MoriartyBot.HIDDEN_CHANNELS
local CONV2_W_SIZE = 2 * MoriartyBot.HIDDEN_CHANNELS * 9
local CONV2_B_SIZE = 2

local conv1_w_t = ffi.typeof("float[$]", CONV1_W_SIZE)
local conv1_b_t = ffi.typeof("float[$]", CONV1_B_SIZE)
local conv2_w_t = ffi.typeof("float[$]", CONV2_W_SIZE)
local conv2_b_t = ffi.typeof("float[$]", CONV2_B_SIZE)

local function create_default_weights()
    local w1 = conv1_w_t()
    local b1 = conv1_b_t()
    local w2 = conv2_w_t()
    local b2 = conv2_b_t()

    for oc = 0, MoriartyBot.HIDDEN_CHANNELS - 1 do
        local is_horiz_detector = (oc < 8)
        b1[oc] = 0.85 + 0.01 * (oc % 8)

        for ic = 0, MoriartyBot.IN_CHANNELS - 1 do
            for k = 0, 8 do
                local idx = ((oc * MoriartyBot.IN_CHANNELS + ic) * 9) + k
                local val = 0.0

                if ic == 0 or ic == 2 then
                    if k == 4 then
                        val = -2.50
                    elseif is_horiz_detector and (k == 3 or k == 5) then
                        val = -0.55
                    elseif (not is_horiz_detector) and (k == 1 or k == 7) then
                        val = -0.55
                    elseif k == 1 or k == 3 or k == 5 or k == 7 then
                        val = -0.08
                    else
                        val = -0.03
                    end
                elseif ic == 1 then
                    if k == 4 then
                        val = -2.00
                    elseif is_horiz_detector and (k == 3 or k == 5) then
                        val = 1.40
                    elseif (not is_horiz_detector) and (k == 1 or k == 7) then
                        val = 1.40
                    end
                elseif ic == 3 then
                    if k == 4 then
                        val = 0.15
                    end
                end

                w1[idx] = val
            end
        end
    end

    b2[0] = -1.20
    b2[1] = -1.60

    for oc = 0, 1 do
        local scale = (oc == 0) and 0.28 or 0.22
        for ic = 0, MoriartyBot.HIDDEN_CHANNELS - 1 do
            local is_horiz_detector = (ic < 8)
            for k = 0, 8 do
                local idx = ((oc * MoriartyBot.HIDDEN_CHANNELS + ic) * 9) + k
                if k == 4 then
                    w2[idx] = scale * 1.40
                elseif is_horiz_detector and (k == 3 or k == 5) then
                    w2[idx] = scale * 0.45
                elseif (not is_horiz_detector) and (k == 1 or k == 7) then
                    w2[idx] = scale * 0.45
                elseif k == 1 or k == 3 or k == 5 or k == 7 then
                    w2[idx] = scale * 0.10
                else
                    w2[idx] = 0.0
                end
            end
        end
    end

    return {
        conv1_w = w1,
        conv1_b = b1,
        conv2_w = w2,
        conv2_b = b2
    }
end

local DEFAULT_WEIGHTS = create_default_weights()

-- ============================================================================
-- 3. FUSED 3x3 CONV2D + LEAKY_RELU HELPERS (ZERO-ALLOCATION FFI)
-- ============================================================================

--- Copy 10x10 channels [C, 100] into zero-padded 12x12 buffer [C, 144]
local function pad_10x10_to_12x12(src_100, dst_144, channels)
    ffi.fill(dst_144, channels * 144 * 4, 0)
    for ch = 0, channels - 1 do
        local s_base = ch * 100
        local d_base = ch * 144
        for r = 0, 9 do
            local s_row = s_base + r * 10
            local d_row = d_base + (r + 1) * 12 + 1
            for c = 0, 9 do
                dst_144[d_row + c] = src_100[s_row + c]
            end
        end
    end
end

--- 3x3 Conv2d over 12x12 padded input -> 10x10 output
local function conv3x3_forward(pad_in, in_ch, out_100, out_ch, w_ptr, b_ptr, apply_leaky_relu, residual_100)
    for oc = 0, out_ch - 1 do
        local out_base = oc * 100
        local bias = b_ptr[oc]
        for i = 0, 99 do
            out_100[out_base + i] = bias
        end

        for ic = 0, in_ch - 1 do
            local in_base = ic * 144
            local w_base = (oc * in_ch + ic) * 9
            local k0, k1, k2 = w_ptr[w_base],     w_ptr[w_base + 1], w_ptr[w_base + 2]
            local k3, k4, k5 = w_ptr[w_base + 3], w_ptr[w_base + 4], w_ptr[w_base + 5]
            local k6, k7, k8 = w_ptr[w_base + 6], w_ptr[w_base + 7], w_ptr[w_base + 8]

            for r = 1, 10 do
                local r12 = in_base + r * 12
                local out_r10 = out_base + (r - 1) * 10
                for c = 1, 10 do
                    local idx = r12 + c
                    local o_idx = out_r10 + (c - 1)
                    out_100[o_idx] = out_100[o_idx]
                        + pad_in[idx - 13] * k0 + pad_in[idx - 12] * k1 + pad_in[idx - 11] * k2
                        + pad_in[idx -  1] * k3 + pad_in[idx     ] * k4 + pad_in[idx +  1] * k5
                        + pad_in[idx + 11] * k6 + pad_in[idx + 12] * k7 + pad_in[idx + 13] * k8
                end
            end
        end

        if residual_100 then
            for i = 0, 99 do
                local v = out_100[out_base + i] + residual_100[out_base + i]
                if apply_leaky_relu and v < 0.0 then v = v * 0.1 end
                out_100[out_base + i] = v
            end
        elseif apply_leaky_relu then
            for i = 0, 99 do
                local v = out_100[out_base + i]
                if v < 0.0 then
                    out_100[out_base + i] = v * 0.1
                end
            end
        end
    end
end

--- Constructor
-- @param parity_offset number (0 or 1, default 0)
-- @param use_linear_kill boolean (optional, default true): Use CSP Assassin mode when active hits exist
-- @param custom_forward_fn function (optional): Custom neural forward callback `fn(input_tensor) -> policy_logits, occ_logits`
function MoriartyBot:init(parity_offset, use_linear_kill, custom_forward_fn)
    self.parity_offset = parity_offset or 0
    if use_linear_kill == nil then
        self.use_linear_kill = true
    else
        self.use_linear_kill = use_linear_kill
    end
    self.custom_forward_fn = custom_forward_fn

    self.deduction = Deduction.new()
    self.combinatorics = Combinatorics.new()
    self.weights = DEFAULT_WEIGHTS
    self.resnet_weights = load_trained_resnet_weights()

    -- Reusable FFI buffers for zero-allocation, JIT-compiled inference
    self._input_buf     = float_tensor_4x100_t()
    self._pad_in_buf    = float_pad_4x144_t()
    self._pad_hid_buf   = float_pad_16x144_t()
    self._pad64_buf     = float_pad_64x144_t()
    self._feat64_a      = float_tensor_64x100_t()
    self._feat64_b      = float_tensor_64x100_t()
    self._head32_buf    = float_tensor_32x100_t()
    self._fc256_buf     = float_vec_256_t()
    self._policy_logits = float_vec_100_t()
    self._occ_logits    = float_vec_100_t()
    self._policy_probs  = float_vec_100_t()
end

--- Reset bot state between games
function MoriartyBot:reset(game_id, parity_offset)
    BaseBot.reset(self, game_id, parity_offset)
end

--- Set a custom neural forward pass function for experimentation
function MoriartyBot:set_forward_fn(fn)
    self.custom_forward_fn = fn
end

--- Build the 4-channel 10x10 input tensor matching `bots/neural_bot.py`
function MoriartyBot:build_input_tensor(board, as_lua_table)
    local buf = self._input_buf
    local pad = self._pad_in_buf
    ffi.fill(buf, ffi.sizeof(buf), 0)
    ffi.fill(pad, ffi.sizeof(pad), 0)

    for r = 0, 9 do
        local row = board[r]
        local r10 = r * 10
        local r12 = (r + 1) * 12
        for c = 0, 9 do
            local spatial_idx = r10 + c
            local pad_idx = r12 + (c + 1)
            local val = row[c]
            if val == GameEngine.CELL_MISS then
                buf[0 * 100 + spatial_idx] = 1.0
                pad[0 * 144 + pad_idx] = 1.0
            elseif val == GameEngine.CELL_HIT then
                buf[1 * 100 + spatial_idx] = 1.0
                pad[1 * 144 + pad_idx] = 1.0
            elseif val == GameEngine.CELL_SUNK then
                buf[2 * 100 + spatial_idx] = 1.0
                pad[2 * 144 + pad_idx] = 1.0
            end

            if (r + c) % 2 == self.parity_offset then
                buf[3 * 100 + spatial_idx] = 1.0
                pad[3 * 144 + pad_idx] = 1.0
            end
        end
    end

    if not as_lua_table then
        return buf
    end

    local tensor = {}
    for ch = 0, 3 do
        tensor[ch] = {}
        for r = 0, 9 do
            tensor[ch][r] = {}
            for c = 0, 9 do
                tensor[ch][r][c] = tonumber(buf[ch * 100 + r * 10 + c])
            end
        end
    end
    return buf, tensor
end

--- Execute the 1M-game trained ResNet forward pass (`SherlockNeuralModel`)
function MoriartyBot:forward_resnet(rw)
    local pad_in = self._pad_in_buf
    local pad64  = self._pad64_buf
    local feat_a = self._feat64_a
    local feat_b = self._feat64_b
    local head32 = self._head32_buf
    local fc256  = self._fc256_buf
    local p_logits = self._policy_logits
    local o_logits = self._occ_logits

    -- 1. Stem: Conv2d(4 -> 64, 3x3, pad=1) + LeakyReLU(0.1)
    conv3x3_forward(pad_in, 4, feat_a, 64, rw + OFFSETS.stem_w, rw + OFFSETS.stem_b, true, nil)

    -- 2. ResBlock 0..3
    local res_w1 = { OFFSETS.res0_w1, OFFSETS.res1_w1, OFFSETS.res2_w1, OFFSETS.res3_w1 }
    local res_b1 = { OFFSETS.res0_b1, OFFSETS.res1_b1, OFFSETS.res2_b1, OFFSETS.res3_b1 }
    local res_w2 = { OFFSETS.res0_w2, OFFSETS.res1_w2, OFFSETS.res2_w2, OFFSETS.res3_w2 }
    local res_b2 = { OFFSETS.res0_b2, OFFSETS.res1_b2, OFFSETS.res2_b2, OFFSETS.res3_b2 }

    for b = 1, 4 do
        pad_10x10_to_12x12(feat_a, pad64, 64)
        conv3x3_forward(pad64, 64, feat_b, 64, rw + res_w1[b], rw + res_b1[b], true, nil)
        pad_10x10_to_12x12(feat_b, pad64, 64)
        conv3x3_forward(pad64, 64, feat_b, 64, rw + res_w2[b], rw + res_b2[b], true, feat_a)
        ffi.copy(feat_a, feat_b, 6400 * 4)
    end

    -- Pad final backbone features [64, 10, 10] -> [64, 12, 12]
    pad_10x10_to_12x12(feat_a, pad64, 64)

    -- 3. Occupancy Head: Conv2d(64 -> 32, 3x3) + LeakyReLU(0.1) -> Conv2d(32 -> 1, 1x1)
    conv3x3_forward(pad64, 64, head32, 32, rw + OFFSETS.occ1_w, rw + OFFSETS.occ1_b, true, nil)
    local occ2_w = rw + OFFSETS.occ2_w
    local occ2_b = rw[OFFSETS.occ2_b]
    for i = 0, 99 do
        local sum = occ2_b
        for ch = 0, 31 do
            sum = sum + head32[ch * 100 + i] * occ2_w[ch]
        end
        o_logits[i] = sum
    end

    -- 4. Policy Head: Conv2d(64 -> 32, 3x3) + LeakyReLU(0.1) -> Linear(3200 -> 256) + LeakyReLU(0.1) -> Linear(256 -> 100)
    conv3x3_forward(pad64, 64, head32, 32, rw + OFFSETS.pol1_w, rw + OFFSETS.pol1_b, true, nil)

    local fc1_w = rw + OFFSETS.fc1_w
    local fc1_b = rw + OFFSETS.fc1_b
    for o = 0, 255 do
        local sum = fc1_b[o]
        local w_row = fc1_w + o * 3200
        for j = 0, 3199 do
            sum = sum + head32[j] * w_row[j]
        end
        if sum < 0.0 then sum = sum * 0.1 end
        fc256[o] = sum
    end

    local fc2_w = rw + OFFSETS.fc2_w
    local fc2_b = rw + OFFSETS.fc2_b
    for o = 0, 99 do
        local sum = fc2_b[o]
        local w_row = fc2_w + o * 256
        for j = 0, 255 do
            sum = sum + fc256[j] * w_row[j]
        end
        p_logits[o] = sum
    end

    return p_logits, o_logits
end

--- Run the Neural Network Forward Pass on the 4-channel 10x10 input tensor
function MoriartyBot:forward(board)
    local x_buf = self:build_input_tensor(board, false)

    if self.custom_forward_fn then
        local p_logits, o_logits = self.custom_forward_fn(x_buf, self)
        for i = 0, 99 do
            self._policy_logits[i] = p_logits[i] or p_logits[i + 1] or 0.0
            if o_logits then
                self._occ_logits[i] = o_logits[i] or o_logits[i + 1] or 0.0
            else
                self._occ_logits[i] = self._policy_logits[i]
            end
        end
        return self._policy_logits, self._occ_logits
    end

    if self.resnet_weights ~= nil then
        return self:forward_resnet(self.resnet_weights)
    end

    -- Fallback lightweight ConvNet if binary file is absent
    local pad_in  = self._pad_in_buf
    local pad_hid = self._pad_hid_buf
    local w1      = self.weights.conv1_w
    local b1      = self.weights.conv1_b
    local w2      = self.weights.conv2_w
    local b2      = self.weights.conv2_b

    for idx = 0, 11 do
        pad_in[idx] = 1.0
        pad_in[11 * 12 + idx] = 1.0
        pad_in[idx * 12] = 1.0
        pad_in[idx * 12 + 11] = 1.0
    end

    ffi.fill(pad_hid, ffi.sizeof(pad_hid), 0)

    for oc = 0, MoriartyBot.HIDDEN_CHANNELS - 1 do
        local out_base = oc * 144
        local b_val = b1[oc]
        for r = 1, 10 do
            local r12 = r * 12
            for c = 1, 10 do
                pad_hid[out_base + r12 + c] = b_val
            end
        end

        for ic = 0, MoriartyBot.IN_CHANNELS - 1 do
            local in_base = ic * 144
            local w_base = (oc * MoriartyBot.IN_CHANNELS + ic) * 9
            local k0, k1, k2 = w1[w_base],     w1[w_base + 1], w1[w_base + 2]
            local k3, k4, k5 = w1[w_base + 3], w1[w_base + 4], w1[w_base + 5]
            local k6, k7, k8 = w1[w_base + 6], w1[w_base + 7], w1[w_base + 8]

            for r = 1, 10 do
                local r12 = r * 12
                for c = 1, 10 do
                    local idx = in_base + r12 + c
                    local sum = pad_hid[out_base + r12 + c]
                        + pad_in[idx - 13] * k0 + pad_in[idx - 12] * k1 + pad_in[idx - 11] * k2
                        + pad_in[idx -  1] * k3 + pad_in[idx     ] * k4 + pad_in[idx +  1] * k5
                        + pad_in[idx + 11] * k6 + pad_in[idx + 12] * k7 + pad_in[idx + 13] * k8
                    pad_hid[out_base + r12 + c] = sum
                end
            end
        end

        for r = 1, 10 do
            local r12 = out_base + r * 12
            for c = 1, 10 do
                local v = pad_hid[r12 + c]
                if v < 0.0 then
                    pad_hid[r12 + c] = 0.0
                end
            end
        end
    end

    local p_logits = self._policy_logits
    local o_logits = self._occ_logits
    local bp = b2[0]
    local bo = b2[1]
    for i = 0, 99 do
        p_logits[i] = bp
        o_logits[i] = bo
    end

    for ic = 0, MoriartyBot.HIDDEN_CHANNELS - 1 do
        local in_base = ic * 144
        local wp_base = ic * 9
        local wo_base = (MoriartyBot.HIDDEN_CHANNELS + ic) * 9

        local p0, p1, p2 = w2[wp_base],     w2[wp_base + 1], w2[wp_base + 2]
        local p3, p4, p5 = w2[wp_base + 3], w2[wp_base + 4], w2[wp_base + 5]
        local p6, p7, p8 = w2[wp_base + 6], w2[wp_base + 7], w2[wp_base + 8]

        local o0, o1, o2 = w2[wo_base],     w2[wo_base + 1], w2[wo_base + 2]
        local o3, o4, o5 = w2[wo_base + 3], w2[wo_base + 4], w2[wo_base + 5]
        local o6, o7, o8 = w2[wo_base + 6], w2[wo_base + 7], w2[wo_base + 8]

        for r = 1, 10 do
            local r12 = in_base + r * 12
            local out_r10 = (r - 1) * 10
            for c = 1, 10 do
                local idx = r12 + c
                local out_idx = out_r10 + (c - 1)
                local v00, v01, v02 = pad_hid[idx - 13], pad_hid[idx - 12], pad_hid[idx - 11]
                local v10, v11, v12 = pad_hid[idx -  1], pad_hid[idx     ], pad_hid[idx +  1]
                local v20, v21, v22 = pad_hid[idx + 11], pad_hid[idx + 12], pad_hid[idx + 13]

                p_logits[out_idx] = p_logits[out_idx]
                    + v00 * p0 + v01 * p1 + v02 * p2
                    + v10 * p3 + v11 * p4 + v12 * p5
                    + v20 * p6 + v21 * p7 + v22 * p8

                o_logits[out_idx] = o_logits[out_idx]
                    + v00 * o0 + v01 * o1 + v02 * o2
                    + v10 * o3 + v11 * o4 + v12 * o5
                    + v20 * o6 + v21 * o7 + v22 * o8
            end
        end
    end

    return p_logits, o_logits
end

--- Compute Softmax distribution over 100 policy logits
local function softmax_100(logits, probs_out)
    local max_l = -1e30
    for i = 0, 99 do
        if logits[i] > max_l then
            max_l = logits[i]
        end
    end
    local sum = 0.0
    for i = 0, 99 do
        local e = math.exp(logits[i] - max_l)
        probs_out[i] = e
        sum = sum + e
    end
    local inv_sum = (sum > 0.0) and (1.0 / sum) or 0.01
    for i = 0, 99 do
        probs_out[i] = probs_out[i] * inv_sum
    end
end

--- Helper: Compute available surviving ship sizes
local function get_available_fleet(fleet_sizes, dead_sizes)
    local dead_map = {}
    if dead_sizes then
        for _, sz in ipairs(dead_sizes) do
            dead_map[sz] = (dead_map[sz] or 0) + 1
        end
    end

    local available = {}
    for _, sz in ipairs(fleet_sizes) do
        if (dead_map[sz] or 0) > 0 then
            dead_map[sz] = dead_map[sz] - 1
        else
            table.insert(available, sz)
        end
    end
    return available
end

--- Generate 10x10 integer heatmap (0..100) from Neural Occupancy Head (matching `NeuralBot.get_heatmap`)
function MoriartyBot:get_heatmap(board, dead_sizes)
    if self.last_heatmap then
        return self.last_heatmap
    end
    local _, occ_logits = self:forward(board)
    local heatmap = self:create_blank_matrix(0)

    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_UNKNOWN then
                local logit = tonumber(occ_logits[r * 10 + c])
                local prob = 1.0 / (1.0 + math.exp(-logit))
                heatmap[r][c] = math.floor(prob * 100.0 + 0.5)
            else
                heatmap[r][c] = 0
            end
        end
    end

    self.last_heatmap = heatmap
    return heatmap
end

--- Core Decision Engine (Exact Lua counterpart of `bots/neural_bot.py`):
function MoriartyBot:get_best_move(board, move_history)
    local solutions = self.deduction:analyze(board, move_history)
    local primary_sol = solutions[1]
    local dead_sizes = primary_sol.dead_sizes
    local has_active = Deduction.has_active_hits(solutions)
    local is_2_dead = Deduction.is_size_dead(solutions, 2)

    local diagnostics = {
        active = primary_sol.active_hits,
        claimed = primary_sol.claimed_pegs,
        dead_ships = primary_sol.dead_ships,
        dead_sizes = dead_sizes,
        solutions_count = #solutions,
        is_two_ship_dead = is_2_dead,
        has_active = has_active,
        moriarty_mode = "Neural Hunt"
    }

    -- 1. ASSASSIN / TARGET RESOLUTION PHASE
    if self.use_linear_kill and has_active then
        local misses_bb = Bitboard.empty()
        for r = 0, 9 do
            for c = 0, 9 do
                if board[r][c] == GameEngine.CELL_MISS then
                    misses_bb:set(r, c)
                end
            end
        end

        local total_target_counts = {}
        local max_target_count = -1

        for _, sol in ipairs(solutions) do
            if #sol.active_hits > 0 then
                local forbidden_bb = misses_bb:copy()
                for _, pt in ipairs(sol.claimed_pegs) do
                    forbidden_bb:set(pt.r, pt.c)
                end

                local active_hits_bb = Bitboard.from_coords(sol.active_hits)
                local available_fleet = get_available_fleet(self.fleet_sizes, sol.dead_sizes)

                local counts, _ = self.combinatorics:compute_target_density(
                    board, available_fleet, forbidden_bb, active_hits_bb
                )

                for idx, cnt in pairs(counts) do
                    local new_cnt = (total_target_counts[idx] or 0) + cnt
                    total_target_counts[idx] = new_cnt
                    if new_cnt > max_target_count then
                        max_target_count = new_cnt
                    end
                end
            end
        end

        local best_targets = {}
        for idx, cnt in pairs(total_target_counts) do
            if cnt == max_target_count then
                local r = math.floor(idx / 10)
                local c = idx % 10
                if board[r][c] == GameEngine.CELL_UNKNOWN then
                    table.insert(best_targets, { x = r, y = c })
                end
            end
        end

        if #best_targets > 0 then
            local pick_idx = ((self.parity_offset * 10007 + #move_history) % #best_targets) + 1
            local target = best_targets[pick_idx]
            local heat = self:create_blank_matrix(0)
            if max_target_count > 0 then
                for idx, cnt in pairs(total_target_counts) do
                    local r = math.floor(idx / 10)
                    local c = idx % 10
                    heat[r][c] = math.floor((cnt / max_target_count) * 100)
                end
            end
            heat[target.x][target.y] = 100
            self.last_heatmap = heat
            diagnostics.moriarty_mode = "Assassin"
            local log = string.format("Moriarty: Assassin (Exact Target Peak=%d)", max_target_count)
            return target, log, dead_sizes, diagnostics
        end
    end

    -- 2. PURE NEURAL HUNTING PHASE
    local policy_logits, occ_logits = self:forward(board)
    softmax_100(policy_logits, self._policy_probs)

    -- Populate occupancy heatmap from occ_logits
    local heatmap = self:create_blank_matrix(0)
    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_UNKNOWN then
                local logit = tonumber(occ_logits[r * 10 + c])
                local prob = 1.0 / (1.0 + math.exp(-logit))
                heatmap[r][c] = math.floor(prob * 100.0 + 0.5)
            end
        end
    end
    self.last_heatmap = heatmap

    local best_score = -1e9
    local best_candidates = {}

    -- First pass: try on-parity checkerboard cells ((r + c) % 2 == self.parity_offset)
    for r = 0, 9 do
        for c = 0, 9 do
            if board[r][c] == GameEngine.CELL_UNKNOWN and (r + c) % 2 == self.parity_offset then
                local score = tonumber(self._policy_probs[r * 10 + c])
                if score > best_score + 1e-7 then
                    best_score = score
                    best_candidates = { { x = r, y = c } }
                elseif math.abs(score - best_score) <= 1e-7 then
                    table.insert(best_candidates, { x = r, y = c })
                end
            end
        end
    end

    -- Second pass: if no valid on-parity cell remains, consider all open cells
    if #best_candidates == 0 then
        for r = 0, 9 do
            for c = 0, 9 do
                if board[r][c] == GameEngine.CELL_UNKNOWN then
                    local score = tonumber(self._policy_probs[r * 10 + c])
                    if score > best_score + 1e-7 then
                        best_score = score
                        best_candidates = { { x = r, y = c } }
                    elseif math.abs(score - best_score) <= 1e-7 then
                        table.insert(best_candidates, { x = r, y = c })
                    end
                end
            end
        end
    end

    local target
    if #best_candidates > 0 then
        local pick_idx = ((self.parity_offset * 10007 + #move_history) % #best_candidates) + 1
        target = best_candidates[pick_idx]
    else
        local unrevealed = self:get_unrevealed_cells(board)
        target = unrevealed[1] or { x = 0, y = 0 }
        best_score = 0.0
    end

    local confidence_pct = best_score * 100.0
    local log = string.format("Moriarty: Neural Hunt (Confidence: %.1f%%)", confidence_pct)
    return target, log, dead_sizes, diagnostics
end

return MoriartyBot

-- lua/bitboard.lua
-- High-Performance 100-Bit Bitboard Engine for Battleship (10x10 Grid)
-- Implemented using LuaJIT FFI with 16-byte packed C structs.

local ffi = require("ffi")
local bit = require("bit")

-- 1. FFI C-TYPE DEFINITION
-- 10x10 board = 100 cells (indexed 0..99: idx = r * 10 + c).
-- lo: Bits 0..63  (Cells (0,0) through (6,3))
-- hi: Bits 64..99 (Cells (6,4) through (9,9) => 36 bits used, bits 0..35)
ffi.cdef[[
typedef struct {
    uint64_t lo;
    uint64_t hi;
} bitboard_t;
]]

-- 2. CONSTANTS & PRECOMPUTED LOOKUP TABLES
local HI_MASK = bit.lshift(1ULL, 36) - 1ULL -- 0xFFFFFFFFF (36 ones)
local FULL_LO = 0xFFFFFFFFFFFFFFFFULL

-- Precomputed single-bit masks for instant O(1) bit index lookups
local MASK_LO = ffi.new("uint64_t[64]")
for i = 0, 63 do
    MASK_LO[i] = bit.lshift(1ULL, i)
end

local MASK_HI = ffi.new("uint64_t[36]")
for i = 0, 35 do
    MASK_HI[i] = bit.lshift(1ULL, i)
end

-- 3. FAST 64-BIT SWAR POPCOUNT
local function popcnt64(x)
    x = x - bit.band(bit.rshift(x, 1), 0x5555555555555555ULL)
    x = bit.band(x, 0x3333333333333333ULL) + bit.band(bit.rshift(x, 2), 0x3333333333333333ULL)
    x = bit.band(x + bit.rshift(x, 4), 0x0F0F0F0F0F0F0F0FULL)
    return tonumber(bit.rshift(x * 0x0101010101010101ULL, 56))
end

-- 4. METATABLE FOR bitboard_t
local BitboardMt = {}
BitboardMt.__index = BitboardMt

--- Set bit at (r, c) in-place
-- @param r number (0..9)
-- @param c number (0..9)
-- @return self
function BitboardMt:set(r, c)
    local idx = r * 10 + c
    if idx < 64 then
        self.lo = bit.bor(self.lo, MASK_LO[idx])
    else
        self.hi = bit.bor(self.hi, MASK_HI[idx - 64])
    end
    return self
end

--- Clear bit at (r, c) in-place
-- @param r number (0..9)
-- @param c number (0..9)
-- @return self
function BitboardMt:clear(r, c)
    local idx = r * 10 + c
    if idx < 64 then
        self.lo = bit.band(self.lo, bit.bnot(MASK_LO[idx]))
    else
        self.hi = bit.band(self.hi, bit.bnot(MASK_HI[idx - 64]))
    end
    return self
end

--- Test if bit at (r, c) is set
-- @param r number (0..9)
-- @param c number (0..9)
-- @return boolean
function BitboardMt:is_set(r, c)
    local idx = r * 10 + c
    if idx < 64 then
        return bit.band(self.lo, MASK_LO[idx]) ~= 0ULL
    else
        return bit.band(self.hi, MASK_HI[idx - 64]) ~= 0ULL
    end
end

--- Check if bitboard is completely empty (0 bits set)
-- @return boolean
function BitboardMt:is_zero()
    return self.lo == 0ULL and self.hi == 0ULL
end

--- Check if self and other share ANY overlapping bits
-- Ultra-fast early-exit without allocating a new struct
-- @param other bitboard_t
-- @return boolean
function BitboardMt:intersects(other)
    return bit.band(self.lo, other.lo) ~= 0ULL or bit.band(self.hi, other.hi) ~= 0ULL
end

--- Check if self completely covers other (self & other == other)
-- @param other bitboard_t
-- @return boolean
function BitboardMt:contains_all(other)
    return bit.band(self.lo, other.lo) == other.lo and bit.band(self.hi, other.hi) == other.hi
end

--- Total number of set bits (0..100)
-- @return number
function BitboardMt:popcnt()
    return popcnt64(self.lo) + popcnt64(self.hi)
end

--- Bitwise AND: returns new bitboard (self & other)
-- @param other bitboard_t
-- @return bitboard_t
function BitboardMt:band(other)
    local res = ffi.new("bitboard_t")
    res.lo = bit.band(self.lo, other.lo)
    res.hi = bit.band(self.hi, other.hi)
    return res
end

--- Bitwise OR: returns new bitboard (self | other)
-- @param other bitboard_t
-- @return bitboard_t
function BitboardMt:bor(other)
    local res = ffi.new("bitboard_t")
    res.lo = bit.bor(self.lo, other.lo)
    res.hi = bit.bor(self.hi, other.hi)
    return res
end

--- Bitwise XOR: returns new bitboard (self ^ other)
-- @param other bitboard_t
-- @return bitboard_t
function BitboardMt:bxor(other)
    local res = ffi.new("bitboard_t")
    res.lo = bit.bxor(self.lo, other.lo)
    res.hi = bit.bxor(self.hi, other.hi)
    return res
end

--- Bitwise NOT: returns new bitboard (~self & MASK_100)
-- @return bitboard_t
function BitboardMt:bnot()
    local res = ffi.new("bitboard_t")
    res.lo = bit.bnot(self.lo)
    res.hi = bit.band(bit.bnot(self.hi), HI_MASK)
    return res
end

--- In-place Bitwise AND (Zero Allocations)
-- @param other bitboard_t
-- @return self
function BitboardMt:band_inplace(other)
    self.lo = bit.band(self.lo, other.lo)
    self.hi = bit.band(self.hi, other.hi)
    return self
end

--- In-place Bitwise OR (Zero Allocations)
-- @param other bitboard_t
-- @return self
function BitboardMt:bor_inplace(other)
    self.lo = bit.bor(self.lo, other.lo)
    self.hi = bit.bor(self.hi, other.hi)
    return self
end

--- Deep copy of this bitboard
-- @return bitboard_t
function BitboardMt:copy()
    local res = ffi.new("bitboard_t")
    res.lo = self.lo
    res.hi = self.hi
    return res
end

--- Convert bitboard to an array of {r, c} coordinates
-- @return table of { {r=..., c=...}, ... }
function BitboardMt:to_coords()
    local coords = {}
    local n = 0
    for r = 0, 9 do
        for c = 0, 9 do
            if self:is_set(r, c) then
                n = n + 1
                coords[n] = {r = r, c = c}
            end
        end
    end
    return coords
end

--- Human-readable 10x10 ASCII grid visualization
-- @return string
function BitboardMt:to_grid_string()
    local lines = {"    0 1 2 3 4 5 6 7 8 9", "  +-------------------+"}
    for r = 0, 9 do
        local row_chars = {}
        for c = 0, 9 do
            row_chars[c + 1] = self:is_set(r, c) and "#" or "."
        end
        lines[#lines + 1] = string.format("%d | %s |", r, table.concat(row_chars, " "))
    end
    lines[#lines + 1] = "  +-------------------+"
    return table.concat(lines, "\n")
end

function BitboardMt:__tostring()
    return string.format("bitboard_t(lo=0x%016X, hi=0x%09X, popcnt=%d)",
        tonumber(self.lo), tonumber(self.hi), self:popcnt())
end

-- 5. REGISTER METATYPE WITH FFI
local bitboard_ctor = ffi.metatype("bitboard_t", BitboardMt)

-- 6. MODULE CONSTRUCTORS & FACTORIES
local Bitboard = {}

--- Create a new bitboard with optional initial lo and hi values
-- @param lo number|uint64_t (optional, default 0ULL)
-- @param hi number|uint64_t (optional, default 0ULL)
-- @return bitboard_t
function Bitboard.new(lo, hi)
    local bb = bitboard_ctor()
    bb.lo = lo and ffi.cast("uint64_t", lo) or 0ULL
    bb.hi = hi and ffi.cast("uint64_t", hi) or 0ULL
    return bb
end

--- Return a new empty bitboard (all 0s)
-- @return bitboard_t
function Bitboard.empty()
    return bitboard_ctor()
end

--- Return a new full bitboard (all 100 cells set to 1)
-- @return bitboard_t
function Bitboard.full()
    local bb = bitboard_ctor()
    bb.lo = FULL_LO
    bb.hi = HI_MASK
    return bb
end

--- Return a bitboard with a single cell set at (r, c)
-- @param r number (0..9)
-- @param c number (0..9)
-- @return bitboard_t
function Bitboard.from_coord(r, c)
    local bb = bitboard_ctor()
    bb:set(r, c)
    return bb
end

--- Return a bitboard populated from an array of coordinates
-- Supports both {{r, c}, ...} and {{x=r, y=c}, ...}
-- @param coords table
-- @return bitboard_t
function Bitboard.from_coords(coords)
    local bb = bitboard_ctor()
    for _, pt in ipairs(coords) do
        local r = pt.r or pt[1] or pt.x
        local c = pt.c or pt[2] or pt.y
        if r and c then
            bb:set(r, c)
        end
    end
    return bb
end

--- Check if input is a valid bitboard_t cdata
-- @param val any
-- @return boolean
function Bitboard.is_bitboard(val)
    return type(val) == "cdata" and ffi.istype("bitboard_t", val)
end

return Bitboard

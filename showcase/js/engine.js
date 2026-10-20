// showcase/js/engine.js
// Battleship AI & Combinatorics Engine — 1:1 Port of the LuaJIT 2.1 Suite
// Includes:
//   - Bitboard100 (4x32-bit SWAR bitboard, ~0.5ns collision & popcount)
//   - BattleshipGame (10x10 Fog-of-War physics, continuous fire, undo stack, ambiguity presets)
//   - DeductionEngine (Chronological Multi-Hypothesis CSP solver for the Blind Sink Identity Problem)
//   - Combinatorics (760 ship bitboards + P1/P2 Disjoint Bayesian Joint Covers)
//   - JointSolver (Exact Joint Density with 1-Ship Analytic SWAR Elimination & Incremental Miss-Subtraction)
//   - Bots: Watson, Sherlock, Mycroft, Moriarty (Trained ResNet)

(function (root) {
  "use strict";

  const CELL_UNKNOWN = 0;
  const CELL_MISS = -1;
  const CELL_HIT = 1;
  const CELL_SUNK = 2;
  const FLEET_SIZES = [5, 4, 3, 3, 2];

  // Precomputed 15,046,987,728 Empty-Board Exact Joint Heatmap (normalized 0..45)
  const EMPTY_BOARD_EXACT_HEATMAP = [
    [ 9, 13, 17, 19, 21, 21, 19, 17, 13,  9],
    [13, 18, 22, 26, 28, 28, 26, 22, 18, 13],
    [17, 22, 28, 33, 35, 35, 33, 28, 22, 17],
    [19, 26, 33, 38, 41, 41, 38, 33, 26, 19],
    [21, 28, 35, 41, 45, 45, 41, 35, 28, 21],
    [21, 28, 35, 41, 45, 45, 41, 35, 28, 21],
    [19, 26, 33, 38, 41, 41, 38, 33, 26, 19],
    [17, 22, 28, 33, 35, 35, 33, 28, 22, 17],
    [13, 18, 22, 26, 28, 28, 26, 22, 18, 13],
    [ 9, 13, 17, 19, 21, 21, 19, 17, 13,  9]
  ];

  // ==========================================================================
  // 1. 100-BIT SWAR BITBOARD (4 x 32-bit Unsigned Words)
  // ==========================================================================
  function popcnt32(v) {
    v = v - ((v >>> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    return (((v + (v >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24;
  }

  class Bitboard {
    constructor(w0 = 0, w1 = 0, w2 = 0, w3 = 0) {
      this.w0 = w0 | 0; // bits 0..31
      this.w1 = w1 | 0; // bits 32..63
      this.w2 = w2 | 0; // bits 64..95
      this.w3 = (w3 & 0xF) | 0; // bits 96..99
    }

    static empty() {
      return new Bitboard(0, 0, 0, 0);
    }

    static fromCoords(coords) {
      const bb = new Bitboard();
      for (let i = 0; i < coords.length; i++) {
        const pt = coords[i];
        bb.set(pt.r !== undefined ? pt.r : pt.x, pt.c !== undefined ? pt.c : pt.y);
      }
      return bb;
    }

    copy() {
      return new Bitboard(this.w0, this.w1, this.w2, this.w3);
    }

    set(r, c) {
      const idx = r * 10 + c;
      const word = idx >>> 5;
      const bit = 1 << (idx & 31);
      if (word === 0) this.w0 |= bit;
      else if (word === 1) this.w1 |= bit;
      else if (word === 2) this.w2 |= bit;
      else this.w3 |= bit;
    }

    isSet(r, c) {
      const idx = r * 10 + c;
      const word = idx >>> 5;
      const bit = 1 << (idx & 31);
      if (word === 0) return (this.w0 & bit) !== 0;
      if (word === 1) return (this.w1 & bit) !== 0;
      if (word === 2) return (this.w2 & bit) !== 0;
      return (this.w3 & bit) !== 0;
    }

    intersects(other) {
      return (
        ((this.w0 & other.w0) |
          (this.w1 & other.w1) |
          (this.w2 & other.w2) |
          (this.w3 & other.w3)) !== 0
      );
    }

    containsAll(sub) {
      return (
        (this.w0 & sub.w0) === sub.w0 &&
        (this.w1 & sub.w1) === sub.w1 &&
        (this.w2 & sub.w2) === sub.w2 &&
        (this.w3 & sub.w3) === sub.w3
      );
    }

    bor(other) {
      return new Bitboard(
        this.w0 | other.w0,
        this.w1 | other.w1,
        this.w2 | other.w2,
        this.w3 | other.w3
      );
    }

    borInplace(other) {
      this.w0 |= other.w0;
      this.w1 |= other.w1;
      this.w2 |= other.w2;
      this.w3 |= other.w3;
    }

    band(other) {
      return new Bitboard(
        this.w0 & other.w0,
        this.w1 & other.w1,
        this.w2 & other.w2,
        this.w3 & other.w3
      );
    }

    bnot() {
      return new Bitboard(~this.w0, ~this.w1, ~this.w2, (~this.w3) & 0xF);
    }

    popcnt() {
      return popcnt32(this.w0) + popcnt32(this.w1) + popcnt32(this.w2) + popcnt32(this.w3);
    }

    toCoords() {
      const res = [];
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (this.isSet(r, c)) res.push({ r, c, x: r, y: c });
        }
      }
      return res;
    }

    toKey() {
      return `${this.w0 >>> 0}_${this.w1 >>> 0}_${this.w2 >>> 0}_${this.w3 >>> 0}`;
    }
  }

  // Precompute column masks on 100-bit grid for SWAR shift-and-count
  const col9 = Bitboard.empty();
  const col8 = Bitboard.empty();
  const col7 = Bitboard.empty();
  const col6 = Bitboard.empty();
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c <= 8; c++) col9.set(r, c);
    for (let c = 0; c <= 7; c++) col8.set(r, c);
    for (let c = 0; c <= 6; c++) col7.set(r, c);
    for (let c = 0; c <= 5; c++) col6.set(r, c);
  }

  // Analytic O(1) SWAR ship placement counter on a 100-bit free mask (f0, f1, f2, f3)
  function countShipOnFreeMask(f0, f1, f2, f3, sz) {
    // 1-bit right shift across 100 bits
    const s1_0 = (f0 >>> 1) | (f1 << 31);
    const s1_1 = (f1 >>> 1) | (f2 << 31);
    const s1_2 = (f2 >>> 1) | (f3 << 31);
    const s1_3 = f3 >>> 1;

    // 10-bit right shift across 100 bits
    const d1_0 = (f0 >>> 10) | (f1 << 22);
    const d1_1 = (f1 >>> 10) | (f2 << 22);
    const d1_2 = (f2 >>> 10) | (f3 << 22);
    const d1_3 = f3 >>> 10;

    if (sz === 2) {
      const h0 = f0 & s1_0 & col9.w0;
      const h1 = f1 & s1_1 & col9.w1;
      const h2 = f2 & s1_2 & col9.w2;
      const h3 = f3 & s1_3 & col9.w3;
      const v0 = f0 & d1_0;
      const v1 = f1 & d1_1;
      const v2 = f2 & d1_2;
      const v3 = f3 & d1_3;
      return (
        popcnt32(h0) + popcnt32(h1) + popcnt32(h2) + popcnt32(h3) +
        popcnt32(v0) + popcnt32(v1) + popcnt32(v2) + popcnt32(v3)
      );
    }

    const s2_0 = (s1_0 >>> 1) | (s1_1 << 31);
    const s2_1 = (s1_1 >>> 1) | (s1_2 << 31);
    const s2_2 = (s1_2 >>> 1) | (s1_3 << 31);
    const s2_3 = s1_3 >>> 1;

    const d2_0 = (d1_0 >>> 10) | (d1_1 << 22);
    const d2_1 = (d1_1 >>> 10) | (d1_2 << 22);
    const d2_2 = (d1_2 >>> 10) | (d1_3 << 22);
    const d2_3 = d1_3 >>> 10;

    if (sz === 3) {
      const h0 = f0 & s1_0 & s2_0 & col8.w0;
      const h1 = f1 & s1_1 & s2_1 & col8.w1;
      const h2 = f2 & s1_2 & s2_2 & col8.w2;
      const h3 = f3 & s1_3 & s2_3 & col8.w3;
      const v0 = f0 & d1_0 & d2_0;
      const v1 = f1 & d1_1 & d2_1;
      const v2 = f2 & d1_2 & d2_2;
      const v3 = f3 & d1_3 & d2_3;
      return (
        popcnt32(h0) + popcnt32(h1) + popcnt32(h2) + popcnt32(h3) +
        popcnt32(v0) + popcnt32(v1) + popcnt32(v2) + popcnt32(v3)
      );
    }

    const s3_0 = (s2_0 >>> 1) | (s2_1 << 31);
    const s3_1 = (s2_1 >>> 1) | (s2_2 << 31);
    const s3_2 = (s2_2 >>> 1) | (s2_3 << 31);
    const s3_3 = s2_3 >>> 1;

    const d3_0 = (d2_0 >>> 10) | (d2_1 << 22);
    const d3_1 = (d2_1 >>> 10) | (d2_2 << 22);
    const d3_2 = (d2_2 >>> 10) | (d2_3 << 22);
    const d3_3 = d2_3 >>> 10;

    if (sz === 4) {
      const h0 = f0 & s1_0 & s2_0 & s3_0 & col7.w0;
      const h1 = f1 & s1_1 & s2_1 & s3_1 & col7.w1;
      const h2 = f2 & s1_2 & s2_2 & s3_2 & col7.w2;
      const h3 = f3 & s1_3 & s2_3 & s3_3 & col7.w3;
      const v0 = f0 & d1_0 & d2_0 & d3_0;
      const v1 = f1 & d1_1 & d2_1 & d3_1;
      const v2 = f2 & d1_2 & d2_2 & d3_2;
      const v3 = f3 & d1_3 & d2_3 & d3_3;
      return (
        popcnt32(h0) + popcnt32(h1) + popcnt32(h2) + popcnt32(h3) +
        popcnt32(v0) + popcnt32(v1) + popcnt32(v2) + popcnt32(v3)
      );
    }

    if (sz === 5) {
      const s4_0 = (s3_0 >>> 1) | (s3_1 << 31);
      const s4_1 = (s3_1 >>> 1) | (s3_2 << 31);
      const s4_2 = (s3_2 >>> 1) | (s3_3 << 31);
      const s4_3 = s3_3 >>> 1;

      const d4_0 = (d3_0 >>> 10) | (d3_1 << 22);
      const d4_1 = (d3_1 >>> 10) | (d3_2 << 22);
      const d4_2 = (d3_2 >>> 10) | (d3_3 << 22);
      const d4_3 = d3_3 >>> 10;

      const h0 = f0 & s1_0 & s2_0 & s3_0 & s4_0 & col6.w0;
      const h1 = f1 & s1_1 & s2_1 & s3_1 & s4_1 & col6.w1;
      const h2 = f2 & s1_2 & s2_2 & s3_2 & s4_2 & col6.w2;
      const h3 = f3 & s1_3 & s2_3 & s3_3 & s4_3 & col6.w3;
      const v0 = f0 & d1_0 & d2_0 & d3_0 & d4_0;
      const v1 = f1 & d1_1 & d2_1 & d3_1 & d4_1;
      const v2 = f2 & d1_2 & d2_2 & d3_2 & d4_2;
      const v3 = f3 & d1_3 & d2_3 & d3_3 & d4_3;
      return (
        popcnt32(h0) + popcnt32(h1) + popcnt32(h2) + popcnt32(h3) +
        popcnt32(v0) + popcnt32(v1) + popcnt32(v2) + popcnt32(v3)
      );
    }
    return 0;
  }

  // ==========================================================================
  // 2. BATTLESHIP GAME ENGINE
  // ==========================================================================
  function createBlankMatrix(val = 0) {
    const m = new Array(10);
    for (let r = 0; r < 10; r++) {
      m[r] = new Array(10).fill(val);
    }
    return m;
  }

  class BattleshipGame {
    constructor() {
      this.fleetSizes = FLEET_SIZES.slice();
      this.totalPegs = 17;
      this.reset();
    }

    reset() {
      this.shipBoard = createBlankMatrix(0);
      this.trackingBoard = createBlankMatrix(CELL_UNKNOWN);
      this.shipPegs = {};
      this.shipHealth = {};
      this.moveHistory = [];
      this.shotNumber = 0;
      this.pegsHit = 0;
    }

    canPlace(r, c, length, isHorizontal) {
      if (isHorizontal) {
        if (c < 0 || c + length > 10 || r < 0 || r >= 10) return false;
        for (let i = 0; i < length; i++) {
          if (this.shipBoard[r][c + i] !== 0) return false;
        }
      } else {
        if (r < 0 || r + length > 10 || c < 0 || c >= 10) return false;
        for (let i = 0; i < length; i++) {
          if (this.shipBoard[r + i][c] !== 0) return false;
        }
      }
      return true;
    }

    placeShipManual(r, c, length, isHorizontal, shipId) {
      if (!this.canPlace(r, c, length, isHorizontal)) return false;
      const pegs = [];
      for (let i = 0; i < length; i++) {
        const pr = isHorizontal ? r : r + i;
        const pc = isHorizontal ? c + i : c;
        this.shipBoard[pr][pc] = shipId;
        pegs.push({ r: pr, c: pc });
      }
      this.shipPegs[shipId] = pegs;
      this.shipHealth[shipId] = length;
      return true;
    }

    placeShipsRandomly(seed = null) {
      this.reset();
      let rngState = seed ? ((seed * 10007 + 12345) % 2147483647) : Math.floor(Math.random() * 2147483646) + 1;
      const nextInt = (n) => {
        rngState = (rngState * 48271) % 2147483647;
        return rngState % n;
      };

      for (let idx = 0; idx < this.fleetSizes.length; idx++) {
        const shipId = idx + 1;
        const len = this.fleetSizes[idx];
        let placed = false;
        while (!placed) {
          const isHoriz = nextInt(2) === 0;
          const r = nextInt(10);
          const c = nextInt(10);
          if (this.placeShipManual(r, c, len, isHoriz, shipId)) {
            placed = true;
          }
        }
      }
    }

    loadPreset(presetId) {
      this.reset();
      this.currentPresetId = presetId;
      if (presetId === 1) {
        // Preset 1: Collinear End-to-End Train (Destroyer 2 + Cruiser 3 in Row D = 5 inline cells!)
        this.placeShipManual(3, 1, 2, true, 5);  // Destroyer (2): D2..D3
        this.placeShipManual(3, 3, 3, true, 3);  // Cruiser (3):   D4..D6 (touches D3 end-to-end!)
        this.placeShipManual(0, 0, 5, true, 1);  // Carrier (5):   A1..A5
        this.placeShipManual(5, 8, 4, false, 2); // Battleship (4): F9..I9
        this.placeShipManual(8, 1, 3, true, 4);  // Submarine (3): I2..I4
      } else if (presetId === 2) {
        // Preset 2: T-Bone / Cross Intersection Ambiguity
        this.placeShipManual(4, 2, 4, true, 2);  // Battleship (4): E3..E6 (horizontal bar)
        this.placeShipManual(1, 4, 3, false, 3); // Cruiser (3):    B5..D5 (vertical stem touching E5!)
        this.placeShipManual(0, 0, 5, false, 1); // Carrier (5):    A1..E1
        this.placeShipManual(7, 1, 3, true, 4);  // Submarine (3):  H2..H4
        this.placeShipManual(8, 7, 2, true, 5);  // Destroyer (2):  I8..I9
      } else if (presetId === 3) {
        // Preset 3: Stacked 5-Ship Parallel Phalanx Wall (All 5 ships side-by-side in Rows D..H)
        this.placeShipManual(3, 2, 5, true, 1);  // Carrier (5):    D3..D7
        this.placeShipManual(4, 2, 4, true, 2);  // Battleship (4): E3..E6
        this.placeShipManual(5, 2, 3, true, 3);  // Cruiser (3):    F3..F5
        this.placeShipManual(6, 2, 3, true, 4);  // Submarine (3):  G3..G5
        this.placeShipManual(7, 2, 2, true, 5);  // Destroyer (2):  H3..H4
      } else if (presetId === 4) {
        // Preset 4: L-Corner & Pocketed Destroyer
        this.placeShipManual(2, 2, 3, false, 3); // Cruiser (3):    C3..E3 (vertical)
        this.placeShipManual(5, 3, 3, true, 4);  // Submarine (3):  F4..F6 (horizontal L-base)
        this.placeShipManual(3, 3, 2, false, 5); // Destroyer (2):  D4..E4 (tucked inside L-pocket!)
        this.placeShipManual(0, 4, 5, true, 1);  // Carrier (5):    A5..A9
        this.placeShipManual(8, 1, 4, true, 2);  // Battleship (4): I2..I5
      } else {
        // Preset 5: Twin Size-3 Staggered Overlap (Cruiser + Submarine)
        this.placeShipManual(3, 3, 3, true, 3);  // Cruiser (3):    D4..D6
        this.placeShipManual(4, 4, 3, true, 4);  // Submarine (3):  E5..E7 (staggered overlap)
        this.placeShipManual(2, 4, 2, true, 5);  // Destroyer (2):  C5..C6 (touching above)
        this.placeShipManual(0, 0, 5, false, 1); // Carrier (5):    A1..E1
        this.placeShipManual(8, 2, 4, true, 2);  // Battleship (4): I3..I6
      }
    }

    firePresetTrapShots(presetId = null) {
      const pid = presetId || this.currentPresetId || 1;
      this.loadPreset(pid);
      let shots = [];
      if (pid === 1) {
        // Collinear train: Hit D3 (Destroyer), then D4, D5, D6 (Sinks Cruiser!), plus Misses at D7 & D1
        shots = [
          { r: 3, c: 2 }, { r: 3, c: 3 }, { r: 3, c: 4 }, { r: 3, c: 5 },
          { r: 3, c: 6 }, { r: 2, c: 2 }
        ];
      } else if (pid === 2) {
        // T-Bone junction: Hit D5 (stem), then sink E3..E6 (horizontal bar)
        shots = [
          { r: 3, c: 4 }, { r: 4, c: 2 }, { r: 4, c: 3 }, { r: 4, c: 4 }, { r: 4, c: 5 }
        ];
      } else if (pid === 3) {
        // Phalanx wall: Vertical slice down Col 4 (D4, E4, F4, G4, H4) + H3 (Sinks Destroyer at bottom!)
        shots = [
          { r: 3, c: 3 }, { r: 4, c: 3 }, { r: 5, c: 3 }, { r: 6, c: 3 },
          { r: 7, c: 2 }, { r: 7, c: 3 }
        ];
      } else if (pid === 4) {
        // L-Corner & Pocketed Destroyer: Hit pocket D4, sink C3..E3, hit F4
        shots = [
          { r: 3, c: 3 }, { r: 2, c: 2 }, { r: 3, c: 2 }, { r: 4, c: 2 }, { r: 5, c: 3 }
        ];
      } else {
        // Twin Size-3 Staggered: Sink Destroyer C5..C6, then hit D5 & E5
        shots = [
          { r: 2, c: 4 }, { r: 2, c: 5 }, { r: 3, c: 4 }, { r: 4, c: 4 }
        ];
      }
      for (const s of shots) {
        this.fireShot(s.r, s.c);
      }
    }

    fireShot(r, c) {
      if (r < 0 || r >= 10 || c < 0 || c >= 10) return "INVALID";
      if (this.trackingBoard[r][c] !== CELL_UNKNOWN) return "ALREADY_FIRED";

      this.shotNumber += 1;
      const shipId = this.shipBoard[r][c];
      let result = "MISS";
      let isSunk = false;

      if (shipId === 0) {
        this.trackingBoard[r][c] = CELL_MISS;
      } else {
        this.pegsHit += 1;
        this.shipHealth[shipId] -= 1;
        if (this.shipHealth[shipId] <= 0) {
          this.trackingBoard[r][c] = CELL_SUNK;
          result = "SUNK";
          isSunk = true;
        } else {
          this.trackingBoard[r][c] = CELL_HIT;
          result = "HIT";
        }
      }

      this.moveHistory.push({
        x: r,
        y: c,
        result,
        isSunk,
        shotNumber: this.shotNumber,
        shipId
      });
      return result;
    }

    undoLastShot() {
      if (this.moveHistory.length === 0) return false;
      const last = this.moveHistory.pop();
      this.trackingBoard[last.x][last.y] = CELL_UNKNOWN;
      this.shotNumber -= 1;
      if (last.result !== "MISS") {
        this.pegsHit -= 1;
        this.shipHealth[last.shipId] += 1;
      }
      return true;
    }

    isGameOver() {
      return this.pegsHit >= this.totalPegs || this.shotNumber >= 100;
    }
  }

  // ==========================================================================
  // 3. CHRONOLOGICAL MULTI-HYPOTHESIS CSP DEDUCTION ENGINE
  // ==========================================================================
  class DeductionEngine {
    constructor(fleetSizes = FLEET_SIZES) {
      this.fleetSizes = fleetSizes.slice();
    }

    getValidPlacements(board, sr, sc, sunkTime, size, pegTimes, claimedBb) {
      const placements = [];

      // 1. Horizontal placements passing through (sr, sc)
      for (let cStart = sc - size + 1; cStart <= sc; cStart++) {
        if (cStart >= 0 && cStart + size - 1 < 10) {
          let valid = true;
          const coords = [];
          const bb = Bitboard.empty();

          for (let i = 0; i < size; i++) {
            const r = sr;
            const c = cStart + i;
            const t = pegTimes[r * 10 + c] || -1;
            if (claimedBb.isSet(r, c)) {
              valid = false;
              break;
            }
            if (r === sr && c === sc) {
              coords.push({ r, c });
              bb.set(r, c);
            } else if (t > 0 && t < sunkTime) {
              coords.push({ r, c });
              bb.set(r, c);
            } else {
              valid = false;
              break;
            }
          }
          if (valid) placements.push({ coords, bb });
        }
      }

      // 2. Vertical placements passing through (sr, sc)
      for (let rStart = sr - size + 1; rStart <= sr; rStart++) {
        if (rStart >= 0 && rStart + size - 1 < 10) {
          let valid = true;
          const coords = [];
          const bb = Bitboard.empty();

          for (let i = 0; i < size; i++) {
            const r = rStart + i;
            const c = sc;
            const t = pegTimes[r * 10 + c] || -1;
            if (claimedBb.isSet(r, c)) {
              valid = false;
              break;
            }
            if (r === sr && c === sc) {
              coords.push({ r, c });
              bb.set(r, c);
            } else if (t > 0 && t < sunkTime) {
              coords.push({ r, c });
              bb.set(r, c);
            } else {
              valid = false;
              break;
            }
          }
          if (valid) placements.push({ coords, bb });
        }
      }

      return placements;
    }

    solveAll(board, sunks, sunkIdx, pegTimes, availableFleet, currentDeadShips, currentDeadSizes, allHitsBb, claimedBb) {
      if (sunkIdx >= sunks.length) {
        const leftoversBb = allHitsBb.band(claimedBb.bnot());
        const numLeftovers = leftoversBb.popcnt();

        if (availableFleet.length > 0 && numLeftovers > 0) {
          let minShipSize = 5;
          for (let i = 0; i < availableFleet.length; i++) {
            if (availableFleet[i] < minShipSize) minShipSize = availableFleet[i];
          }

          const leftoverCoords = leftoversBb.toCoords();
          for (let i = 0; i < leftoverCoords.length; i++) {
            const { r, c } = leftoverCoords[i];

            let left = c;
            while (left > 0 && (board[r][left - 1] === 0 || board[r][left - 1] === 1) && !claimedBb.isSet(r, left - 1)) {
              left--;
            }
            let right = c;
            while (right < 9 && (board[r][right + 1] === 0 || board[r][right + 1] === 1) && !claimedBb.isSet(r, right + 1)) {
              right++;
            }
            const horizLen = right - left + 1;
            let horizUnknowns = 0;
            for (let col = left; col <= right; col++) {
              if (board[r][col] === 0) horizUnknowns++;
            }

            let up = r;
            while (up > 0 && (board[up - 1][c] === 0 || board[up - 1][c] === 1) && !claimedBb.isSet(up - 1, c)) {
              up--;
            }
            let down = r;
            while (down < 9 && (board[down + 1][c] === 0 || board[down + 1][c] === 1) && !claimedBb.isSet(down + 1, c)) {
              down++;
            }
            const vertLen = down - up + 1;
            let vertUnknowns = 0;
            for (let row = up; row <= down; row++) {
              if (board[row][c] === 0) vertUnknowns++;
            }

            const validHoriz = horizLen >= minShipSize && horizUnknowns > 0;
            const validVert = vertLen >= minShipSize && vertUnknowns > 0;
            if (!validHoriz && !validVert) {
              return { score: Infinity, solutions: [] };
            }
          }
        }

        const deadShipsCopy = currentDeadShips.map((s) => s.map((pt) => ({ r: pt.r, c: pt.c })));
        const deadSizesCopy = currentDeadSizes.slice();

        return {
          score: numLeftovers,
          solutions: [
            {
              deadShips: deadShipsCopy,
              deadSizes: deadSizesCopy,
              claimedBb: claimedBb.copy(),
              leftoversBb: leftoversBb.copy(),
              score: numLeftovers
            }
          ]
        };
      }

      const sunk = sunks[sunkIdx];
      const sunkTime = pegTimes[sunk.r * 10 + sunk.c] || Infinity;

      const uniqueSizes = [];
      const seenSizes = {};
      for (let i = 0; i < availableFleet.length; i++) {
        const sz = availableFleet[i];
        if (!seenSizes[sz]) {
          seenSizes[sz] = true;
          uniqueSizes.push(sz);
        }
      }
      uniqueSizes.sort((a, b) => b - a);

      let bestScore = Infinity;
      let bestSolutions = [];

      for (let u = 0; u < uniqueSizes.length; u++) {
        const size = uniqueSizes[u];
        const candidates = this.getValidPlacements(board, sunk.r, sunk.c, sunkTime, size, pegTimes, claimedBb);

        for (let p = 0; p < candidates.length; p++) {
          const placement = candidates[p];
          if (!claimedBb.intersects(placement.bb)) {
            const nextClaimed = claimedBb.bor(placement.bb);
            currentDeadShips.push(placement.coords);
            currentDeadSizes.push(size);

            const nextFleet = [];
            let removed = false;
            for (let f = 0; f < availableFleet.length; f++) {
              if (availableFleet[f] === size && !removed) {
                removed = true;
              } else {
                nextFleet.push(availableFleet[f]);
              }
            }

            const child = this.solveAll(
              board,
              sunks,
              sunkIdx + 1,
              pegTimes,
              nextFleet,
              currentDeadShips,
              currentDeadSizes,
              allHitsBb,
              nextClaimed
            );

            if (child.solutions.length > 0) {
              if (child.score < bestScore) {
                bestScore = child.score;
                bestSolutions = child.solutions.slice();
              } else if (child.score === bestScore) {
                for (let s = 0; s < child.solutions.length; s++) {
                  bestSolutions.push(child.solutions[s]);
                }
              }
            }

            currentDeadShips.pop();
            currentDeadSizes.pop();
          }
        }
      }

      return { score: bestScore, solutions: bestSolutions };
    }

    analyze(board, moveHistory) {
      const allHitsBb = Bitboard.empty();
      const sunks = [];
      const pegTimes = new Int32Array(100);

      if (moveHistory) {
        for (let i = 0; i < moveHistory.length; i++) {
          const m = moveHistory[i];
          if (m.result !== "MISS") {
            pegTimes[m.x * 10 + m.y] = m.shotNumber || i + 1;
          }
        }
      }

      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const val = board[r][c];
          if (val === CELL_HIT || val === CELL_SUNK) {
            allHitsBb.set(r, c);
          }
          if (val === CELL_SUNK) {
            sunks.push({ r, c });
          }
        }
      }

      if (sunks.length === 0) {
        return [
          {
            activeHits: allHitsBb.toCoords(),
            deadShips: [],
            claimedPegs: [],
            deadSizes: [],
            score: allHitsBb.popcnt()
          }
        ];
      }

      sunks.sort((a, b) => (pegTimes[a.r * 10 + a.c] || 0) - (pegTimes[b.r * 10 + b.c] || 0));

      const { solutions: rawSolutions } = this.solveAll(
        board,
        sunks,
        0,
        pegTimes,
        this.fleetSizes.slice(),
        [],
        [],
        allHitsBb,
        Bitboard.empty()
      );

      if (rawSolutions.length === 0) {
        const fbDeadShips = [];
        const fbClaimed = Bitboard.empty();
        for (let i = 0; i < sunks.length; i++) {
          fbDeadShips.push([sunks[i]]);
          fbClaimed.set(sunks[i].r, sunks[i].c);
        }
        const fbActive = allHitsBb.band(fbClaimed.bnot()).toCoords();
        return [
          {
            activeHits: fbActive,
            deadShips: fbDeadShips,
            claimedPegs: fbClaimed.toCoords(),
            deadSizes: [],
            score: fbActive.length
          }
        ];
      }

      const parsed = [];
      const seenKeys = {};

      for (let i = 0; i < rawSolutions.length; i++) {
        const sol = rawSolutions[i];
        sol.deadSizes.sort((a, b) => a - b);
        const key = `${sol.deadSizes.join(",")}|${sol.claimedBb.toKey()}`;
        if (!seenKeys[key]) {
          seenKeys[key] = true;
          parsed.push({
            activeHits: sol.leftoversBb.toCoords(),
            deadShips: sol.deadShips,
            claimedPegs: sol.claimedBb.toCoords(),
            deadSizes: sol.deadSizes,
            score: sol.score
          });
        }
      }

      return parsed;
    }

    static isSizeDead(solutions, size) {
      if (!solutions || solutions.length === 0) return false;
      for (let i = 0; i < solutions.length; i++) {
        if (!solutions[i].deadSizes.includes(size)) return false;
      }
      return true;
    }

    static hasActiveHits(solutions) {
      if (!solutions || solutions.length === 0) return false;
      for (let i = 0; i < solutions.length; i++) {
        if (solutions[i].activeHits.length > 0) return true;
      }
      return false;
    }

    static buildDiagnostics(solutions) {
      if (!solutions || solutions.length === 0) {
        return {
          active: [],
          claimed: [],
          deadShips: [],
          deadSizes: [],
          solutionsCount: 0,
          isTwoShipDead: false,
          hasActive: false
        };
      }

      const primary = solutions[0];
      const hasActive = DeductionEngine.hasActiveHits(solutions);
      const is2Dead = DeductionEngine.isSizeDead(solutions, 2);

      if (solutions.length === 1) {
        return {
          active: primary.activeHits,
          claimed: primary.claimedPegs,
          deadShips: primary.deadShips,
          deadSizes: primary.deadSizes,
          solutionsCount: 1,
          isTwoShipDead: is2Dead,
          hasActive
        };
      }

      // Multi-hypothesis consensus when 2+ explanations tie:
      // 1. Active Hits = UNION across all tied hypotheses (every disputed hit gets a cyan ring)
      // 2. Claimed Pegs = INTERSECTION across all tied hypotheses (only 100% agreed-upon dead cells are locked)
      const activeBb = Bitboard.empty();
      let consensusClaimedBb = Bitboard.fromCoords(primary.claimedPegs);

      for (let s = 0; s < solutions.length; s++) {
        const sol = solutions[s];
        for (let i = 0; i < sol.activeHits.length; i++) {
          activeBb.set(sol.activeHits[i].r, sol.activeHits[i].c);
        }
        const solClaimedBb = Bitboard.fromCoords(sol.claimedPegs);
        consensusClaimedBb = consensusClaimedBb.band(solClaimedBb);
      }

      // Trim each deadShip hull to only the cells guaranteed dead in 100% of tied hypotheses
      const consensusDeadShips = [];
      for (let i = 0; i < primary.deadShips.length; i++) {
        const hull = primary.deadShips[i];
        const trimmed = [];
        for (let j = 0; j < hull.length; j++) {
          if (consensusClaimedBb.isSet(hull[j].r, hull[j].c)) {
            trimmed.push({ r: hull[j].r, c: hull[j].c });
          }
        }
        if (trimmed.length > 0) {
          consensusDeadShips.push(trimmed);
        }
      }

      return {
        active: activeBb.toCoords(),
        claimed: consensusClaimedBb.toCoords(),
        deadShips: consensusDeadShips,
        deadSizes: primary.deadSizes,
        solutionsCount: solutions.length,
        isTwoShipDead: is2Dead,
        hasActive
      };
    }
  }

  // ==========================================================================
  // 4. COMBINATORICS (760 Precomputed Placements + P1/P2 Joint Covers)
  // ==========================================================================
  class Combinatorics {
    constructor() {
      this.placementsBySize = {};
      this.initPlacements();
    }

    initPlacements() {
      const sizes = [5, 4, 3, 2];
      for (let s = 0; s < sizes.length; s++) {
        const size = sizes[s];
        const list = [];
        // Horizontal
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c <= 10 - size; c++) {
            const bb = Bitboard.empty();
            const coords = [];
            for (let i = 0; i < size; i++) {
              bb.set(r, c + i);
              coords.push({ r, c: c + i });
            }
            list.push({ size, mask: bb, coords });
          }
        }
        // Vertical
        for (let c = 0; c < 10; c++) {
          for (let r = 0; r <= 10 - size; r++) {
            const bb = Bitboard.empty();
            const coords = [];
            for (let i = 0; i < size; i++) {
              bb.set(r + i, c);
              coords.push({ r: r + i, c });
            }
            list.push({ size, mask: bb, coords });
          }
        }
        this.placementsBySize[size] = list;
      }
    }

    getPlacements(size) {
      return this.placementsBySize[size] || [];
    }

    computeHuntDensity(availableFleet, forbiddenBb) {
      const density = createBlankMatrix(0);
      let maxD = 0;
      for (let s = 0; s < availableFleet.length; s++) {
        const plist = this.placementsBySize[availableFleet[s]] || [];
        for (let i = 0; i < plist.length; i++) {
          const p = plist[i];
          if (!p.mask.intersects(forbiddenBb)) {
            for (let k = 0; k < p.coords.length; k++) {
              const pt = p.coords[k];
              const d = density[pt.r][pt.c] + 1;
              density[pt.r][pt.c] = d;
              if (d > maxD) maxD = d;
            }
          }
        }
      }
      return { density, maxD };
    }

    computeTargetDensity(board, availableFleet, forbiddenBb, activeHitsBb) {
      const counts = {};
      let maxCount = 0;

      const addCount = (r, c, weight = 1) => {
        if (board[r][c] === CELL_UNKNOWN) {
          const idx = r * 10 + c;
          const cur = (counts[idx] || 0) + weight;
          counts[idx] = cur;
          if (cur > maxCount) maxCount = cur;
        }
      };

      // PHASE 1: Single-Ship Exact Covers (P1)
      const singleCovers = [];
      for (let s = 0; s < availableFleet.length; s++) {
        const plist = this.placementsBySize[availableFleet[s]] || [];
        for (let i = 0; i < plist.length; i++) {
          const p = plist[i];
          if (!p.mask.intersects(forbiddenBb) && p.mask.containsAll(activeHitsBb)) {
            let hasUnknown = false;
            for (let k = 0; k < p.coords.length; k++) {
              const pt = p.coords[k];
              if (board[pt.r][pt.c] === CELL_UNKNOWN) {
                hasUnknown = true;
                break;
              }
            }
            if (hasUnknown) singleCovers.push(p);
          }
        }
      }

      if (singleCovers.length > 0) {
        for (let i = 0; i < singleCovers.length; i++) {
          const coords = singleCovers[i].coords;
          for (let k = 0; k < coords.length; k++) {
            addCount(coords[k].r, coords[k].c, 1);
          }
        }
        return { counts, maxCount };
      }

      // PHASE 2: Disjoint 2-Ship Joint Covers (P1, P2)
      const candidates = [];
      for (let shipIdx = 0; shipIdx < availableFleet.length; shipIdx++) {
        const size = availableFleet[shipIdx];
        const plist = this.placementsBySize[size] || [];
        for (let i = 0; i < plist.length; i++) {
          const p = plist[i];
          if (!p.mask.intersects(forbiddenBb) && p.mask.intersects(activeHitsBb)) {
            let hasUnknown = false;
            for (let k = 0; k < p.coords.length; k++) {
              const pt = p.coords[k];
              if (board[pt.r][pt.c] === CELL_UNKNOWN) {
                hasUnknown = true;
                break;
              }
            }
            if (hasUnknown) {
              candidates.push({
                shipIdx,
                size,
                mask: p.mask,
                coords: p.coords,
                hitOverlap: p.mask.band(activeHitsBb)
              });
            }
          }
        }
      }

      const pairCovers = [];
      for (let i = 0; i < candidates.length; i++) {
        const c1 = candidates[i];
        for (let j = i + 1; j < candidates.length; j++) {
          const c2 = candidates[j];
          if (c1.shipIdx !== c2.shipIdx && !c1.mask.intersects(c2.mask)) {
            const unionHits = c1.hitOverlap.bor(c2.hitOverlap);
            if (unionHits.containsAll(activeHitsBb)) {
              pairCovers.push([c1, c2]);
            }
          }
        }
      }

      if (pairCovers.length > 0) {
        for (let i = 0; i < pairCovers.length; i++) {
          const [c1, c2] = pairCovers[i];
          for (let k = 0; k < c1.coords.length; k++) addCount(c1.coords[k].r, c1.coords[k].c, 1);
          for (let k = 0; k < c2.coords.length; k++) addCount(c2.coords[k].r, c2.coords[k].c, 1);
        }
        return { counts, maxCount };
      }

      // PHASE 3: Weighted Fallback (3+ Touching Ships)
      for (let i = 0; i < candidates.length; i++) {
        const c = candidates[i];
        const hc = c.hitOverlap.popcnt();
        const w = hc * hc;
        for (let k = 0; k < c.coords.length; k++) {
          addCount(c.coords[k].r, c.coords[k].c, w);
        }
      }

      return { counts, maxCount };
    }
  }

  // ==========================================================================
  // 5. EXACT JOINT DENSITY SOLVER (SWAR Dual-Pass + Incremental Miss-Subtraction)
  // ==========================================================================
  class JointSolver {
    constructor(combinatorics) {
      this.combinatorics = combinatorics || new Combinatorics();
    }

    computeExactJointDensity(availableFleet, forbiddenBb) {
      if (!availableFleet || availableFleet.length === 0) {
        return { density: createBlankMatrix(0), totalConfigs: 0, maxCount: 0 };
      }

      const sortedFleet = availableFleet.slice().sort((a, b) => b - a);
      const k = sortedFleet.length;

      if (k >= 5) {
        const uniqueFleet = [];
        const seen = {};
        for (let i = 0; i < sortedFleet.length; i++) {
          if (!seen[sortedFleet[i]]) {
            seen[sortedFleet[i]] = true;
            uniqueFleet.push(sortedFleet[i]);
          }
        }
        const { density, maxD } = this.combinatorics.computeHuntDensity(uniqueFleet, forbiddenBb);
        return { density, totalConfigs: 1, maxCount: maxD };
      }

      const lists = new Array(k);
      const weights = new Array(k);
      const isSame = new Array(k);

      for (let i = 0; i < k; i++) {
        const sz = sortedFleet[i];
        const raw = this.combinatorics.getPlacements(sz);
        const flt = [];
        for (let j = 0; j < raw.length; j++) {
          if (!raw[j].mask.intersects(forbiddenBb)) {
            flt.push(raw[j]);
          }
        }
        if (flt.length === 0) {
          return { density: createBlankMatrix(0), totalConfigs: 0, maxCount: 0 };
        }
        lists[i] = flt;
        weights[i] = new Float64Array(flt.length);
        isSame[i] = i > 0 && sortedFleet[i] === sortedFleet[i - 1];
      }

      let totalConfigs = 0;

      if (k === 1) {
        const L1 = lists[0];
        const w1 = weights[0];
        totalConfigs = L1.length;
        for (let j = 0; j < totalConfigs; j++) w1[j] = 1;
      } else if (k === 2) {
        const L1 = lists[0], L2 = lists[1];
        const w1 = weights[0], w2 = weights[1];
        const same = isSame[1];

        for (let i = 0; i < L1.length; i++) {
          const m1 = L1[i].mask;
          const startJ = same ? i + 1 : 0;
          let countForI = 0;
          for (let j = startJ; j < L2.length; j++) {
            const m2 = L2[j].mask;
            if (((m1.w0 & m2.w0) | (m1.w1 & m2.w1) | (m1.w2 & m2.w2) | (m1.w3 & m2.w3)) === 0) {
              countForI++;
              w2[j] += 1;
            }
          }
          if (countForI > 0) {
            w1[i] += countForI;
            totalConfigs += countForI;
          }
        }
      } else if (k === 3) {
        const L1 = lists[0], L2 = lists[1], L3 = lists[2];
        const w1 = weights[0], w2 = weights[1], w3 = weights[2];
        const same2 = isSame[1], same3 = isSame[2];

        for (let i = 0; i < L1.length; i++) {
          const m1 = L1[i].mask;
          const startJ = same2 ? i + 1 : 0;
          for (let j = startJ; j < L2.length; j++) {
            const m2 = L2[j].mask;
            if (((m1.w0 & m2.w0) | (m1.w1 & m2.w1) | (m1.w2 & m2.w2) | (m1.w3 & m2.w3)) === 0) {
              const m12_0 = m1.w0 | m2.w0;
              const m12_1 = m1.w1 | m2.w1;
              const m12_2 = m1.w2 | m2.w2;
              const m12_3 = m1.w3 | m2.w3;
              const startK = same3 ? j + 1 : 0;
              let countForJ = 0;
              for (let l = startK; l < L3.length; l++) {
                const m3 = L3[l].mask;
                if (((m12_0 & m3.w0) | (m12_1 & m3.w1) | (m12_2 & m3.w2) | (m12_3 & m3.w3)) === 0) {
                  countForJ++;
                  w3[l] += 1;
                }
              }
              if (countForJ > 0) {
                w2[j] += countForJ;
                w1[i] += countForJ;
                totalConfigs += countForJ;
              }
            }
          }
        }
      } else if (k === 4) {
        const L1 = lists[0], L2 = lists[1], L3 = lists[2], L4 = lists[3];
        const w1 = weights[0], w2 = weights[1], w3 = weights[2], w4 = weights[3];
        const same2 = isSame[1], same3 = isSame[2], same4 = isSame[3];
        const sz1 = sortedFleet[0], sz4 = sortedFleet[3];
        const fb0 = forbiddenBb.w0, fb1 = forbiddenBb.w1, fb2 = forbiddenBb.w2, fb3 = forbiddenBb.w3;

        if (!same4) {
          // PASS 1: Loop (S1, S2, S3), evaluate S4 analytically in O(1) via SWAR shifts
          for (let i = 0; i < L1.length; i++) {
            const m1 = L1[i].mask;
            const startJ = same2 ? i + 1 : 0;
            for (let j = startJ; j < L2.length; j++) {
              const m2 = L2[j].mask;
              if (((m1.w0 & m2.w0) | (m1.w1 & m2.w1) | (m1.w2 & m2.w2) | (m1.w3 & m2.w3)) === 0) {
                const m12_0 = m1.w0 | m2.w0;
                const m12_1 = m1.w1 | m2.w1;
                const m12_2 = m1.w2 | m2.w2;
                const m12_3 = m1.w3 | m2.w3;
                const startK = same3 ? j + 1 : 0;
                for (let kIdx = startK; kIdx < L3.length; kIdx++) {
                  const m3 = L3[kIdx].mask;
                  if (((m12_0 & m3.w0) | (m12_1 & m3.w1) | (m12_2 & m3.w2) | (m12_3 & m3.w3)) === 0) {
                    const f0 = ~(m12_0 | m3.w0 | fb0);
                    const f1 = ~(m12_1 | m3.w1 | fb1);
                    const f2 = ~(m12_2 | m3.w2 | fb2);
                    const f3 = (~(m12_3 | m3.w3 | fb3)) & 0xF;

                    const cnt = countShipOnFreeMask(f0, f1, f2, f3, sz4);
                    if (cnt > 0) {
                      w3[kIdx] += cnt;
                      w2[j] += cnt;
                      w1[i] += cnt;
                      totalConfigs += cnt;
                    }
                  }
                }
              }
            }
          }

          // PASS 2: Loop (S2, S3, S4), evaluate S1 analytically in O(1) for w4 weights
          for (let j = 0; j < L2.length; j++) {
            const m2 = L2[j].mask;
            const startK = same3 ? j + 1 : 0;
            for (let kIdx = startK; kIdx < L3.length; kIdx++) {
              const m3 = L3[kIdx].mask;
              if (((m2.w0 & m3.w0) | (m2.w1 & m3.w1) | (m2.w2 & m3.w2) | (m2.w3 & m3.w3)) === 0) {
                const m23_0 = m2.w0 | m3.w0;
                const m23_1 = m2.w1 | m3.w1;
                const m23_2 = m2.w2 | m3.w2;
                const m23_3 = m2.w3 | m3.w3;
                for (let l = 0; l < L4.length; l++) {
                  const m4 = L4[l].mask;
                  if (((m23_0 & m4.w0) | (m23_1 & m4.w1) | (m23_2 & m4.w2) | (m23_3 & m4.w3)) === 0) {
                    const f0 = ~(m23_0 | m4.w0 | fb0);
                    const f1 = ~(m23_1 | m4.w1 | fb1);
                    const f2 = ~(m23_2 | m4.w2 | fb2);
                    const f3 = (~(m23_3 | m4.w3 | fb3)) & 0xF;

                    const cnt = countShipOnFreeMask(f0, f1, f2, f3, sz1);
                    if (cnt > 0) {
                      w4[l] += cnt;
                    }
                  }
                }
              }
            }
          }
        } else {
          // Symmetric identical sizes (e.g. [5, 4, 3, 3])
          for (let i = 0; i < L1.length; i++) {
            const m1 = L1[i].mask;
            const startJ = same2 ? i + 1 : 0;
            for (let j = startJ; j < L2.length; j++) {
              const m2 = L2[j].mask;
              if (((m1.w0 & m2.w0) | (m1.w1 & m2.w1) | (m1.w2 & m2.w2) | (m1.w3 & m2.w3)) === 0) {
                const m12_0 = m1.w0 | m2.w0;
                const m12_1 = m1.w1 | m2.w1;
                const m12_2 = m1.w2 | m2.w2;
                const m12_3 = m1.w3 | m2.w3;
                const startK = same3 ? j + 1 : 0;
                for (let kIdx = startK; kIdx < L3.length; kIdx++) {
                  const m3 = L3[kIdx].mask;
                  if (((m12_0 & m3.w0) | (m12_1 & m3.w1) | (m12_2 & m3.w2) | (m12_3 & m3.w3)) === 0) {
                    const m123_0 = m12_0 | m3.w0;
                    const m123_1 = m12_1 | m3.w1;
                    const m123_2 = m12_2 | m3.w2;
                    const m123_3 = m12_3 | m3.w3;
                    let countForK = 0;
                    for (let l = kIdx + 1; l < L4.length; l++) {
                      const m4 = L4[l].mask;
                      if (((m123_0 & m4.w0) | (m123_1 & m4.w1) | (m123_2 & m4.w2) | (m123_3 & m4.w3)) === 0) {
                        countForK++;
                        w4[l] += 1;
                      }
                    }
                    if (countForK > 0) {
                      w3[kIdx] += countForK;
                      w2[j] += countForK;
                      w1[i] += countForK;
                      totalConfigs += countForK;
                    }
                  }
                }
              }
            }
          }
        }
      }

      const density = createBlankMatrix(0);
      let maxCount = 0;
      for (let i = 0; i < k; i++) {
        const plist = lists[i];
        const wlist = weights[i];
        for (let j = 0; j < plist.length; j++) {
          const w = wlist[j];
          if (w > 0) {
            const coords = plist[j].coords;
            for (let cIdx = 0; cIdx < coords.length; cIdx++) {
              const pt = coords[cIdx];
              const d = density[pt.r][pt.c] + w;
              density[pt.r][pt.c] = d;
              if (d > maxCount) maxCount = d;
            }
          }
        }
      }

      return { density, totalConfigs, maxCount };
    }

    computeRuinedDensity(availableFleet, forbiddenBb, missR, missC) {
      const dRuined = createBlankMatrix(0);
      let zRuined = 0;
      if (!availableFleet || availableFleet.length === 0) {
        return { dRuined, zRuined };
      }

      const seenSz = {};
      for (let i = 0; i < availableFleet.length; i++) {
        const sz = availableFleet[i];
        if (!seenSz[sz]) {
          seenSz[sz] = true;
          const allP = this.combinatorics.getPlacements(sz);
          for (let j = 0; j < allP.length; j++) {
            const p = allP[j];
            if (p.mask.isSet(missR, missC) && !p.mask.intersects(forbiddenBb)) {
              const subFleet = [];
              let removed = false;
              for (let s = 0; s < availableFleet.length; s++) {
                if (availableFleet[s] === sz && !removed) {
                  removed = true;
                } else {
                  subFleet.push(availableFleet[s]);
                }
              }
              const pinnedBb = forbiddenBb.bor(p.mask);
              const { density: dSub, totalConfigs: zSub } = this.computeExactJointDensity(subFleet, pinnedBb);
              if (zSub > 0) {
                zRuined += zSub;
                for (let k = 0; k < p.coords.length; k++) {
                  const pt = p.coords[k];
                  dRuined[pt.r][pt.c] += zSub;
                }
                for (let r = 0; r < 10; r++) {
                  for (let c = 0; c < 10; c++) {
                    if (dSub[r][c] > 0) dRuined[r][c] += dSub[r][c];
                  }
                }
              }
            }
          }
        }
      }

      return { dRuined, zRuined };
    }

    subtractDensity(dBase, zBase, dRuined, zRuined) {
      const dNext = createBlankMatrix(0);
      let maxCount = 0;
      const zNext = Math.max(0, zBase - zRuined);
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const val = Math.max(0, dBase[r][c] - dRuined[r][c]);
          dNext[r][c] = val;
          if (val > maxCount) maxCount = val;
        }
      }
      return { density: dNext, totalConfigs: zNext, maxCount };
    }
  }

  // ==========================================================================
  // 6. BOT ARCHITECTURES (WATSON, SHERLOCK, MYCROFT, MORIARTY)
  // ==========================================================================
  function getAvailableFleet(fleetSizes, deadSizes) {
    const deadMap = {};
    if (deadSizes) {
      for (let i = 0; i < deadSizes.length; i++) {
        const sz = deadSizes[i];
        deadMap[sz] = (deadMap[sz] || 0) + 1;
      }
    }
    const available = [];
    for (let i = 0; i < fleetSizes.length; i++) {
      const sz = fleetSizes[i];
      if ((deadMap[sz] || 0) > 0) {
        deadMap[sz]--;
      } else {
        available.push(sz);
      }
    }
    return available;
  }

  function isCellOnStride(r, c, stride, parityOffset) {
    return (r + c) % stride === (parityOffset % stride);
  }

  function normalizeHeatmap(matrix) {
    const out = createBlankMatrix(0);
    let maxV = 0;
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 10; c++) {
        if (matrix[r][c] > maxV) maxV = matrix[r][c];
      }
    }
    if (maxV > 0) {
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          out[r][c] = Math.floor((matrix[r][c] / maxV) * 100);
        }
      }
    }
    return out;
  }

  // Shared Bayesian Joint Target Mode (P1, P2) across tied CSP solutions
  function solveTargetMode(board, moveHistory, solutions, missesBb, combinatorics, parityOffset, openingFlip = 0) {
    const totalTargetCounts = {};
    let maxTargetCount = 0;

    for (let s = 0; s < solutions.length; s++) {
      const sol = solutions[s];
      if (sol.activeHits.length > 0) {
        const forbiddenBb = missesBb.copy();
        for (let i = 0; i < sol.claimedPegs.length; i++) {
          forbiddenBb.set(sol.claimedPegs[i].r, sol.claimedPegs[i].c);
        }
        const activeHitsBb = Bitboard.fromCoords(sol.activeHits);
        const availableFleet = getAvailableFleet(FLEET_SIZES, sol.deadSizes);
        const { counts } = combinatorics.computeTargetDensity(board, availableFleet, forbiddenBb, activeHitsBb);

        for (const idxStr in counts) {
          const idx = Number(idxStr);
          const cur = (totalTargetCounts[idx] || 0) + counts[idx];
          totalTargetCounts[idx] = cur;
          if (cur > maxTargetCount) maxTargetCount = cur;
        }
      }
    }

    const heatmap = createBlankMatrix(0);
    if (maxTargetCount > 0) {
      for (const idxStr in totalTargetCounts) {
        const idx = Number(idxStr);
        const r = Math.floor(idx / 10);
        const c = idx % 10;
        heatmap[r][c] = Math.floor((totalTargetCounts[idx] / maxTargetCount) * 100);
      }
    }

    const bestTargets = [];
    for (const idxStr in totalTargetCounts) {
      const idx = Number(idxStr);
      if (totalTargetCounts[idx] === maxTargetCount) {
        const r = Math.floor(idx / 10);
        const c = idx % 10;
        if (board[r][c] === CELL_UNKNOWN) {
          bestTargets.push({ x: r, y: c });
        }
      }
    }

    if (bestTargets.length > 0) {
      const pickIdx = (((parityOffset + openingFlip) * 10007 + moveHistory.length) % bestTargets.length);
      return {
        target: bestTargets[pickIdx],
        heatmap,
        maxTargetCount,
        ties: bestTargets.length
      };
    }
    return null;
  }

  // --------------------------------------------------------------------------
  // 6A. WATSON BOT (Chronological CSP + Unique Marginal Parity Hunt)
  // --------------------------------------------------------------------------
  class WatsonBot {
    constructor(parityOffset = 0, sharedCombinatorics = null, sharedDeduction = null) {
      this.name = "Watson";
      this.parityOffset = parityOffset;
      this.combinatorics = sharedCombinatorics || new Combinatorics();
      this.deduction = sharedDeduction || new DeductionEngine();
      this.lastHeatmap = createBlankMatrix(0);
    }

    reset(gameId = 1, parityOffset = 0) {
      this.parityOffset = parityOffset;
      this.lastHeatmap = createBlankMatrix(0);
    }

    getBestMove(board, moveHistory) {
      const solutions = this.deduction.analyze(board, moveHistory);
      const primary = solutions[0];
      const hasActive = DeductionEngine.hasActiveHits(solutions);
      const is2Dead = DeductionEngine.isSizeDead(solutions, 2);

      const missesBb = Bitboard.empty();
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (board[r][c] === CELL_MISS) missesBb.set(r, c);
        }
      }

      const diagnostics = DeductionEngine.buildDiagnostics(solutions);

      if (hasActive) {
        const res = solveTargetMode(board, moveHistory, solutions, missesBb, this.combinatorics, this.parityOffset, 0);
        if (res) {
          this.lastHeatmap = res.heatmap;
          return {
            target: res.target,
            log: `Target Cover (peak=${res.maxTargetCount})`,
            deadSizes: primary.deadSizes,
            diagnostics,
            heatmap: this.lastHeatmap
          };
        }
      }

      const stride = is2Dead ? 3 : 2;
      const totalHunt = createBlankMatrix(0);

      for (let s = 0; s < solutions.length; s++) {
        const sol = solutions[s];
        const forbiddenBb = missesBb.copy();
        for (let i = 0; i < sol.claimedPegs.length; i++) {
          forbiddenBb.set(sol.claimedPegs[i].r, sol.claimedPegs[i].c);
        }
        const avail = getAvailableFleet(FLEET_SIZES, sol.deadSizes);
        const seen = {};
        const uniqueFleet = [];
        for (let i = 0; i < avail.length; i++) {
          if (!seen[avail[i]]) {
            seen[avail[i]] = true;
            uniqueFleet.push(avail[i]);
          }
        }
        const { density } = this.combinatorics.computeHuntDensity(uniqueFleet, forbiddenBb);
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            if (board[r][c] === CELL_UNKNOWN) {
              totalHunt[r][c] += density[r][c];
            }
          }
        }
      }

      this.lastHeatmap = normalizeHeatmap(totalHunt);

      let bestScore = -1;
      let candidates = [];
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (board[r][c] === CELL_UNKNOWN && isCellOnStride(r, c, stride, this.parityOffset)) {
            const d = totalHunt[r][c];
            if (d > bestScore) {
              bestScore = d;
              candidates = [{ x: r, y: c }];
            } else if (d === bestScore && d >= 0) {
              candidates.push({ x: r, y: c });
            }
          }
        }
      }

      if (bestScore <= 0 || candidates.length === 0) {
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            if (board[r][c] === CELL_UNKNOWN) {
              const d = totalHunt[r][c];
              if (d > bestScore) {
                bestScore = d;
                candidates = [{ x: r, y: c }];
              } else if (d === bestScore) {
                candidates.push({ x: r, y: c });
              }
            }
          }
        }
      }

      const pickIdx = ((this.parityOffset * 10007 + moveHistory.length) % Math.max(1, candidates.length));
      const target = candidates[pickIdx] || { x: 4, y: 4 };
      return {
        target,
        log: `Marginal Parity (Stride-${stride})`,
        deadSizes: primary.deadSizes,
        diagnostics,
        heatmap: this.lastHeatmap
      };
    }
  }

  // --------------------------------------------------------------------------
  // 6B. SHERLOCK BOT (Exact Joint Distribution + Incremental Miss-Subtraction)
  // --------------------------------------------------------------------------
  class SherlockBot {
    constructor(parityOffset = 0, sharedCombinatorics = null, sharedDeduction = null, sharedJoint = null) {
      this.name = "Sherlock";
      this.parityOffset = parityOffset;
      this.openingFlip = 0;
      this.combinatorics = sharedCombinatorics || new Combinatorics();
      this.deduction = sharedDeduction || new DeductionEngine();
      this.jointSolver = sharedJoint || new JointSolver(this.combinatorics);
      this.cachedHuntState = null;
      this.lastHeatmap = createBlankMatrix(0);
    }

    reset(gameId = 1, parityOffset = 0) {
      this.parityOffset = parityOffset;
      this.openingFlip = gameId % 2;
      this.cachedHuntState = null;
      this.lastHeatmap = createBlankMatrix(0);
    }

    computeJointHunt(board, moveHistory, solutions, missesBb) {
      const primary = solutions[0];
      const lastMove = moveHistory.length > 0 ? moveHistory[moveHistory.length - 1] : null;
      const primaryForbiddenBb = missesBb.copy();
      for (let i = 0; i < primary.claimedPegs.length; i++) {
        primaryForbiddenBb.set(primary.claimedPegs[i].r, primary.claimedPegs[i].c);
      }
      const primaryAvail = getAvailableFleet(FLEET_SIZES, primary.deadSizes);

      let canIncremental = false;
      if (
        this.cachedHuntState &&
        solutions.length === 1 &&
        lastMove &&
        lastMove.result === "MISS" &&
        primaryAvail.length <= 4
      ) {
        const prevFleet = this.cachedHuntState.fleet;
        if (
          prevFleet.length === primaryAvail.length &&
          prevFleet.every((v, idx) => v === primaryAvail[idx])
        ) {
          const prevFbb = this.cachedHuntState.forbiddenBb;
          const testFbb = prevFbb.copy();
          testFbb.set(lastMove.x, lastMove.y);
          if (testFbb.toKey() === primaryForbiddenBb.toKey() && !prevFbb.isSet(lastMove.x, lastMove.y)) {
            canIncremental = true;
          }
        }
      }

      let combinedCounts = createBlankMatrix(0);
      let maxHuntVal = 0;
      let totalConfigs = 0;
      let modeTag = "Exact Joint";

      if (canIncremental) {
        const { dRuined, zRuined } = this.jointSolver.computeRuinedDensity(
          primaryAvail,
          this.cachedHuntState.forbiddenBb,
          lastMove.x,
          lastMove.y
        );
        const { density: dNext, totalConfigs: zNext, maxCount } = this.jointSolver.subtractDensity(
          this.cachedHuntState.density,
          this.cachedHuntState.configs,
          dRuined,
          zRuined
        );
        combinedCounts = dNext;
        totalConfigs = zNext;
        maxHuntVal = 0;
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            if (board[r][c] === CELL_UNKNOWN && combinedCounts[r][c] > maxHuntVal) {
              maxHuntVal = combinedCounts[r][c];
            }
          }
        }
        this.cachedHuntState = {
          fleet: primaryAvail.slice(),
          forbiddenBb: primaryForbiddenBb.copy(),
          density: dNext,
          configs: zNext
        };
        modeTag = "Miss-Subtract";
      } else {
        const seenHyp = {};
        for (let s = 0; s < solutions.length; s++) {
          const sol = solutions[s];
          const fbb = missesBb.copy();
          for (let i = 0; i < sol.claimedPegs.length; i++) {
            fbb.set(sol.claimedPegs[i].r, sol.claimedPegs[i].c);
          }
          const avail = getAvailableFleet(FLEET_SIZES, sol.deadSizes);
          let density, configs;

          if (avail.length <= 4) {
            const key = `${avail.join(",")}|${fbb.toKey()}`;
            if (seenHyp[key]) {
              density = seenHyp[key].d;
              configs = seenHyp[key].c;
            } else {
              const res = this.jointSolver.computeExactJointDensity(avail, fbb);
              density = res.density;
              configs = res.totalConfigs;
              seenHyp[key] = { d: density, c: configs };
            }
          } else {
            const seen = {};
            const uniqueFleet = [];
            for (let i = 0; i < avail.length; i++) {
              if (!seen[avail[i]]) {
                seen[avail[i]] = true;
                uniqueFleet.push(avail[i]);
              }
            }
            const res = this.combinatorics.computeHuntDensity(uniqueFleet, fbb);
            density = res.density;
            configs = res.maxD;
          }

          totalConfigs += configs;
          for (let r = 0; r < 10; r++) {
            for (let c = 0; c < 10; c++) {
              if (board[r][c] === CELL_UNKNOWN) {
                const d = combinedCounts[r][c] + density[r][c];
                combinedCounts[r][c] = d;
                if (d > maxHuntVal) maxHuntVal = d;
              }
            }
          }
        }

        if (solutions.length === 1 && primaryAvail.length <= 4) {
          this.cachedHuntState = {
            fleet: primaryAvail.slice(),
            forbiddenBb: primaryForbiddenBb.copy(),
            density: combinedCounts,
            configs: totalConfigs
          };
        } else {
          this.cachedHuntState = null;
        }
      }

      return { combinedCounts, maxHuntVal, totalConfigs, modeTag };
    }

    getBestMove(board, moveHistory) {
      const solutions = this.deduction.analyze(board, moveHistory);
      const primary = solutions[0];
      const hasActive = DeductionEngine.hasActiveHits(solutions);
      const is2Dead = DeductionEngine.isSizeDead(solutions, 2);

      const missesBb = Bitboard.empty();
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (board[r][c] === CELL_MISS) missesBb.set(r, c);
        }
      }

      const diagnostics = DeductionEngine.buildDiagnostics(solutions);

      if (hasActive) {
        this.cachedHuntState = null;
        const res = solveTargetMode(board, moveHistory, solutions, missesBb, this.combinatorics, this.parityOffset, this.openingFlip);
        if (res) {
          this.lastHeatmap = res.heatmap;
          return {
            target: res.target,
            log: `Joint Cover P1,P2 (peak=${res.maxTargetCount})`,
            deadSizes: primary.deadSizes,
            diagnostics,
            heatmap: this.lastHeatmap
          };
        }
      }

      const stride = is2Dead ? 3 : 2;

      if (moveHistory.length === 0) {
        const copy = createBlankMatrix(0);
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            copy[r][c] = Math.round((EMPTY_BOARD_EXACT_HEATMAP[r][c] / 45) * 100);
          }
        }
        this.lastHeatmap = copy;
        const pair = this.parityOffset === 0 ? [{ x: 4, y: 4 }, { x: 5, y: 5 }] : [{ x: 5, y: 4 }, { x: 4, y: 5 }];
        const target = pair[this.openingFlip % 2];
        return {
          target,
          log: `15.05B Exact Prior (Stride-${stride})`,
          deadSizes: primary.deadSizes,
          diagnostics,
          heatmap: this.lastHeatmap
        };
      }

      const { combinedCounts, maxHuntVal, modeTag } = this.computeJointHunt(board, moveHistory, solutions, missesBb);
      const heatmap = createBlankMatrix(0);
      if (maxHuntVal > 0) {
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            if (board[r][c] === CELL_UNKNOWN) {
              heatmap[r][c] = Math.floor((combinedCounts[r][c] / maxHuntVal) * 100);
            }
          }
        }
      }
      this.lastHeatmap = heatmap;

      let bestScore = -1;
      let candidates = [];
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (board[r][c] === CELL_UNKNOWN && isCellOnStride(r, c, stride, this.parityOffset)) {
            const d = combinedCounts[r][c];
            if (d > bestScore) {
              bestScore = d;
              candidates = [{ x: r, y: c }];
            } else if (d === bestScore && d >= 0) {
              candidates.push({ x: r, y: c });
            }
          }
        }
      }

      if (bestScore <= 0 || candidates.length === 0) {
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            if (board[r][c] === CELL_UNKNOWN) {
              const d = combinedCounts[r][c];
              if (d > bestScore) {
                bestScore = d;
                candidates = [{ x: r, y: c }];
              } else if (d === bestScore) {
                candidates.push({ x: r, y: c });
              }
            }
          }
        }
      }

      const pickIdx = (((this.parityOffset + this.openingFlip) * 10007 + moveHistory.length) % Math.max(1, candidates.length));
      const target = candidates[pickIdx] || { x: 4, y: 4 };
      return {
        target,
        log: `${modeTag} (Stride-${stride})`,
        deadSizes: primary.deadSizes,
        diagnostics,
        heatmap: this.lastHeatmap
      };
    }
  }

  // --------------------------------------------------------------------------
  // 6C. MYCROFT BOT (Shannon Entropy H(p) + Exact Joint + Ocean Openness)
  // --------------------------------------------------------------------------
  const INV_LN2 = 1.0 / Math.log(2.0);
  const STAR_OFFSETS = [
    [-1, 0], [1, 0], [0, -1], [0, 1],
    [-2, 0], [2, 0], [0, -2], [0, 2]
  ];

  class MycroftBot extends SherlockBot {
    constructor(parityOffset = 0, sharedCombinatorics = null, sharedDeduction = null, sharedJoint = null) {
      super(parityOffset, sharedCombinatorics, sharedDeduction, sharedJoint);
      this.name = "Mycroft";
      this.wP = 2.5;
      this.wEntropy = 0.35;
      this.wOpen = 0.15;
    }

    evaluateCellUtility(board, r, c, rawP) {
      let p = rawP || 0;
      if (p < 1e-5) p = 1e-5;
      else if (p > 1.0 - 1e-5) p = 1.0 - 1e-5;

      const entropy = -(p * (Math.log(p) * INV_LN2) + (1.0 - p) * (Math.log(1.0 - p) * INV_LN2));
      let openCount = 0;
      for (let i = 0; i < 8; i++) {
        const nr = r + STAR_OFFSETS[i][0];
        const nc = c + STAR_OFFSETS[i][1];
        if (nr >= 0 && nr < 10 && nc >= 0 && nc < 10 && board[nr][nc] === CELL_UNKNOWN) {
          openCount++;
        }
      }
      const openFactor = openCount / 8.0;
      return this.wP * (rawP || 0) + this.wEntropy * entropy + this.wOpen * openFactor;
    }

    getBestMove(board, moveHistory) {
      const solutions = this.deduction.analyze(board, moveHistory);
      const primary = solutions[0];
      const hasActive = DeductionEngine.hasActiveHits(solutions);
      const is2Dead = DeductionEngine.isSizeDead(solutions, 2);

      const missesBb = Bitboard.empty();
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (board[r][c] === CELL_MISS) missesBb.set(r, c);
        }
      }

      const diagnostics = DeductionEngine.buildDiagnostics(solutions);

      if (hasActive) {
        this.cachedHuntState = null;
        const res = solveTargetMode(board, moveHistory, solutions, missesBb, this.combinatorics, this.parityOffset, this.openingFlip);
        if (res) {
          this.lastHeatmap = res.heatmap;
          return {
            target: res.target,
            log: `Joint Cover P1,P2 (peak=${res.maxTargetCount})`,
            deadSizes: primary.deadSizes,
            diagnostics,
            heatmap: this.lastHeatmap
          };
        }
      }

      const stride = is2Dead ? 3 : 2;
      const pMap = createBlankMatrix(0);
      let modeTag = "Shannon H(p)";

      if (moveHistory.length === 0) {
        const copy = createBlankMatrix(0);
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            const v = EMPTY_BOARD_EXACT_HEATMAP[r][c];
            pMap[r][c] = v / 45.0;
            copy[r][c] = Math.round((v / 45.0) * 100);
          }
        }
        this.lastHeatmap = copy;
      } else {
        const { combinedCounts, maxHuntVal, modeTag: jTag } = this.computeJointHunt(board, moveHistory, solutions, missesBb);
        modeTag = jTag === "Miss-Subtract" ? "H(p) + Miss-Sub" : "H(p) + Exact Joint";
        const heatmap = createBlankMatrix(0);
        if (maxHuntVal > 0) {
          for (let r = 0; r < 10; r++) {
            for (let c = 0; c < 10; c++) {
              if (board[r][c] === CELL_UNKNOWN) {
                const p = combinedCounts[r][c] / maxHuntVal;
                pMap[r][c] = p;
                heatmap[r][c] = Math.floor(p * 100);
              }
            }
          }
        }
        this.lastHeatmap = heatmap;
      }

      let bestScore = -1e9;
      let candidates = [];
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (board[r][c] === CELL_UNKNOWN && isCellOnStride(r, c, stride, this.parityOffset)) {
            const score = this.evaluateCellUtility(board, r, c, pMap[r][c]);
            if (score > bestScore + 1e-6) {
              bestScore = score;
              candidates = [{ x: r, y: c }];
            } else if (Math.abs(score - bestScore) <= 1e-6) {
              candidates.push({ x: r, y: c });
            }
          }
        }
      }

      if (candidates.length === 0) {
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            if (board[r][c] === CELL_UNKNOWN) {
              const score = this.evaluateCellUtility(board, r, c, pMap[r][c]);
              if (score > bestScore + 1e-6) {
                bestScore = score;
                candidates = [{ x: r, y: c }];
              } else if (Math.abs(score - bestScore) <= 1e-6) {
                candidates.push({ x: r, y: c });
              }
            }
          }
        }
      }

      const pickIdx = (((this.parityOffset + this.openingFlip) * 10007 + moveHistory.length) % Math.max(1, candidates.length));
      const target = candidates[pickIdx] || { x: 4, y: 4 };
      return {
        target,
        log: `${modeTag} (Score=${bestScore.toFixed(2)})`,
        deadSizes: primary.deadSizes,
        diagnostics,
        heatmap: this.lastHeatmap
      };
    }
  }

  // --------------------------------------------------------------------------
  // 6D. MORIARTY BOT (Trained 4-ResBlock Neural Network + CSP Assassin)
  // --------------------------------------------------------------------------
  const MORIARTY_OFFSETS = {
    stem_w: 0,       stem_b: 2304,
    res0_w1: 2368,   res0_b1: 39232,
    res0_w2: 39296,  res0_b2: 76160,
    res1_w1: 76224,  res1_b1: 113088,
    res1_w2: 113152, res1_b2: 150016,
    res2_w1: 150080, res2_b1: 186944,
    res2_w2: 187008, res2_b2: 223872,
    res3_w1: 223936, res3_b1: 260800,
    res3_w2: 260864, res3_b2: 297728,
    occ1_w: 297792,  occ1_b: 316224,
    occ2_w: 316256,  occ2_b: 316288,
    pol1_w: 316289,  pol1_b: 334721,
    fc1_w: 334753,   fc1_b: 1153953,
    fc2_w: 1154209,  fc2_b: 1179809
  };

  function decodeFloat16Buffer(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== "MF16") {
      throw new Error("Invalid Moriarty FP16 weight file header: " + magic);
    }
    const count = view.getUint32(4, true);
    const out = new Float32Array(count);
    const u16 = new Uint16Array(arrayBuffer, 8, count);

    for (let i = 0; i < count; i++) {
      const h = u16[i];
      const s = (h & 0x8000) >> 15;
      const e = (h & 0x7C00) >> 10;
      const f = h & 0x03FF;
      let val;
      if (e === 0) {
        val = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
      } else if (e === 0x1F) {
        val = f ? NaN : (s ? -Infinity : Infinity);
      } else {
        val = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
      }
      out[i] = val;
    }
    return out;
  }

  function pad10x10To12x12(src100, dst144, channels) {
    dst144.fill(0);
    for (let ch = 0; ch < channels; ch++) {
      const sBase = ch * 100;
      const dBase = ch * 144;
      for (let r = 0; r < 10; r++) {
        const sRow = sBase + r * 10;
        const dRow = dBase + (r + 1) * 12 + 1;
        for (let c = 0; c < 10; c++) {
          dst144[dRow + c] = src100[sRow + c];
        }
      }
    }
  }

  function conv3x3ForwardJS(padIn, inCh, out100, outCh, weights, wOffset, bOffset, applyLeakyRelu, residual100) {
    for (let oc = 0; oc < outCh; oc++) {
      const outBase = oc * 100;
      const bias = weights[bOffset + oc];
      for (let i = 0; i < 100; i++) {
        out100[outBase + i] = bias;
      }

      for (let ic = 0; ic < inCh; ic++) {
        const inBase = ic * 144;
        const wBase = wOffset + (oc * inCh + ic) * 9;
        const k0 = weights[wBase],     k1 = weights[wBase + 1], k2 = weights[wBase + 2];
        const k3 = weights[wBase + 3], k4 = weights[wBase + 4], k5 = weights[wBase + 5];
        const k6 = weights[wBase + 6], k7 = weights[wBase + 7], k8 = weights[wBase + 8];

        for (let r = 1; r <= 10; r++) {
          const r12 = inBase + r * 12;
          const outR10 = outBase + (r - 1) * 10;
          for (let c = 1; c <= 10; c++) {
            const idx = r12 + c;
            const oIdx = outR10 + (c - 1);
            out100[oIdx] +=
              padIn[idx - 13] * k0 + padIn[idx - 12] * k1 + padIn[idx - 11] * k2 +
              padIn[idx -  1] * k3 + padIn[idx     ] * k4 + padIn[idx +  1] * k5 +
              padIn[idx + 11] * k6 + padIn[idx + 12] * k7 + padIn[idx + 13] * k8;
          }
        }
      }

      if (residual100) {
        for (let i = 0; i < 100; i++) {
          let v = out100[outBase + i] + residual100[outBase + i];
          if (applyLeakyRelu && v < 0) v *= 0.1;
          out100[outBase + i] = v;
        }
      } else if (applyLeakyRelu) {
        for (let i = 0; i < 100; i++) {
          const v = out100[outBase + i];
          if (v < 0) out100[outBase + i] = v * 0.1;
        }
      }
    }
  }

  class MoriartyBot {
    constructor(parityOffset = 0, sharedCombinatorics = null, sharedDeduction = null) {
      this.name = "Moriarty";
      this.parityOffset = parityOffset;
      this.combinatorics = sharedCombinatorics || new Combinatorics();
      this.deduction = sharedDeduction || new DeductionEngine();
      this.resnetWeights = null;

      // Preallocated typed buffers
      this._padIn = new Float32Array(4 * 144);
      this._pad64 = new Float32Array(64 * 144);
      this._featA = new Float32Array(64 * 100);
      this._featB = new Float32Array(64 * 100);
      this._head32 = new Float32Array(32 * 100);
      this._fc256 = new Float32Array(256);
      this._policyLogits = new Float32Array(100);
      this._occLogits = new Float32Array(100);
      this._policyProbs = new Float32Array(100);
      this.lastHeatmap = createBlankMatrix(0);
    }

    setWeights(float32Weights) {
      this.resnetWeights = float32Weights;
    }

    reset(gameId = 1, parityOffset = 0) {
      this.parityOffset = parityOffset;
      this.lastHeatmap = createBlankMatrix(0);
    }

    forward(board) {
      const padIn = this._padIn;
      padIn.fill(0);

      for (let r = 0; r < 10; r++) {
        const r12 = (r + 1) * 12;
        for (let c = 0; c < 10; c++) {
          const padIdx = r12 + (c + 1);
          const val = board[r][c];
          if (val === CELL_MISS) padIn[0 * 144 + padIdx] = 1.0;
          else if (val === CELL_HIT) padIn[1 * 144 + padIdx] = 1.0;
          else if (val === CELL_SUNK) padIn[2 * 144 + padIdx] = 1.0;

          if ((r + c) % 2 === this.parityOffset) {
            padIn[3 * 144 + padIdx] = 1.0;
          }
        }
      }

      const rw = this.resnetWeights;
      const pLogits = this._policyLogits;
      const oLogits = this._occLogits;

      if (!rw) {
        // Lightweight fallback if weights are still streaming
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            const idx = r * 10 + c;
            const prior = EMPTY_BOARD_EXACT_HEATMAP[r][c] / 45.0;
            oLogits[idx] = Math.log(prior / (1.0 - prior + 1e-5));
            pLogits[idx] = prior * 3.0;
          }
        }
        return { pLogits, oLogits };
      }

      const pad64 = this._pad64;
      const featA = this._featA;
      const featB = this._featB;
      const head32 = this._head32;
      const fc256 = this._fc256;

      // 1. Stem: Conv2d(4->64, 3x3) + LeakyReLU(0.1)
      conv3x3ForwardJS(padIn, 4, featA, 64, rw, MORIARTY_OFFSETS.stem_w, MORIARTY_OFFSETS.stem_b, true, null);

      // 2. 4 Residual Blocks
      const resW1 = [MORIARTY_OFFSETS.res0_w1, MORIARTY_OFFSETS.res1_w1, MORIARTY_OFFSETS.res2_w1, MORIARTY_OFFSETS.res3_w1];
      const resB1 = [MORIARTY_OFFSETS.res0_b1, MORIARTY_OFFSETS.res1_b1, MORIARTY_OFFSETS.res2_b1, MORIARTY_OFFSETS.res3_b1];
      const resW2 = [MORIARTY_OFFSETS.res0_w2, MORIARTY_OFFSETS.res1_w2, MORIARTY_OFFSETS.res2_w2, MORIARTY_OFFSETS.res3_w2];
      const resB2 = [MORIARTY_OFFSETS.res0_b2, MORIARTY_OFFSETS.res1_b2, MORIARTY_OFFSETS.res2_b2, MORIARTY_OFFSETS.res3_b2];

      for (let b = 0; b < 4; b++) {
        pad10x10To12x12(featA, pad64, 64);
        conv3x3ForwardJS(pad64, 64, featB, 64, rw, resW1[b], resB1[b], true, null);
        pad10x10To12x12(featB, pad64, 64);
        conv3x3ForwardJS(pad64, 64, featB, 64, rw, resW2[b], resB2[b], true, featA);
        featA.set(featB);
      }

      pad10x10To12x12(featA, pad64, 64);

      // 3. Occupancy Head
      conv3x3ForwardJS(pad64, 64, head32, 32, rw, MORIARTY_OFFSETS.occ1_w, MORIARTY_OFFSETS.occ1_b, true, null);
      const occ2W = MORIARTY_OFFSETS.occ2_w;
      const occ2B = rw[MORIARTY_OFFSETS.occ2_b];
      for (let i = 0; i < 100; i++) {
        let sum = occ2B;
        for (let ch = 0; ch < 32; ch++) {
          sum += head32[ch * 100 + i] * rw[occ2W + ch];
        }
        oLogits[i] = sum;
      }

      // 4. Policy Head
      conv3x3ForwardJS(pad64, 64, head32, 32, rw, MORIARTY_OFFSETS.pol1_w, MORIARTY_OFFSETS.pol1_b, true, null);
      const fc1W = MORIARTY_OFFSETS.fc1_w;
      const fc1B = MORIARTY_OFFSETS.fc1_b;
      for (let o = 0; o < 256; o++) {
        let sum = rw[fc1B + o];
        const wRow = fc1W + o * 3200;
        for (let j = 0; j < 3200; j++) {
          sum += head32[j] * rw[wRow + j];
        }
        if (sum < 0) sum *= 0.1;
        fc256[o] = sum;
      }

      const fc2W = MORIARTY_OFFSETS.fc2_w;
      const fc2B = MORIARTY_OFFSETS.fc2_b;
      for (let o = 0; o < 100; o++) {
        let sum = rw[fc2B + o];
        const wRow = fc2W + o * 256;
        for (let j = 0; j < 256; j++) {
          sum += fc256[j] * rw[wRow + j];
        }
        pLogits[o] = sum;
      }

      return { pLogits, oLogits };
    }

    getBestMove(board, moveHistory) {
      const solutions = this.deduction.analyze(board, moveHistory);
      const primary = solutions[0];
      const hasActive = DeductionEngine.hasActiveHits(solutions);
      const is2Dead = DeductionEngine.isSizeDead(solutions, 2);

      const missesBb = Bitboard.empty();
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (board[r][c] === CELL_MISS) missesBb.set(r, c);
        }
      }

      const diagnostics = DeductionEngine.buildDiagnostics(solutions);

      const { pLogits, oLogits } = this.forward(board);
      const heatmap = createBlankMatrix(0);
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (board[r][c] === CELL_UNKNOWN) {
            const logit = oLogits[r * 10 + c];
            const prob = 1.0 / (1.0 + Math.exp(-logit));
            heatmap[r][c] = Math.round(prob * 100);
          }
        }
      }

      // 1. Assassin Mode when active hits exist
      if (hasActive) {
        const res = solveTargetMode(board, moveHistory, solutions, missesBb, this.combinatorics, this.parityOffset, 0);
        if (res) {
          heatmap[res.target.x][res.target.y] = 100;
          this.lastHeatmap = heatmap;
          return {
            target: res.target,
            log: `Assassin Override (peak=${res.maxTargetCount})`,
            deadSizes: primary.deadSizes,
            diagnostics,
            heatmap: this.lastHeatmap
          };
        }
      }

      this.lastHeatmap = heatmap;

      // Softmax over policy logits
      let maxL = -1e30;
      for (let i = 0; i < 100; i++) {
        if (pLogits[i] > maxL) maxL = pLogits[i];
      }
      let sumE = 0;
      for (let i = 0; i < 100; i++) {
        const e = Math.exp(pLogits[i] - maxL);
        this._policyProbs[i] = e;
        sumE += e;
      }
      for (let i = 0; i < 100; i++) {
        this._policyProbs[i] /= (sumE || 1);
      }

      let bestScore = -1e9;
      let candidates = [];
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (board[r][c] === CELL_UNKNOWN && (r + c) % 2 === this.parityOffset) {
            const sc = this._policyProbs[r * 10 + c];
            if (sc > bestScore + 1e-7) {
              bestScore = sc;
              candidates = [{ x: r, y: c }];
            } else if (Math.abs(sc - bestScore) <= 1e-7) {
              candidates.push({ x: r, y: c });
            }
          }
        }
      }

      if (candidates.length === 0) {
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            if (board[r][c] === CELL_UNKNOWN) {
              const sc = this._policyProbs[r * 10 + c];
              if (sc > bestScore + 1e-7) {
                bestScore = sc;
                candidates = [{ x: r, y: c }];
              } else if (Math.abs(sc - bestScore) <= 1e-7) {
                candidates.push({ x: r, y: c });
              }
            }
          }
        }
      }

      const pickIdx = ((this.parityOffset * 10007 + moveHistory.length) % Math.max(1, candidates.length));
      const target = candidates[pickIdx] || { x: 4, y: 4 };
      return {
        target,
        log: `Neural Policy (${(bestScore * 100).toFixed(1)}% conf)`,
        deadSizes: primary.deadSizes,
        diagnostics,
        heatmap: this.lastHeatmap
      };
    }
  }

  const Engine = {
    CELL_UNKNOWN,
    CELL_MISS,
    CELL_HIT,
    CELL_SUNK,
    FLEET_SIZES,
    EMPTY_BOARD_EXACT_HEATMAP,
    Bitboard,
    countShipOnFreeMask,
    createBlankMatrix,
    BattleshipGame,
    DeductionEngine,
    Combinatorics,
    JointSolver,
    WatsonBot,
    SherlockBot,
    MycroftBot,
    MoriartyBot,
    decodeFloat16Buffer
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = Engine;
  } else {
    root.BattleshipEngine = Engine;
  }
})(typeof window !== "undefined" ? window : globalThis);

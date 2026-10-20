// showcase/js/test_engine.js
// Automated Verification Suite for showcase/js/engine.js
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const Engine = require("./engine.js");

console.log("=== Running Showcase JS Engine Verification Suite ===");

// 1. Test Bitboard & SWAR Analytic Ship Placement Counter
{
  const fbb = Engine.Bitboard.empty();
  fbb.set(2, 3);
  fbb.set(7, 8);
  const inv = fbb.bnot();
  const comb = new Engine.Combinatorics();

  for (const sz of [2, 3, 4, 5]) {
    const swarCount = Engine.countShipOnFreeMask(inv.w0, inv.w1, inv.w2, inv.w3, sz);
    const listCount = comb.getPlacements(sz).filter((p) => !p.mask.intersects(fbb)).length;
    assert.strictEqual(swarCount, listCount, `SWAR count mismatch for size ${sz}`);
  }
  console.log("[PASS] 1. Bitboard100 & Analytic SWAR Ship Placement Counter verified.");
}

// 2. Test Exact Joint Solver & Incremental Miss-Subtraction Identity (Z_{t+1} == Z_t - Z_ruined)
{
  const comb = new Engine.Combinatorics();
  const solver = new Engine.JointSolver(comb);
  const fbb = Engine.Bitboard.empty();
  fbb.set(0, 0);
  fbb.set(4, 4);

  const fleet = [5, 4, 3, 2];
  const t0 = performance.now();
  const base = solver.computeExactJointDensity(fleet, fbb);
  const coldMs = performance.now() - t0;

  const missR = 3, missC = 6;
  const t1 = performance.now();
  const ruined = solver.computeRuinedDensity(fleet, fbb, missR, missC);
  const sub = solver.subtractDensity(base.density, base.totalConfigs, ruined.dRuined, ruined.zRuined);
  const warmMs = performance.now() - t1;

  const fbbNext = fbb.copy();
  fbbNext.set(missR, missC);
  const direct = solver.computeExactJointDensity(fleet, fbbNext);

  assert.strictEqual(sub.totalConfigs, direct.totalConfigs, "Total configs after miss-subtraction must match direct solve");
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      assert.strictEqual(sub.density[r][c], direct.density[r][c], `Cell (${r},${c}) density mismatch`);
    }
  }
  console.log(`[PASS] 2. Exact Joint Solver & Incremental Miss-Subtraction verified (Cold 4-ship: ${coldMs.toFixed(1)}ms, Warm Miss-Sub: ${warmMs.toFixed(2)}ms).`);
}

// 3. Test CSP Blind-Sink Deduction Engine
{
  const deduction = new Engine.DeductionEngine();
  const board = Engine.createBlankMatrix(0);
  board[4][4] = Engine.CELL_HIT;
  board[4][5] = Engine.CELL_SUNK;
  board[3][4] = Engine.CELL_HIT;
  const history = [
    { x: 3, y: 4, result: "HIT", shotNumber: 1 },
    { x: 4, y: 4, result: "HIT", shotNumber: 2 },
    { x: 4, y: 5, result: "SUNK", shotNumber: 3 }
  ];
  const sols = deduction.analyze(board, history);
  assert.strictEqual(sols.length, 1, "Should deduce unique dead ship");
  assert.deepStrictEqual(sols[0].deadSizes, [2], "Dead ship is length 2");
  assert.strictEqual(sols[0].activeHits.length, 1, "1 leftover active hit at (3,4)");
  assert.strictEqual(sols[0].activeHits[0].r, 3);
  assert.strictEqual(sols[0].activeHits[0].c, 4);
  console.log("[PASS] 3. Chronological CSP Blind-Sink Deduction verified.");
}

// 4. Test All 4 Bots (Watson, Sherlock, Mycroft, Moriarty with FP16 ResNet weights)
{
  const fp16Path = path.join(__dirname, "..", "assets", "moriarty_fp16.bin");
  const buf = fs.readFileSync(fp16Path);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const weights = Engine.decodeFloat16Buffer(ab);
  assert.strictEqual(weights.length, 1179909, "Decoded FP16 weights count must equal 1,179,909");

  const comb = new Engine.Combinatorics();
  const ded = new Engine.DeductionEngine();
  const joint = new Engine.JointSolver(comb);

  const watson = new Engine.WatsonBot(0, comb, ded);
  const sherlock = new Engine.SherlockBot(0, comb, ded, joint);
  const mycroft = new Engine.MycroftBot(0, comb, ded, joint);
  const moriarty = new Engine.MoriartyBot(0, comb, ded);
  moriarty.setWeights(weights);

  const game = new Engine.BattleshipGame();
  game.loadPreset(2);

  for (const bot of [watson, sherlock, mycroft, moriarty]) {
    const t0 = performance.now();
    const res = bot.getBestMove(game.trackingBoard, game.moveHistory);
    const dt = performance.now() - t0;
    assert(res.target.x >= 0 && res.target.x < 10 && res.target.y >= 0 && res.target.y < 10);
    console.log(`       -> ${bot.name.padEnd(9)} | Target: (${res.target.x},${res.target.y}) | ${res.log} | ${dt.toFixed(2)} ms`);
  }
  console.log("[PASS] 4. All 4 bots (Watson, Sherlock, Mycroft, Moriarty) verified.");
}

console.log("\nALL SHOWCASE JS ENGINE TESTS PASSED!");

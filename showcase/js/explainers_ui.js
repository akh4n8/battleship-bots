// showcase/js/explainers_ui.js
// Visual-First Interactive Explainers for:
//   Section 2: Paired-Board Tournament Leaderboard Chart
//   Section 3: 5-Layer CSP Blind-Sink Deduction (Side-by-Side Triptych)
//   Section 4: Full-Width Combination Gallery & Subtraction Visuals:
//              1. Marginal vs. Exact Joint Hunt
//              2. Speedup #1: Last-Ship Subtraction Shortcut
//              3. Speedup #2: Incremental Miss-Subtraction
//              4. Target Mode (Last): Hits, Linear Kill & 2-Ship Cover

(function (root) {
  "use strict";

  function setupCanvas(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  // ==========================================================================
  // HELPER: Draw 5x5 Mini-Board for Section 3 CSP Triptych
  // ==========================================================================
  function drawMiniBoard5x5(ctx, w, h, cfg) {
    ctx.fillStyle = "#161820";
    ctx.fillRect(0, 0, w, h);

    const N = 5;
    const cs = cfg.cs || 36;
    const ox = Math.floor((w - N * cs) / 2) + 6;
    const oy = 24;

    ctx.fillStyle = "#9AA0B4";
    ctx.font = "600 10px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < N; i++) {
      ctx.fillText(String(i + 2), ox + i * cs + cs / 2, oy - 10);
      ctx.fillText(String.fromCharCode(66 + i), ox - 12, oy + i * cs + cs / 2);
    }

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const x = ox + c * cs;
        const y = oy + r * cs;
        ctx.fillStyle = "#0A3C78";
        ctx.fillRect(x, y, cs, cs);
        ctx.strokeStyle = "#485468";
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cs, cs);
      }
    }

    if (cfg.targets) {
      for (const t of cfg.targets) {
        const x = ox + t.c * cs;
        const y = oy + t.r * cs;
        ctx.fillStyle = "rgba(255, 50, 50, 0.85)";
        ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
        ctx.strokeStyle = "#FFF03C";
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4);
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "700 10.5px Inter, system-ui, sans-serif";
        ctx.fillText(String(t.label || "100"), x + cs / 2, y + cs / 2);
      }
    }

    if (cfg.hulls) {
      for (const h of cfg.hulls) {
        const minC = Math.min(...h.cells.map((p) => p.c));
        const maxC = Math.max(...h.cells.map((p) => p.c));
        const minR = Math.min(...h.cells.map((p) => p.r));
        const maxR = Math.max(...h.cells.map((p) => p.r));
        const hx = ox + minC * cs + 3;
        const hy = oy + minR * cs + 3;
        const hw = (maxC - minC + 1) * cs - 6;
        const hh = (maxR - minR + 1) * cs - 6;

        ctx.save();
        if (h.dashed) ctx.setLineDash([5, 3]);
        ctx.fillStyle = h.fill || "rgba(180, 190, 205, 0.28)";
        ctx.strokeStyle = h.stroke || "#C8D2E0";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.roundRect(hx, hy, hw, hh, 7);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }
    }

    if (cfg.errorCells) {
      for (const ec of cfg.errorCells) {
        const x = ox + ec.c * cs;
        const y = oy + ec.r * cs;
        ctx.fillStyle = "rgba(255, 50, 50, 0.35)";
        ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
        ctx.strokeStyle = "#FF5252";
        ctx.lineWidth = 2.5;
        ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4);
      }
    }

    if (cfg.misses) {
      for (const m of cfg.misses) {
        const cx = ox + m.c * cs + cs / 2;
        const cy = oy + m.r * cs + cs / 2;
        ctx.fillStyle = "#DCDCDC";
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (cfg.pegs) {
      for (const p of cfg.pegs) {
        const x = ox + p.c * cs;
        const y = oy + p.r * cs;
        ctx.strokeStyle = p.type === "SUNK" ? "#3296FF" : "#FF3232";
        ctx.lineWidth = p.type === "SUNK" ? 3.5 : 2.8;
        const pad = Math.round(cs * 0.24);
        ctx.beginPath();
        ctx.moveTo(x + pad, y + pad);
        ctx.lineTo(x + cs - pad, y + cs - pad);
        ctx.moveTo(x + cs - pad, y + pad);
        ctx.lineTo(x + pad, y + cs - pad);
        ctx.stroke();

        if (p.t !== undefined) {
          ctx.fillStyle = "rgba(12, 14, 20, 0.88)";
          ctx.fillRect(x + 2, y + cs - 12, 24, 10);
          ctx.fillStyle = p.type === "SUNK" ? "#6EC6FF" : "#FFD56B";
          ctx.font = "700 8.5px Inter, system-ui, sans-serif";
          ctx.textAlign = "left";
          ctx.fillText(`t=${p.t}`, x + 3, y + cs - 6.5);
          ctx.textAlign = "center";
        }
      }
    }

    if (cfg.activeHits) {
      for (const ah of cfg.activeHits) {
        const x = ox + ah.c * cs;
        const y = oy + ah.r * cs;
        ctx.strokeStyle = "#00E5FF";
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.roundRect(x + 2.5, y + 2.5, cs - 5, cs - 5, 4);
        ctx.stroke();
      }
    }

    if (cfg.footerText) {
      ctx.fillStyle = cfg.footerColor || "#9AA0B4";
      ctx.font = "700 10px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(cfg.footerText, w / 2, oy + N * cs + 14);
    }
  }

  // ==========================================================================
  // SECTION 3 DATA: 5 PROGRESSIVE CSP LAYERS (SIDE-BY-SIDE TRIPTYCH)
  // ==========================================================================
  const CSP_LAYERS_DATA = [
    {
      banner: "<strong>Layer 1 — Minimize Leftover Active Hits (Occam's Razor):</strong> When a ship sinks in a line of hits, prefer the ship size that explains the most hits and leaves the fewest unexplained active squares.",
      raw: {
        pegs: [
          { r: 2, c: 1, t: 3, type: "HIT" },
          { r: 2, c: 2, t: 4, type: "HIT" },
          { r: 2, c: 3, t: 5, type: "SUNK" }
        ],
        misses: [{ r: 2, c: 0 }, { r: 2, c: 4 }],
        caption: "3 collinear hits in Row D ending in a blind SUNK at D5 (t=5)."
      },
      bad: {
        header: "2. ❌ Rejected: Size-2 Destroyer [D4, D5]",
        pegs: [
          { r: 2, c: 1, t: 3, type: "HIT" },
          { r: 2, c: 2, t: 4, type: "HIT" },
          { r: 2, c: 3, t: 5, type: "SUNK" }
        ],
        misses: [{ r: 2, c: 0 }, { r: 2, c: 4 }],
        hulls: [{ cells: [{ r: 2, c: 2 }, { r: 2, c: 3 }], stroke: "#FF5252", fill: "rgba(255,82,82,0.22)", dashed: true }],
        activeHits: [{ r: 2, c: 1 }],
        footerText: "❌ Score = 1 Leftover Hit (Hallucinates 2nd ship!)",
        footerColor: "#FF8A80",
        caption: "Claiming only a Size-2 Destroyer leaves D3 as a phantom 2nd ship (1 leftover active hit)."
      },
      good: {
        header: "3. ✅ Accepted: Size-3 Cruiser [D3, D4, D5]",
        pegs: [
          { r: 2, c: 1, t: 3, type: "HIT" },
          { r: 2, c: 2, t: 4, type: "HIT" },
          { r: 2, c: 3, t: 5, type: "SUNK" }
        ],
        misses: [{ r: 2, c: 0 }, { r: 2, c: 4 }],
        hulls: [{ cells: [{ r: 2, c: 1 }, { r: 2, c: 2 }, { r: 2, c: 3 }], stroke: "#3CFF78", fill: "rgba(60,255,120,0.22)" }],
        footerText: "✅ Score = 0 Leftover Hits (Minimal Explanation)",
        footerColor: "#3CFF78",
        caption: "A Size-3 Cruiser explains all 3 hits cleanly (0 leftovers). All 3 cells are locked as dead!"
      }
    },
    {
      banner: "<strong>Layer 2 — Chronological Order (hit.t ≤ sunk.t):</strong> A ship cannot sink at turn <code>t=6</code> if one of its cells wasn't hit until turn <code>t=9</code>! Any hit fired AFTER a sink belongs to a different ship.",
      raw: {
        pegs: [
          { r: 2, c: 1, t: 4, type: "HIT" },
          { r: 2, c: 2, t: 6, type: "SUNK" },
          { r: 2, c: 3, t: 9, type: "HIT" }
        ],
        misses: [{ r: 1, c: 2 }, { r: 3, c: 2 }],
        caption: "D4 sank at t=6. Later at t=9, the bot fired at D5 and got another HIT!"
      },
      bad: {
        header: "2. ❌ Rejected: Size-3 Cruiser [D3, D4, D5]",
        pegs: [
          { r: 2, c: 1, t: 4, type: "HIT" },
          { r: 2, c: 2, t: 6, type: "SUNK" },
          { r: 2, c: 3, t: 9, type: "HIT" }
        ],
        misses: [{ r: 1, c: 2 }, { r: 3, c: 2 }],
        hulls: [{ cells: [{ r: 2, c: 1 }, { r: 2, c: 2 }, { r: 2, c: 3 }], stroke: "#FF5252", fill: "rgba(255,82,82,0.22)", dashed: true }],
        errorCells: [{ r: 2, c: 3 }],
        footerText: "❌ Violates Time: D5 (t=9) > Sunk (t=6)!",
        footerColor: "#FF8A80",
        caption: "Impossible! If [D3,D4,D5] were one Cruiser, it could not have sunk at t=6 before D5 was hit at t=9."
      },
      good: {
        header: "3. ✅ Accepted: Size-2 [D3, D4] + Active D5",
        pegs: [
          { r: 2, c: 1, t: 4, type: "HIT" },
          { r: 2, c: 2, t: 6, type: "SUNK" },
          { r: 2, c: 3, t: 9, type: "HIT" }
        ],
        misses: [{ r: 1, c: 2 }, { r: 3, c: 2 }],
        hulls: [{ cells: [{ r: 2, c: 1 }, { r: 2, c: 2 }], stroke: "#3CFF78", fill: "rgba(60,255,120,0.22)" }],
        activeHits: [{ r: 2, c: 3 }],
        targets: [{ r: 2, c: 4, label: "100" }],
        footerText: "✅ Locks [D3,D4] (t≤6) & Isolates D5 (t=9)",
        footerColor: "#3CFF78",
        caption: "Locks [D3,D4] as the dead Size-2 Destroyer (t=4,6) and isolates D5 (t=9) as a live wounded ship!"
      }
    },
    {
      banner: "<strong>Layer 3 — Boxed-In Leftover Rejection:</strong> Every leftover active hit must still have enough open room (and at least 1 unhit water cell) to fit a surviving ship. If a corpse boxes in a hit, that corpse is impossible!",
      raw: {
        pegs: [
          { r: 2, c: 1, t: 2, type: "HIT" },
          { r: 2, c: 2, t: 4, type: "HIT" },
          { r: 2, c: 3, t: 5, type: "HIT" },
          { r: 2, c: 4, t: 7, type: "SUNK" }
        ],
        misses: [{ r: 2, c: 0 }, { r: 1, c: 1 }, { r: 3, c: 1 }],
        caption: "4 hits in Row D. Notice D3 (t=2) is surrounded by 3 Misses (D2, C3, E3)!"
      },
      bad: {
        header: "2. ❌ Rejected: Size-3 Cruiser [D4, D5, D6]",
        pegs: [
          { r: 2, c: 1, t: 2, type: "HIT" },
          { r: 2, c: 2, t: 4, type: "HIT" },
          { r: 2, c: 3, t: 5, type: "HIT" },
          { r: 2, c: 4, t: 7, type: "SUNK" }
        ],
        misses: [{ r: 2, c: 0 }, { r: 1, c: 1 }, { r: 3, c: 1 }],
        hulls: [{ cells: [{ r: 2, c: 2 }, { r: 2, c: 3 }, { r: 2, c: 4 }], stroke: "#FF5252", fill: "rgba(255,82,82,0.22)", dashed: true }],
        errorCells: [{ r: 2, c: 1 }],
        footerText: "❌ Leftover Hit D3 is 100% Boxed-In!",
        footerColor: "#FF8A80",
        caption: "If [D4,D5,D6] is a dead Cruiser, leftover hit D3 is trapped on all 4 sides by Misses and the corpse!"
      },
      good: {
        header: "3. ✅ Accepted: Size-4 Battleship [D3..D6]",
        pegs: [
          { r: 2, c: 1, t: 2, type: "HIT" },
          { r: 2, c: 2, t: 4, type: "HIT" },
          { r: 2, c: 3, t: 5, type: "HIT" },
          { r: 2, c: 4, t: 7, type: "SUNK" }
        ],
        misses: [{ r: 2, c: 0 }, { r: 1, c: 1 }, { r: 3, c: 1 }],
        hulls: [{ cells: [{ r: 2, c: 1 }, { r: 2, c: 2 }, { r: 2, c: 3 }, { r: 2, c: 4 }], stroke: "#3CFF78", fill: "rgba(60,255,120,0.22)" }],
        footerText: "✅ Zero Stranded Hits (Or Size-2 [D5,D6] + Vert D4)",
        footerColor: "#3CFF78",
        caption: "Only explanations where D3 is part of a horizontal ship (or D4 extends vertically) are physically valid!"
      }
    },
    {
      banner: "<strong>Layer 4 — Fleet Inventory &amp; Disjoint Corpses:</strong> Standard Battleship only has ONE Size-2 Destroyer (`{5, 4, 3, 3, 2}`). When a second ship sinks, the solver cannot claim a second Size-2 Destroyer or overlap an existing corpse.",
      raw: {
        pegs: [
          { r: 0, c: 1, t: 2, type: "HIT" },
          { r: 0, c: 2, t: 3, type: "SUNK" },
          { r: 2, c: 1, t: 6, type: "HIT" },
          { r: 2, c: 2, t: 7, type: "HIT" },
          { r: 2, c: 3, t: 8, type: "SUNK" }
        ],
        misses: [{ r: 0, c: 0 }, { r: 0, c: 3 }],
        caption: "Sunk #1 at B4 (t=3) is boxed in as Size-2. Later, Sunk #2 occurs at D5 (t=8)."
      },
      bad: {
        header: "2. ❌ Rejected: Second Size-2 Destroyer [D4, D5]",
        pegs: [
          { r: 0, c: 1, t: 2, type: "HIT" },
          { r: 0, c: 2, t: 3, type: "SUNK" },
          { r: 2, c: 1, t: 6, type: "HIT" },
          { r: 2, c: 2, t: 7, type: "HIT" },
          { r: 2, c: 3, t: 8, type: "SUNK" }
        ],
        misses: [{ r: 0, c: 0 }, { r: 0, c: 3 }],
        hulls: [
          { cells: [{ r: 0, c: 1 }, { r: 0, c: 2 }], stroke: "#C8D2E0", fill: "rgba(180,190,205,0.22)" },
          { cells: [{ r: 2, c: 2 }, { r: 2, c: 3 }], stroke: "#FF5252", fill: "rgba(255,82,82,0.22)", dashed: true }
        ],
        activeHits: [{ r: 2, c: 1 }],
        footerText: "❌ Illegal Fleet: Only ONE Size-2 Ship Exists!",
        footerColor: "#FF8A80",
        caption: "Since Sunk #1 [B3,B4] already used the only Size-2 Destroyer, Sunk #2 cannot also be Size 2!"
      },
      good: {
        header: "3. ✅ Accepted: Size-2 [B3,B4] + Size-3 [D3..D5]",
        pegs: [
          { r: 0, c: 1, t: 2, type: "HIT" },
          { r: 0, c: 2, t: 3, type: "SUNK" },
          { r: 2, c: 1, t: 6, type: "HIT" },
          { r: 2, c: 2, t: 7, type: "HIT" },
          { r: 2, c: 3, t: 8, type: "SUNK" }
        ],
        misses: [{ r: 0, c: 0 }, { r: 0, c: 3 }],
        hulls: [
          { cells: [{ r: 0, c: 1 }, { r: 0, c: 2 }], stroke: "#3CFF78", fill: "rgba(60,255,120,0.22)" },
          { cells: [{ r: 2, c: 1 }, { r: 2, c: 2 }, { r: 2, c: 3 }], stroke: "#3CFF78", fill: "rgba(60,255,120,0.22)" }
        ],
        footerText: "✅ Valid Inventory: {2, 3} Dead → Upgrade to Stride 3!",
        footerColor: "#3CFF78",
        caption: "Sunk #2 must be at least Size 3 [D3,D4,D5], locking all 3 cells and upgrading Hunt Parity to Stride 3!"
      }
    },
    {
      banner: "<strong>Layer 5 — Tied-Hypothesis Consensus:</strong> What if two valid explanations tie with the exact same number of leftover hits? Instead of guessing one and being wrong, the solver keeps BOTH and only locks squares they agree on!",
      raw: {
        pegs: [
          { r: 1, c: 2, t: 2, type: "HIT" },
          { r: 2, c: 1, t: 4, type: "HIT" },
          { r: 2, c: 2, t: 6, type: "SUNK" }
        ],
        misses: [{ r: 2, c: 3 }, { r: 3, c: 2 }],
        caption: "L-Corner with 3 pegs ending in SUNK at D4 (t=6). Did [C4,D4] (vert) or [D3,D4] (horiz) sink?"
      },
      bad: {
        header: "2. ❌ Dangerous: Guessing Only Horizontal [D3, D4]",
        pegs: [
          { r: 1, c: 2, t: 2, type: "HIT" },
          { r: 2, c: 1, t: 4, type: "HIT" },
          { r: 2, c: 2, t: 6, type: "SUNK" }
        ],
        misses: [{ r: 2, c: 3 }, { r: 3, c: 2 }],
        hulls: [{ cells: [{ r: 2, c: 1 }, { r: 2, c: 2 }], stroke: "#FF5252", fill: "rgba(255,82,82,0.22)", dashed: true }],
        activeHits: [{ r: 1, c: 2 }],
        footerText: "❌ Ignores D3 if Vertical [C4,D4] Actually Sank!",
        footerColor: "#FF8A80",
        caption: "Both H1 (Horiz [D3,D4]) and H2 (Vert [C4,D4]) leave 1 active hit. Guessing H1 abandons D3 if H2 is true!"
      },
      good: {
        header: "3. ✅ Accepted: Keep Both Tied Hypotheses (H1 ∪ H2)",
        pegs: [
          { r: 1, c: 2, t: 2, type: "HIT" },
          { r: 2, c: 1, t: 4, type: "HIT" },
          { r: 2, c: 2, t: 6, type: "SUNK" }
        ],
        misses: [{ r: 2, c: 3 }, { r: 3, c: 2 }],
        hulls: [{ cells: [{ r: 2, c: 2 }], stroke: "#3CFF78", fill: "rgba(60,255,120,0.25)" }],
        activeHits: [{ r: 1, c: 2 }, { r: 2, c: 1 }],
        targets: [{ r: 0, c: 2, label: "95" }, { r: 2, c: 0, label: "95" }],
        footerText: "✅ Locks Shared D4 & Targets Both C4 and D3!",
        footerColor: "#3CFF78",
        caption: "Locks D4 (dead in both H1 & H2) and evaluates target density for BOTH C4 and D3 until resolved!"
      }
    }
  ];

  // ==========================================================================
  // SECTION 4 HELPERS: DRAW MINI COMBINATION CARD & SUMMED EQUATION BOARDS
  // ==========================================================================
  function drawComboMiniCanvas(canvas, rows, cols, combo, boardState) {
    const w = 128;
    const h = rows === 2 ? 78 : 96;
    const ctx = setupCanvas(canvas, w, h);

    ctx.fillStyle = "#12151E";
    ctx.fillRect(0, 0, w, h);

    const cs = Math.min(Math.floor((w - 16) / cols), Math.floor((h - 12) / rows));
    const ox = Math.floor((w - cols * cs) / 2);
    const oy = Math.floor((h - rows * cs) / 2);

    // Draw grid cells
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = ox + c * cs;
        const y = oy + r * cs;
        ctx.fillStyle = "#0A3264";
        ctx.fillRect(x, y, cs, cs);
        ctx.strokeStyle = "#343E54";
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cs, cs);
      }
    }

    // Draw ships in this combination
    if (combo.ships) {
      for (const s of combo.ships) {
        const minR = Math.min(...s.cells.map((p) => p.r));
        const maxR = Math.max(...s.cells.map((p) => p.r));
        const minC = Math.min(...s.cells.map((p) => p.c));
        const maxC = Math.max(...s.cells.map((p) => p.c));
        const sx = ox + minC * cs + 2.5;
        const sy = oy + minR * cs + 2.5;
        const sw = (maxC - minC + 1) * cs - 5;
        const sh = (maxR - minR + 1) * cs - 5;

        ctx.fillStyle = s.fill || "rgba(0, 229, 255, 0.55)";
        ctx.strokeStyle = s.stroke || "#00E5FF";
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.roundRect(sx, sy, sw, sh, 4);
        ctx.fill();
        ctx.stroke();
      }
    }

    // Draw collision/overlap cell X if ships overlap or hit a Miss
    if (combo.conflictCells) {
      for (const cc of combo.conflictCells) {
        const x = ox + cc.c * cs;
        const y = oy + cc.r * cs;
        ctx.fillStyle = "rgba(255, 50, 50, 0.55)";
        ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
        ctx.strokeStyle = "#FF5252";
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 1.5, y + 1.5, cs - 3, cs - 3);
      }
    }

    // Draw board misses
    if (boardState.misses) {
      for (const m of boardState.misses) {
        ctx.fillStyle = "#DCDCDC";
        ctx.beginPath();
        ctx.arc(ox + m.c * cs + cs / 2, oy + m.r * cs + cs / 2, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw board hits
    if (boardState.hits) {
      for (const ht of boardState.hits) {
        const x = ox + ht.c * cs;
        const y = oy + ht.r * cs;
        ctx.strokeStyle = "#FF3232";
        ctx.lineWidth = 2.2;
        const p = 5;
        ctx.beginPath();
        ctx.moveTo(x + p, y + p);
        ctx.lineTo(x + cs - p, y + cs - p);
        ctx.moveTo(x + cs - p, y + p);
        ctx.lineTo(x + p, y + cs - p);
        ctx.stroke();
      }
    }
  }

  // Draw the Single Interactive Heatmap on the right side of Section 4
  function drawInteractiveHeatmap(canvas, rows, cols, viewMode, allMat, subMat, validMat, boardState, hoverCell, hoverComboCells, footerText) {
    const w = 320;
    const h = 250;
    const ctx = setupCanvas(canvas, w, h);

    ctx.fillStyle = "#161820";
    ctx.fillRect(0, 0, w, h);

    const cs = Math.min(Math.floor((w - 64) / cols), Math.floor((h - 60) / rows));
    const ox = Math.floor((w - cols * cs) / 2) + 8;
    const oy = Math.floor((h - rows * cs) / 2) - 4;

    // Axis labels
    ctx.fillStyle = "#9AA0B4";
    ctx.font = "600 11.5px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let c = 0; c < cols; c++) {
      ctx.fillText(String(c + 1), ox + c * cs + cs / 2, oy - 13);
    }
    for (let r = 0; r < rows; r++) {
      ctx.fillText(String.fromCharCode(65 + r), ox - 15, oy + r * cs + cs / 2);
    }

    const activeMat = viewMode === "all" ? allMat : validMat;

    // Find max score on unknown cells for highlighting best target
    let maxScoreUnknown = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const isMiss = boardState.misses && boardState.misses.some((m) => m.r === r && m.c === c);
        const isHit = boardState.hits && boardState.hits.some((ht) => ht.r === r && ht.c === c);
        const isFixed = boardState.fixedShip && boardState.fixedShip.some((fs) => fs.r === r && fs.c === c);
        if (!isMiss && !isHit && !isFixed && activeMat[r][c] > maxScoreUnknown) {
          maxScoreUnknown = activeMat[r][c];
        }
      }
    }

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = ox + c * cs;
        const y = oy + r * cs;
        const isMiss = boardState.misses && boardState.misses.some((m) => m.r === r && m.c === c);
        const isHit = boardState.hits && boardState.hits.some((ht) => ht.r === r && ht.c === c);
        const isFixed = boardState.fixedShip && boardState.fixedShip.some((fs) => fs.r === r && fs.c === c);

        const aVal = allMat[r][c];
        const sVal = subMat[r][c];
        const vVal = validMat[r][c];
        const shownVal = viewMode === "all" ? aVal : vVal;

        const isHoveredCell = hoverCell && hoverCell.r === r && hoverCell.c === c;
        const isHoveredComboCell = hoverComboCells && hoverComboCells.some((pt) => pt.r === r && pt.c === c);
        const isBest = !isMiss && !isHit && !isFixed && maxScoreUnknown > 0 && shownVal === maxScoreUnknown;

        if (isMiss) {
          ctx.fillStyle = "#1E222D";
        } else if (isFixed) {
          ctx.fillStyle = "rgba(92, 107, 192, 0.50)";
        } else if (isHoveredComboCell) {
          ctx.fillStyle = "rgba(255, 240, 60, 0.28)";
        } else if (isBest) {
          ctx.fillStyle = viewMode === "all" ? "rgba(0, 229, 255, 0.32)" : "rgba(60, 255, 120, 0.32)";
        } else {
          ctx.fillStyle = shownVal > 0 ? "rgba(0, 229, 255, 0.14)" : "#0A2E5C";
        }
        ctx.fillRect(x, y, cs, cs);

        if (isHoveredCell || isHoveredComboCell) {
          ctx.strokeStyle = "#FFF03C";
          ctx.lineWidth = 2.5;
        } else if (isBest) {
          ctx.strokeStyle = viewMode === "all" ? "#00E5FF" : "#3CFF78";
          ctx.lineWidth = 2;
        } else {
          ctx.strokeStyle = "#444E66";
          ctx.lineWidth = 1;
        }
        ctx.strokeRect(x, y, cs, cs);

        const cx = x + cs / 2;
        const cy = y + cs / 2;

        if (isMiss) {
          ctx.fillStyle = "#FF5252";
          ctx.font = "700 10px Inter, system-ui, sans-serif";
          ctx.fillText("MISS", cx, cy);
        } else if (isHit) {
          ctx.fillStyle = "#FF3232";
          ctx.font = "700 10.5px Inter, system-ui, sans-serif";
          ctx.fillText("🔥 HIT", cx, cy);
        } else if (isFixed) {
          ctx.fillStyle = "#C5CAE9";
          ctx.font = "700 9.5px Inter, system-ui, sans-serif";
          ctx.fillText("SHIP 1", cx, cy - 7);
          ctx.fillStyle = "#FFFFFF";
          ctx.font = "700 12px Inter, system-ui, sans-serif";
          ctx.fillText(viewMode === "all" ? String(aVal) : "—", cx, cy + 8);
        } else {
          // Show big clean count in center, and small "-X" badge in corner if any were subtracted
          ctx.fillStyle = shownVal > 0 ? (isBest ? "#3CFF78" : "#FFFFFF") : "#6E7891";
          ctx.font = "700 18px Inter, system-ui, sans-serif";
          ctx.fillText(String(shownVal), cx, cy - (viewMode === "valid" && sVal > 0 ? 4 : 0));

          if (viewMode === "valid" && sVal > 0) {
            ctx.fillStyle = "#FF8A80";
            ctx.font = "600 9.5px Inter, system-ui, sans-serif";
            ctx.fillText(`(${aVal} − ${sVal})`, cx, cy + 13);
          }
        }
      }
    }

    if (footerText) {
      ctx.fillStyle = "#9AA0B4";
      ctx.font = "600 10.5px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(footerText, w / 2, h - 10);
    }

    return { ox, oy, cs, rows, cols };
  }

  // ==========================================================================
  // SECTION 4 SCENARIO GENERATORS (UNIFIED 2x3 BOARD FOR TABS 1-3 + 3x4 FOR TAB 4)
  // ==========================================================================

  // Shared 7 placements of a Size-2 Destroyer on a 2x3 board
  const ALL_7_SIZE2_SLOTS_2X3 = [
    { label: "Slot #1", cells: [{ r: 0, c: 0 }, { r: 0, c: 1 }] },
    { label: "Slot #2", cells: [{ r: 0, c: 1 }, { r: 0, c: 2 }] },
    { label: "Slot #3", cells: [{ r: 1, c: 0 }, { r: 1, c: 1 }] },
    { label: "Slot #4", cells: [{ r: 1, c: 1 }, { r: 1, c: 2 }] },
    { label: "Slot #5", cells: [{ r: 0, c: 0 }, { r: 1, c: 0 }] },
    { label: "Slot #6", cells: [{ r: 0, c: 1 }, { r: 1, c: 1 }] },
    { label: "Slot #7", cells: [{ r: 0, c: 2 }, { r: 1, c: 2 }] }
  ];

  // TAB 0: Marginal vs. Exact Joint Hunt (2x3 Grid, Fleet = {Ship 1: Size 3, Ship 2: Size 2}, Miss at B3)
  function buildTab0MarginalVsJoint() {
    const rows = 2, cols = 3;
    const misses = [{ r: 1, c: 2 }]; // B3 is a Miss
    const ship1Cells = [{ r: 0, c: 0 }, { r: 0, c: 1 }, { r: 0, c: 2 }]; // Forced into Row A!

    // The 5 placements of Ship 2 (Size 2) that don't touch Miss B3:
    const ship2Placements = [
      { label: "Slot #1", cells: [{ r: 0, c: 0 }, { r: 0, c: 1 }], reason: "❌ Collides at A1,A2" },
      { label: "Slot #2", cells: [{ r: 0, c: 1 }, { r: 0, c: 2 }], reason: "❌ Collides at A2,A3" },
      { label: "Slot #3", cells: [{ r: 1, c: 0 }, { r: 1, c: 1 }], reason: "✅ Fits in Row B!" },
      { label: "Slot #5", cells: [{ r: 0, c: 0 }, { r: 1, c: 0 }], reason: "❌ Collides at A1" },
      { label: "Slot #6", cells: [{ r: 0, c: 1 }, { r: 1, c: 1 }], reason: "❌ Collides at A2" }
    ];

    const combos = ship2Placements.map((b) => {
      const conflicts = b.cells.filter((pt) => pt.r === 0);
      const valid = conflicts.length === 0;
      return {
        label: b.label,
        valid,
        reason: b.reason,
        ships: [
          { cells: ship1Cells, fill: "rgba(92, 107, 192, 0.42)", stroke: "#7986CB" },
          { cells: b.cells, fill: valid ? "rgba(60, 255, 120, 0.55)" : "rgba(255, 82, 82, 0.45)", stroke: valid ? "#3CFF78" : "#FF5252" }
        ],
        conflictCells: conflicts,
        countCells: b.cells
      };
    });

    return {
      rows,
      cols,
      boardState: { misses, fixedShip: ship1Cells },
      banner: "<strong>1. Marginal vs. Exact Joint Hunt:</strong> Because B3 is a Miss, <b>Ship 1 (Size 3, purple)</b> is forced into Row A. Watson's Marginal Hunt still counts all <b>5</b> placements of Ship 2 (Size 2); Sherlock's Joint Hunt subtracts the <b>4</b> that crash into Ship 1!",
      galleryTitle: "5 Placements of Ship 2 (Size 2) vs. Forced Ship 1 in Row A",
      gallerySummary: "5 Marginal − 4 Colliding (❌) = 1 Valid (✅)",
      heatmapTitleValid: "Sherlock Joint Heatmap (✅ Valid)",
      heatmapTitleAll: "Watson Marginal Heatmap (All 5)",
      heatmapFooter: "Hover any square to highlight its cards on the left",
      heatmapCaption: "After subtracting the 4 red ❌ collisions, only Slot #3 [B1, B2] remains!",
      combos
    };
  }

  // TAB 1: Speedup #1 — Last-Ship Subtraction Shortcut (Same 2x3 Grid, All 7 Empty-Board Slots of Ship 2)
  function buildTab1LastShipShortcut(ship1PosIdx) {
    const rows = 2, cols = 3;
    const ship1Cells = ship1PosIdx === 0
      ? [{ r: 0, c: 0 }, { r: 0, c: 1 }, { r: 0, c: 2 }]
      : [{ r: 1, c: 0 }, { r: 1, c: 1 }, { r: 1, c: 2 }];

    const combos = ALL_7_SIZE2_SLOTS_2X3.map((slot) => {
      const conflicts = slot.cells.filter((pt) =>
        ship1Cells.some((s1p) => s1p.r === pt.r && s1p.c === pt.c)
      );
      const valid = conflicts.length === 0;
      return {
        label: slot.label,
        valid,
        reason: valid ? "✅ Open Water" : "❌ Touches Ship 1",
        ships: [
          { cells: ship1Cells, fill: "rgba(92, 107, 192, 0.42)", stroke: "#7986CB" },
          { cells: slot.cells, fill: valid ? "rgba(60, 255, 120, 0.55)" : "rgba(255, 82, 82, 0.45)", stroke: valid ? "#3CFF78" : "#FF5252" }
        ],
        conflictCells: conflicts,
        countCells: slot.cells
      };
    });

    return {
      rows,
      cols,
      boardState: { fixedShip: ship1Cells },
      banner: "<strong>2. Last-Ship Subtraction Shortcut:</strong> An empty 2×3 board has <b>7 total slots</b> for the last ship (Size 2). When Ship 1 (purple) is placed, we don't loop over all 7 slots—we know Ship 1 blocks <b>5 slots</b> (2 horiz + 3 vert), so <code>7 − 5 = 2 Valid Slots</code> in <code>O(1)</code>!",
      galleryTitle: "All 7 Empty-Board Slots of the Last Ship (Size 2)",
      gallerySummary: "7 Empty-Board Slots − 5 Blocked (❌) = 2 Valid (✅)",
      heatmapTitleValid: "Remaining Valid Slots (7 − 5 = 2)",
      heatmapTitleAll: "Precomputed Empty Board (All 7)",
      heatmapFooter: "Hover any square to see which of the 7 slots cover it",
      heatmapCaption: "Subtracting the 5 red ❌ slots that touch Ship 1 leaves the exact 2 green ✅ slots in the open row!",
      combos
    };
  }

  // TAB 2: Speedup #2 — Incremental Miss-Subtraction (Same 7 Slots on 2x3 Grid! Click Any Square to Fire a Miss!)
  function buildTab2MissSubtraction(missCoord) {
    const rows = 2, cols = 3;
    const mName = `${String.fromCharCode(65 + missCoord.r)}${missCoord.c + 1}`;
    let ruinedCount = 0;

    const combos = ALL_7_SIZE2_SLOTS_2X3.map((slot) => {
      const hitsMiss = slot.cells.some((pt) => pt.r === missCoord.r && pt.c === missCoord.c);
      if (hitsMiss) ruinedCount++;
      return {
        label: slot.label,
        valid: !hitsMiss,
        reason: hitsMiss ? `❌ Hits Miss ${mName}` : `✅ Avoids ${mName}`,
        ships: [
          {
            cells: slot.cells,
            fill: !hitsMiss ? "rgba(60, 255, 120, 0.55)" : "rgba(255, 82, 82, 0.45)",
            stroke: !hitsMiss ? "#3CFF78" : "#FF5252"
          }
        ],
        conflictCells: hitsMiss ? [missCoord] : [],
        countCells: slot.cells
      };
    });

    const validCount = ALL_7_SIZE2_SLOTS_2X3.length - ruinedCount;

    return {
      rows,
      cols,
      boardState: { misses: [missCoord] },
      banner: `<strong>3. Incremental Miss-Subtraction (Click Any Square on the Heatmap to Fire a Miss!):</strong> Before shooting, the empty 2×3 board has <b>7 valid slots</b>. When you fire a <b>MISS at ${mName}</b>, we only find the <b>${ruinedCount} slots</b> touching ${mName} (marked ❌) and subtract <code>−1</code> from their squares!`,
      galleryTitle: `Same 7 Slots — Firing a MISS at ${mName} Crosses Out ${ruinedCount} Slots`,
      gallerySummary: `7 Previous Slots − ${ruinedCount} Touching ${mName} (❌) = ${validCount} Valid (✅)`,
      heatmapTitleValid: `Updated Heatmap (7 − ${ruinedCount} = ${validCount} Valid)`,
      heatmapTitleAll: "Previous Turn Heatmap (All 7 Slots)",
      heatmapFooter: "👆 Click any square on this Heatmap to move the MISS!",
      heatmapCaption: `Only the ${ruinedCount} red ❌ slots touching ${mName} are subtracted from the cached board!`,
      combos
    };
  }

  // TAB 3 (LAST): Target Mode — 3 Interactive Steps (1 Hit -> 2 Hits Linear Kill -> Blocked Ends 2-Ship Flip)
  function buildTab3TargetLinearKill(stageIdx) {
    const rows = 3, cols = 4;

    // The 6 single-ship placements (Size 3 & Size 2) that pass through Hit #1 (B2)
    const base6Covers = [
      { label: "Slot #1", cells: [{ r: 0, c: 1 }, { r: 1, c: 1 }, { r: 2, c: 1 }], coversB3: false },
      { label: "Slot #2", cells: [{ r: 0, c: 1 }, { r: 1, c: 1 }], coversB3: false },
      { label: "Slot #3", cells: [{ r: 1, c: 1 }, { r: 2, c: 1 }], coversB3: false },
      { label: "Slot #4", cells: [{ r: 1, c: 0 }, { r: 1, c: 1 }], coversB3: false },
      { label: "Slot #5", cells: [{ r: 1, c: 0 }, { r: 1, c: 1 }, { r: 1, c: 2 }], coversB3: true },
      { label: "Slot #6", cells: [{ r: 1, c: 1 }, { r: 1, c: 2 }, { r: 1, c: 3 }], coversB3: true }
    ];

    if (stageIdx === 0) {
      // Step 1: Single Hit at B2 -> All 6 placements covering B2 are valid!
      const hits = [{ r: 1, c: 1 }];
      const combos = base6Covers.map((c) => ({
        label: c.label,
        valid: true,
        reason: "✅ Covers Hit B2",
        ships: [{ cells: c.cells, fill: "rgba(60, 255, 120, 0.55)", stroke: "#3CFF78" }],
        conflictCells: [],
        countCells: c.cells
      }));

      return {
        rows,
        cols,
        boardState: { hits },
        banner: "<strong>4. Target Mode (Step 1 — Single Hit at B2):</strong> When a single Hit lands at <b>B2</b>, all <b>6 ship placements</b> passing through B2 are valid (✅), creating a cross-shaped target pattern around B2.",
        galleryTitle: "All 6 Ship Placements Covering Hit #1 (B2)",
        gallerySummary: "6 Covering B2 − 0 Crossed Out = 6 Valid (✅)",
        heatmapTitleValid: "Step 1: Cross Target Pattern Around B2",
        heatmapTitleAll: "All 6 Placements Covering B2",
        heatmapFooter: "Next: Click 'Step 2: 2nd Hit at B3' above to trigger Linear Kill!",
        heatmapCaption: "With 1 Hit at B2, neighbors A2, C2, B1, and B3 all get high counts.",
        combos
      };
    } else if (stageIdx === 1) {
      // Step 2: 2nd Hit lands at B3 -> Crosses out the 4 placements that miss B3 -> Linear Kill on B1 & B4!
      const hits = [{ r: 1, c: 1 }, { r: 1, c: 2 }];
      const combos = base6Covers.map((c) => ({
        label: c.label,
        valid: c.coversB3,
        reason: c.coversB3 ? "✅ Covers B2 & B3!" : "❌ Misses Hit B3",
        ships: [
          {
            cells: c.cells,
            fill: c.coversB3 ? "rgba(60, 255, 120, 0.55)" : "rgba(255, 82, 82, 0.45)",
            stroke: c.coversB3 ? "#3CFF78" : "#FF5252"
          }
        ],
        conflictCells: c.coversB3 ? [] : [{ r: 1, c: 2 }],
        countCells: c.cells
      }));

      return {
        rows,
        cols,
        boardState: { hits },
        banner: "<strong>4. Target Mode (Step 2 — 2nd Hit at B3 → Linear Kill):</strong> Once Hit #2 lands at <b>B3</b>, any single ship must cover <i>both</i> B2 and B3. The <b>4 placements that miss B3 are crossed out (❌)</b>, locking onto inline ends <b>B1 &amp; B4</b>!",
        galleryTitle: "Same 6 Placements — Tested Against 2nd Hit at B3",
        gallerySummary: "6 Covers of B2 − 4 Missing B3 (❌) = 2 Inline Covers (✅)",
        heatmapTitleValid: "Step 2: Linear Kill Spikes Ends B1 & B4!",
        heatmapTitleAll: "Before Hit #2 (All 6 Covers of B2)",
        heatmapFooter: "Next: Click 'Step 3: Ends B1 & B4 Miss' to see the 90° flip!",
        heatmapCaption: "Subtracting the 4 red ❌ placements zeroes out A2 & C2 and spikes inline ends B1 & B4!",
        combos
      };
    } else {
      // Step 3: Inline Ends B1 and B4 BOTH MISS -> Horizontal 1-ship covers crossed out -> 4 Parallel Vertical 2-Ship Pairs!
      const hits = [{ r: 1, c: 1 }, { r: 1, c: 2 }];
      const misses = [{ r: 1, c: 0 }, { r: 1, c: 3 }];

      const cands = [
        {
          label: "Horiz #5",
          valid: false,
          reason: "❌ Blocked by Miss B1",
          ships: [{ cells: [{ r: 1, c: 0 }, { r: 1, c: 1 }, { r: 1, c: 2 }], fill: "rgba(255,82,82,0.45)", stroke: "#FF5252" }],
          conflictCells: [{ r: 1, c: 0 }],
          countCells: [{ r: 1, c: 0 }, { r: 1, c: 1 }, { r: 1, c: 2 }]
        },
        {
          label: "Horiz #6",
          valid: false,
          reason: "❌ Blocked by Miss B4",
          ships: [{ cells: [{ r: 1, c: 1 }, { r: 1, c: 2 }, { r: 1, c: 3 }], fill: "rgba(255,82,82,0.45)", stroke: "#FF5252" }],
          conflictCells: [{ r: 1, c: 3 }],
          countCells: [{ r: 1, c: 1 }, { r: 1, c: 2 }, { r: 1, c: 3 }]
        },
        {
          label: "Vert Pair #1",
          valid: true,
          reason: "✅ 2 Parallel Ships",
          ships: [
            { cells: [{ r: 0, c: 1 }, { r: 1, c: 1 }], fill: "rgba(60,255,120,0.55)", stroke: "#3CFF78" },
            { cells: [{ r: 0, c: 2 }, { r: 1, c: 2 }], fill: "rgba(0,229,255,0.55)", stroke: "#00E5FF" }
          ],
          conflictCells: [],
          countCells: [{ r: 0, c: 1 }, { r: 1, c: 1 }, { r: 0, c: 2 }, { r: 1, c: 2 }]
        },
        {
          label: "Vert Pair #2",
          valid: true,
          reason: "✅ 2 Parallel Ships",
          ships: [
            { cells: [{ r: 0, c: 1 }, { r: 1, c: 1 }], fill: "rgba(60,255,120,0.55)", stroke: "#3CFF78" },
            { cells: [{ r: 1, c: 2 }, { r: 2, c: 2 }], fill: "rgba(0,229,255,0.55)", stroke: "#00E5FF" }
          ],
          conflictCells: [],
          countCells: [{ r: 0, c: 1 }, { r: 1, c: 1 }, { r: 1, c: 2 }, { r: 2, c: 2 }]
        },
        {
          label: "Vert Pair #3",
          valid: true,
          reason: "✅ 2 Parallel Ships",
          ships: [
            { cells: [{ r: 1, c: 1 }, { r: 2, c: 1 }], fill: "rgba(60,255,120,0.55)", stroke: "#3CFF78" },
            { cells: [{ r: 0, c: 2 }, { r: 1, c: 2 }], fill: "rgba(0,229,255,0.55)", stroke: "#00E5FF" }
          ],
          conflictCells: [],
          countCells: [{ r: 1, c: 1 }, { r: 2, c: 1 }, { r: 0, c: 2 }, { r: 1, c: 2 }]
        },
        {
          label: "Vert Pair #4",
          valid: true,
          reason: "✅ 2 Parallel Ships",
          ships: [
            { cells: [{ r: 1, c: 1 }, { r: 2, c: 1 }], fill: "rgba(60,255,120,0.55)", stroke: "#3CFF78" },
            { cells: [{ r: 1, c: 2 }, { r: 2, c: 2 }], fill: "rgba(0,229,255,0.55)", stroke: "#00E5FF" }
          ],
          conflictCells: [],
          countCells: [{ r: 1, c: 1 }, { r: 2, c: 1 }, { r: 1, c: 2 }, { r: 2, c: 2 }]
        }
      ];

      return {
        rows,
        cols,
        boardState: { hits, misses },
        banner: "<strong>4. Target Mode (Step 3 — Blocked Ends Twist):</strong> When both inline ends <b>B1 &amp; B4 MISS</b>, both horizontal 1-ship covers are crossed out (❌)! The solver pairs <b>2 parallel vertical ships</b> (✅), flipping target priority 90° to <b>A2, C2, A3, C3</b>!",
        galleryTitle: "2 Blocked Horizontal Covers (❌) → 4 Parallel Vertical 2-Ship Pairs (✅)",
        gallerySummary: "6 Candidates − 2 Blocked Horiz (❌) = 4 Vertical Pairs (✅)",
        heatmapTitleValid: "Step 3: Target Flips 90° to A2, C2, A3, C3!",
        heatmapTitleAll: "All 6 Candidates (Before Misses at B1 & B4)",
        heatmapFooter: "Hover A2, C2, A3, or C3 to see which vertical pairs cover it",
        heatmapCaption: "Because B1 and B4 missed, B2 and B3 MUST belong to two parallel vertical ships side-by-side!",
        combos: cands
      };
    }
  }

  // ==========================================================================
  // MAIN EXPLAINERS UI CONTROLLER
  // ==========================================================================
  class ExplainersUI {
    constructor() {
      // Section 3 state
      this.cspLayerIdx = 0;

      // Section 4 state
      this.algoTabIdx = 0;
      this.algoViewMode = "valid"; // "valid" (After Subtraction) or "all" (Before Subtraction)
      this.tab1ShipPos = 0;
      this.tab2MissCoord = { r: 1, c: 1 }; // Default Miss at B2 on 2x3 grid
      this.tab3TargetStage = 1;            // Default to Step 2 (Linear Kill) so the subtraction is immediately visible
      this.hoverCell = null;               // { r, c } when hovering Heatmap
      this.hoverComboIdx = null;           // index when hovering a Mini-Board card
      this.lastHeatmapGeom = null;
      this.currentSection4Data = null;
      this.currentMatrices = null;
    }

    init() {
      this.bindEvents();
      this.renderAll();
    }

    bindEvents() {
      // Section 3: 5 Layer Tabs
      const layerTabs = document.getElementById("csp-layer-tabs");
      if (layerTabs) {
        layerTabs.querySelectorAll("button[data-layer]").forEach((btn) => {
          btn.addEventListener("click", () => {
            this.cspLayerIdx = Number(btn.getAttribute("data-layer")) || 0;
            layerTabs.querySelectorAll("button[data-layer]").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            this.renderCSPLayers();
          });
        });
      }

      // Section 4: 4 Algorithm Tabs
      const algoTabs = document.getElementById("algo-tab-buttons");
      if (algoTabs) {
        algoTabs.querySelectorAll("button[data-algo]").forEach((btn) => {
          btn.addEventListener("click", () => {
            this.algoTabIdx = Number(btn.getAttribute("data-algo")) || 0;
            this.hoverCell = null;
            this.hoverComboIdx = null;
            algoTabs.querySelectorAll("button[data-algo]").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            this.renderSection4();
          });
        });
      }

      // Section 4: Before / After Subtraction Toggle on Heatmap
      const viewModeGroup = document.getElementById("algo-view-mode-group");
      if (viewModeGroup) {
        viewModeGroup.querySelectorAll("button[data-viewmode]").forEach((btn) => {
          btn.addEventListener("click", () => {
            this.algoViewMode = btn.getAttribute("data-viewmode") || "valid";
            viewModeGroup.querySelectorAll("button[data-viewmode]").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            this.refreshSection4Interactive();
          });
        });
      }

      // Section 4: Hover & Click on the Single Interactive Heatmap
      const heatCan = document.getElementById("canvas-algo-heatmap");
      if (heatCan) {
        const getGridCellFromEvent = (e) => {
          if (!this.lastHeatmapGeom) return null;
          const rect = heatCan.getBoundingClientRect();
          const mx = (e.clientX - rect.left) * (320 / rect.width);
          const my = (e.clientY - rect.top) * (250 / rect.height);
          const { ox, oy, cs, rows, cols } = this.lastHeatmapGeom;
          const c = Math.floor((mx - ox) / cs);
          const r = Math.floor((my - oy) / cs);
          if (r >= 0 && r < rows && c >= 0 && c < cols) {
            return { r, c };
          }
          return null;
        };

        heatCan.addEventListener("mousemove", (e) => {
          const cell = getGridCellFromEvent(e);
          const prev = this.hoverCell;
          const changed = (!prev && cell) || (prev && !cell) || (prev && cell && (prev.r !== cell.r || prev.c !== cell.c));
          if (changed) {
            this.hoverCell = cell;
            this.hoverComboIdx = null;
            this.refreshSection4Interactive();
          }
        });

        heatCan.addEventListener("mouseleave", () => {
          if (this.hoverCell) {
            this.hoverCell = null;
            this.refreshSection4Interactive();
          }
        });

        heatCan.addEventListener("click", (e) => {
          if (this.algoTabIdx !== 2) return;
          const cell = getGridCellFromEvent(e);
          if (cell) {
            this.tab2MissCoord = cell;
            this.hoverCell = cell;
            this.renderSection4();
          }
        });
      }
    }

    renderAll() {
      this.renderBenchmarkChart();
      this.renderCSPLayers();
      this.renderSection4();
    }

    // ------------------------------------------------------------------------
    // SECTION 3: 5-LAYER SIDE-BY-SIDE TRIPTYCH
    // ------------------------------------------------------------------------
    renderCSPLayers() {
      const layer = CSP_LAYERS_DATA[this.cspLayerIdx];
      if (!layer) return;

      const bannerEl = document.getElementById("csp-layer-banner");
      if (bannerEl) bannerEl.innerHTML = layer.banner;

      const hBad = document.getElementById("header-layer-bad");
      const hGood = document.getElementById("header-layer-good");
      if (hBad) hBad.textContent = layer.bad.header;
      if (hGood) hGood.textContent = layer.good.header;

      const cRaw = document.getElementById("caption-layer-raw");
      const cBad = document.getElementById("caption-layer-bad");
      const cGood = document.getElementById("caption-layer-good");
      if (cRaw) cRaw.textContent = layer.raw.caption;
      if (cBad) cBad.textContent = layer.bad.caption;
      if (cGood) cGood.textContent = layer.good.caption;

      const canRaw = document.getElementById("canvas-layer-raw");
      const canBad = document.getElementById("canvas-layer-bad");
      const canGood = document.getElementById("canvas-layer-good");
      if (canRaw) drawMiniBoard5x5(setupCanvas(canRaw, 250, 230), 250, 230, layer.raw);
      if (canBad) drawMiniBoard5x5(setupCanvas(canBad, 250, 230), 250, 230, layer.bad);
      if (canGood) drawMiniBoard5x5(setupCanvas(canGood, 250, 230), 250, 230, layer.good);
    }

    // ------------------------------------------------------------------------
    // SECTION 4: INTERACTIVE MINI-BOARDS + LIVE HEATMAP
    // ------------------------------------------------------------------------
    renderSection4() {
      let data;
      if (this.algoTabIdx === 0) {
        data = buildTab0MarginalVsJoint();
      } else if (this.algoTabIdx === 1) {
        data = buildTab1LastShipShortcut(this.tab1ShipPos);
      } else if (this.algoTabIdx === 2) {
        data = buildTab2MissSubtraction(this.tab2MissCoord);
      } else {
        data = buildTab3TargetLinearKill(this.tab3TargetStage);
      }
      this.currentSection4Data = data;

      // Render contextual subcontrols if applicable
      const subBar = document.getElementById("algo-subcontrols");
      if (subBar) {
        subBar.innerHTML = "";
        if (this.algoTabIdx === 1) {
          ["Ship 1 in Row A", "Ship 1 in Row B"].forEach((lbl, idx) => {
            const b = document.createElement("button");
            b.className = "btn" + (this.tab1ShipPos === idx ? " active" : "");
            b.textContent = lbl;
            b.addEventListener("click", () => {
              this.tab1ShipPos = idx;
              this.renderSection4();
            });
            subBar.appendChild(b);
          });
        } else if (this.algoTabIdx === 2) {
          [
            { r: 0, c: 1, lbl: "Miss at A2" },
            { r: 1, c: 1, lbl: "Miss at B2" },
            { r: 1, c: 2, lbl: "Miss at B3" }
          ].forEach((m) => {
            const b = document.createElement("button");
            const isAct = this.tab2MissCoord.r === m.r && this.tab2MissCoord.c === m.c;
            b.className = "btn" + (isAct ? " active" : "");
            b.textContent = m.lbl;
            b.addEventListener("click", () => {
              this.tab2MissCoord = { r: m.r, c: m.c };
              this.renderSection4();
            });
            subBar.appendChild(b);
          });
        } else if (this.algoTabIdx === 3) {
          [
            "Step 1: 1 Hit (B2)",
            "Step 2: 2nd Hit (B3) → Linear Kill",
            "Step 3: Ends B1 & B4 Miss → 2-Ship Flip"
          ].forEach((lbl, idx) => {
            const b = document.createElement("button");
            b.className = "btn" + (this.tab3TargetStage === idx ? " active" : "");
            b.textContent = lbl;
            b.addEventListener("click", () => {
              this.tab3TargetStage = idx;
              this.renderSection4();
            });
            subBar.appendChild(b);
          });
        }
      }

      // Update text banners
      const bannerEl = document.getElementById("algo-rule-banner");
      if (bannerEl) bannerEl.innerHTML = data.banner;

      const gTitle = document.getElementById("combo-gallery-title");
      const gSum = document.getElementById("combo-gallery-summary");
      if (gTitle) gTitle.textContent = data.galleryTitle;
      if (gSum) gSum.textContent = data.gallerySummary;

      // Render Mini-Board Combination Gallery
      const galleryEl = document.getElementById("algo-combo-gallery");
      if (galleryEl) {
        galleryEl.innerHTML = "";
        data.combos.forEach((combo, idx) => {
          const card = document.createElement("div");
          card.className = "mini-combo-card " + (combo.valid ? "valid" : "ruined");
          card.setAttribute("data-combo-idx", String(idx));

          const top = document.createElement("div");
          top.className = "mini-combo-top";
          top.innerHTML = `<span>${combo.label}</span><span>${combo.valid ? "✅ +1" : "❌ −1"}</span>`;

          const can = document.createElement("canvas");
          can.style.borderRadius = "5px";

          const reason = document.createElement("div");
          reason.className = "mini-combo-reason";
          reason.textContent = combo.reason;

          card.appendChild(top);
          card.appendChild(can);
          card.appendChild(reason);

          card.addEventListener("mouseenter", () => {
            this.hoverComboIdx = idx;
            this.hoverCell = null;
            this.refreshSection4Interactive();
          });
          card.addEventListener("mouseleave", () => {
            if (this.hoverComboIdx === idx) {
              this.hoverComboIdx = null;
              this.refreshSection4Interactive();
            }
          });

          galleryEl.appendChild(card);
          drawComboMiniCanvas(can, data.rows, data.cols, combo, data.boardState);
        });
      }

      // Compute exact cell-by-cell matrices from the combos in the gallery
      const allMat = Array.from({ length: data.rows }, () => new Array(data.cols).fill(0));
      const subMat = Array.from({ length: data.rows }, () => new Array(data.cols).fill(0));
      const validMat = Array.from({ length: data.rows }, () => new Array(data.cols).fill(0));

      for (const combo of data.combos) {
        for (const pt of combo.countCells) {
          allMat[pt.r][pt.c] += 1;
          if (!combo.valid) {
            subMat[pt.r][pt.c] += 1;
          } else {
            validMat[pt.r][pt.c] += 1;
          }
        }
      }
      this.currentMatrices = { allMat, subMat, validMat };

      this.refreshSection4Interactive();
    }

    refreshSection4Interactive() {
      const data = this.currentSection4Data;
      const mats = this.currentMatrices;
      if (!data || !mats) return;

      const { allMat, subMat, validMat } = mats;
      const hTitle = document.getElementById("algo-heatmap-title");
      const hCaption = document.getElementById("algo-heatmap-caption");
      const eqPill = document.getElementById("algo-heatmap-equation");

      if (hTitle) {
        hTitle.textContent = this.algoViewMode === "all" ? data.heatmapTitleAll : data.heatmapTitleValid;
      }
      if (hCaption) {
        hCaption.textContent = data.heatmapCaption;
      }

      // Update card highlighting in the gallery when hovering a Heatmap cell or a Card
      const galleryEl = document.getElementById("algo-combo-gallery");
      if (galleryEl) {
        const cards = galleryEl.querySelectorAll(".mini-combo-card");
        cards.forEach((card, idx) => {
          const combo = data.combos[idx];
          card.classList.remove("highlight-hit", "dimmed");
          if (this.hoverCell) {
            const coversHover = combo.countCells.some(
              (pt) => pt.r === this.hoverCell.r && pt.c === this.hoverCell.c
            );
            if (coversHover) {
              card.classList.add("highlight-hit");
            } else {
              card.classList.add("dimmed");
            }
          } else if (this.hoverComboIdx !== null) {
            if (idx === this.hoverComboIdx) {
              card.classList.add("highlight-hit");
            } else {
              card.classList.add("dimmed");
            }
          }
        });
      }

      // Update live equation pill
      if (eqPill) {
        if (this.hoverCell) {
          const { r, c } = this.hoverCell;
          const cellName = `${String.fromCharCode(65 + r)}${c + 1}`;
          const aVal = allMat[r][c];
          const sVal = subMat[r][c];
          const vVal = validMat[r][c];
          eqPill.innerHTML = `Square <b>${cellName}</b>: <code>${aVal} Total − ${sVal} Crossed Out (❌) = <span style="color:#3CFF78;font-weight:700;">${vVal} Valid (✅)</span></code>`;
        } else if (this.hoverComboIdx !== null) {
          const combo = data.combos[this.hoverComboIdx];
          const coords = combo.countCells.map((p) => `${String.fromCharCode(65 + p.r)}${p.c + 1}`).join(", ");
          eqPill.innerHTML = `<b>${combo.label}</b> (${combo.reason}): ${combo.valid ? `<span style="color:#3CFF78;">Adds +1</span>` : `<span style="color:#FF8A80;">Subtracted (−1)</span>`} on <code>[${coords}]</code>`;
        } else {
          eqPill.innerHTML = `Hover any square on the Heatmap (or any card on the left) to inspect its exact math.`;
        }
      }

      const hoverComboCells = this.hoverComboIdx !== null && data.combos[this.hoverComboIdx]
        ? data.combos[this.hoverComboIdx].countCells
        : null;

      const heatCan = document.getElementById("canvas-algo-heatmap");
      if (heatCan) {
        this.lastHeatmapGeom = drawInteractiveHeatmap(
          heatCan,
          data.rows,
          data.cols,
          this.algoViewMode,
          allMat,
          subMat,
          validMat,
          data.boardState,
          this.hoverCell,
          hoverComboCells,
          data.heatmapFooter
        );
      }
    }

    // ------------------------------------------------------------------------
    // SECTION 2: PAIRED-BOARD TOURNAMENT BENCHMARK CHART
    // ------------------------------------------------------------------------
    renderBenchmarkChart() {
      const canvas = document.getElementById("canvas-benchmark-chart");
      if (!canvas) return;
      const ctx = setupCanvas(canvas, 860, 220);

      ctx.fillStyle = "#161820";
      ctx.fillRect(0, 0, 860, 220);

      const data = [
        { name: "Mycroft (Entropy + Exact Joint)", mean: 45.01, ci: "±0.16", p90: 56, outliers: "2.0%", speed: "0.08–1.2 ms", color: "#00E5FF" },
        { name: "Watson (CSP + Unique Marginal)",  mean: 45.11, ci: "±0.16", p90: 56, outliers: "1.4%", speed: "0.05 ms",     color: "#3CFF78" },
        { name: "Sherlock (Exact Joint Solver)",   mean: 45.14, ci: "±0.17", p90: 56, outliers: "2.1%", speed: "0.08–1.2 ms", color: "#FFC864" },
        { name: "Moriarty (Trained ResNet + CSP)", mean: 45.62, ci: "±0.45", p90: 57, outliers: "2.2%", speed: "18.5 ms",     color: "#B388FF" },
        { name: "Lestrade (Naive Marginal Base)",  mean: 56.09, ci: "±0.58", p90: 70, outliers: "29.3%", speed: "0.001 ms",   color: "#FF5252" }
      ];

      const leftLabelW = 240;
      const rightStatsW = 220;
      const barAreaW = 860 - leftLabelW - rightStatsW - 30;
      const minScale = 40.0;
      const maxScale = 60.0;

      ctx.fillStyle = "#9AA0B4";
      ctx.font = "600 11px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("Bot Architecture", 16, 22);
      ctx.fillText("Average Shots to Clear Fleet (Lower is Better)", leftLabelW, 22);
      ctx.fillText("95% CI  |  Outliers >60  |  Turn Speed", 860 - rightStatsW, 22);

      data.forEach((row, i) => {
        const y = 42 + i * 34;

        ctx.fillStyle = "#E1E5F0";
        ctx.font = "600 12.5px Inter, system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(row.name, 16, y + 14);

        ctx.fillStyle = "#222634";
        ctx.beginPath();
        ctx.roundRect(leftLabelW, y + 2, barAreaW, 20, 5);
        ctx.fill();

        const ratio = Math.max(0.05, Math.min(1.0, (row.mean - minScale) / (maxScale - minScale)));
        const bw = ratio * barAreaW;
        ctx.fillStyle = row.color;
        ctx.beginPath();
        ctx.roundRect(leftLabelW, y + 2, bw, 20, 5);
        ctx.fill();

        ctx.fillStyle = "#101218";
        ctx.font = "700 11.5px Inter, system-ui, sans-serif";
        ctx.fillText(`${row.mean.toFixed(2)} shots`, leftLabelW + 10, y + 15);

        ctx.fillStyle = "#B8C0D4";
        ctx.font = "500 12px Inter, system-ui, sans-serif";
        ctx.fillText(`${row.ci}   |   ${row.outliers.padStart(5)}   |   ${row.speed}`, 860 - rightStatsW, y + 15);
      });
    }
  }

  root.ExplainersUI = ExplainersUI;
})(typeof window !== "undefined" ? window : globalThis);

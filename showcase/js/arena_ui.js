// showcase/js/arena_ui.js
// Interactive Multi-Bot Arena UI (Master Gameboard + 4-Bot Quad & 2-Bot Dual Heatmap Views)
// Supports all classic showcase controls: [1][2][3] Presets, [R] Randomize, [Click] Fire/Place,
// [W][S][M][O] Bot Shoot, [A] Auto-Fire, [U] Undo, [H] Toggle Ships & Diagnostics, [N] Setup, [C] Theme.

(function (root) {
  "use strict";

  const Engine = root.BattleshipEngine;

  const THEMES = {
    classic: {
      label: "Classic Tactical",
      bg: "#19191E",
      panelBg: "#212229",
      gridLine: "#646464",
      water: "#0A3C78",
      ship: "#3C463C",
      ghostValid: "rgba(0, 255, 0, 0.47)",
      ghostInval: "rgba(255, 0, 0, 0.47)",
      label: "#B4B4B4",
      header: "#E1E1EB",
      telemetry: "#FFC864",
      victory: "#3CFF78",
      pegMiss: "#DCDCDC",
      pegHit: "#FF3232",
      pegSunk: "#3296FF",
      diagClaimed: "rgba(180, 190, 205, 0.92)",
      diagActive: "#00E5FF",
      targetRet: "#FFF03C",
      heatStops: [
        [10, 60, 120],
        [50, 120, 160],
        [80, 160, 120],
        [180, 180, 60],
        [220, 120, 40],
        [255, 50, 50]
      ]
    },
    baker_street: {
      label: "221B Gaslight",
      bg: "#120E0C",
      panelBg: "#1C1613",
      gridLine: "#735C3E",
      water: "#18262C",
      ship: "#4E3A28",
      ghostValid: "rgba(110, 210, 110, 0.47)",
      ghostInval: "rgba(220, 60, 50, 0.47)",
      label: "#CDB994",
      header: "#F2E0BA",
      telemetry: "#F5BC52",
      victory: "#96EB82",
      pegMiss: "#E8DABE",
      pegHit: "#E13A2D",
      pegSunk: "#F5BE46",
      diagClaimed: "rgba(235, 195, 115, 0.92)",
      diagActive: "#69E1D2",
      targetRet: "#FFE678",
      heatStops: [
        [24, 38, 44],
        [42, 78, 76],
        [110, 122, 64],
        [188, 142, 52],
        [218, 92, 42],
        [205, 38, 38]
      ]
    },
    sonar_crt: {
      label: "Cold War Sonar",
      bg: "#060E0A",
      panelBg: "#0B1912",
      gridLine: "#1C693E",
      water: "#081C12",
      ship: "#18482E",
      ghostValid: "rgba(40, 255, 120, 0.45)",
      ghostInval: "rgba(255, 60, 60, 0.45)",
      label: "#6ED791",
      header: "#96FAB4",
      telemetry: "#BEFF6E",
      victory: "#50FF96",
      pegMiss: "#82D2A0",
      pegHit: "#FF961E",
      pegSunk: "#28F5DC",
      diagClaimed: "rgba(120, 255, 160, 0.92)",
      diagActive: "#FFE63C",
      targetRet: "#FFFF8C",
      heatStops: [
        [8, 28, 18],
        [16, 72, 42],
        [36, 130, 64],
        [95, 190, 55],
        [205, 210, 45],
        [255, 145, 35]
      ]
    },
    abyssal: {
      label: "Abyssal Biolum",
      bg: "#0A0B18",
      panelBg: "#121428",
      gridLine: "#444E8A",
      water: "#101630",
      ship: "#342C5C",
      ghostValid: "rgba(0, 245, 190, 0.45)",
      ghostInval: "rgba(255, 40, 110, 0.45)",
      label: "#AAB6E6",
      header: "#E1E8FF",
      telemetry: "#6EF0FF",
      victory: "#50FFC8",
      pegMiss: "#C3CDF0",
      pegHit: "#FF3791",
      pegSunk: "#00EBFF",
      diagClaimed: "rgba(190, 165, 255, 0.92)",
      diagActive: "#FFE14B",
      targetRet: "#50FFDC",
      heatStops: [
        [16, 22, 48],
        [52, 32, 108],
        [118, 36, 138],
        [188, 52, 122],
        [238, 108, 68],
        [252, 198, 58]
      ]
    }
  };

  const THEME_KEYS = ["classic", "baker_street", "sonar_crt", "abyssal"];

  function sampleHeatColor(val, maxVal, stops) {
    if (maxVal <= 0 || val <= 0) {
      const s0 = stops[0];
      return { css: `rgb(${s0[0]},${s0[1]},${s0[2]})`, r: s0[0], g: s0[1], b: s0[2] };
    }
    const ratio = Math.max(0, Math.min(1, val / maxVal));
    const numActive = stops.length - 1;
    const scaled = 1.0 + ratio * (numActive - 1);
    const idxLo = Math.min(stops.length - 2, Math.max(1, Math.floor(scaled)));
    const idxHi = Math.min(stops.length - 1, idxLo + 1);
    const t = scaled - idxLo;
    const c1 = stops[idxLo];
    const c2 = stops[idxHi];
    const r = Math.round(c1[0] + (c2[0] - c1[0]) * t);
    const g = Math.round(c1[1] + (c2[1] - c1[1]) * t);
    const b = Math.round(c1[2] + (c2[2] - c1[2]) * t);
    return { css: `rgb(${r},${g},${b})`, r, g, b };
  }

  function setupHiDPICanvas(canvas, logicalW, logicalH) {
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(logicalW * dpr) || canvas.height !== Math.round(logicalH * dpr)) {
      canvas.width = Math.round(logicalW * dpr);
      canvas.height = Math.round(logicalH * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function drawRoundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
  }

  class ArenaController {
    constructor() {
      this.comb = new Engine.Combinatorics();
      this.ded = new Engine.DeductionEngine();
      this.joint = new Engine.JointSolver(this.comb);

      this.bots = {
        watson: new Engine.WatsonBot(0, this.comb, this.ded),
        sherlock: new Engine.SherlockBot(0, this.comb, this.ded, this.joint),
        mycroft: new Engine.MycroftBot(0, this.comb, this.ded, this.joint),
        moriarty: new Engine.MoriartyBot(0, this.comb, this.ded)
      };

      this.botMeta = {
        watson: { key: "watson", name: "Watson", badge: "CSP + Marginal Parity", hotkey: "W", avgShots: "45.11" },
        sherlock: { key: "sherlock", name: "Sherlock", badge: "Exact Joint Solver", hotkey: "S", avgShots: "45.14" },
        mycroft: { key: "mycroft", name: "Mycroft", badge: "Shannon Entropy + Joint", hotkey: "M", avgShots: "45.01" },
        moriarty: { key: "moriarty", name: "Moriarty", badge: "Trained ResNet + Assassin", hotkey: "O", avgShots: "46.85" }
      };

      this.game = new Engine.BattleshipGame();
      this.gameId = 1;
      this.phase = "PLAY"; // Start immediately on a randomized board so visitors can click & play right away
      this.shipsToPlace = [];
      this.currentShipId = 1;
      this.isHorizontal = true;
      this.showShips = true;
      this.themeKey = "classic";
      this.viewMode = "quad"; // "quad" (4 bots) or "dual" (2 bots side-by-side)
      this.dualSlots = ["sherlock", "mycroft"];
      this.autoPlay = false;
      this.autoBotKey = "mycroft";
      this.autoTimer = null;

      this.hoverCell = { r: -1, c: -1 };
      this.botOutputs = {};
      this.diagnostics = {};
    }

    async init() {
      this.bindDOM();
      this.startRandomGame();
      this.loadMoriartyWeights();
      this.renderAll();
    }

    async loadMoriartyWeights() {
      const statusBadge = document.getElementById("moriarty-weights-status");
      try {
        const resp = await fetch("./assets/moriarty_fp16.bin");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const buf = await resp.arrayBuffer();
        const f32 = Engine.decodeFloat16Buffer(buf);
        this.bots.moriarty.setWeights(f32);
        if (statusBadge) {
          statusBadge.textContent = "ResNet Ready";
          statusBadge.classList.add("ready");
        }
        this.recomputeBots();
        this.renderAll();
      } catch (err) {
        console.warn("Moriarty weights load fallback:", err);
        if (statusBadge) statusBadge.textContent = "Fallback Weights";
      }
    }

    resetBotsForNewGame() {
      const parity = this.gameId % 2;
      for (const k of Object.keys(this.bots)) {
        this.bots[k].reset(this.gameId, parity);
      }
    }

    startManualSetup() {
      this.stopAutoPlay();
      this.game.reset();
      this.phase = "SETUP";
      this.shipsToPlace = Engine.FLEET_SIZES.slice();
      this.currentShipId = 1;
      this.isHorizontal = true;
      this.gameId++;
      this.resetBotsForNewGame();
      this.botOutputs = {};
      this.diagnostics = {};
      this.updateStatusHeader();
      this.renderAll();
    }

    startRandomGame() {
      this.stopAutoPlay();
      this.gameId++;
      this.game.placeShipsRandomly(this.gameId);
      this.phase = "PLAY";
      this.shipsToPlace = [];
      this.resetBotsForNewGame();
      this.recomputeBots();
      this.updateStatusHeader();
      this.renderAll();
    }

    loadPreset(presetId) {
      this.stopAutoPlay();
      this.gameId++;
      this.currentPresetId = presetId;
      this.game.loadPreset(presetId);
      this.phase = "PLAY";
      this.shipsToPlace = [];
      this.resetBotsForNewGame();
      this.recomputeBots();
      const names = {
        1: "Preset #1 (Collinear End-to-End Train)",
        2: "Preset #2 (T-Bone Junction)",
        3: "Preset #3 (5-Ship Phalanx Wall)",
        4: "Preset #4 (L-Corner & Pocketed Destroyer)",
        5: "Preset #5 (Twin Size-3 Staggered Overlap)"
      };
      this.updateStatusHeader(`${names[presetId] || `Preset #${presetId}`} — Click [⚡ Fire Trap] or shoot manually`);
      this.renderAll();
    }

    fireTrapShots() {
      this.stopAutoPlay();
      const pid = this.currentPresetId || 1;
      this.gameId++;
      this.resetBotsForNewGame();
      this.game.firePresetTrapShots(pid);
      this.phase = "PLAY";
      this.shipsToPlace = [];
      this.recomputeBots();
      this.updateStatusHeader(`Preset #${pid} Trap Shots Fired! Compare bot deductions →`);
      this.renderAll();
    }

    recomputeBots() {
      if (this.phase !== "PLAY" || this.game.isGameOver()) {
        if (this.game.isGameOver()) {
          const sols = this.ded.analyze(this.game.trackingBoard, this.game.moveHistory);
          this.diagnostics = Engine.DeductionEngine.buildDiagnostics(sols);
        }
        return;
      }

      const board = this.game.trackingBoard;
      const history = this.game.moveHistory;

      // Run Watson
      const tW0 = performance.now();
      const outW = this.bots.watson.getBestMove(board, history);
      outW.ms = performance.now() - tW0;
      this.botOutputs.watson = outW;

      // Run Sherlock
      const tS0 = performance.now();
      const outS = this.bots.sherlock.getBestMove(board, history);
      outS.ms = performance.now() - tS0;
      this.botOutputs.sherlock = outS;

      // Share Sherlock's exact joint hunt cache with Mycroft so we don't double-compute identical joint density!
      if (this.bots.sherlock.cachedHuntState) {
        this.bots.mycroft.cachedHuntState = {
          fleet: this.bots.sherlock.cachedHuntState.fleet.slice(),
          forbiddenBb: this.bots.sherlock.cachedHuntState.forbiddenBb.copy(),
          density: this.bots.sherlock.cachedHuntState.density,
          configs: this.bots.sherlock.cachedHuntState.configs
        };
      }

      // Run Mycroft
      const tM0 = performance.now();
      const outM = this.bots.mycroft.getBestMove(board, history);
      outM.ms = performance.now() - tM0;
      this.botOutputs.mycroft = outM;

      // Run Moriarty
      const tO0 = performance.now();
      const outO = this.bots.moriarty.getBestMove(board, history);
      outO.ms = performance.now() - tO0;
      this.botOutputs.moriarty = outO;

      this.diagnostics = outS.diagnostics || {};
    }

    fireBotShot(botKey) {
      if (this.phase !== "PLAY" || this.game.isGameOver()) return;
      const out = this.botOutputs[botKey];
      if (!out || !out.target) return;
      this.game.fireShot(out.target.x, out.target.y);
      this.recomputeBots();
      this.updateStatusHeader(`${this.botMeta[botKey].name} fired (${String.fromCharCode(65 + out.target.x)}${out.target.y + 1})`);
      this.renderAll();
      if (this.game.isGameOver()) {
        this.stopAutoPlay();
      }
    }

    undoShot() {
      this.stopAutoPlay();
      if (this.phase !== "PLAY") return;
      if (this.game.undoLastShot()) {
        // Clear subtractive caches on undo so joint state refreshes cleanly
        this.bots.sherlock.cachedHuntState = null;
        this.bots.mycroft.cachedHuntState = null;
        this.recomputeBots();
        this.updateStatusHeader("Undid last shot");
        this.renderAll();
      }
    }

    toggleAutoPlay(botKey = null) {
      if (botKey) this.autoBotKey = botKey;
      if (this.autoPlay) {
        this.stopAutoPlay();
      } else {
        if (this.phase !== "PLAY" || this.game.isGameOver()) return;
        this.autoPlay = true;
        const btn = document.getElementById("btn-autoplay");
        if (btn) btn.classList.add("active");
        this.autoTimer = setInterval(() => {
          if (this.phase !== "PLAY" || this.game.isGameOver()) {
            this.stopAutoPlay();
            return;
          }
          this.fireBotShot(this.autoBotKey);
        }, 120);
      }
      this.updateStatusHeader();
    }

    stopAutoPlay() {
      this.autoPlay = false;
      if (this.autoTimer) {
        clearInterval(this.autoTimer);
        this.autoTimer = null;
      }
      const btn = document.getElementById("btn-autoplay");
      if (btn) btn.classList.remove("active");
    }

    cycleTheme() {
      const idx = THEME_KEYS.indexOf(this.themeKey);
      this.themeKey = THEME_KEYS[(idx + 1) % THEME_KEYS.length];
      document.documentElement.style.setProperty("--arena-bg", THEMES[this.themeKey].bg);
      document.documentElement.style.setProperty("--arena-panel", THEMES[this.themeKey].panelBg);
      const themeBtn = document.getElementById("btn-theme");
      if (themeBtn) themeBtn.textContent = `[C] Theme: ${THEMES[this.themeKey].label}`;
      this.renderAll();
      if (root.ExplainersController) {
        root.ExplainersController.renderAll(THEMES[this.themeKey]);
      }
    }

    setViewMode(mode) {
      this.viewMode = mode;
      const gridContainer = document.getElementById("bot-heatmaps-grid");
      const dualControls = document.getElementById("dual-slot-selectors");
      const btnQuad = document.getElementById("btn-view-quad");
      const btnDual = document.getElementById("btn-view-dual");

      if (mode === "quad") {
        gridContainer.classList.remove("dual-mode");
        gridContainer.classList.add("quad-mode");
        if (dualControls) dualControls.classList.add("hidden");
        if (btnQuad) btnQuad.classList.add("active");
        if (btnDual) btnDual.classList.remove("active");
      } else {
        gridContainer.classList.remove("quad-mode");
        gridContainer.classList.add("dual-mode");
        if (dualControls) dualControls.classList.remove("hidden");
        if (btnQuad) btnQuad.classList.remove("active");
        if (btnDual) btnDual.classList.add("active");
      }
      this.updateVisibleCards();
      this.renderAll();
    }

    updateVisibleCards() {
      const cards = document.querySelectorAll(".bot-heatmap-card");
      cards.forEach((card) => {
        const key = card.getAttribute("data-bot");
        if (this.viewMode === "quad") {
          card.classList.remove("hidden");
        } else {
          if (this.dualSlots.includes(key)) {
            card.classList.remove("hidden");
          } else {
            card.classList.add("hidden");
          }
        }
      });
    }

    updateStatusHeader(actionNote = "") {
      const statusEl = document.getElementById("master-board-status");
      if (!statusEl) return;

      if (this.phase === "SETUP") {
        const nextLen = this.shipsToPlace[0] || 0;
        statusEl.textContent = `SETUP MODE — Place length-${nextLen} ship (${this.isHorizontal ? "Horiz" : "Vert"}) | Right-Click/[T]: Rotate | [R]: Randomize`;
        statusEl.className = "board-status-pill setup";
      } else if (this.game.isGameOver()) {
        statusEl.textContent = `FLEET CLEARED IN ${this.game.shotNumber} SHOTS! Press [R] to Randomize or [N] for Manual Setup.`;
        statusEl.className = "board-status-pill victory";
      } else {
        const tieNote = (this.diagnostics.solutionsCount && this.diagnostics.solutionsCount > 1)
          ? ` (${this.diagnostics.solutionsCount} tied hypotheses)`
          : "";
        const deadStr = (this.diagnostics.deadSizes && this.diagnostics.deadSizes.length > 0)
          ? `Locked Sunk: [${this.diagnostics.deadSizes.join(", ")}]${tieNote}`
          : "No ships sunk yet";
        const note = actionNote ? ` • ${actionNote}` : "";
        statusEl.textContent = `Shots: ${this.game.shotNumber}  |  Hits: ${this.game.pegsHit}/17  |  ${deadStr}${note}`;
        statusEl.className = "board-status-pill play";
      }
    }

    bindDOM() {
      const masterCanvas = document.getElementById("master-board-canvas");
      if (masterCanvas) {
        masterCanvas.addEventListener("mousemove", (e) => {
          const rect = masterCanvas.getBoundingClientRect();
          const scaleX = 420 / rect.width;
          const scaleY = 420 / rect.height;
          const mx = (e.clientX - rect.left) * scaleX;
          const my = (e.clientY - rect.top) * scaleY;
          const margin = 32;
          const cs = 38;
          const c = Math.floor((mx - margin) / cs);
          const r = Math.floor((my - margin) / cs);
          const valid = r >= 0 && r < 10 && c >= 0 && c < 10;
          const nr = valid ? r : -1;
          const nc = valid ? c : -1;
          if (nr !== this.hoverCell.r || nc !== this.hoverCell.c) {
            this.hoverCell = { r: nr, c: nc };
            this.renderMasterBoard();
          }
        });

        masterCanvas.addEventListener("mouseleave", () => {
          this.hoverCell = { r: -1, c: -1 };
          this.renderMasterBoard();
        });

        masterCanvas.addEventListener("contextmenu", (e) => {
          e.preventDefault();
          if (this.phase === "SETUP") {
            this.isHorizontal = !this.isHorizontal;
            this.updateStatusHeader();
            this.renderMasterBoard();
          }
        });

        masterCanvas.addEventListener("mousedown", (e) => {
          if (e.button === 2) return;
          const { r, c } = this.hoverCell;
          if (r < 0 || r >= 10 || c < 0 || c >= 10) return;

          if (this.phase === "SETUP" && this.shipsToPlace.length > 0) {
            const len = this.shipsToPlace[0];
            if (this.game.placeShipManual(r, c, len, this.isHorizontal, this.currentShipId)) {
              this.shipsToPlace.shift();
              this.currentShipId++;
              if (this.shipsToPlace.length === 0) {
                this.phase = "PLAY";
                this.recomputeBots();
              }
              this.updateStatusHeader();
              this.renderAll();
            }
          } else if (this.phase === "PLAY" && !this.game.isGameOver()) {
            const res = this.game.fireShot(r, c);
            if (res !== "ALREADY_FIRED" && res !== "INVALID") {
              this.recomputeBots();
              this.updateStatusHeader(`Manual shot (${String.fromCharCode(65 + r)}${c + 1}): ${res}`);
              this.renderAll();
            }
          }
        });
      }

      // Bind toolbar buttons
      const bindClick = (id, fn) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("click", fn);
      };

      bindClick("btn-preset-1", () => this.loadPreset(1));
      bindClick("btn-preset-2", () => this.loadPreset(2));
      bindClick("btn-preset-3", () => this.loadPreset(3));
      bindClick("btn-preset-4", () => this.loadPreset(4));
      bindClick("btn-preset-5", () => this.loadPreset(5));
      bindClick("btn-fire-trap", () => this.fireTrapShots());
      bindClick("btn-randomize", () => this.startRandomGame());
      bindClick("btn-setup", () => this.startManualSetup());
      bindClick("btn-undo", () => this.undoShot());
      bindClick("btn-toggle-ships", () => {
        this.showShips = !this.showShips;
        const b = document.getElementById("btn-toggle-ships");
        if (b) b.classList.toggle("active", this.showShips);
        this.renderMasterBoard();
      });
      bindClick("btn-autoplay", () => this.toggleAutoPlay());
      bindClick("btn-theme", () => this.cycleTheme());
      bindClick("btn-view-quad", () => this.setViewMode("quad"));
      bindClick("btn-view-dual", () => this.setViewMode("dual"));

      const autoSelect = document.getElementById("select-autobot");
      if (autoSelect) {
        autoSelect.addEventListener("change", (e) => {
          this.autoBotKey = e.target.value;
        });
      }

      const slot1Select = document.getElementById("select-slot-1");
      const slot2Select = document.getElementById("select-slot-2");
      const updateSlots = () => {
        if (slot1Select && slot2Select) {
          let s1 = slot1Select.value;
          let s2 = slot2Select.value;
          if (s1 === s2) {
            const all = ["watson", "sherlock", "mycroft", "moriarty"];
            s2 = all.find((k) => k !== s1) || "mycroft";
            slot2Select.value = s2;
          }
          this.dualSlots = [s1, s2];
          this.updateVisibleCards();
          this.renderAll();
        }
      };
      if (slot1Select) slot1Select.addEventListener("change", updateSlots);
      if (slot2Select) slot2Select.addEventListener("change", updateSlots);

      // Bind individual bot shoot buttons
      for (const botKey of ["watson", "sherlock", "mycroft", "moriarty"]) {
        bindClick(`btn-shoot-${botKey}`, () => this.fireBotShot(botKey));
      }

      // Global Keyboard Controls matching the original showcase
      window.addEventListener("keydown", (e) => {
        if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA")) {
          return;
        }
        const key = e.key.toLowerCase();
        if (key === "1") this.loadPreset(1);
        else if (key === "2") this.loadPreset(2);
        else if (key === "3") this.loadPreset(3);
        else if (key === "4") this.loadPreset(4);
        else if (key === "5") this.loadPreset(5);
        else if (key === "f") this.fireTrapShots();
        else if (key === "r") this.startRandomGame();
        else if (key === "n") this.startManualSetup();
        else if (key === "u") this.undoShot();
        else if (key === "h") {
          this.showShips = !this.showShips;
          const b = document.getElementById("btn-toggle-ships");
          if (b) b.classList.toggle("active", this.showShips);
          this.renderMasterBoard();
        } else if (key === "t" && this.phase === "SETUP") {
          this.isHorizontal = !this.isHorizontal;
          this.updateStatusHeader();
          this.renderMasterBoard();
        } else if (key === "w") this.fireBotShot("watson");
        else if (key === "s") this.fireBotShot("sherlock");
        else if (key === "m") this.fireBotShot("mycroft");
        else if (key === "o") this.fireBotShot("moriarty");
        else if (key === "a") this.toggleAutoPlay();
        else if (key === "c") this.cycleTheme();
      });
    }

    renderAll() {
      this.renderMasterBoard();
      const visibleBots = this.viewMode === "quad" ? ["watson", "sherlock", "mycroft", "moriarty"] : this.dualSlots;
      for (const botKey of visibleBots) {
        this.renderBotHeatmap(botKey);
      }
    }

    renderMasterBoard() {
      const canvas = document.getElementById("master-board-canvas");
      if (!canvas) return;
      const ctx = setupHiDPICanvas(canvas, 420, 420);
      const theme = THEMES[this.themeKey];
      const margin = 32;
      const cs = 38;

      ctx.fillStyle = theme.panelBg;
      ctx.fillRect(0, 0, 420, 420);

      // Labels A..J and 1..10
      ctx.fillStyle = theme.label;
      ctx.font = "600 12px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let i = 0; i < 10; i++) {
        ctx.fillText(String(i + 1), margin + i * cs + cs / 2, margin - 14);
        ctx.fillText(String.fromCharCode(65 + i), margin - 15, margin + i * cs + cs / 2);
      }

      // Water grid cells
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const x = margin + c * cs;
          const y = margin + r * cs;
          ctx.fillStyle = theme.water;
          ctx.fillRect(x, y, cs, cs);
          ctx.strokeStyle = theme.gridLine;
          ctx.lineWidth = 1;
          ctx.strokeRect(x, y, cs, cs);
        }
      }

      // Draw ships if in SETUP or showShips is enabled
      if (this.phase === "SETUP" || this.showShips) {
        for (const sid of Object.keys(this.game.shipPegs)) {
          const pegs = this.game.shipPegs[sid];
          if (pegs && pegs.length > 0) {
            let minR = 99, maxR = -1, minC = 99, maxC = -1;
            for (const pt of pegs) {
              if (pt.r < minR) minR = pt.r;
              if (pt.r > maxR) maxR = pt.r;
              if (pt.c < minC) minC = pt.c;
              if (pt.c > maxC) maxC = pt.c;
            }
            const sx = margin + minC * cs + 4;
            const sy = margin + minR * cs + 4;
            const sw = (maxC - minC + 1) * cs - 8;
            const sh = (maxR - minR + 1) * cs - 8;
            ctx.fillStyle = theme.ship;
            drawRoundedRect(ctx, sx, sy, sw, sh, 8);
            ctx.fill();
          }
        }
      }

      // Setup Ghost Ship Preview
      if (this.phase === "SETUP" && this.shipsToPlace.length > 0 && this.hoverCell.r >= 0) {
        const len = this.shipsToPlace[0];
        const { r, c } = this.hoverCell;
        const valid = this.game.canPlace(r, c, len, this.isHorizontal);
        const sw = (this.isHorizontal ? len : 1) * cs - 8;
        const sh = (this.isHorizontal ? 1 : len) * cs - 8;
        ctx.fillStyle = valid ? theme.ghostValid : theme.ghostInval;
        drawRoundedRect(ctx, margin + c * cs + 4, margin + r * cs + 4, sw, sh, 8);
        ctx.fill();
      }

      // Hover highlight in PLAY mode
      if (this.phase === "PLAY" && !this.game.isGameOver() && this.hoverCell.r >= 0) {
        const { r, c } = this.hoverCell;
        if (this.game.trackingBoard[r][c] === Engine.CELL_UNKNOWN) {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
          ctx.lineWidth = 2;
          ctx.strokeRect(margin + c * cs + 2, margin + r * cs + 2, cs - 4, cs - 4);
        }
      }

      // Draw CSP Dead Ship Locked Outlines when showShips is on
      if (this.showShips && this.diagnostics && this.diagnostics.deadShips) {
        ctx.strokeStyle = theme.diagClaimed;
        ctx.lineWidth = 2.5;
        for (const shipPegs of this.diagnostics.deadShips) {
          if (shipPegs && shipPegs.length > 0) {
            let minR = 99, maxR = -1, minC = 99, maxC = -1;
            for (const pt of shipPegs) {
              if (pt.r < minR) minR = pt.r;
              if (pt.r > maxR) maxR = pt.r;
              if (pt.c < minC) minC = pt.c;
              if (pt.c > maxC) maxC = pt.c;
            }
            const sx = margin + minC * cs + 3;
            const sy = margin + minR * cs + 3;
            const sw = (maxC - minC + 1) * cs - 6;
            const sh = (maxR - minR + 1) * cs - 6;
            drawRoundedRect(ctx, sx, sy, sw, sh, 7);
            ctx.stroke();
          }
        }
      }

      // Draw Pegs (Misses, Hits, Sunk) and Active CSP Hit Rings
      const activeList = (this.showShips && this.diagnostics && this.diagnostics.active) ? this.diagnostics.active : [];
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const x = margin + c * cs;
          const y = margin + r * cs;
          const cx = x + cs / 2;
          const cy = y + cs / 2;
          const val = this.game.trackingBoard[r][c];

          if (val === Engine.CELL_MISS) {
            ctx.fillStyle = theme.pegMiss;
            ctx.beginPath();
            ctx.arc(cx, cy, cs / 6, 0, Math.PI * 2);
            ctx.fill();
          } else if (val === Engine.CELL_HIT) {
            ctx.strokeStyle = theme.pegHit;
            ctx.lineWidth = 3;
            const pad = 9;
            ctx.beginPath();
            ctx.moveTo(x + pad, y + pad);
            ctx.lineTo(x + cs - pad, y + cs - pad);
            ctx.moveTo(x + cs - pad, y + pad);
            ctx.lineTo(x + pad, y + cs - pad);
            ctx.stroke();
          } else if (val === Engine.CELL_SUNK) {
            ctx.strokeStyle = theme.pegSunk;
            ctx.lineWidth = 4;
            const pad = 8;
            ctx.beginPath();
            ctx.moveTo(x + pad, y + pad);
            ctx.lineTo(x + cs - pad, y + cs - pad);
            ctx.moveTo(x + cs - pad, y + pad);
            ctx.lineTo(x + pad, y + cs - pad);
            ctx.stroke();
          }

          if (activeList.some((pt) => pt.r === r && pt.c === c)) {
            ctx.strokeStyle = theme.diagActive;
            ctx.lineWidth = 2;
            drawRoundedRect(ctx, x + 3, y + 3, cs - 6, cs - 6, 4);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(cx, cy, cs * 0.26, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      }
    }

    renderBotHeatmap(botKey) {
      const canvas = document.getElementById(`heatmap-canvas-${botKey}`);
      if (!canvas) return;
      const size = this.viewMode === "dual" ? 360 : 270;
      const ctx = setupHiDPICanvas(canvas, size, size);
      const theme = THEMES[this.themeKey];
      const margin = 24;
      const cs = (size - margin - 8) / 10;

      ctx.fillStyle = theme.panelBg;
      ctx.fillRect(0, 0, size, size);

      // Axis labels
      ctx.fillStyle = theme.label;
      ctx.font = "600 10px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let i = 0; i < 10; i++) {
        ctx.fillText(String(i + 1), margin + i * cs + cs / 2, margin - 10);
        ctx.fillText(String.fromCharCode(65 + i), margin - 11, margin + i * cs + cs / 2);
      }

      const out = this.botOutputs[botKey];
      const heat = (out && out.heatmap) ? out.heatmap : Engine.createBlankMatrix(0);
      let maxHeat = 0;
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (heat[r][c] > maxHeat) maxHeat = heat[r][c];
        }
      }

      const fontSize = this.viewMode === "dual" ? 11 : 9.5;
      ctx.font = `700 ${fontSize}px Inter, system-ui, sans-serif`;

      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const x = margin + c * cs;
          const y = margin + r * cs;
          const val = heat[r][c] || 0;
          const col = sampleHeatColor(val, maxHeat, theme.heatStops);

          ctx.fillStyle = col.css;
          ctx.fillRect(x, y, cs, cs);

          if (val > 0) {
            const lum = (0.299 * col.r + 0.587 * col.g + 0.114 * col.b) / 255;
            ctx.fillStyle = lum > 0.66 ? "#121318" : "#FFFFFF";
            ctx.fillText(String(val), x + cs / 2, y + cs / 2 + 0.5);
          } else if (this.game.trackingBoard[r][c] !== Engine.CELL_UNKNOWN) {
            // Subtle marker for already-fired cells on the heatmap
            const bVal = this.game.trackingBoard[r][c];
            ctx.fillStyle = bVal === Engine.CELL_MISS ? "rgba(255,255,255,0.2)" : "rgba(255,80,80,0.45)";
            ctx.beginPath();
            ctx.arc(x + cs / 2, y + cs / 2, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }

          ctx.strokeStyle = theme.gridLine;
          ctx.lineWidth = 0.75;
          ctx.strokeRect(x, y, cs, cs);
        }
      }

      // Draw Target Reticle on Bot's Recommended Cell
      if (out && out.target && !this.game.isGameOver() && this.phase === "PLAY") {
        const tx = margin + out.target.y * cs;
        const ty = margin + out.target.x * cs;
        ctx.strokeStyle = theme.targetRet;
        ctx.lineWidth = 2.5;
        ctx.strokeRect(tx + 1.5, ty + 1.5, cs - 3, cs - 3);
      }

      // Update telemetry footer on card
      const metaEl = document.getElementById(`bot-telemetry-${botKey}`);
      if (metaEl) {
        if (out && out.target && !this.game.isGameOver()) {
          const coordStr = `${String.fromCharCode(65 + out.target.x)}${out.target.y + 1}`;
          metaEl.textContent = `Target: ${coordStr} • ${out.log} (${out.ms.toFixed(1)} ms)`;
        } else {
          metaEl.textContent = this.game.isGameOver() ? "Fleet Eliminated" : "Awaiting Setup...";
        }
      }
    }
  }

  root.ArenaController = ArenaController;
})(typeof window !== "undefined" ? window : globalThis);

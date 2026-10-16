# 🚢 Battleship AI & Bots Laboratory (LuaJIT 2.1 + Love2D)

A high-performance, zero-allocation algorithmic research and benchmarking suite for **Battleship** ($10 \times 10$ grid, 5-ship fleet, Continuous Fire, strict Fog of War), architected in **LuaJIT 2.1 FFI** and **Love2D**, with **PyTorch** reserved for neural network training in `ml/`.

---

## 🏆 Canonical Sherlock Hierarchy (Paired-Board Tournament)

All bots are evaluated on identical master boards under strict Fog of War (blind sinks) and continuous fire:

| Rank | Bot | File | Canonical Strategy | Avg Shots | Outliers ($>60$) |
| :---: | :--- | :--- | :--- | :---: | :---: |
| 🏆 **Rank 4** | **`MycroftBot`** | [`lua/bots/mycroft.lua`](lua/bots/mycroft.lua) | Exact Shannon Binary Entropy $H(p)$ + Radius-2 Ocean Openness Utility + Joint Covers ($P_1, P_2$) | **45.19** | ~2.15% |
| **Rank 3** | **`SherlockBot`** | [`lua/bots/sherlock.lua`](lua/bots/sherlock.lua) | Pure Exact Bayesian Joint Covers ($P_1, P_2$) + 15.05B Empty-Board Prior + Dynamic Stride Parity | **45.38** | ~1.85% |
| **Rank 2** | **`WatsonBot`** | [`lua/bots/watson.lua`](lua/bots/watson.lua) | Multi-Hypothesis Chronological CSP Deduction + Dynamic Stride Parity ($2 \to 3$) + Exact Target | **45.41** | ~2.05% |
| **Rank 1** | **`LestradeBot`** | [`lua/bots/lestrade.lua`](lua/bots/lestrade.lua) | Pure Single-Cell Marginal Overlap Density (`1 + overlap_hits`), No Parity, No CSP | **56.07** | ~29.5% |
| **Tier 2** | **`AdlerBot`** | [`lua/bots/adler.lua`](lua/bots/adler.lua) | Empirical Bayesian MAP Human Profiler ($\text{Likelihood} \times \text{Prior}^\gamma$) + Anti-Human Defense | **32.37** *(vs biased)* / **45.42** *(uniform)* | ~2.15% |
| **Tier 2** | **`MoriartyBot`** | [`lua/bots/moriarty.lua`](lua/bots/moriarty.lua) | Pure Essential 4-Channel Spatial ConvNet (Policy + Occupancy Heads) + CSP Assassin Mode | **48.61** | ~5.0% |
| **Lab** | **`MCMCProbabilityBot`** | [`lua/lab/mcmc.lua`](lua/lab/mcmc.lua) | 100,000-Step Metropolis-Hastings Markov Chain Monte Carlo Bitboard Sampler (**10.8M steps/s**) | **45.48** | ~2.40% |

---

## 📂 Repository Structure

```text
battleship-bots/
├── lua/
│   ├── bitboard.lua             # 100-bit C struct bitboard via LuaJIT FFI (0.43 ns/op)
│   ├── game_engine.lua          # BattleshipGame physics, continuous fire, Fog of War, undo ledger
│   ├── deduction.lua            # Multi-hypothesis chronological CSP corpse-locking solver
│   ├── combinatorics.lua        # 760 precomputed ship bitboards & disjoint 2-ship joint covers (P1, P2)
│   ├── benchmark.lua            # Multi-core POSIX fork() paired-board tournament runner
│   ├── bots/
│   │   ├── base_bot.lua         # Abstract BaseBot contract, OOP inheritance, dynamic parity nets
│   │   ├── lestrade.lua         # Rank 1: LestradeBot
│   │   ├── watson.lua           # Rank 2: WatsonBot
│   │   ├── sherlock.lua         # Rank 3: SherlockBot
│   │   ├── mycroft.lua          # Rank 4: MycroftBot (Project Champion)
│   │   ├── adler.lua            # Tier 2: AdlerBot (Empirical Bayesian MAP Human Profiler)
│   │   └── moriarty.lua         # Tier 2: MoriartyBot (Pure Essential Neural Network Bot)
│   ├── lab/
│   │   └── mcmc.lua             # Metropolis-Hastings MCMC Joint Probability Sampler
│   └── test/                    # Automated Unit & Benchmark Test Suite (LuaJIT)
├── love/
│   ├── conf.lua                 # Love2D window & high-DPI display configuration
│   └── main.lua                 # 60fps Interactive Live Play & Forensic Replay Visualizer
└── ml/                          # PyTorch Neural Network & PPO RL Training Suite
```

---

## 🚀 Quickstart

### 1. Run Multi-Core Paired-Board Tournaments (`LuaJIT`)
```bash
luajit lua/benchmark.lua --games 2000 --bots lestrade watson sherlock mycroft adler moriarty
```

### 2. Run Interactive Love2D Visualizer (`LIVE` & `REPLAY` Modes)
```bash
# Live play & real-time probability heatmap diagnostics
love love --bot mycroft --mode live

# Forensic outlier timeline replay scrubber
love love --bot sherlock --mode replay
```
* **Hotkeys in Visualizer:**
  * `[1]..[7]` or `[TAB]`: Hot-swap active bot (`Mycroft`, `Sherlock`, `Watson`, `Lestrade`, `Adler`, `Moriarty`, `MCMC`) on the current board state
  * `[M]`: Toggle between `LIVE` Play and `REPLAY` Forensic Scrubber
  * `[SPACE]`: Step single AI shot | `[A]`: Toggle continuous auto-fire | `[U / LEFT]`: Undo shot | `[H]`: Show/hide ships | `[F]`: Instant randomized rematch

### 3. Run Automated Unit Test Suite
```bash
for t in lua/test/test_*.lua; do
  luajit -e 'package.path = "./?.lua;./lua/?.lua;" .. package.path' "$t"
done
```

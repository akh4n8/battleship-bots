# 🚢 Battleship AI & Bots Laboratory

A high-performance algorithmic research and benchmarking suite for the game of **Battleship** ($10 \times 10$ grid, 5-ship fleet, Continuous Fire, strict Fog of War).

This repository contains the Python algorithmic engines, Monte Carlo simulation harnesses, neural networks, and interactive visualizers used to research and push the theoretical mathematical limits of blind spatial search.

---

## 🏆 Project Leaderboard (100,000 Paired-Board Tournament)

All bots tested under identical master boards, strict Fog of War (blind sinks), and continuous fire:

| Rank | Bot | Architecture | Avg Shots | 95% CI | Outliers ($>60$) | Speed |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **`PorlockBot`** | Exact Shannon Information Entropy + Ocean Volume Slicing | **45.131** | $\pm 0.027$ | 2.62% | ~1,000 games/s |
| 🥈 | **`Sherrinford-Apex`** | Exact Multi-Hypothesis Joint Cover + Dynamic Stride Parity | **45.209** | $\pm 0.027$ | 2.65% | ~1,000 games/s |
| 🥉 | **`SherlockBot`** | Multi-Hypothesis CSP Deduction + Exact Target + Dynamic Parity | **45.330** | $\pm 0.038$ | 2.45% | ~1,500 games/s |
| 4 | **`Sherrinford-Parity`** | Pure Exact Bayes + Dynamic Stride Parity | **45.502** | $\pm 0.089$ | 2.58% | ~1,000 games/s |
| 5 | **`Sherrinford-Deduction`** | Multi-Hypothesis Tied CSP + Exact Joint Target | **45.665** | $\pm 0.087$ | **2.36%** | ~1,000 games/s |
| 6 | **`MycroftBot`** | 100,000-Step Monte Carlo MCMC Joint Probability | **45.790** | $\pm 0.121$ | 4.78% | ~1 game/s |
| 7 | **Classic `SherlockBot`** | Chronological CSP + Heuristic Linear Assassin | **45.885** | $\pm 0.039$ | 3.62% | ~1,500 games/s |

---

## 🧠 The Arthur Conan Doyle AI Hierarchy

### 1. `PorlockBot` (The All-Time Champion)
* **The Math:** Exact Information-Theoretic Bayesian Expected Utility Engine.
* **The Strategy:** Evaluates candidate hunt cells using Claude Shannon's binary information entropy $H(p) = -p \log_2 p - (1-p) \log_2(1-p)$ and an 8-cell radius-2 ocean openness star pattern to cleanly bisect open water.
$$\text{Utility}(r, c) = 2.5 \cdot p(r, c) + 0.35 \cdot H(p) + 0.15 \cdot \text{Openness}(r, c)$$
* Solves active wounded hits using exact single-ship and joint 2-ship disjoint bitboard covers ($P_1, P_2$). Dynamically shifts from Stride-2 checkerboard to Stride-3 diagonal $(r+2c) \pmod 3$ once the 2-ship is confirmed dead.

### 2. `Sherrinford-Apex`
* **The Math:** Pure Exact Multi-Hypothesis Joint Cover + Dynamic Stride Parity.
* **The Strategy:** Fuses multi-hypothesis tied CSP deduction-locked corpses with pure exact bitboard density. Zero heuristics. Resolves touching and overlapping ship clusters organically via exact 2-ship joint probability fields.

### 3. `SherlockBot`
* **The Math:** Chronological Constraint Satisfaction Solver (CSP) + Exact Combinatorial Target Density + Dynamic Stride Parity.
* **The Strategy:** Solves historical board states as a chronological CSP, isolating dead corpses and active hits across all tied minimum-leftover hypotheses. Evaluates exact target density across active hits and hunts via gravitational parity sweeps with sub-millisecond execution speed.

### 4. `MycroftBot`
* **The Math:** Metropolis-Hastings Markov Chain Monte Carlo (MCMC).
* **The Strategy:** Simulates up to 100,000 valid 5-ship board configurations using 128-bit bitboards, sampling joint non-overlapping spatial probability.

---

## 🚀 Quickstart

### Prerequisites
* Python 3.10+
* Pygame, PyTorch (optional, for NeuralBot / Moriarty models)

### 1. Interactive Pygame Visualizer & Replay Viewer
Play against the bots in real-time or watch diagnostic heatmaps update live:
```bash
# Play against the champion PorlockBot with live heatmap diagnostics
python main.py --bot porlock

# Play against SherlockBot
python main.py --bot sherlock

# Play against Sherrinford-Apex
python main.py --bot apex
```

### 2. Run Single-Bot Benchmarks
Run multi-core Monte Carlo simulations:
```bash
# Benchmark PorlockBot across 1,000 games
python benchmark.py --bot porlock --games 1000

# Benchmark Sherlock across 5,000 games
python benchmark.py --bot sherlock --games 5000
```

### 3. Run Paired-Board Tournaments
Run head-to-head titan showdowns where bots play the exact same randomized boards:
```bash
python benchmarks/tournament_sherrinford_variants.py --games 10000 --bots porlock apex sherlock
```

---

## ⚖️ Game Rules

* **Grid:** $10 \times 10$ (100 cells).
* **Fleet:** 5 ships (Carrier: 5, Battleship: 4, Cruiser: 3, Submarine: 3, Destroyer: 2). Total pegs = 17.
* **Turn Mechanics:** Continuous Fire (a player keeps shooting as long as they get a Hit or Sunk; turn ends on Miss).
* **Fog of War (Blind Sink):** Ship identity/length is never revealed upon sinking. Previous hits on sunken ships remain marked as generic hits; only the final lethal shot is marked "Sunk".

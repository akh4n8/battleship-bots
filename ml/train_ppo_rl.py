import os
import sys
import time
import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.train_sherlock_nn import ResConvBlock
from ml.rl_battleship_env import DecomposedBattleshipEnv

class SherlockActorCritic(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # Stem and Residual Backbone matching SherlockNeuralModel
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.res_layers = nn.Sequential(
            ResConvBlock(64),
            ResConvBlock(64),
            ResConvBlock(64),
            ResConvBlock(64)
        )
        # Head 1: Occupancy Head (matches SherlockNeuralModel for seamless heatmap generation)
        self.occupancy_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        # Head 2: Policy Head (Actor: 100-way categorical distribution)
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 100)
        )
        # Head 3: Value Head (Critic: Expected remaining return)
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 10 * 10, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.stem(x)
        features = self.res_layers(features)
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_logits, values

    def get_action_and_value(self, x, action_mask, action=None):
        policy_logits, value = self.forward(x)
        # Action masking: set already-fired tiles to -1e9
        masked_logits = policy_logits + (1.0 - action_mask) * -1e9
        dist = torch.distributions.Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

def safe_save(model, path):
    try:
        torch.save(model.state_dict(), path)
        return path
    except Exception as e:
        tmp_path = os.path.join('/tmp', os.path.basename(path))
        print(f"⚠️ Could not write to {path} ({e}). Saved to {tmp_path}")
        torch.save(model.state_dict(), tmp_path)
        return tmp_path

def safe_onnx_export(model, dummy_input, path, opset_version=18):
    try:
        torch.onnx.export(
            model,
            dummy_input,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['board_input'],
            output_names=['policy_logits', 'value']
        )
        return path
    except Exception as e:
        tmp_path = os.path.join('/tmp', os.path.basename(path))
        print(f"⚠️ Could not export ONNX to {path} ({e}). Exporting to {tmp_path}")
        torch.onnx.export(
            model,
            dummy_input,
            tmp_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['board_input'],
            output_names=['policy_logits', 'value']
        )
        return tmp_path

def train_ppo():
    parser = argparse.ArgumentParser(description="PPO Reinforcement Learning for Battleship Neural Bot")
    parser.add_argument("--total-games", type=int, default=25000, help="Total games to simulate (25,000)")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel environments on CPU/GPU")
    parser.add_argument("--rollout-steps", type=int, default=16, help="Steps per environment per rollout")
    parser.add_argument("--base-weights", type=str, default=os.path.join(DATA_DIR, "sherlock_nn_best.pth"))
    parser.add_argument("--output-dir", type=str, default=DATA_DIR, help="Directory to save weights and checkpoints")
    parser.add_argument("--lr", type=float, default=1e-4, help="PPO learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO surrogate clipping epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.005, help="Entropy bonus coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--update-epochs", type=int, default=2, help="PPO update epochs per rollout")
    parser.add_argument("--minibatch-size", type=int, default=128, help="PPO mini-batch size")
    parser.add_argument("--eval-interval-games", type=int, default=1000, help="Games between evaluation telemetry")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("⚔️ PPO REINFORCEMENT LEARNING ENGINE: BATTLESHIP NEURAL BOT")
    print(f"   Target Games       : {args.total_games:,}")
    print(f"   Hardware Device    : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Parallel Envs      : {args.num_envs}")
    print(f"   Rollout Buffer     : {args.num_envs * args.rollout_steps:,} transitions per iteration")
    print(f"   Learning Rate      : {args.lr}")
    print(f"   Base Warm-Start    : {args.base_weights}")
    print("=" * 70)

    # 1. Initialize Decomposed Batched Environment
    env = DecomposedBattleshipEnv(num_envs=args.num_envs, device=str(device))

    # 2. Initialize Actor-Critic Network
    model = SherlockActorCritic(in_channels=4).to(device)

    # Warm-start from 1,000,000 Sherlock foundation model
    if os.path.exists(args.base_weights):
        base_state = torch.load(args.base_weights, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(base_state, strict=False)
        print(f"✅ Successfully warm-started Actor policy from {args.base_weights}")
        print(f"   (Loaded stem, res_layers, occupancy_head, and policy_head; initialized Critic value_head)")
    else:
        print("⚠️ Base weights not found! Training Actor-Critic from scratch.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Rollout buffer allocations [T, N, ...]
    T = args.rollout_steps
    N = args.num_envs
    obs_buffer = torch.zeros((T, N, 4, 10, 10), dtype=torch.float32, device=device)
    mask_buffer = torch.zeros((T, N, 100), dtype=torch.float32, device=device)
    actions_buffer = torch.zeros((T, N), dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros((T, N), dtype=torch.float32, device=device)
    rewards_buffer = torch.zeros((T, N), dtype=torch.float32, device=device)
    dones_buffer = torch.zeros((T, N), dtype=torch.bool, device=device)
    values_buffer = torch.zeros((T, N), dtype=torch.float32, device=device)

    total_games_completed = 0
    recent_game_shots = deque(maxlen=5000)
    recent_game_baselines = deque(maxlen=5000)
    recent_game_regrets = deque(maxlen=5000)
    best_avg_shots = float('inf')
    best_regret = float('-inf')

    output_weights_path = os.path.join(args.output_dir, "moriarty_rl_best.pth")
    checkpoint_path = os.path.join(args.output_dir, "moriarty_rl_checkpoint.pth")

    start_time = time.time()
    next_eval_milestone = args.eval_interval_games
    iteration = 0

    while total_games_completed < args.total_games:
        iteration += 1
        model.eval()

        # Phase 1: Collect Rollout Transitions
        for t in range(T):
            obs_buffer[t] = env.states.clone()
            mask_buffer[t] = env.action_masks.clone()

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(env.states, env.action_masks)

            actions_buffer[t] = action
            logprobs_buffer[t] = logprob
            values_buffer[t] = value

            # Step environment
            next_obs, rewards, dones, completed = env.step(action)

            rewards_buffer[t] = rewards
            dones_buffer[t] = dones

            for g_info in completed:
                total_games_completed += 1
                recent_game_shots.append(g_info['total_shots'])
                recent_game_baselines.append(g_info['baseline_shots'])
                recent_game_regrets.append(g_info['regret'])

        # Compute GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            _, next_value = model.forward(env.states)

        advantages = torch.zeros_like(rewards_buffer)
        lastgaelam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = (~dones_buffer[t]).float()
                nextvalues = next_value
            else:
                nextnonterminal = (~dones_buffer[t]).float()
                nextvalues = values_buffer[t + 1]

            delta = rewards_buffer[t] + args.gamma * nextvalues * nextnonterminal - values_buffer[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values_buffer

        # Flatten batch for PPO updates
        b_obs = obs_buffer.view(-1, 4, 10, 10)
        b_masks = mask_buffer.view(-1, 100)
        b_actions = actions_buffer.view(-1)
        b_logprobs = logprobs_buffer.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = values_buffer.view(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Phase 2: PPO Gradient Updates
        model.train()
        batch_size = T * N
        inds = np.arange(batch_size)

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_inds], b_masks[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Clipped surrogate objective
                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped Value Loss
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_eps, args.clip_eps
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss + (args.value_coef * v_loss) - (args.entropy_coef * entropy_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # Telemetry & Checkpointing
        if total_games_completed >= next_eval_milestone:
            next_eval_milestone += args.eval_interval_games
            elapsed = time.time() - start_time
            games_per_sec = total_games_completed / elapsed if elapsed > 0 else 0

            rolling_avg = np.mean(recent_game_shots) if recent_game_shots else 0.0
            rolling_baseline = np.mean(recent_game_baselines) if recent_game_baselines else 0.0
            rolling_regret = np.mean(recent_game_regrets) if recent_game_regrets else 0.0
            beat_baseline_pct = (np.mean([r > 0 for r in recent_game_regrets]) * 100) if recent_game_regrets else 0.0

            print(f"[{total_games_completed:6,d} / {args.total_games:,} Games] "
                  f"({elapsed:.1f}s | {games_per_sec:,.0f} g/s) | "
                  f"Agent: {rolling_avg:.2f} | Base: {rolling_baseline:.2f} | "
                  f"Regret: {rolling_regret:+.2f} ({beat_baseline_pct:.1f}% beat) | "
                  f"PG: {pg_loss.item():.4f} | V: {v_loss.item():.4f}")

            # Save checkpoint
            checkpoint_path = safe_save(model, checkpoint_path)

            if rolling_avg < best_avg_shots and len(recent_game_shots) >= 1000:
                best_avg_shots = rolling_avg
                best_regret = rolling_regret
                output_weights_path = safe_save(model, output_weights_path)
                print(f"  -> ⭐ Saved New Best RL Model to {output_weights_path} "
                      f"(Avg: {rolling_avg:.2f} shots | Regret: {rolling_regret:+.2f})")

    print("\n" + "=" * 70)
    print(f"🏆 PPO REINFORCEMENT LEARNING COMPLETE! ({total_games_completed:,} Games)")
    print(f"   Best Rolling Avg Shots : {best_avg_shots:.2f}")
    print(f"   Best Rolling Regret    : {best_regret:+.2f}")
    print(f"   Final Saved Weights    : {output_weights_path}")
    print("=" * 70)

    # Save final best if not yet saved
    if not os.path.exists(output_weights_path):
        output_weights_path = safe_save(model, output_weights_path)

    # Export ONNX model
    onnx_path = os.path.join(args.output_dir, "moriarty_rl.onnx")
    print(f"Exporting RL ONNX model to {onnx_path}...")
    try:
        model.eval()
        dummy_input = torch.randn(1, 4, 10, 10, device=device)
        onnx_path = safe_onnx_export(model, dummy_input, onnx_path, opset_version=18)
        print(f"✅ Successfully exported RL ONNX model to {onnx_path}")
    except Exception as e:
        print(f"Warning during ONNX export: {e}")

if __name__ == "__main__":
    train_ppo()

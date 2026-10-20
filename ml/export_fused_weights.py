#!/usr/bin/env python3
"""
ml/export_fused_weights.py
Loads the 1M-game trained SherlockNeuralModel (data/sherlock_nn_best.pth),
folds all BatchNorm2d layers directly into their preceding Conv2d weights & biases,
verifies exact numerical parity (< 1e-5 max abs diff), and exports:
  1. data/moriarty_fused.bin       (Float32 binary for LuaJIT FFI MoriartyBot)
  2. showcase/assets/moriarty_fp16.bin (Float16 binary for fast web showcase loading)
"""

import os
import sys
import struct
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.train_sherlock_nn import SherlockNeuralModel


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fold BatchNorm2d parameters into Conv2d weight and bias:
        W_fused = W * (gamma / sqrt(running_var + eps))
        b_fused = (b - running_mean) * (gamma / sqrt(running_var + eps)) + beta
    Returns (W_fused_np [out_c, in_c, kH, kW], b_fused_np [out_c]) as float32 numpy arrays.
    """
    with torch.no_grad():
        w = conv.weight.detach().clone().cpu().float()
        if conv.bias is not None:
            b = conv.bias.detach().clone().cpu().float()
        else:
            b = torch.zeros(w.size(0), dtype=torch.float32)

        mean = bn.running_mean.detach().clone().cpu().float()
        var = bn.running_var.detach().clone().cpu().float()
        gamma = bn.weight.detach().clone().cpu().float()
        beta = bn.bias.detach().clone().cpu().float()
        eps = bn.eps

        scale = gamma / torch.sqrt(var + eps)
        w_fused = w * scale.view(-1, 1, 1, 1)
        b_fused = (b - mean) * scale + beta

        return w_fused.numpy().astype(np.float32), b_fused.numpy().astype(np.float32)


def extract_all_fused_layers(model: SherlockNeuralModel):
    """
    Extracts all layers in strict deterministic order:
      1. stem: Conv2d(4->64, 3x3) + BN
      2. res0..res3: each has conv1(64->64, 3x3)+BN and conv2(64->64, 3x3)+BN
      3. occ_conv1: Conv2d(64->32, 3x3) + BN
      4. occ_conv2: Conv2d(32->1, 1x1) (no BN)
      5. pol_conv1: Conv2d(64->32, 3x3) + BN
      6. pol_fc1: Linear(3200->256)
      7. pol_fc2: Linear(256->100)
    """
    layers = []

    # 1. Stem (4 -> 64, 3x3)
    w_stem, b_stem = fuse_conv_bn(model.stem[0], model.stem[1])
    layers.append(("stem_conv", w_stem, b_stem))

    # 2. Residual blocks 0..3
    for idx, block in enumerate(model.res_layers):
        w1, b1 = fuse_conv_bn(block.conv1, block.bn1)
        w2, b2 = fuse_conv_bn(block.conv2, block.bn2)
        layers.append((f"res{idx}_conv1", w1, b1))
        layers.append((f"res{idx}_conv2", w2, b2))

    # 3. Occupancy head
    w_occ1, b_occ1 = fuse_conv_bn(model.occupancy_head[0], model.occupancy_head[1])
    layers.append(("occ_conv1", w_occ1, b_occ1))

    occ_conv2 = model.occupancy_head[3]
    w_occ2 = occ_conv2.weight.detach().cpu().float().numpy().astype(np.float32)
    b_occ2 = occ_conv2.bias.detach().cpu().float().numpy().astype(np.float32)
    layers.append(("occ_conv2", w_occ2, b_occ2))

    # 4. Policy head
    w_pol1, b_pol1 = fuse_conv_bn(model.policy_head[0], model.policy_head[1])
    layers.append(("pol_conv1", w_pol1, b_pol1))

    pol_fc1 = model.policy_head[4]
    w_fc1 = pol_fc1.weight.detach().cpu().float().numpy().astype(np.float32)
    b_fc1 = pol_fc1.bias.detach().cpu().float().numpy().astype(np.float32)
    layers.append(("pol_fc1", w_fc1, b_fc1))

    pol_fc2 = model.policy_head[6]
    w_fc2 = pol_fc2.weight.detach().cpu().float().numpy().astype(np.float32)
    b_fc2 = pol_fc2.bias.detach().cpu().float().numpy().astype(np.float32)
    layers.append(("pol_fc2", w_fc2, b_fc2))

    return layers


def numpy_leaky_relu(x, slope=0.1):
    return np.where(x >= 0.0, x, x * slope)


def numpy_conv2d_3x3_same(x, w, b):
    """
    x: [C_in, 10, 10]
    w: [C_out, C_in, 3, 3]
    b: [C_out]
    returns: [C_out, 10, 10]
    """
    c_in, h, width = x.shape
    c_out = w.shape[0]
    padded = np.zeros((c_in, h + 2, width + 2), dtype=np.float32)
    padded[:, 1:h + 1, 1:width + 1] = x

    out = np.broadcast_to(b[:, None, None], (c_out, h, width)).copy()
    for kh in range(3):
        for kw in range(3):
            patch = padded[:, kh:kh + h, kw:kw + width]  # [C_in, 10, 10]
            out += np.tensordot(w[:, :, kh, kw], patch, axes=([1], [0]))
    return out


def verify_fused_numpy_forward(layers, x_tensor):
    """
    Runs pure numpy fused forward pass on x_tensor [1, 4, 10, 10] and returns (occ_logits [100], pol_logits [100]).
    """
    layer_map = {name: (w, b) for name, w, b in layers}
    x = x_tensor[0].cpu().numpy().astype(np.float32)

    # Stem
    w, b = layer_map["stem_conv"]
    feat = numpy_leaky_relu(numpy_conv2d_3x3_same(x, w, b), 0.1)

    # 4 ResBlocks
    for idx in range(4):
        w1, b1 = layer_map[f"res{idx}_conv1"]
        w2, b2 = layer_map[f"res{idx}_conv2"]
        h = numpy_leaky_relu(numpy_conv2d_3x3_same(feat, w1, b1), 0.1)
        h = numpy_conv2d_3x3_same(h, w2, b2)
        feat = numpy_leaky_relu(h + feat, 0.1)

    # Occupancy head
    w_o1, b_o1 = layer_map["occ_conv1"]
    occ_h = numpy_leaky_relu(numpy_conv2d_3x3_same(feat, w_o1, b_o1), 0.1)
    w_o2, b_o2 = layer_map["occ_conv2"]  # [1, 32, 1, 1]
    occ_out = np.tensordot(w_o2[:, :, 0, 0], occ_h, axes=([1], [0])) + b_o2[:, None, None]
    occ_logits = occ_out.reshape(100)

    # Policy head
    w_p1, b_p1 = layer_map["pol_conv1"]
    pol_h = numpy_leaky_relu(numpy_conv2d_3x3_same(feat, w_p1, b_p1), 0.1)
    flat = pol_h.reshape(3200)
    w_fc1, b_fc1 = layer_map["pol_fc1"]
    fc1 = numpy_leaky_relu(w_fc1 @ flat + b_fc1, 0.1)
    w_fc2, b_fc2 = layer_map["pol_fc2"]
    pol_logits = w_fc2 @ fc1 + b_fc2

    return occ_logits, pol_logits


def main():
    pth_path = os.path.join(PROJECT_ROOT, "data", "sherlock_nn_best.pth")
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Missing {pth_path}")

    model = SherlockNeuralModel(in_channels=4)
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    layers = extract_all_fused_layers(model)

    # Verify parity across 5 random test boards
    torch.manual_seed(42)
    max_occ_diff = 0.0
    max_pol_diff = 0.0
    for _ in range(5):
        test_x = torch.randint(0, 2, (1, 4, 10, 10)).float()
        with torch.no_grad():
            pt_occ, pt_pol = model(test_x)
        np_occ, np_pol = verify_fused_numpy_forward(layers, test_x)
        occ_diff = float(np.max(np.abs(pt_occ[0].numpy() - np_occ)))
        pol_diff = float(np.max(np.abs(pt_pol[0].numpy() - np_pol)))
        max_occ_diff = max(max_occ_diff, occ_diff)
        max_pol_diff = max(max_pol_diff, pol_diff)

    print(f"[Parity Check] Max Occupancy Logit Diff: {max_occ_diff:.2e}")
    print(f"[Parity Check] Max Policy Logit Diff   : {max_pol_diff:.2e}")
    assert max_occ_diff < 1e-4 and max_pol_diff < 1e-4, "Fused forward pass parity check failed!"

    # Concatenate all weights & biases in exact order: [w, b, w, b, ...]
    flat_fp32 = []
    total_floats = 0
    print("\nLayer Layout (Float32 / Float16 Offset Table):")
    for name, w, b in layers:
        w_flat = w.reshape(-1)
        b_flat = b.reshape(-1)
        print(f"  {name:14s} | W: {str(w.shape):18s} ({len(w_flat):7d}) | B: {str(b.shape):10s} ({len(b_flat):4d}) | offset={total_floats}")
        flat_fp32.append(w_flat)
        flat_fp32.append(b_flat)
        total_floats += len(w_flat) + len(b_flat)

    all_fp32 = np.concatenate(flat_fp32, axis=0).astype(np.float32)
    all_fp16 = all_fp32.astype(np.float16)

    # Verify FP16 precision as well
    layers_fp16 = [(name, w.astype(np.float16).astype(np.float32), b.astype(np.float16).astype(np.float32)) for name, w, b in layers]
    fp16_occ, fp16_pol = verify_fused_numpy_forward(layers_fp16, test_x)
    print(f"[FP16 Check]   Max FP16 vs FP32 Policy Diff: {np.max(np.abs(np_pol - fp16_pol)):.4f}")

    # Save Float32 binary for LuaJIT FFI
    out_fp32_path = os.path.join(PROJECT_ROOT, "data", "moriarty_fused.bin")
    with open(out_fp32_path, "wb") as f:
        f.write(struct.pack("<4sI", b"MOR1", total_floats))
        f.write(all_fp32.tobytes())

    # Save Float16 binary for Web Showcase
    showcase_assets = os.path.join(PROJECT_ROOT, "showcase", "assets")
    os.makedirs(showcase_assets, exist_ok=True)
    out_fp16_path = os.path.join(showcase_assets, "moriarty_fp16.bin")
    with open(out_fp16_path, "wb") as f:
        f.write(struct.pack("<4sI", b"MF16", total_floats))
        f.write(all_fp16.tobytes())

    print(f"\n[SUCCESS] Total floats: {total_floats:,}")
    print(f"  -> Wrote Float32 binary: {out_fp32_path} ({os.path.getsize(out_fp32_path) / 1024:.1f} KB)")
    print(f"  -> Wrote Float16 binary: {out_fp16_path} ({os.path.getsize(out_fp16_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

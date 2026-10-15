import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# ==========================================
# 1. SHARDED DATASET LOADER
# ==========================================
class SherlockShardedDataset(Dataset):
    def __init__(self, data_dir, max_shards=None):
        self.shard_files = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
        if not self.shard_files:
            raise FileNotFoundError(f"No shard_*.npz files found in {data_dir}. Run generate_sherlock_data.py first!")

        if max_shards:
            self.shard_files = self.shard_files[:max_shards]

        print(f"Loading {len(self.shard_files)} shards into memory...")
        all_inputs = []
        all_occupancies = []
        all_policies = []

        for idx, f in enumerate(self.shard_files):
            data = np.load(f)
            all_inputs.append(data['inputs'])
            all_occupancies.append(data['occupancy_targets'])
            all_policies.append(data['policy_targets'])

        self.inputs = np.concatenate(all_inputs, axis=0)
        self.occupancies = np.concatenate(all_occupancies, axis=0)
        self.policies = np.concatenate(all_policies, axis=0)

        print(f"Dataset ready: {len(self.inputs):,} total board states across {len(self.shard_files)} shards.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.inputs[idx]),          # [4, 10, 10]
            torch.from_numpy(self.occupancies[idx]),     # [10, 10]
            torch.tensor(self.policies[idx], dtype=torch.long) # scalar int (0..99)
        )

# ==========================================
# 2. SHERLOCK RESIDUAL CONVNET
# ==========================================
class ResConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class SherlockNeuralModel(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # Initial projection: 4 channels -> 64 channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Residual backbone preserving 10x10 resolution
        self.res_layers = nn.Sequential(
            ResConvBlock(64),
            ResConvBlock(64),
            ResConvBlock(64),
            ResConvBlock(64)
        )

        # Head 1: Occupancy Heatmap (10x10 binary fleet locations)
        self.occupancy_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1) # [B, 1, 10, 10]
        )

        # Head 2: Next Shot Policy (100-way categorical distribution)
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 100) # [B, 100]
        )

    def forward(self, x):
        features = self.stem(x)
        features = self.res_layers(features)

        # [B, 100] occupancy logits
        occ_logits = self.occupancy_head(features).squeeze(1).view(x.size(0), 100)

        # [B, 100] policy logits
        policy_logits = self.policy_head(features)

        return occ_logits, policy_logits

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model():
    parser = argparse.ArgumentParser(description="Train Neural Network on 1M Sherlock Games")
    parser.add_argument("--data-dir", type=str, default=os.path.join(DATA_DIR, "sherlock_1m_dataset"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-shards", type=int, default=None, help="Limit number of shards loaded (for quick testing)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(f"🧠 TRAINING SHERLOCK NEURAL NETWORK")
    print(f"   Device         : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Epochs         : {args.epochs}")
    print(f"   Batch Size     : {args.batch_size}")
    print(f"   Learning Rate  : {args.lr}")
    print("=" * 65)

    dataset = SherlockShardedDataset(args.data_dir, max_shards=args.max_shards)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == 'cuda'), num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == 'cuda'), num_workers=2)

    model = SherlockNeuralModel(in_channels=4).to(device)

    # Losses
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_weights_path = os.path.join(DATA_DIR, "sherlock_nn_best.pth")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_occ_loss = 0.0
        total_pol_loss = 0.0
        correct_policies = 0
        total_samples = 0

        start_time = time.time()
        for batch_x, batch_occ, batch_pol in train_loader:
            batch_x = batch_x.to(device)
            batch_occ_flat = batch_occ.view(batch_occ.size(0), -1).to(device) # [B, 100]
            batch_pol = batch_pol.to(device) # [B]

            optimizer.zero_grad()
            occ_logits, pol_logits = model(batch_x)

            loss_occ = bce_loss_fn(occ_logits, batch_occ_flat)
            loss_pol = ce_loss_fn(pol_logits, batch_pol)
            loss = loss_occ + (0.5 * loss_pol)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_occ_loss += loss_occ.item() * batch_x.size(0)
            total_pol_loss += loss_pol.item() * batch_x.size(0)

            preds = pol_logits.argmax(dim=-1)
            correct_policies += (preds == batch_pol).sum().item()
            total_samples += batch_x.size(0)

        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_correct = 0

        with torch.no_grad():
            for val_x, val_occ, val_pol in val_loader:
                val_x = val_x.to(device)
                val_occ_flat = val_occ.view(val_occ.size(0), -1).to(device)
                val_pol = val_pol.to(device)

                occ_logits, pol_logits = model(val_x)
                loss_occ = bce_loss_fn(occ_logits, val_occ_flat)
                loss_pol = ce_loss_fn(pol_logits, val_pol)
                loss = loss_occ + (0.5 * loss_pol)

                val_loss += loss.item() * val_x.size(0)
                preds = pol_logits.argmax(dim=-1)
                val_correct += (preds == val_pol).sum().item()
                val_samples += val_x.size(0)

        epoch_time = time.time() - start_time
        train_acc = (correct_policies / total_samples) * 100
        val_acc = (val_correct / val_samples) * 100
        avg_train_loss = total_loss / total_samples
        avg_val_loss = val_loss / val_samples

        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.1f}%) | "
              f"Val Loss: {avg_val_loss:.4f} (Val Acc: {val_acc:.1f}%)")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_weights_path)
            print(f"  -> ⭐ Saved best model checkpoint to {best_weights_path}")

    print("\n✅ Training Complete!")

    # Export to ONNX
    print("Exporting ONNX model for fast deployment...")
    model.load_state_dict(torch.load(best_weights_path, map_location='cpu'))
    model.to('cpu')
    model.eval()
    dummy_input = torch.randn(1, 4, 10, 10)
    onnx_path = os.path.join(DATA_DIR, "sherlock_nn.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['board_input'],
        output_names=['occupancy_logits', 'policy_logits']
    )
    print(f"✅ Exported ONNX to {onnx_path}")

if __name__ == "__main__":
    train_model()

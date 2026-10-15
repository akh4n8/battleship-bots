import torch
import torch.nn as nn
import math

# ==========================================
# 1. SPATIAL FEATURE EXTRACTOR (CNN)
# ==========================================
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        # A 10x10 grid is small. We use padding='same' (padding=1 for kernel 3)
        # to ensure we don't crush the spatial dimensions too quickly.
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Pool down to 5x5 to aggregate regional features
            nn.MaxPool2d(2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        
        # 128 channels * 5x5 grid = 3200 features
        self.proj = nn.Linear(128 * 5 * 5, d_model)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        return self.proj(x)

# ==========================================
# 2. POSITIONAL ENCODING
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10): # max_len is the sequence length (5)
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: [1, max_len, d_model]

    def forward(self, x):
        # x shape: [Batch, SeqLen, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# ==========================================
# 3. THE COGNITIVE TRANSFORMER
# ==========================================
class CognitiveBattleshipModel(nn.Module):
    def __init__(self, in_channels=5, d_model=256, n_heads=4, n_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # 1. Spatial Processing
        self.feature_extractor = SpatialFeatureExtractor(in_channels, d_model)
        
        # 2. Temporal Processing
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=512, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Decoding to 100 Logits (The 10x10 Board)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 100) # Output 100 raw logits
        )

    def forward(self, x):
        """
        x shape: [Batch, SeqLen, Channels, Height, Width]
        """
        B, S, C, H, W = x.shape
        
        # Flatten Batch and SeqLen so the CNN can process every game independently
        x = x.view(B * S, C, H, W)
        
        # Extract features [B*S, d_model]
        spatial_features = self.feature_extractor(x)
        
        # Reshape back to Sequence [B, S, d_model]
        sequence_features = spatial_features.view(B, S, self.d_model)
        
        # Add temporal awareness
        sequence_features = self.pos_encoder(sequence_features)
        
        # Pass through the Transformer
        memory = self.transformer(sequence_features)
        
        # Extract the state from the MOST RECENT game in the sequence (index -1)
        # The Transformer has already mixed the context of games T-5 through T-2 into this state.
        latest_state = memory[:, -1, :]
        
        # Decode into 100 logits
        logits = self.decoder(latest_state)
        
        return logits

# ==========================================
# EXPORT HELPERS
# ==========================================
def export_to_onnx(model, model_name="moriarty_defense.onnx", in_channels=7, seq_len=5):
    """
    Exports the PyTorch model to ONNX, strictly defining static shapes. 
    This is required to ensure a smooth transition to TFLite for Android.
    """
    model.eval()
    
    # Create a dummy tensor matching the exact shape expected by the model
    dummy_input = torch.randn(1, seq_len, in_channels, 10, 10)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        model_name,
        export_params=True,
        opset_version=14,  # Opset 14 handles Transformers well
        do_constant_folding=True,
        input_names=['input_sequence'],
        output_names=['board_logits'],
    )
    print(f"[SUCCESS] Successfully exported {model_name}. Ready for onnx2tf!")

if __name__ == "__main__":
    # Test the Defensive Model (7 Channels)
    print("Testing Defense Architecture...")
    defense_model = CognitiveBattleshipModel(in_channels=7)
    dummy_input = torch.randn(2, 5, 7, 10, 10) # Batch of 2
    output = defense_model(dummy_input)
    print(f"Output Shape: {output.shape} (Expected: [2, 100])")
    
    # Export for Android
    export_to_onnx(defense_model, "moriarty_defense.onnx", in_channels=7)

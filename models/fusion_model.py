# models/fusion_model.py
import torch
import torch.nn as nn
import logging

# Import the encoders and decoder we just built
from models.visual_model import VisualEncoder
from models.audio_model import AudioEncoder, AudioDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VisioVoxModel")

class VisioVox(nn.Module):
    def __init__(self, visual_embedding_dim=128):
        super(VisioVox, self).__init__()
        
        logger.info("Initializing VisioVox Master Architecture...")
        
        # 1. The Sub-Networks
        self.visual_encoder = VisualEncoder(embedding_dim=visual_embedding_dim)
        self.audio_encoder = AudioEncoder()
        self.audio_decoder = AudioDecoder()
        
        # 2. The Fusion Block
        # Audio bottleneck outputs 512 channels. Visual outputs 128 channels.
        # When concatenated, we get 512 + 128 = 640 channels.
        # We need to reduce this back to 512 so the AudioDecoder can accept it.
        self.fusion_block = nn.Sequential(
            nn.Conv2d(512 + visual_embedding_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, mixed_spectrogram, lip_frames):
        """
        mixed_spectrogram shape: [Batch, 1, Freq, Time]
        lip_frames shape:        [Batch, 1, 112, 112]
        """
        # 1. Extract Visual Features
        # v_emb shape: [Batch, 128]
        v_emb = self.visual_encoder(lip_frames)
        
        # 2. Extract Audio Features
        # a_bottleneck shape: [Batch, 512, F_reduced, T_reduced]
        a_bottleneck, skips = self.audio_encoder(mixed_spectrogram)
        
        # 3. Spatial Expansion of Visual Features
        # We need to stretch the [Batch, 128] vector to match the spatial dims of the audio bottleneck.
        # Add spatial dimensions: [Batch, 128, 1, 1]
        v_emb_expanded = v_emb.unsqueeze(-1).unsqueeze(-1)
        # Expand to match audio height and width: [Batch, 128, F_reduced, T_reduced]
        v_emb_expanded = v_emb_expanded.expand(-1, -1, a_bottleneck.size(2), a_bottleneck.size(3))
        
        # 4. Concatenate along the channel dimension
        # fused shape: [Batch, 640, F_reduced, T_reduced]
        fused = torch.cat([a_bottleneck, v_emb_expanded], dim=1)
        
        # 5. Blend the features back to 512 channels
        # blended shape: [Batch, 512, F_reduced, T_reduced]
        blended = self.fusion_block(fused)
        
        # 6. Decode back into a mask
        # mask shape: [Batch, 1, Freq, Time]
        predicted_mask = self.audio_decoder(blended, skips)
        
        return predicted_mask

# --- Standalone End-to-End Test ---
if __name__ == "__main__":
    try:
        model = VisioVox()
        
        # Create Dummy Data
        batch_size = 2
        dummy_spectrogram = torch.randn(batch_size, 1, 256, 128) # 256 freq bins, 128 time steps
        dummy_lip_frames = torch.randn(batch_size, 1, 112, 112)  # 112x112 grayscale image
        
        logger.info(f"Input Spectrogram Shape: {dummy_spectrogram.shape}")
        logger.info(f"Input Lip Frames Shape: {dummy_lip_frames.shape}")
        
        logger.info("Running forward pass (this might take a few seconds on CPU)...")
        
        # Pass through the full architecture
        output_mask = model(dummy_spectrogram, dummy_lip_frames)
        
        logger.info(f"Final Output Mask Shape: {output_mask.shape}")
        
        if output_mask.shape == dummy_spectrogram.shape:
             logger.info("VICTORY! The End-to-End architecture is mathematically sound.")
             
    except Exception as e:
        logger.error(f"Error during full model testing: {e}")
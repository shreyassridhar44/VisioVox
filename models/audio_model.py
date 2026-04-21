# models/audio_model.py
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AudioModel")

class AudioEncoder(nn.Module):
    """Compresses the mixed spectrogram and saves skip connections for the U-Net."""
    def __init__(self):
        super(AudioEncoder, self).__init__()
        
        # Input: [Batch Size, Channels=1, Frequencies=256, Time Steps]
        self.enc1 = self._conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2)) # Out: [32, 128, T/2]
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2)) # Out: [64, 64, T/4]
        
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2)) # Out: [128, 32, T/8]
        
        self.enc4 = self._conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2)) # Out: [256, 16, T/16]
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Forward pass while saving skip connections for the Decoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        bottleneck = self.bottleneck(p4)
        
        return bottleneck, [e1, e2, e3, e4]

class AudioDecoder(nn.Module):
    """Upsamples features back into a spectrogram mask."""
    def __init__(self):
        super(AudioDecoder, self).__init__()
        
        # Note: In Step 3.3, we will fuse the visual features into the bottleneck.
        # For now, this decoder assumes the bottleneck is 512 channels.
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(512, 256) # 256 (upsampled) + 256 (skip connection)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128) 
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64) 
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32) 
        
        # Final output is a 1-channel mask predicting values between 0 and 1 (Sigmoid)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connections):
        e1, e2, e3, e4 = skip_connections
        
        d4 = self.upconv4(x)
        d4 = self._pad_to_match(d4, e4) # Fix dimension mismatches
        d4 = torch.cat([d4, e4], dim=1) # Concatenate skip connection
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = self._pad_to_match(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = self._pad_to_match(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = self._pad_to_match(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        mask = self.final_conv(d1)
        return mask

    def _pad_to_match(self, x, target):
        """Pads tensor x to perfectly match the spatial dimensions of the target tensor."""
        diffY = target.size()[2] - x.size()[2]
        diffX = target.size()[3] - x.size()[3]
        
        # Pad: (left, right, top, bottom)
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        return x

# --- Standalone Test ---
if __name__ == "__main__":
    try:
        encoder = AudioEncoder()
        decoder = AudioDecoder()
        logger.info("Audio Encoder and Decoder initialized successfully.")
        
        # Create a dummy mixed spectrogram: [Batch=2, Channels=1, Freq=256, Time=128]
        dummy_spec = torch.randn(2, 1, 256, 128)
        logger.info(f"Input Spectrogram Shape: {dummy_spec.shape}")
        
        # 1. Encode
        bottleneck, skips = encoder(dummy_spec)
        logger.info(f"Bottleneck Shape: {bottleneck.shape}")
             
        # 2. Decode (Bypassing fusion for this test)
        mask = decoder(bottleneck, skips)
        logger.info(f"Output Mask Shape: {mask.shape}")
        
        if dummy_spec.shape == mask.shape:
             logger.info("Success! The predicted mask shape perfectly matches the input spectrogram.")
             
    except Exception as e:
        logger.error(f"Error during model testing: {e}")
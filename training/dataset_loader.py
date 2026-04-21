# training/dataset_loader.py
import os
import torch
import librosa
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("DatasetLoader")

class VisioVoxDataset(Dataset):
    def __init__(self, metadata_path: str, audio_length_sec: float = 3.0, sample_rate: int = 16000):
        self.metadata = pd.read_csv(metadata_path)
        self.sample_rate = sample_rate
        # Lock audio to exact number of samples (e.g., 3 seconds * 16000 = 48000)
        self.target_samples = int(audio_length_sec * sample_rate)
        
        # STFT Parameters
        self.n_fft = 510  # n_fft//2 + 1 = 256 frequency bins (perfect for our U-Net)
        self.hop_length = 160
        self.win_length = 400
        self.window = torch.hann_window(self.win_length)

    def __len__(self):
        return len(self.metadata)

    def _process_audio(self, audio_path):
        """Loads audio, pads/truncates to target length, and computes STFT Spectrogram."""
        wave, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Truncate or Pad to ensure uniform batch sizes
        if len(wave) > self.target_samples:
            wave = wave[:self.target_samples]
        else:
            padding = self.target_samples - len(wave)
            wave = np.pad(wave, (0, padding), mode='constant')
            
        # Convert to PyTorch tensor
        wave_tensor = torch.tensor(wave)
        
        # Compute STFT (Spectrogram)
        stft = torch.stft(
            wave_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        
        # We only need the magnitude (absolute value) for the U-Net Mask
        magnitude = torch.abs(stft).unsqueeze(0) # Shape: [1, 256, TimeSteps]
        return magnitude

    def _get_lip_frame(self, lips_dir):
        """Grabs the middle frame from the target speaker's lip directory."""
        lip_path_obj = Path(lips_dir)
        frames = sorted(list(lip_path_obj.glob("*.jpg")))
        
        if not frames:
            # Fallback if a folder is empty (shouldn't happen with our valid_samples check)
            return torch.zeros(1, 112, 112)
            
        # Pick the middle frame to ensure the mouth is likely active
        middle_idx = len(frames) // 2
        frame_path = frames[middle_idx]
        
        # Read grayscale and normalize between 0 and 1
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        
        # Shape: [1, 112, 112]
        img_tensor = torch.tensor(img).unsqueeze(0)
        return img_tensor

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 1. Load Mixed Audio (Input)
        mixed_spec = self._process_audio(row['mixed_audio_path'])
        
        # 2. Load Target Audio (Ground Truth we want the model to isolate)
        target_spec = self._process_audio(row['target_audio_path'])
        
        # 3. Load Target Lips (Visual Cue)
        target_lips = self._get_lip_frame(row['target_lips_dir'])
        
        return mixed_spec, target_spec, target_lips

# --- Standalone Local Test ---
if __name__ == "__main__":
    METADATA_CSV = "data/processed/dataset_metadata.csv"
    
    try:
        logger.info("Initializing Dataset...")
        dataset = VisioVoxDataset(metadata_path=METADATA_CSV)
        logger.info(f"Total mixtures in dataset: {len(dataset)}")
        
        # Create a DataLoader to test batching
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        logger.info("Fetching a single batch to verify tensor shapes...")
        mixed_batch, target_batch, lips_batch = next(iter(dataloader))
        
        logger.info(f"Mixed Spectrogram Batch Shape: {mixed_batch.shape}")
        logger.info(f"Target Spectrogram Batch Shape: {target_batch.shape}")
        logger.info(f"Lip Frames Batch Shape: {lips_batch.shape}")
        
        # Verification checks
        assert mixed_batch.shape[1] == 1, "Spectrogram should have 1 channel"
        assert mixed_batch.shape[2] == 256, "Spectrogram should have 256 frequency bins"
        assert lips_batch.shape[2] == 112 and lips_batch.shape[3] == 112, "Lips must be 112x112"
        
        logger.info("All tensor shapes are verified. Dataset Loader is ready for training!")
        
    except Exception as e:
        logger.error(f"Error testing Dataset Loader: {e}")
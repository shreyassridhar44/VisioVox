# preprocessing/generate_mixtures.py
import os
import random
import torch
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("MixtureGenerator")

def generate_mixtures(audio_dir: str, lips_dir: str, mixtures_dir: str, metadata_path: str, num_mixtures: int = 200):
    audio_path_obj = Path(audio_dir)
    lips_path_obj = Path(lips_dir)
    mixtures_path_obj = Path(mixtures_dir)
    
    mixtures_path_obj.mkdir(parents=True, exist_ok=True)
    
    # Get all valid audio files that also have corresponding lip frames
    valid_samples = []
    for audio_file in audio_path_obj.glob("*.wav"):
        video_id = audio_file.stem
        lip_folder = lips_path_obj / video_id
        
        # Only use this sample if we successfully extracted lip frames for it
        if lip_folder.exists() and len(list(lip_folder.glob("*.jpg"))) > 0:
            valid_samples.append(audio_file)
            
    if len(valid_samples) < 2:
        logger.error("Not enough valid audio/lip pairs to create mixtures.")
        return

    logger.info(f"Found {len(valid_samples)} valid audio samples. Generating {num_mixtures} mixtures...")
    
    metadata = []
    
    for i in range(num_mixtures):
        # Randomly select two DIFFERENT speakers
        target_audio_path, interference_audio_path = random.sample(valid_samples, 2)
        
        # The target speaker is the one whose lips we will show to the model
        target_id = target_audio_path.stem
        interference_id = interference_audio_path.stem
        mix_id = f"mix_{i:04d}_{target_id}_AND_{interference_id}"
        
        # --- THE FIX: Using librosa/soundfile to bypass torchaudio Windows DLL issues ---
        target_wave_np, sr = librosa.load(target_audio_path, sr=16000, mono=True)
        interference_wave_np, _ = librosa.load(interference_audio_path, sr=16000, mono=True)
        
        # Convert numpy arrays to PyTorch tensors and add the channel dimension (1, Length)
        target_wave = torch.tensor(target_wave_np).unsqueeze(0)
        interference_wave = torch.tensor(interference_wave_np).unsqueeze(0)
        
        # Truncate both to the length of the shorter one so they overlap perfectly
        min_length = min(target_wave.shape[1], interference_wave.shape[1])
        target_wave = target_wave[:, :min_length]
        interference_wave = interference_wave[:, :min_length]
        
        # Mix the audio (Simple addition)
        mixed_wave = target_wave + interference_wave
        
        # Normalize the mixture to prevent audio clipping (values outside -1.0 to 1.0)
        max_val = torch.max(torch.abs(mixed_wave))
        if max_val > 1.0:
            mixed_wave = mixed_wave / max_val
            
        # Save the mixed audio using soundfile
        mix_output_path = mixtures_path_obj / f"{mix_id}.wav"
        sf.write(mix_output_path, mixed_wave.squeeze(0).numpy(), sr)
        
        # Record this in our dataset metadata
        metadata.append({
            "mix_id": mix_id,
            "mixed_audio_path": str(mix_output_path),
            "target_audio_path": str(target_audio_path),
            "target_lips_dir": str(lips_path_obj / target_id),
            "duration_samples": min_length
        })
        
        if (i + 1) % 50 == 0:
            logger.info(f"Generated {i + 1}/{num_mixtures} mixtures...")

    # Save metadata to CSV
    df = pd.DataFrame(metadata)
    df.to_csv(metadata_path, index=False)
    logger.info(f"Mixture generation complete! Metadata saved to {metadata_path}")

if __name__ == "__main__":
    AUDIO_DIR = "data/raw/audio"
    LIPS_DIR = "data/processed/lips"
    MIXTURES_DIR = "data/processed/mixtures"
    METADATA_PATH = "data/processed/dataset_metadata.csv"
    
    # We generate 200 mixtures from our valid debug videos
    generate_mixtures(AUDIO_DIR, LIPS_DIR, MIXTURES_DIR, METADATA_PATH, num_mixtures=200)
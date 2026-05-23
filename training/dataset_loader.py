# training/dataset_loader.py  -  VisioVox v4
# Loads ALL 25 lip frames per clip (not just the middle frame)
# Also loads interferer audio and lips for PIT + contrastive loss

import random
import numpy as np
import pandas as pd
import cv2
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path

SAMPLE_RATE    = 16000
N_FFT          = 510
HOP_LENGTH     = 160
WIN_LENGTH     = 400
TARGET_SAMPLES = 48000
NUM_FRAMES     = 25
FREQ_MASK_MAX  = 30
TIME_MASK_MAX  = 30
N_FREQ_MASKS   = 2
N_TIME_MASKS   = 2


def spec_augment(spec):
    spec = spec.clone()
    _, F, T = spec.shape
    for _ in range(N_FREQ_MASKS):
        f  = random.randint(0, FREQ_MASK_MAX)
        f0 = random.randint(0, max(F - f, 1))
        spec[:, f0:f0 + f, :] = 0.0
    for _ in range(N_TIME_MASKS):
        t  = random.randint(0, TIME_MASK_MAX)
        t0 = random.randint(0, max(T - t, 1))
        spec[:, :, t0:t0 + t] = 0.0
    return spec


class VisioVoxDataset(Dataset):
    def __init__(self, metadata_path: str, augment=False):
        self.meta    = pd.read_csv(metadata_path)
        self.window  = torch.hann_window(WIN_LENGTH)
        self.augment = augment

    def __len__(self):
        return len(self.meta)

    def _spec(self, path):
        w, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        if len(w) >= TARGET_SAMPLES:
            w = w[:TARGET_SAMPLES]
        else:
            w = np.pad(w, (0, TARGET_SAMPLES - len(w)))
        t    = torch.tensor(w, dtype=torch.float32)
        s    = torch.stft(t, n_fft=N_FFT, hop_length=HOP_LENGTH,
                          win_length=WIN_LENGTH, window=self.window,
                          return_complex=True)
        spec = torch.abs(s).unsqueeze(0)   # [1, 256, T]
        if self.augment:
            spec = spec_augment(spec)
        return spec

    def _lips(self, d):
        paths  = sorted(Path(d).glob("*.jpg"))
        frames = []
        for p in paths[:NUM_FRAMES]:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            frames.append(
                cv2.resize(img, (112, 112)).astype(np.float32) / 255.0
                if img is not None else np.zeros((112, 112), np.float32)
            )
        while len(frames) < NUM_FRAMES:
            frames.append(np.zeros((112, 112), np.float32))
        return torch.tensor(np.stack(frames)).unsqueeze(1)  # [25, 1, 112, 112]

    def __getitem__(self, idx):
        r = self.meta.iloc[idx]
        try:
            return (
                self._spec(r["mixed_audio_path"]),
                self._spec(r["target_audio_path"]),
                self._lips(r["target_lips_dir"]),
                self._spec(r["interfere_audio"]),
                self._lips(r["interfere_lips"]),
            )
        except Exception:
            ds = torch.zeros(1, 256, 301)
            dl = torch.zeros(NUM_FRAMES, 1, 112, 112)
            return ds, ds, dl, ds, dl
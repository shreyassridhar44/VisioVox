# create_mixtures_devA.py
# Run after preprocess_devA.py finishes:
#   python create_mixtures_devA.py
#
# Creates 5000 speaker mixtures from the processed Dev A clips.
# Each mixture: target speaker audio + different interference speaker.
# Saves mixture WAVs + complete training metadata CSV.

import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────
PROJECT_DIR   = "E:\\visiovox"
MANIFEST_PATH = "D:\\visiovox_data\\devA_manifest.csv"
OUTPUT_DIR    = "D:\\visiovox_data\\devA_processed\\mixtures"
METADATA_OUT  = "D:\\visiovox_data\\devA_training_metadata.csv"
NUM_MIXTURES  = 500000
SAMPLE_RATE   = 16000
CLIP_SAMPLES  = int(3.0 * SAMPLE_RATE)

random.seed(42)
np.random.seed(42)

# ── Load manifest ─────────────────────────────────────────────
assert os.path.exists(MANIFEST_PATH), (
    f"Manifest not found: {MANIFEST_PATH}\n"
    "Run preprocess_devA.py first."
)

manifest = pd.read_csv(MANIFEST_PATH)
print(f"Manifest loaded  : {len(manifest):,} clips")
print(f"Unique speakers  : {manifest['speaker'].nunique()}")

# Group clips by speaker
by_speaker = {}
for _, row in manifest.iterrows():
    spk = row["speaker"]
    if spk not in by_speaker:
        by_speaker[spk] = []
    by_speaker[spk].append(row)

# Only use speakers with at least 2 clips
valid_speakers = [s for s in by_speaker if len(by_speaker[s]) >= 2]
print(f"Speakers with 2+ clips: {len(valid_speakers)}")
assert len(valid_speakers) >= 2, "Need at least 2 valid speakers"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper ────────────────────────────────────────────────────
def load_clip_3s(audio_path: str) -> np.ndarray:
    wave, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if len(wave) >= CLIP_SAMPLES:
        start = random.randint(0, len(wave) - CLIP_SAMPLES)
        return wave[start:start + CLIP_SAMPLES].astype(np.float32)
    return np.pad(wave, (0, CLIP_SAMPLES - len(wave))).astype(np.float32)

# ── Create mixtures ───────────────────────────────────────────
records = []
skipped = 0

print(f"\nCreating {NUM_MIXTURES:,} mixtures...")

for i in tqdm(range(NUM_MIXTURES), desc="Creating mixtures"):
    # Two different speakers
    spk_t, spk_i = random.sample(valid_speakers, 2)
    row_t = random.choice(by_speaker[spk_t])
    row_i = random.choice(by_speaker[spk_i])

    mix_id   = f"mix_{i:05d}_{spk_t}_{spk_i}"
    mix_path = os.path.join(OUTPUT_DIR, f"{mix_id}.wav")

    try:
        t_wave = load_clip_3s(row_t["audio_path"])
        i_wave = load_clip_3s(row_i["audio_path"])

        # Random SNR between -5 and +5 dB
        snr_db   = random.uniform(-5.0, 5.0)
        snr_lin  = 10 ** (snr_db / 20.0)
        i_scaled = i_wave * snr_lin
        mixed    = t_wave + i_scaled

        # Normalise
        peak = np.max(np.abs(mixed))
        if peak > 0.95:
            scale    = 0.95 / peak
            mixed    = mixed    * scale
            t_wave   = t_wave   * scale
            i_scaled = i_scaled * scale

        sf.write(mix_path, mixed, SAMPLE_RATE)

        records.append({
            "mix_id":            mix_id,
            "mixed_audio_path":  mix_path,
            "target_audio_path": row_t["audio_path"],
            "target_lips_dir":   row_t["lips_dir"],
            "target_speaker":    spk_t,
            "interfere_audio":   row_i["audio_path"],
            "interfere_lips":    row_i["lips_dir"],
            "interfere_speaker": spk_i,
            "snr_db":            round(snr_db, 2),
        })

    except Exception as e:
        skipped += 1
        if skipped <= 5:
            tqdm.write(f"  SKIP mix_{i}: {e}")

# ── Save ──────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(METADATA_OUT, index=False)

mix_size_mb = sum(
    os.path.getsize(r["mixed_audio_path"])
    for r in records if os.path.exists(r["mixed_audio_path"])
) / (1024 * 1024)

print()
print("=" * 55)
print(f"Mixture creation complete!")
print(f"  Created       : {len(records):,}")
print(f"  Skipped       : {skipped}")
print(f"  Metadata CSV  : {METADATA_OUT}")
print(f"  Mixture WAVs  : {mix_size_mb:.0f} MB")
print()

# ── Estimate upload size ──────────────────────────────────────
import shutil
audio_size = sum(
    f.stat().st_size
    for f in Path(os.path.join(PROJECT_DIR, "data/devA_processed/audio")).rglob("*.wav")
) / (1024 ** 3) if os.path.exists(
    os.path.join(PROJECT_DIR, "data/devA_processed/audio")) else 0

lips_size = sum(
    f.stat().st_size
    for f in Path(os.path.join(PROJECT_DIR, "data/devA_processed/lips")).rglob("*.jpg")
) / (1024 ** 3) if os.path.exists(
    os.path.join(PROJECT_DIR, "data/devA_processed/lips")) else 0

print(f"  Audio WAVs size  : {audio_size:.1f} GB")
print(f"  Lip JPGs size    : {lips_size:.1f} GB")
print(f"  Mixtures size    : {mix_size_mb/1024:.1f} GB")
print(f"  Total to upload  : {audio_size + lips_size + mix_size_mb/1024:.1f} GB")
print()
print("NEXT STEP:")
print("  Upload data/devA_processed/ and data/devA_training_metadata.csv")
print("  to Google Drive at: My Drive/capstone/")
print("  Use Google Drive desktop app for large uploads.")
print("=" * 55)
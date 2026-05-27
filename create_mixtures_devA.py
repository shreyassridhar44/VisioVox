import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────
MANIFEST_PATH  = "D:\\visiovox_data\\devA_manifest.csv"
OUTPUT_DIR     = "E:\\visiovox\\data\\mixtures"
METADATA_OUT   = "E:\\visiovox\\data\\devA_training_metadata.csv"
PROGRESS_FILE  = "E:\\visiovox\\data\\mixtures_progress.txt"
NUM_MIXTURES   = 500000
SAMPLE_RATE    = 16000
CLIP_SAMPLES   = int(3.0 * SAMPLE_RATE)   # 48000 samples = 3 seconds

random.seed(42)
np.random.seed(42)

# ── Load manifest ─────────────────────────────────────────────
assert os.path.exists(MANIFEST_PATH), f"Manifest not found: {MANIFEST_PATH}"

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

valid_speakers = [s for s in by_speaker if len(by_speaker[s]) >= 2]
print(f"Speakers with 2+ clips: {len(valid_speakers)}")
assert len(valid_speakers) >= 2, "Need at least 2 valid speakers"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METADATA_OUT), exist_ok=True)


def load_clip_3s(audio_path: str) -> np.ndarray:
    wave, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if len(wave) >= CLIP_SAMPLES:
        start = random.randint(0, len(wave) - CLIP_SAMPLES)
        return wave[start:start + CLIP_SAMPLES].astype(np.float32)
    return np.pad(wave, (0, CLIP_SAMPLES - len(wave))).astype(np.float32)


def main():
    # ── Resume logic ──────────────────────────────────────────
    start_i  = 0
    records  = []

    if os.path.exists(METADATA_OUT) and os.path.exists(PROGRESS_FILE):
        try:
            records = pd.read_csv(METADATA_OUT).to_dict("records")
            with open(PROGRESS_FILE, "r") as f:
                start_i = int(f.read().strip())
            print(f"Resuming from mixture {start_i:,} ({len(records):,} already created)")
        except Exception:
            print("Could not resume -- starting fresh")
            start_i = 0
            records = []
    else:
        print("Starting fresh")

    if start_i >= NUM_MIXTURES:
        print("All mixtures already created.")
        return

    skipped = 0

    print(f"\nCreating mixtures {start_i:,} to {NUM_MIXTURES:,}...")

    # Re-seed at start_i so we get the same sequence if resuming
    random.seed(42 + start_i)
    np.random.seed(42 + start_i)

    for i in tqdm(range(start_i, NUM_MIXTURES), desc="Creating mixtures"):
        spk_t, spk_i = random.sample(valid_speakers, 2)
        row_t = random.choice(by_speaker[spk_t])
        row_i = random.choice(by_speaker[spk_i])

        mix_id   = f"mix_{i:06d}_{spk_t}_{spk_i}"
        mix_path = os.path.join(OUTPUT_DIR, f"{mix_id}.wav")

        try:
            t_wave = load_clip_3s(row_t["audio_path"])
            i_wave = load_clip_3s(row_i["audio_path"])

            snr_db   = random.uniform(-5.0, 5.0)
            snr_lin  = 10 ** (snr_db / 20.0)
            i_scaled = i_wave * snr_lin
            mixed    = t_wave + i_scaled

            peak = np.max(np.abs(mixed))
            if peak > 0.95:
                scale    = 0.95 / peak
                mixed    = mixed  * scale
                t_wave   = t_wave * scale
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

        # Save progress every 1000 mixtures
        if (i + 1) % 1000 == 0:
            pd.DataFrame(records).to_csv(METADATA_OUT, index=False)
            with open(PROGRESS_FILE, "w") as f:
                f.write(str(i + 1))
            tqdm.write(f"  Saved -- {i+1:,} mixtures done")

    # Final save
    pd.DataFrame(records).to_csv(METADATA_OUT, index=False)
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(NUM_MIXTURES))

    print()
    print("=" * 55)
    print("Mixture creation complete!")
    print(f"  Created  : {len(records):,}")
    print(f"  Skipped  : {skipped}")
    print(f"  Metadata : {METADATA_OUT}")
    print("=" * 55)
    print("\nNext: python training/train.py")


if __name__ == "__main__":
    main()
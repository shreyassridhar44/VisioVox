# scan_devA.py
# Run from /Users/sharanmalali/visiovoxnew with venv active:
#   python scan_devA.py

import os
from pathlib import Path

DEV_A_PATH = "C:\\Users\\samar\\Downloads\\dev\\mp4"

dev_path = Path(DEV_A_PATH)
assert dev_path.exists(), f"Path not found: {DEV_A_PATH}\nChange DEV_A_PATH to match your actual Dev A location."

print(f"Scanning: {DEV_A_PATH}")
print()

speakers      = []
total_clips   = 0
total_size_mb = 0

speaker_dirs = sorted([d for d in dev_path.iterdir() if d.is_dir()])
print(f"Speakers found: {len(speaker_dirs)}")

for spk_dir in speaker_dirs:
    spk_clips = 0
    for yt_dir in spk_dir.iterdir():
        if not yt_dir.is_dir():
            continue
        clips = list(yt_dir.glob("*.mp4"))
        spk_clips   += len(clips)
        total_clips += len(clips)
        for clip in clips:
            total_size_mb += clip.stat().st_size / (1024 * 1024)
    speakers.append((spk_dir.name, spk_clips))

print("First 20 speakers:")
for name, count in speakers[:20]:
    print(f"  {name}: {count} clips")
if len(speakers) > 20:
    print(f"  ... and {len(speakers)-20} more")

print()
print("=" * 50)
print(f"Total speakers : {len(speakers)}")
print(f"Total clips    : {total_clips:,}")
print(f"Total size     : {total_size_mb/1024:.1f} GB")
print()

# M4 has 10 cores — with 6 workers, ~3s per clip / 6 parallel = 0.5s effective
import multiprocessing
cpu_count  = multiprocessing.cpu_count()
workers    = max(1, cpu_count - 2)
eff_secs   = total_clips * 3.0 / workers
eff_hours  = eff_secs / 3600
print(f"Your M4 has {cpu_count} logical cores")
print(f"With {workers} workers: ~{eff_hours:.1f} hours estimated")
print()
print("Run: python preprocess_devA.py")
print("=" * 50)
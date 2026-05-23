from pathlib import Path

parts = [
    "vox2_dev_mp4_partaa~",
    "vox2_dev_mp4_partab~",
    "vox2_dev_mp4_partac~",
    "vox2_dev_mp4_partad~",
    "vox2_dev_mp4_partae~",
    "vox2_dev_mp4_partaf~",
    "vox2_dev_mp4_partag~",
    "vox2_dev_mp4_partah~",
]

total = 0
for p in parts:
    path = Path(f"E:\\{p}\\dev\\mp4")
    if path.exists():
        clips = list(path.rglob("*.mp4"))
        print(f"{p}: {len(clips):,} clips")
        total += len(clips)
    else:
        print(f"{p}: NOT FOUND")

print(f"\nTOTAL: {total:,} clips")
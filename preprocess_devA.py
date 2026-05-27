import os
import cv2
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# ── Config ────────────────────────────────────────────────────
PARTS = [
    "vox2_dev_mp4_partaa~",
    "vox2_dev_mp4_partab~",
    "vox2_dev_mp4_partac~",
    "vox2_dev_mp4_partad~",
    "vox2_dev_mp4_partae~",
    "vox2_dev_mp4_partaf~",
    "vox2_dev_mp4_partag~",
    "vox2_dev_mp4_partah~",
]

PROJECT_DIR  = "E:\\visiovox"
OUTPUT_DIR   = "D:\\visiovox_data\\devA_processed"
MANIFEST_OUT = "D:\\visiovox_data\\devA_manifest.csv"
FFMPEG       = r"C:\Users\Admin\Downloads\ffmpeg-8.1.1-essentials_build\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"

SAMPLE_RATE    = 16000
NUM_LIP_FRAMES = 25
CROP_SIZE      = 112
DNN_CONF       = 0.4
NUM_WORKERS    = 16
BATCH_SIZE     = 100

PROTOTXT   = os.path.join(PROJECT_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(PROJECT_DIR, "res10_300x300_ssd.caffemodel")


def process_clip(args):
    clip_path, clip_id, speaker, output_dir = args

    audio_dir = os.path.join(output_dir, "audio", speaker)
    lips_dir  = os.path.join(output_dir, "lips", speaker,
                             Path(clip_path).stem)
    os.makedirs(audio_dir, exist_ok=True)
    audio_out = os.path.join(audio_dir, f"{Path(clip_path).stem}.wav")

    try:
        # ── Audio ─────────────────────────────────────────────
        if not os.path.exists(audio_out):
            result = subprocess.run(
                [FFMPEG, "-y", "-i", clip_path,
                 "-ac", "1", "-ar", str(SAMPLE_RATE), audio_out],
                capture_output=True, timeout=30
            )
            if result.returncode != 0 or not os.path.exists(audio_out):
                return None, clip_id

        # ── Lips ──────────────────────────────────────────────
        existing = list(Path(lips_dir).glob("*.jpg")) if os.path.exists(lips_dir) else []
        if len(existing) >= 10:
            return {
                "clip_id":    clip_id,
                "speaker":    speaker,
                "audio_path": audio_out,
                "lips_dir":   lips_dir,
                "clip_path":  clip_path,
            }, None

        net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return None, clip_id

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 5:
            cap.release()
            return None, clip_id

        start_f = int(total * 0.1)
        end_f   = int(total * 0.9)
        if end_f - start_f < NUM_LIP_FRAMES:
            start_f, end_f = 0, total

        indices = [
            int(start_f + (end_f - start_f) * i / NUM_LIP_FRAMES)
            for i in range(NUM_LIP_FRAMES)
        ]

        os.makedirs(lips_dir, exist_ok=True)
        saved = 0

        for fn in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

            blob = cv2.dnn.blobFromImage(
                cv2.resize(enhanced, (300, 300)),
                1.0, (300, 300), (104, 177, 123)
            )
            net.setInput(blob)
            dets = net.forward()

            best_conf, best_box = 0, None
            for i in range(dets.shape[2]):
                conf = float(dets[0, 0, i, 2])
                if conf > best_conf and conf >= DNN_CONF:
                    best_conf = conf
                    box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)
                    if x2 > x1 and y2 > y1:
                        best_box = (x1, y1, x2, y2)

            if best_box is None:
                continue

            x1, y1, x2, y2 = best_box
            lip_cx = int((x1 + x2) / 2)
            lip_cy = int(y1 + (y2 - y1) * 0.75)
            half   = CROP_SIZE // 2
            lx1 = max(0, lip_cx - half); lx2 = min(w, lip_cx + half)
            ly1 = max(0, lip_cy - half); ly2 = min(h, lip_cy + half)

            if lx2 - lx1 != CROP_SIZE or ly2 - ly1 != CROP_SIZE:
                continue

            lip = cv2.cvtColor(frame[ly1:ly2, lx1:lx2], cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(lips_dir, f"frame_{saved:05d}.jpg"), lip)
            saved += 1

        cap.release()

        if saved < 10:
            if os.path.exists(audio_out):
                os.remove(audio_out)
            return None, clip_id

        return {
            "clip_id":    clip_id,
            "speaker":    speaker,
            "audio_path": audio_out,
            "lips_dir":   lips_dir,
            "clip_path":  clip_path,
        }, None

    except Exception as e:
        return None, f"{clip_id}: {str(e)}"


def main():
    os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
    manifest_path = Path(MANIFEST_OUT)

    # Resume
    if manifest_path.exists() and manifest_path.stat().st_size > 10:
        try:
            existing = pd.read_csv(manifest_path)
            done_ids = set(existing["clip_id"].tolist())
            records  = existing.to_dict("records")
            print(f"Resuming -- {len(done_ids):,} clips already done")
        except Exception:
            done_ids = set()
            records  = []
    else:
        done_ids = set()
        records  = []

    # Collect all clips
    all_clips = []
    for part in PARTS:
        dev_path = Path(f"E:\\{part}\\dev\\mp4")
        if not dev_path.exists():
            print(f"WARNING: {dev_path} not found, skipping")
            continue
        for spk_dir in sorted(dev_path.iterdir()):
            if not spk_dir.is_dir():
                continue
            for yt_dir in sorted(spk_dir.iterdir()):
                if not yt_dir.is_dir():
                    continue
                for clip in sorted(yt_dir.glob("*.mp4")):
                    clip_id = f"{spk_dir.name}_{yt_dir.name}_{clip.stem}"
                    if clip_id not in done_ids:
                        all_clips.append((
                            str(clip),
                            clip_id,
                            spk_dir.name,
                            OUTPUT_DIR,
                        ))

    print(f"Total clips : {len(all_clips) + len(done_ids):,}")
    print(f"Already done: {len(done_ids):,}")
    print(f"Remaining   : {len(all_clips):,}")
    print(f"Workers     : {NUM_WORKERS}")
    print()

    if not all_clips:
        print("Nothing to do.")
        return

    success = 0
    failed  = 0

    with tqdm(total=len(all_clips), desc="Processing") as pbar:
        # Process in batches to avoid overwhelming Windows
        for batch_start in range(0, len(all_clips), BATCH_SIZE):
            batch = all_clips[batch_start: batch_start + BATCH_SIZE]

            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(process_clip, args): args[1]
                           for args in batch}
                for future in concurrent.futures.as_completed(futures):
                    record, error = future.result()
                    if record:
                        records.append(record)
                        success += 1
                    else:
                        failed += 1

                    pbar.update(1)
                    pbar.set_postfix(ok=success, fail=failed)

            # Save after every batch
            if (batch_start // BATCH_SIZE) % 5 == 0:
                pd.DataFrame(records).to_csv(manifest_path, index=False)
                tqdm.write(f"  Saved -- success={success:,} failed={failed:,}")

    # Final save
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    print()
    print("=" * 55)
    print("Processing complete!")
    print(f"  Successful : {success:,}")
    print(f"  Failed     : {failed:,}")
    print(f"  Total done : {len(records):,}")
    print("=" * 55)


if __name__ == "__main__":
    main()
import os
import sys
import cv2
import subprocess
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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
OUTPUT_DIR   = os.path.join(PROJECT_DIR, "data\\devA_processed")
MANIFEST_OUT = os.path.join(PROJECT_DIR, "data\\devA_manifest.csv")

SAMPLE_RATE    = 16000
NUM_LIP_FRAMES = 25
CROP_SIZE      = 112
DNN_CONF       = 0.4
FACE_PAD       = 30
NUM_WORKERS    = 12

PROTOTXT   = os.path.join(PROJECT_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(PROJECT_DIR, "res10_300x300_ssd.caffemodel")

# ── Download face detector models if needed ───────────────────
def ensure_face_detector():
    import urllib.request
    if not os.path.exists(PROTOTXT):
        print("Downloading face detector prototxt...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            PROTOTXT
        )
    if not os.path.exists(CAFFEMODEL):
        print("Downloading face detector caffemodel...")
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            CAFFEMODEL
        )
    print("Face detector models ready.")

# ── Audio extraction using system ffmpeg ──────────────────────
def extract_audio(video_path: str, audio_out: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-ac", "1", "-ar", str(SAMPLE_RATE),
             audio_out],
            capture_output=True,
            timeout=30
        )
        return result.returncode == 0 and os.path.exists(audio_out)
    except Exception:
        return False

# ── Lip frame extraction ──────────────────────────────────────
def extract_lip_frames(video_path: str, lips_out_dir: str,
                       n_frames: int = NUM_LIP_FRAMES) -> bool:
    # Each worker process creates its own net (can't share across processes)
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 5:
        cap.release()
        return False

    # Sample n_frames evenly across middle 80% of clip
    start_frame = int(total * 0.1)
    end_frame   = int(total * 0.9)
    if end_frame - start_frame < n_frames:
        start_frame = 0
        end_frame   = total

    sample_indices = [
        int(start_frame + (end_frame - start_frame) * i / n_frames)
        for i in range(n_frames)
    ]

    os.makedirs(lips_out_dir, exist_ok=True)
    saved = 0

    for frame_num in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect face
        h, w = frame.shape[:2]
        lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        blob = cv2.dnn.blobFromImage(
            cv2.resize(enhanced, (300, 300)),
            1.0, (300, 300), (104, 177, 123)
        )
        net.setInput(blob)
        dets = net.forward()

        best_conf = 0
        best_box  = None
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

        # Extract lip crop
        x1, y1, x2, y2 = best_box
        lip_cx = int((x1 + x2) / 2)
        lip_cy = int(y1 + (y2 - y1) * 0.75)
        half   = CROP_SIZE // 2
        lx1 = max(0, lip_cx - half); lx2 = min(w, lip_cx + half)
        ly1 = max(0, lip_cy - half); ly2 = min(h, lip_cy + half)

        if lx2 - lx1 != CROP_SIZE or ly2 - ly1 != CROP_SIZE:
            continue

        lip = cv2.cvtColor(frame[ly1:ly2, lx1:lx2], cv2.COLOR_BGR2GRAY)
        cv2.imwrite(
            os.path.join(lips_out_dir, f"frame_{saved:05d}.jpg"),
            lip
        )
        saved += 1

    cap.release()
    return saved >= 10  # need at least 10 frames

# ── Worker function (runs in separate process) ────────────────
def process_clip(args):
    clip_path, clip_id, speaker, output_dir = args

    audio_dir = os.path.join(output_dir, "audio", speaker)
    lips_dir  = os.path.join(output_dir, "lips", speaker,
                              Path(clip_path).stem)

    os.makedirs(audio_dir, exist_ok=True)
    audio_out = os.path.join(audio_dir, f"{Path(clip_path).stem}.wav")

    try:
        # Extract audio
        if not os.path.exists(audio_out):
            ok = extract_audio(clip_path, audio_out)
            if not ok:
                return None, clip_id

        # Extract lip frames
        lip_frames = list(Path(lips_dir).glob("*.jpg")) if os.path.exists(lips_dir) else []
        if len(lip_frames) < 10:
            ok = extract_lip_frames(clip_path, lips_dir)
            if not ok:
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

# ── Main ──────────────────────────────────────────────────────
def main():
    ensure_face_detector()

    os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
    manifest_path = Path(MANIFEST_OUT)

    # Resume from existing manifest
    if manifest_path.exists() and manifest_path.stat().st_size > 10:
        try:
            existing = pd.read_csv(manifest_path)
            done_ids = set(existing["clip_id"].tolist())
            records  = existing.to_dict("records")
            print(f"Resuming -- {len(done_ids):,} clips already done")
        except Exception:
            print("Manifest corrupt -- starting fresh")
            done_ids = set()
            records  = []
    else:
        done_ids = set()
        records  = []

    # Collect clips from ALL 8 parts
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

    print(f"Total clips across all 8 parts : {len(all_clips) + len(done_ids):,}")
    print(f"Already done                   : {len(done_ids):,}")
    print(f"Remaining to process           : {len(all_clips):,}")
    print(f"Workers                        : {NUM_WORKERS}")
    print()

    if not all_clips:
        print("Nothing to do -- all clips already processed.")
        return

    success = 0
    failed  = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_clip, args): args[1]
                   for args in all_clips}

        with tqdm(total=len(all_clips), desc="Processing") as pbar:
            for future in as_completed(futures):
                record, error = future.result()
                if record:
                    records.append(record)
                    success += 1
                else:
                    failed += 1
                    if failed <= 5:
                        tqdm.write(f"  SKIP: {error}")

                if (success + failed) % 500 == 0:
                    pd.DataFrame(records).to_csv(manifest_path, index=False)
                    tqdm.write(
                        f"  Progress saved -- "
                        f"success={success:,} failed={failed:,}"
                    )

                pbar.update(1)

    # Final save
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    print()
    print("=" * 55)
    print(f"Processing complete!")
    print(f"  Successful : {success:,}")
    print(f"  Failed     : {failed:,}")
    print(f"  Total done : {len(records):,}")
    print(f"  Manifest   : {MANIFEST_OUT}")
    print()
    print("Next: python create_mixtures_devA.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
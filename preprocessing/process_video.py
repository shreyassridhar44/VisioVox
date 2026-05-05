# preprocessing/process_video.py
#
# v3 — Fixed for Windows + Python 3.13 + mediapipe 0.10.x
#
# Architecture:
#   1. OpenCV DNN (ResNet-SSD)  → detects face bounding boxes in full frame
#   2. FaceTracker              → assigns stable face_id (0,1,2) across frames
#   3. MediaPipe FaceLandmarker → run on the TIGHT face crop (not full frame)
#                                 → gives precise lip landmark coordinates
#   4. Lip crop saved as 112×112 grayscale
#   5. Face thumbnail saved once per speaker for the frontend
#
# Why this approach:
#   MediaPipe FaceLandmarker silently returns 0 detections on full dark frames
#   on Windows / Python 3.13 (known regression in mediapipe 0.10.x).
#   Running it on a tight 300×300 face crop fixes this completely.

import os
import cv2
import ffmpeg
import urllib.request
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from utils.logger import get_logger

logger = get_logger("VideoProcessor")

# ── Constants ────────────────────────────────────────────────────────────────
MAX_FACES   = 3
CROP_SIZE   = 112    # lip crop square (must match model input)
MATCH_DIST  = 120    # px — max distance to consider "same face" across frames
DNN_CONF    = 0.4    # minimum DNN detection confidence
FACE_PAD    = 30     # pixels of padding around DNN face box for landmark crop

# ── File paths (auto-downloaded if missing) ──────────────────────────────────
PROTOTXT_PATH    = Path("deploy.prototxt")
CAFFEMODEL_PATH  = Path("res10_300x300_ssd.caffemodel")
MP_MODEL_PATH    = "models/face_landmarker.task"

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_file(path: Path, url: str, label: str):
    if not path.exists():
        logger.info(f"Downloading {label}...")
        urllib.request.urlretrieve(url, path)
        logger.info(f"  Saved to {path}")


_ensure_file(PROTOTXT_PATH,   PROTOTXT_URL,   "face detector prototxt")
_ensure_file(CAFFEMODEL_PATH, CAFFEMODEL_URL, "face detector caffemodel")
_ensure_file(Path(MP_MODEL_PATH), MP_MODEL_URL, "MediaPipe face landmarker")


# ── OpenCV DNN face detector (ResNet-SSD, robust on dark frames) ─────────────
_dnn_net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))


def _detect_faces_dnn(frame: np.ndarray) -> list:
    """
    Returns list of (x1, y1, x2, y2) face bounding boxes in the original frame.
    Applies CLAHE contrast enhancement before detection for dark frames.
    """
    h, w = frame.shape[:2]

    # CLAHE on luminance channel to handle dark frames
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    frame_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_enhanced, (300, 300)),
        1.0, (300, 300), (104, 177, 123)
    )
    _dnn_net.setInput(blob)
    detections = _dnn_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < DNN_CONF:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes


# ── MediaPipe lip landmarker (runs on tight face crop) ───────────────────────
_mp_base    = mp_python.BaseOptions(model_asset_path=MP_MODEL_PATH)
_mp_options = vision.FaceLandmarkerOptions(
    base_options=_mp_base,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
_mp_detector = vision.FaceLandmarker.create_from_options(_mp_options)


def _get_lip_crop_from_box(frame: np.ndarray, box: tuple):
    """
    Given a full frame and a DNN face box (x1,y1,x2,y2):
      1. Crops the face region (with padding) to a 300x300 image
      2. Runs MediaPipe FaceLandmarker on that crop
      3. Maps lip landmarks back to original frame coordinates
      4. Returns (lip_gray_112x112, face_center) tuple

    If MediaPipe fails on the crop, falls back to DNN box geometry
    to estimate lip position. face_center always returned for tracker.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    face_center = (cx, cy)

    # Padded crop for MediaPipe
    px1 = max(0, x1 - FACE_PAD)
    py1 = max(0, y1 - FACE_PAD)
    px2 = min(w, x2 + FACE_PAD)
    py2 = min(h, y2 + FACE_PAD)

    face_crop = frame[py1:py2, px1:px2]
    if face_crop.size == 0:
        return None, face_center

    crop_h, crop_w = face_crop.shape[:2]
    face_300 = cv2.resize(face_crop, (300, 300))

    rgb_crop = cv2.cvtColor(face_300, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
    results  = _mp_detector.detect(mp_image)

    if not results.face_landmarks:
        # Fallback: estimate lip as lower-third of face box
        lip_cx = int(cx)
        lip_cy = int(y1 + (y2 - y1) * 0.75)
        half   = CROP_SIZE // 2
        lx1 = max(0, lip_cx - half); lx2 = min(w, lip_cx + half)
        ly1 = max(0, lip_cy - half); ly2 = min(h, lip_cy + half)
        if lx2 - lx1 == CROP_SIZE and ly2 - ly1 == CROP_SIZE:
            lip_crop = frame[ly1:ly2, lx1:lx2]
            lip_gray = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
            return lip_gray, face_center
        return None, face_center

    # Map normalised landmark coords from 300x300 crop → original frame
    lm = results.face_landmarks[0]

    def lm_to_orig(lm_point):
        fx = lm_point.x * crop_w + px1
        fy = lm_point.y * crop_h + py1
        return fx, fy

    xl, _  = lm_to_orig(lm[78])
    xr, _  = lm_to_orig(lm[308])
    _, yt  = lm_to_orig(lm[13])
    _, yb  = lm_to_orig(lm[14])

    lip_cx = int((xl + xr) / 2)
    lip_cy = int((yt + yb) / 2)
    half   = CROP_SIZE // 2

    lx1 = lip_cx - half; lx2 = lip_cx + half
    ly1 = lip_cy - half; ly2 = lip_cy + half

    if not (lx1 >= 0 and lx2 <= w and ly1 >= 0 and ly2 <= h):
        return None, face_center

    lip_region = frame[ly1:ly2, lx1:lx2]
    lip_gray   = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
    return lip_gray, face_center


# ── Face tracker ─────────────────────────────────────────────────────────────
class FaceTracker:
    """Assigns stable face_id (0,1,2) to face centers across frames."""

    def __init__(self, max_faces=MAX_FACES, match_dist=MATCH_DIST):
        self.max_faces  = max_faces
        self.match_dist = match_dist
        self.known      = {}   # face_id -> last center
        self._next_id   = 0

    @staticmethod
    def _dist(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def update(self, centers):
        assignment = {}
        used_ids   = set()
        for idx, center in enumerate(centers):
            best_id, best_dist = None, float("inf")
            for fid, kc in self.known.items():
                if fid in used_ids:
                    continue
                d = self._dist(center, kc)
                if d < best_dist and d < self.match_dist:
                    best_dist, best_id = d, fid
            if best_id is None:
                if self._next_id < self.max_faces:
                    best_id = self._next_id
                    self._next_id += 1
                else:
                    continue
            assignment[idx]     = best_id
            used_ids.add(best_id)
            self.known[best_id] = center
        return assignment

    @property
    def num_faces_seen(self):
        return self._next_id


# ── Thumbnail helper ──────────────────────────────────────────────────────────
def _save_thumbnail(frame, box, path):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    pad  = 20
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(w, x2+pad); y2 = min(h, y2+pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        cv2.imwrite(str(path), cv2.resize(crop, (112, 112)))


# ── Public functions ──────────────────────────────────────────────────────────
def extract_audio(video_path: Path, output_audio_path: Path,
                  sample_rate: int = 16000) -> bool:
    """Extracts 16 kHz mono WAV. Unchanged from v1."""
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(output_audio_path), ac=1, ar=sample_rate)
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except ffmpeg.Error as e:
        logger.error(
            f"FFmpeg failed on {video_path.name}: "
            f"{e.stderr.decode('utf-8') if e.stderr else str(e)}"
        )
        return False


def extract_faces_and_lips(
    video_path:    Path,
    lips_base_dir: Path,
    thumbs_dir:    Path,
) -> int:
    """
    Processes every frame with DNN face detection + MediaPipe lip landmarks.
    Saves lip crops to lips_base_dir/face_{id}/frame_NNNNN.jpg
    Saves thumbnails to thumbs_dir/face_{id}.jpg
    Returns number of distinct faces tracked.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open: {video_path}")
        return 0

    tracker      = FaceTracker()
    thumb_saved  = set()
    frame_counts = {}
    frame_idx    = 0

    lips_base_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = _detect_faces_dnn(frame)

        if boxes:
            centers    = [((x1+x2)/2.0, (y1+y2)/2.0) for x1,y1,x2,y2 in boxes]
            assignment = tracker.update(centers)

            for box_idx, face_id in assignment.items():
                box = boxes[box_idx]
                lip_gray, _ = _get_lip_crop_from_box(frame, box)

                if lip_gray is not None:
                    lip_dir = lips_base_dir / f"face_{face_id}"
                    lip_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(lip_dir / f"frame_{frame_idx:05d}.jpg"),
                        lip_gray
                    )
                    frame_counts[face_id] = frame_counts.get(face_id, 0) + 1

                if face_id not in thumb_saved:
                    _save_thumbnail(
                        frame, box,
                        thumbs_dir / f"face_{face_id}.jpg"
                    )
                    thumb_saved.add(face_id)
                    logger.info(f"Saved thumbnail → face_{face_id}.jpg")

        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info(f"  Processed {frame_idx} frames...")

    cap.release()
    num_faces = tracker.num_faces_seen
    logger.info(
        f"Done: {num_faces} face(s) over {frame_idx} frames. "
        f"Lip frames: {frame_counts}"
    )
    return num_faces


def process_video_for_inference(
    video_path:  Path,
    output_root: Path,
) -> dict:
    """
    Entry point for the FastAPI backend.
    Creates mixed_audio.wav, lips/face_N/, thumbs/face_N.jpg
    Returns dict with paths and face metadata.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    audio_path = output_root / "mixed_audio.wav"
    lips_dir   = output_root / "lips"
    thumbs_dir = output_root / "thumbs"

    logger.info(f"[1/2] Extracting audio from {video_path.name}...")
    if not extract_audio(video_path, audio_path):
        raise RuntimeError(f"Audio extraction failed for {video_path}")

    logger.info("[2/2] Detecting faces and extracting lip crops...")
    num_faces = extract_faces_and_lips(video_path, lips_dir, thumbs_dir)

    if num_faces == 0:
        raise RuntimeError(
            "No faces detected. Ensure speakers are clearly visible."
        )

    logger.info(f"Pre-processing complete. Faces: {num_faces}")
    return {
        "audio_path": str(audio_path),
        "lips_dir":   str(lips_dir),
        "thumbs_dir": str(thumbs_dir),
        "num_faces":  num_faces,
        "face_ids":   list(range(num_faces)),
    }


def process_dataset(videos_dir: str, audio_dir: str, lips_dir: str):
    """Dataset preprocessing for training. Single-speaker videos → face_0."""
    video_path_obj = Path(videos_dir)
    audio_path_obj = Path(audio_dir)
    lips_path_obj  = Path(lips_dir)

    audio_path_obj.mkdir(parents=True, exist_ok=True)
    lips_path_obj.mkdir(parents=True, exist_ok=True)

    videos = list(video_path_obj.glob("*.mp4"))
    logger.info(f"Processing {len(videos)} videos...")

    for idx, video_path in enumerate(videos, 1):
        video_id = video_path.stem
        logger.info(f"[{idx}/{len(videos)}] {video_id}")
        extract_audio(video_path, audio_path_obj / f"{video_id}.wav")
        extract_faces_and_lips(
            video_path,
            lips_path_obj / video_id,
            lips_path_obj / "thumbs",
        )

    logger.info("Dataset processing complete.")


if __name__ == "__main__":
    process_dataset("data/raw/videos", "data/raw/audio", "data/processed/lips")
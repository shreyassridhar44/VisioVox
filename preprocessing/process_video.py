# preprocessing/process_video.py
import os
import cv2
import ffmpeg
import urllib.request
import numpy as np
from pathlib import Path

# MediaPipe Tasks API (Required for Python 3.13+)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from utils.logger import get_logger

logger = get_logger("VideoProcessor")

# --- Auto-Download MediaPipe Model ---
MODEL_PATH = "models/face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    logger.info("Downloading MediaPipe Face Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    logger.info("Model downloaded successfully!")

# --- Initialize MediaPipe ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def extract_audio(video_path: Path, output_audio_path: Path, sample_rate: int = 16000):
    """Extracts 16kHz mono audio from a video file."""
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
        logger.error(f"FFmpeg failed on {video_path.name}: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return False

def extract_and_crop_lips(video_path: Path, output_dir: Path, crop_size: int = 112):
    """Reads video, detects lips via Tasks API, crops 112x112 region, saves grayscale."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_idx = 0
    success = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB and format for MediaPipe Tasks API
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = detector.detect(mp_image)
        
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            h, w, _ = frame.shape
            
            # Calculate mouth center (13=Upper lip, 14=Lower lip, 78=Left corner, 308=Right corner)
            x_left = landmarks[78].x * w
            x_right = landmarks[308].x * w
            y_top = landmarks[13].y * h
            y_bottom = landmarks[14].y * h
            
            center_x = int((x_left + x_right) / 2)
            center_y = int((y_top + y_bottom) / 2)
            
            # Define crop boundaries
            half_size = crop_size // 2
            y1, y2 = center_y - half_size, center_y + half_size
            x1, x2 = center_x - half_size, center_x + half_size
            
            if y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
                lip_crop = frame[y1:y2, x1:x2]
                lip_gray = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
                
                frame_path = output_dir / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(frame_path), lip_gray)
            else:
                logger.warning(f"Frame {frame_idx} in {video_path.name}: Lips out of bounds.")
        else:
             logger.debug(f"Frame {frame_idx} in {video_path.name}: No face detected.")
             
        frame_idx += 1
        
    cap.release()
    return success

def process_dataset(videos_dir: str, audio_dir: str, lips_dir: str):
    video_path_obj = Path(videos_dir)
    audio_path_obj = Path(audio_dir)
    lips_path_obj = Path(lips_dir)
    
    audio_path_obj.mkdir(parents=True, exist_ok=True)
    lips_path_obj.mkdir(parents=True, exist_ok=True)
    
    videos = list(video_path_obj.glob("*.mp4"))
    total_videos = len(videos)
    logger.info(f"Starting processing pipeline for {total_videos} videos...")
    
    for idx, video_path in enumerate(videos, 1):
        video_id = video_path.stem 
        audio_output = audio_path_obj / f"{video_id}.wav"
        lip_output_dir = lips_path_obj / video_id
        
        logger.info(f"[{idx}/{total_videos}] Processing {video_id}...")
        
        extract_audio(video_path, audio_output)
        extract_and_crop_lips(video_path, lip_output_dir)
        
    logger.info("Dataset processing complete!")

if __name__ == "__main__":
    VIDEOS_DIR = "data/raw/videos"
    AUDIO_DIR = "data/raw/audio"
    LIPS_DIR = "data/processed/lips"
    
    process_dataset(VIDEOS_DIR, AUDIO_DIR, LIPS_DIR)
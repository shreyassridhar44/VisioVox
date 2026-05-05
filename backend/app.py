# backend/app.py
#
# VisioVox FastAPI Backend
#
# Endpoints:
#   POST /upload                      → upload video, start preprocessing
#   GET  /status/{job_id}             → poll job progress
#   GET  /thumbnails/{job_id}/{face_id} → face thumbnail image
#   POST /process/{job_id}            → run inference + ASR for a face_id
#   GET  /audio/{job_id}/{face_id}    → serve isolated WAV
#   GET  /captions/{job_id}/{face_id} → serve .vtt caption file
#   GET  /video/{job_id}              → serve original uploaded video
#   GET  /health                      → server + model status
#
# Run with:
#   cd VisioVox-main
#   uvicorn backend.app:app --reload --port 8000

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uuid
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.job_manager import (
    create_job, get_job, update_job,
    STATUS_PREPROCESSING, STATUS_SEPARATING,
    STATUS_TRANSCRIBING, STATUS_DONE, STATUS_ERROR,
)
from backend.caption import generate_captions
from preprocessing.process_video import process_video_for_inference
from inference.inference import VisioVoxSeparator
from utils.logger import get_logger

logger = get_logger("Backend")

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/visiovox_epoch_20.pth")
JOBS_DIR        = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VisioVox API",
    description="Targeted speaker extraction from multi-speaker videos",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor   = ThreadPoolExecutor(max_workers=2)
_separator: VisioVoxSeparator = None


@app.on_event("startup")
async def load_model():
    global _separator
    if not CHECKPOINT_PATH.exists():
        logger.warning(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "/process endpoint will fail until the checkpoint is placed there."
        )
        return
    loop       = asyncio.get_event_loop()
    _separator = await loop.run_in_executor(
        _executor,
        lambda: VisioVoxSeparator(str(CHECKPOINT_PATH))
    )
    logger.info("VisioVox model loaded and ready.")


# ── Background workers ────────────────────────────────────────────────────────

def _run_preprocessing(job_id: str):
    """Runs in a thread pool. Preprocesses the uploaded video."""
    job = get_job(job_id)
    update_job(job_id,
               status=STATUS_PREPROCESSING,
               progress_msg="Detecting faces and extracting lip crops...")
    try:
        result = process_video_for_inference(
            video_path=Path(job.video_path),
            output_root=Path(job.output_dir),
        )
        update_job(
            job_id,
            status="ready",
            progress_msg=f"Preprocessing complete. Found {result['num_faces']} speaker(s).",
            num_faces=result["num_faces"],
            face_ids=result["face_ids"],
            audio_path=result["audio_path"],
            lips_dir=result["lips_dir"],
            thumbs_dir=result["thumbs_dir"],
        )
        logger.info(f"[{job_id[:8]}] Preprocessing done. Faces: {result['num_faces']}")
    except Exception as e:
        logger.error(f"[{job_id[:8]}] Preprocessing failed: {e}")
        update_job(job_id,
                   status=STATUS_ERROR,
                   progress_msg="Preprocessing failed.",
                   error_detail=str(e))


def _run_separation_and_asr(job_id: str, face_id: int):
    """Runs in a thread pool. Separates audio and generates captions for one face."""
    job     = get_job(job_id)
    out_dir = Path(job.output_dir)

    if _separator is None:
        update_job(job_id,
                   status=STATUS_ERROR,
                   error_detail="Model not loaded. Check checkpoint path.")
        return

    lips_dir = out_dir / "lips" / f"face_{face_id}"
    wav_out  = out_dir / f"speaker_{face_id}.wav"
    vtt_out  = out_dir / f"speaker_{face_id}.vtt"

    try:
        # Inference
        update_job(job_id,
                   status=STATUS_SEPARATING,
                   progress_msg=f"Isolating speaker {face_id}...")
        _separator.separate(
            mixed_audio_path=job.audio_path,
            lips_dir=str(lips_dir),
            output_path=str(wav_out),
            face_id=face_id,
        )

        # ASR
        update_job(job_id,
                   status=STATUS_TRANSCRIBING,
                   progress_msg=f"Generating captions for speaker {face_id}...")
        generate_captions(str(wav_out), str(vtt_out))

        # Store result
        results          = dict(job.speaker_results)
        results[face_id] = {"wav": str(wav_out), "vtt": str(vtt_out)}
        update_job(job_id,
                   status=STATUS_DONE,
                   progress_msg=f"Speaker {face_id} ready.",
                   speaker_results=results)
        logger.info(f"[{job_id[:8]}] face_{face_id} done.")

    except Exception as e:
        logger.error(f"[{job_id[:8]}] face_{face_id} failed: {e}")
        update_job(job_id,
                   status=STATUS_ERROR,
                   progress_msg=f"Processing failed for speaker {face_id}.",
                   error_detail=str(e))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file. Starts preprocessing immediately in the background.
    Returns job_id — poll GET /status/{job_id} to track progress.
    """
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(400, "Unsupported file type. Upload MP4, MOV, AVI, MKV, or WebM.")

    # ── Single uuid created here, used everywhere ──────────────────────────
    job_id  = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    video_path = job_dir / f"input{Path(file.filename).suffix}"
    contents   = await file.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    logger.info(f"[{job_id[:8]}] Uploaded: {file.filename} ({len(contents)//1024} KB)")

    # Pass the SAME job_id to create_job
    create_job(job_id=job_id, video_path=str(video_path), output_dir=str(job_dir))

    # Start preprocessing in background
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_preprocessing, job_id)

    return {
        "job_id":  job_id,
        "message": "Upload successful. Preprocessing started.",
        "status":  STATUS_PREPROCESSING,
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Poll this to track job progress."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    response = {
        "job_id":       job_id,
        "status":       job.status,
        "progress_msg": job.progress_msg,
        "num_faces":    job.num_faces,
        "face_ids":     job.face_ids,
    }
    if job.status == STATUS_ERROR:
        response["error"] = job.error_detail
    if job.status == STATUS_DONE:
        response["processed_faces"] = list(job.speaker_results.keys())

    return response


@app.get("/thumbnails/{job_id}/{face_id}")
async def get_thumbnail(job_id: str, face_id: int):
    """Returns the face thumbnail JPEG for a given speaker."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job.thumbs_dir is None:
        raise HTTPException(400, "Preprocessing not complete yet.")

    thumb_path = Path(job.thumbs_dir) / f"face_{face_id}.jpg"
    if not thumb_path.exists():
        raise HTTPException(404, f"Thumbnail for face_{face_id} not found.")

    return FileResponse(str(thumb_path), media_type="image/jpeg")


@app.post("/process/{job_id}")
async def process_speaker(job_id: str, face_id: int):
    """
    Triggers inference + ASR for a specific speaker.
    Call when the user clicks a speaker card.
    Poll GET /status/{job_id} for progress.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job.status not in ("ready", STATUS_DONE):
        raise HTTPException(400, f"Job not ready. Current status: {job.status}")
    if face_id not in job.face_ids:
        raise HTTPException(400, f"face_id {face_id} not in this job. Available: {job.face_ids}")

    if face_id in job.speaker_results:
        return {"job_id": job_id, "face_id": face_id,
                "message": "Already processed.", "status": STATUS_DONE}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_separation_and_asr, job_id, face_id)

    return {
        "job_id":  job_id,
        "face_id": face_id,
        "message": f"Processing speaker {face_id}. Poll /status/{job_id} for progress.",
        "status":  STATUS_SEPARATING,
    }


@app.get("/audio/{job_id}/{face_id}")
async def get_audio(job_id: str, face_id: int):
    """Streams the isolated speaker WAV."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if face_id not in job.speaker_results:
        raise HTTPException(400, f"Speaker {face_id} not processed yet.")

    wav_path = Path(job.speaker_results[face_id]["wav"])
    if not wav_path.exists():
        raise HTTPException(404, "Audio file missing on disk.")

    return FileResponse(str(wav_path), media_type="audio/wav",
                        headers={"Accept-Ranges": "bytes"})


@app.get("/captions/{job_id}/{face_id}")
async def get_captions(job_id: str, face_id: int):
    """Returns the .vtt subtitle file for a speaker."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if face_id not in job.speaker_results:
        raise HTTPException(400, f"Speaker {face_id} not processed yet.")

    vtt_path = Path(job.speaker_results[face_id]["vtt"])
    if not vtt_path.exists():
        raise HTTPException(404, "Caption file missing on disk.")

    return FileResponse(str(vtt_path), media_type="text/vtt",
                        headers={"Content-Type": "text/vtt"})


@app.get("/video/{job_id}")
async def get_video(job_id: str):
    """Streams the original uploaded video (muted in the frontend player)."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")

    video_path = Path(job.video_path)
    if not video_path.exists():
        raise HTTPException(404, "Video file missing on disk.")

    media_types = {
        ".mp4": "video/mp4", ".mov": "video/quicktime",
        ".avi": "video/avi", ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    media_type = media_types.get(video_path.suffix.lower(), "video/mp4")

    return FileResponse(str(video_path), media_type=media_type,
                        headers={"Accept-Ranges": "bytes"})


@app.get("/health")
async def health():
    """Quick server + model status check."""
    return {
        "status":       "ok",
        "model_loaded": _separator is not None,
        "checkpoint":   str(CHECKPOINT_PATH),
    }
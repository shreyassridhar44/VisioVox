# backend/job_manager.py

import uuid
from dataclasses import dataclass, field
from typing import Optional

STATUS_UPLOADED      = "uploaded"
STATUS_PREPROCESSING = "preprocessing"
STATUS_SEPARATING    = "separating"
STATUS_TRANSCRIBING  = "transcribing"
STATUS_DONE          = "done"
STATUS_ERROR         = "error"


@dataclass
class Job:
    job_id:          str
    video_path:      str
    output_dir:      str
    status:          str = STATUS_UPLOADED
    progress_msg:    str = "Video uploaded successfully"
    num_faces:       int = 0
    face_ids:        list = field(default_factory=list)
    audio_path:      Optional[str] = None
    lips_dir:        Optional[str] = None
    thumbs_dir:      Optional[str] = None
    speaker_results: dict = field(default_factory=dict)
    error_detail:    Optional[str] = None


_jobs: dict[str, Job] = {}


def create_job(job_id: str, video_path: str, output_dir: str) -> Job:
    """Creates a job with an EXPLICITLY provided job_id (no internal uuid generation)."""
    job = Job(job_id=job_id, video_path=video_path, output_dir=output_dir)
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> Optional[Job]:
    job = _jobs.get(job_id)
    if job is None:
        return None
    for key, value in kwargs.items():
        setattr(job, key, value)
    return job
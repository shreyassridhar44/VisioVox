# preprocessing/sample_data.py
import os
import shutil
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("DataSampler")

def sample_debug_dataset(source_root: str, dest_root: str, num_speakers: int = 20, videos_per_speaker: int = 5):
    """
    Crawls the extracted VoxCeleb directory, extracts a specific number of speakers 
    and videos, and copies them to the project data directory.
    """
    source_path = Path(source_root)
    dest_path = Path(dest_root)
    
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_path}")
        return

    logger.info(f"Scanning {source_path} for speakers...")
    
    # VoxCeleb structure is typically: mp4/idXXXXX/YouTube_ID/00001.mp4
    # Find all speaker directories (folders starting with 'id')
    speaker_dirs = [d for d in source_path.iterdir() if d.is_dir() and d.name.startswith("id")]
    
    if not speaker_dirs:
        logger.error("No speaker directories found. Check your source path.")
        return

    logger.info(f"Found {len(speaker_dirs)} total speakers. Sampling {num_speakers}...")
    
    speakers_processed = 0
    videos_copied = 0

    for speaker_dir in speaker_dirs:
        if speakers_processed >= num_speakers:
            break
            
        speaker_id = speaker_dir.name
        
        # Find all .mp4 files for this speaker recursively
        mp4_files = list(speaker_dir.rglob("*.mp4"))
        
        if len(mp4_files) < videos_per_speaker:
            logger.warning(f"Speaker {speaker_id} only has {len(mp4_files)} videos. Skipping to find a better one.")
            continue
            
        # Take the first N videos
        selected_videos = mp4_files[:videos_per_speaker]
        
        for video_path in selected_videos:
            # Create a flat, unique filename: idXXXXX_YouTubeID_00001.mp4
            # video_path.parts usually looks like: (..., 'id10270', 'x6uYq...', '00001.mp4')
            youtube_id = video_path.parent.name
            clip_id = video_path.name
            
            new_filename = f"{speaker_id}_{youtube_id}_{clip_id}"
            dest_file = dest_path / new_filename
            
            shutil.copy2(video_path, dest_file)
            videos_copied += 1
            
        speakers_processed += 1
        logger.info(f"Processed speaker {speakers_processed}/{num_speakers}: {speaker_id}")

    logger.info(f"Sampling complete! Copied {videos_copied} videos to {dest_path}")

if __name__ == "__main__":
    # UPDATE THIS PATH to wherever your extracted mp4 folder is
    SOURCE_DIRECTORY = r"C:\Users\SHREYAS S\Downloads\vox2_dev_mp4_partaa~\dev\mp4" 
    DESTINATION_DIRECTORY = "data/raw/videos"
    
    sample_debug_dataset(
        source_root=SOURCE_DIRECTORY,
        dest_root=DESTINATION_DIRECTORY,
        num_speakers=20,
        videos_per_speaker=5
    )
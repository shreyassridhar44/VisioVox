# backend/caption.py
#
# Generates speaker-specific captions from an isolated WAV file.
# Uses OpenAI Whisper (runs fully locally, no API key needed).
#
# Output format: WebVTT (.vtt) — the standard subtitle format supported
# natively by the HTML5 <track> element, which our React player uses.
#
# VTT format example:
#   WEBVTT
#
#   00:00:01.000 --> 00:00:04.200
#   Hello and welcome to the panel discussion.
#
#   00:00:04.800 --> 00:00:07.500
#   Thank you for having me.

import whisper
import os
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("Caption")

# Load Whisper once at module level — avoids reloading for every request.
# "base" model: ~140MB, good speed/accuracy balance for a prototype.
# Upgrade to "small" or "medium" on the supercomputer.
logger.info("Loading Whisper base model...")
_whisper_model = whisper.load_model("base")
logger.info("Whisper ready.")


def _seconds_to_vtt_timestamp(seconds: float) -> str:
    """Converts float seconds to VTT timestamp format HH:MM:SS.mmm"""
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def generate_captions(wav_path: str, output_vtt_path: str) -> str:
    """
    Transcribes wav_path using Whisper and writes a .vtt subtitle file.

    Args:
        wav_path        : path to the isolated speaker WAV (16kHz mono)
        output_vtt_path : where to save the .vtt file

    Returns:
        output_vtt_path (str) — path to the generated .vtt file
    """
    logger.info(f"Transcribing: {wav_path}")

    result = _whisper_model.transcribe(
        wav_path,
        language="en",        # set to None for auto-detect if needed
        word_timestamps=False, # segment-level is enough for captions
        verbose=False,
    )

    segments = result.get("segments", [])
    logger.info(f"  Got {len(segments)} caption segments")

    # Build VTT content
    lines = ["WEBVTT", ""]   # VTT header + blank line

    for i, seg in enumerate(segments):
        start = _seconds_to_vtt_timestamp(seg["start"])
        end   = _seconds_to_vtt_timestamp(seg["end"])
        text  = seg["text"].strip()

        if not text:
            continue

        lines.append(f"{i + 1}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")   # blank line between cues

    vtt_content = "\n".join(lines)

    Path(output_vtt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write(vtt_content)

    logger.info(f"  Saved captions → {output_vtt_path}")
    return output_vtt_path
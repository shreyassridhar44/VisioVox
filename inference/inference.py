# inference/inference.py
#
# UPGRADED from Colab inference.py — two key changes:
#
# 1. CHUNKED INFERENCE (main upgrade)
#    The original processed the entire audio as one STFT → model → iSTFT pass.
#    This breaks on real videos because:
#      a) Audio longer than the training clips (3s) produces spectrograms with
#         different Time dimensions, causing shape mismatches in the U-Net decoder.
#      b) Long audio eats GPU/CPU memory.
#    Fix: Split audio into overlapping 3-second chunks, run model on each,
#    reconstruct with overlap-add using a Hann window (standard OLA technique).
#
# 2. BEST LIP FRAME SELECTION PER CHUNK
#    The original used a single static lip image for the whole audio.
#    Now we pick the lip frame whose timestamp is closest to each chunk's midpoint.
#    This means the visual cue actually matches the audio being processed.
#
# Everything else (STFT params, smart noise gate, model loading) is IDENTICAL
# to your Colab inference.py so the trained weights work without any changes.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import librosa
import cv2
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path

from models.fusion_model import VisioVox

# ── STFT parameters — MUST match training exactly ────────────────────────────
SAMPLE_RATE = 16000
N_FFT       = 510
HOP_LENGTH  = 160
WIN_LENGTH  = 400

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_DURATION  = 3.0     # seconds — matches your training clip length
OVERLAP_RATIO   = 0.5     # 50% overlap between chunks
CHUNK_SAMPLES   = int(CHUNK_DURATION * SAMPLE_RATE)          # 48000 samples
HOP_SAMPLES     = int(CHUNK_SAMPLES * (1 - OVERLAP_RATIO))   # 24000 samples

# ── Video frame rate (used for lip frame ↔ audio time mapping) ───────────────
VIDEO_FPS = 30.0


class VisioVoxSeparator:
    def __init__(self, checkpoint_path: str,
                 device: str = None):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")

        self.model = VisioVox().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=True)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.window = torch.hann_window(WIN_LENGTH).to(self.device)
        print("Model loaded successfully.")

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _load_audio(self, path: str) -> np.ndarray:
        wave, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return wave.astype(np.float32)

    def _chunk_audio(self, wave: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """
        Splits waveform into overlapping 3-second chunks.
        Pads the last chunk with zeros if needed.

        Returns list of (chunk_waveform, start_sample_index).
        """
        chunks = []
        start  = 0
        while start < len(wave):
            end   = start + CHUNK_SAMPLES
            chunk = wave[start:end]
            if len(chunk) < CHUNK_SAMPLES:
                # Pad final chunk
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            chunks.append((chunk, start))
            start += HOP_SAMPLES
        return chunks

    def _audio_to_spectrogram(self, wave_np: np.ndarray):
        """
        np.ndarray (CHUNK_SAMPLES,) → magnitude [1,1,F,T], phase [1,F,T]
        F = N_FFT//2 + 1 = 256
        """
        wave_t = torch.tensor(wave_np).unsqueeze(0).to(self.device)
        stft   = torch.stft(
            wave_t, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH, window=self.window,
            return_complex=True
        )
        magnitude = torch.abs(stft).unsqueeze(1)   # [1,1,F,T]
        phase     = torch.angle(stft)               # [1,F,T]
        return magnitude, phase

    def _spectrogram_to_audio(self, magnitude, phase) -> np.ndarray:
        """
        magnitude [1,1,F,T], phase [1,F,T] → np.ndarray (CHUNK_SAMPLES,)
        """
        separated_stft = torch.polar(magnitude.squeeze(1), phase)  # [1,F,T]
        wave = torch.istft(
            separated_stft, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH, window=self.window,
            length=CHUNK_SAMPLES
        )
        return wave.squeeze(0).cpu().numpy()

    # ── Lip helpers ───────────────────────────────────────────────────────────

    def _load_lip_frames(self, lips_dir: Path) -> dict[int, np.ndarray]:
        """
        Loads all lip frame images from lips_dir/frame_NNNNN.jpg.
        Returns {frame_number: normalised float32 array (112,112)}.
        """
        frames = {}
        for p in sorted(lips_dir.glob("frame_*.jpg")):
            try:
                frame_num = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames[frame_num] = img.astype(np.float32) / 255.0
        return frames

    def _get_best_lip_frame(
        self,
        lip_frames: dict[int, np.ndarray],
        chunk_start_sample: int,
    ) -> np.ndarray:
        """
        Picks the lip frame closest in time to the midpoint of the current chunk.

        chunk_start_sample: sample index in the full waveform where this chunk starts.
        Returns a (112,112) float32 array.
        """
        chunk_mid_sec   = (chunk_start_sample + CHUNK_SAMPLES / 2) / SAMPLE_RATE
        target_frame    = int(chunk_mid_sec * VIDEO_FPS)

        if not lip_frames:
            return np.zeros((112, 112), dtype=np.float32)

        # Find the frame number with minimum distance to target
        best_frame_num  = min(lip_frames.keys(),
                              key=lambda f: abs(f - target_frame))
        return lip_frames[best_frame_num]

    def _lip_to_tensor(self, lip_np: np.ndarray) -> torch.Tensor:
        """(112,112) float32 → [1,1,112,112] tensor on device."""
        return (
            torch.tensor(lip_np)
            .unsqueeze(0).unsqueeze(0)
            .to(self.device)
        )

    # ── Smart noise gate (identical to Colab version) ─────────────────────────

    @staticmethod
    def _apply_noise_gate(mask: torch.Tensor) -> torch.Tensor:
        sharpened  = mask ** 1.2
        smart_mask = torch.where(
            sharpened < 0.20,
            sharpened * 0.01,
            sharpened
        )
        return smart_mask

    # ── Main separation function ──────────────────────────────────────────────

    def separate(
        self,
        mixed_audio_path: str,
        lips_dir: str,
        output_path: str,
        face_id: int = 0,
    ) -> str:
        """
        Full pipeline: mixed audio + lip frames directory → isolated speaker WAV.

        Args:
            mixed_audio_path : path to the mixed 16kHz mono WAV
            lips_dir         : path to the face's lip frames directory
                               e.g. test_output/lips/face_0/
            output_path      : where to save the isolated WAV
            face_id          : informational, used for logging only

        Returns:
            output_path (str)
        """
        print(f"\n[face_{face_id}] Loading audio: {mixed_audio_path}")
        wave = self._load_audio(mixed_audio_path)
        print(f"  Audio: {len(wave)/SAMPLE_RATE:.2f}s "
              f"({len(wave)} samples)")

        print(f"[face_{face_id}] Loading lip frames from: {lips_dir}")
        lips_dir_path = Path(lips_dir)
        lip_frames    = self._load_lip_frames(lips_dir_path)
        print(f"  Loaded {len(lip_frames)} lip frames")

        if not lip_frames:
            raise RuntimeError(
                f"No lip frames found in {lips_dir}. "
                "Run process_video_for_inference first."
            )

        # ── Chunked inference with overlap-add ────────────────────────────────
        chunks          = self._chunk_audio(wave)
        output_buffer   = np.zeros(len(wave) + CHUNK_SAMPLES, dtype=np.float32)
        weight_buffer   = np.zeros_like(output_buffer)
        hann_win        = np.hanning(CHUNK_SAMPLES).astype(np.float32)

        print(f"[face_{face_id}] Running inference on {len(chunks)} chunks...")

        with torch.no_grad():
            for i, (chunk_wave, start_sample) in enumerate(chunks):

                # 1. Get the lip frame closest to this chunk's midpoint
                lip_np     = self._get_best_lip_frame(lip_frames, start_sample)
                lips_tensor = self._lip_to_tensor(lip_np)

                # 2. Convert chunk to spectrogram
                mixed_mag, mixed_phase = self._audio_to_spectrogram(chunk_wave)

                # 3. Model forward pass → mask
                predicted_mask = self.model(mixed_mag, lips_tensor)
                predicted_mask = torch.clamp(predicted_mask, 0.0, 1.0)

                # 4. Smart noise gate
                smart_mask = self._apply_noise_gate(predicted_mask)

                # 5. Apply mask to magnitude
                separated_mag = mixed_mag.squeeze(1) * smart_mask.squeeze(1)

                # 6. iSTFT → waveform chunk
                chunk_out = self._spectrogram_to_audio(separated_mag, mixed_phase)

                # 7. Overlap-add with Hann window
                end_sample = start_sample + CHUNK_SAMPLES
                output_buffer[start_sample:end_sample] += chunk_out * hann_win
                weight_buffer[start_sample:end_sample] += hann_win

                if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                    print(f"  Chunk {i+1}/{len(chunks)} done")

        # ── Normalise by overlap weights and trim to original length ──────────
        # Avoid division by zero at edges
        weight_buffer = np.where(weight_buffer < 1e-8, 1.0, weight_buffer)
        output_wave   = output_buffer / weight_buffer
        output_wave   = output_wave[:len(wave)]   # trim padding

        # ── Save ──────────────────────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, output_wave, SAMPLE_RATE)
        print(f"[face_{face_id}] Saved isolated audio → {output_path}")

        return output_path


# ── CLI entry point (for quick testing) ───────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisioVox inference")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--mix",        required=True,
                        help="Path to mixed audio WAV")
    parser.add_argument("--lips_dir",   required=True,
                        help="Path to lip frames directory (face_N/)")
    parser.add_argument("--out",        required=True,
                        help="Output WAV path")
    parser.add_argument("--face_id",    type=int, default=0,
                        help="Face ID (for logging)")
    args = parser.parse_args()

    separator = VisioVoxSeparator(checkpoint_path=args.checkpoint)
    separator.separate(
        mixed_audio_path=args.mix,
        lips_dir=args.lips_dir,
        output_path=args.out,
        face_id=args.face_id,
    )
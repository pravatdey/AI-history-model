"""
EchoMimic HF Space Avatar Engine

Generates realistic lip-synced talking head videos using EchoMimic
via free HuggingFace Space API (fffiloni/EchoMimic).

EchoMimic uses a diffusion-based approach for high-quality facial animation.
Handles long audio by chunking into segments, generating clips, and concatenating.

HF Space: https://huggingface.co/spaces/fffiloni/EchoMimic
"""

import os
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EchoMimicConfig:
    """Configuration for EchoMimic HF Space generation."""
    hf_space_id: str = "fffiloni/EchoMimic"

    # EchoMimic generation settings
    width: int = 512
    height: int = 512
    seed: int = 420
    facemask_dilation: float = 0.1
    facecrop_dilation: float = 0.5
    context_frames: int = 12
    context_overlap: int = 3
    cfg: float = 2.5
    steps: int = 30
    sample_rate: int = 16000
    fps: int = 24

    # Chunking for long audio
    max_chunk_seconds: float = 30.0
    max_retries: int = 3
    retry_delay: int = 15


class EchoMimicEngine:
    """
    Generates talking head videos using EchoMimic HF Space.

    For long audio (>30s), splits into chunks, generates each,
    and concatenates with ffmpeg.
    """

    def __init__(self, config: EchoMimicConfig = None):
        self.config = config or EchoMimicConfig()
        logger.info(f"EchoMimicEngine initialized: space={self.config.hf_space_id}")

    def is_available(self) -> bool:
        """Check if gradio_client is installed."""
        try:
            from gradio_client import Client
            return True
        except ImportError:
            logger.debug("gradio_client not installed")
            return False

    def generate(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
    ) -> dict:
        """
        Generate a talking head video using EchoMimic HF Space.

        Returns:
            dict with keys: success, video_path, duration, error
        """
        if not self.is_available():
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": "gradio_client not installed",
            }

        audio_duration = self._get_audio_duration(audio_path)
        if audio_duration <= 0:
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": f"Could not determine audio duration: {audio_path}",
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if audio_duration <= self.config.max_chunk_seconds:
            return self._generate_single(audio_path, image_path, output_path)

        return self._generate_chunked(audio_path, image_path, output_path, audio_duration)

    def _generate_single(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
    ) -> dict:
        """Generate a single video clip via EchoMimic HF Space."""
        try:
            from gradio_client import Client, handle_file

            logger.info(f"Connecting to HF Space: {self.config.hf_space_id}")
            client = Client(self.config.hf_space_id)

            # Ensure audio is WAV
            wav_audio = self._ensure_wav(audio_path)
            if not wav_audio:
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": f"Failed to convert audio to WAV: {audio_path}",
                }

            logger.info("Submitting to EchoMimic HF Space...")
            start_time = time.time()

            # Calculate max frames from audio duration
            audio_dur = self._get_audio_duration(wav_audio)
            max_length = int(audio_dur * self.config.fps) + 50  # extra frames buffer

            result = client.predict(
                handle_file(os.path.abspath(image_path)),       # Reference Image
                handle_file(os.path.abspath(wav_audio)),        # Input Audio
                self.config.width,                               # Width
                self.config.height,                              # Height
                max_length,                                      # Length (frames)
                self.config.seed,                                # Seed
                self.config.facemask_dilation,                   # Facemask Dilation
                self.config.facecrop_dilation,                   # Facecrop Dilation
                self.config.context_frames,                      # Context Frames
                self.config.context_overlap,                     # Context Overlap
                self.config.cfg,                                 # CFG
                self.config.steps,                               # Steps
                self.config.sample_rate,                         # Sample Rate
                self.config.fps,                                 # FPS
                "cuda",                                          # Device
                api_name="/generate_video",
            )

            elapsed = time.time() - start_time
            logger.info(f"EchoMimic generation completed in {elapsed:.0f}s")

            video_file = str(result) if result else None
            if not video_file or not Path(video_file).exists():
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": f"EchoMimic returned no video. Result: {result}",
                }

            shutil.copy2(video_file, output_path)
            duration = self._get_video_duration(output_path)
            logger.info(f"EchoMimic video saved: {output_path} ({duration:.1f}s)")

            return {
                "success": True,
                "video_path": output_path,
                "duration": duration,
                "error": None,
            }

        except Exception as e:
            logger.error(f"EchoMimic generation failed: {e}", exc_info=True)
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": f"EchoMimic error: {e}",
            }

    def _generate_chunked(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        total_duration: float,
    ) -> dict:
        """Split long audio into chunks, generate each, concatenate."""
        chunk_dur = self.config.max_chunk_seconds
        num_chunks = math.ceil(total_duration / chunk_dur)
        logger.info(
            f"Generating chunked video: {total_duration:.0f}s audio → "
            f"{num_chunks} chunks of {chunk_dur:.0f}s"
        )

        chunk_dir = Path(output_path).parent / "echomimic_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_videos = []
        for i in range(num_chunks):
            start = i * chunk_dur
            end = min(start + chunk_dur, total_duration)
            chunk_audio = str(chunk_dir / f"chunk_{i:03d}.wav")

            # Extract audio chunk with ffmpeg
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start), "-to", str(end),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                chunk_audio,
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)

            if not Path(chunk_audio).exists():
                logger.warning(f"Failed to extract chunk {i}")
                continue

            chunk_video = str(chunk_dir / f"chunk_{i:03d}.mp4")
            logger.info(f"Generating chunk {i+1}/{num_chunks} ({start:.0f}s - {end:.0f}s)")

            result = None
            for attempt in range(1, self.config.max_retries + 1):
                result = self._generate_single(chunk_audio, image_path, chunk_video)
                if result["success"]:
                    break
                logger.warning(f"Chunk {i+1} attempt {attempt} failed: {result['error']}")
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)

            if result and result["success"]:
                chunk_videos.append(chunk_video)
            else:
                logger.error(f"Failed to generate chunk {i+1} after retries")

        if not chunk_videos:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": "All chunks failed",
            }

        # Concatenate chunks with ffmpeg
        if len(chunk_videos) == 1:
            shutil.copy2(chunk_videos[0], output_path)
        else:
            concat_file = str(chunk_dir / "concat.txt")
            with open(concat_file, "w") as f:
                for v in chunk_videos:
                    f.write(f"file '{v}'\n")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_file, "-c", "copy", output_path,
            ]
            subprocess.run(cmd, capture_output=True, timeout=300)

        shutil.rmtree(chunk_dir, ignore_errors=True)

        if Path(output_path).exists():
            duration = self._get_video_duration(output_path)
            return {
                "success": True,
                "video_path": output_path,
                "duration": duration,
                "error": None,
            }
        return {
            "success": False, "video_path": "", "duration": 0,
            "error": "FFmpeg concatenation failed",
        }

    def _ensure_wav(self, audio_path: str) -> Optional[str]:
        """Convert audio to WAV if needed."""
        if audio_path.lower().endswith(".wav"):
            return audio_path
        wav_path = audio_path.rsplit(".", 1)[0] + "_echomimic.wav"
        try:
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                wav_path,
            ]
            subprocess.run(cmd, capture_output=True, timeout=120, check=True)
            return wav_path if Path(wav_path).exists() else None
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None

    @staticmethod
    def _get_audio_duration(path: str) -> float:
        """Get audio duration in seconds."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(path)
            return len(audio) / 1000.0
        except Exception:
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries",
                     "format=duration", "-of", "csv=p=0", path],
                    capture_output=True, text=True, timeout=30,
                )
                return float(result.stdout.strip())
            except Exception:
                return 0.0

    @staticmethod
    def _get_video_duration(path: str) -> float:
        """Get video duration in seconds."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries",
                 "format=duration", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

"""
Wan2.2 S2V (Speech-to-Video) Avatar Engine

Generates cinema-quality realistic talking head videos using Wan2.2-S2V
via free HuggingFace Space API (Wan-AI/Wan2.2-S2V).

Wan2.2-S2V is a 14B parameter model by Alibaba that produces the most
realistic AI talking head videos available for free — comparable to HeyGen/Synthesia.

Features:
  - Cinema-grade facial animation from a single photo + audio
  - Natural head movements, expressions, and eye contact
  - High-quality lip sync
  - Free via HuggingFace Space (ZeroGPU)

HF Space: https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V
GitHub: https://github.com/Wan-Video/Wan2.2
"""

import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Wan2SVConfig:
    """Configuration for Wan2.2 S2V HF Space generation."""
    hf_space_id: str = "Wan-AI/Wan2.2-S2V"
    hf_token: str = ""

    # Generation settings
    resolution: str = "480P"          # 480P or 720P (480P is faster, 720P better quality)
    sample_steps: int = 25            # Diffusion steps (more = better, slower)
    cfg_scale: float = 3.5            # Guidance scale
    seed: int = -1                    # -1 for random
    fps: int = 24

    # Chunking for long audio
    max_chunk_seconds: float = 30.0   # ZeroGPU has timeout limits
    max_retries: int = 3
    retry_delay: float = 15.0

    # Multiple spaces to try (fallback if primary is down)
    hf_space_ids: list = field(default_factory=lambda: [
        "Wan-AI/Wan2.2-S2V",
        "hf-audio/Wan2.2-S2V",
    ])


class Wan2SVEngine:
    """
    Generates talking head videos using Wan2.2 S2V HF Space.

    For long audio (>30s), splits into chunks, generates each,
    and concatenates with ffmpeg.
    """

    def __init__(self, config: Optional[Wan2SVConfig] = None):
        self.config = config or Wan2SVConfig()
        logger.info(f"Wan2SVEngine initialized: space={self.config.hf_space_id}")

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
        Generate a talking head video using Wan2.2 S2V HF Space.

        Returns:
            dict with keys: success, video_path, duration, error
        """
        if not self.is_available():
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": "gradio_client not installed. Run: pip install gradio_client",
            }

        audio_duration = self._get_audio_duration(audio_path)
        if audio_duration <= 0:
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": f"Could not determine audio duration: {audio_path}",
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if audio_duration <= self.config.max_chunk_seconds:
            return self._generate_with_retry(audio_path, image_path, output_path)

        return self._generate_chunked(audio_path, image_path, output_path, audio_duration)

    # ──────────────────────────────────────────────────────────────────────
    #  Single generation with retry across multiple spaces
    # ──────────────────────────────────────────────────────────────────────

    def _generate_with_retry(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
    ) -> dict:
        """Generate video with retries, cycling through available spaces."""
        spaces = self.config.hf_space_ids or [self.config.hf_space_id]
        last_error = ""

        for space_id in spaces:
            logger.info(f"Trying Wan2.2 S2V HF Space: {space_id}")
            for attempt in range(1, self.config.max_retries + 1):
                logger.info(f"  Attempt {attempt}/{self.config.max_retries}")
                result = self._generate_single(audio_path, image_path, output_path, space_id)
                if result["success"]:
                    return result
                last_error = result.get("error", "Unknown error")
                logger.warning(f"  Attempt {attempt} failed: {last_error}")
                if "GPU quota" in last_error or "exceeded" in last_error.lower():
                    logger.warning("  GPU quota exceeded — trying next space")
                    break
                if "not defined" in last_error or "TypeError" in last_error:
                    logger.info(f"  Skipping remaining retries for {space_id} (API mismatch)")
                    break
                if attempt < self.config.max_retries:
                    logger.info(f"  Retrying in {self.config.retry_delay:.0f}s...")
                    time.sleep(self.config.retry_delay)

        return {
            "success": False, "video_path": "", "duration": 0,
            "error": f"Wan2.2 S2V failed on all spaces: {last_error}",
        }

    def _generate_single(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        space_id: Optional[str] = None,
    ) -> dict:
        """Generate a single video clip via Wan2.2 S2V HF Space API."""
        try:
            from gradio_client import Client, handle_file

            space = space_id or self.config.hf_space_id
            hf_token = self.config.hf_token or os.environ.get("HF_TOKEN", "")

            logger.info(f"Connecting to Wan2.2 S2V: {space} (auth={'yes' if hf_token else 'no'})")
            client = Client(space, token=hf_token or None)

            # Ensure audio is WAV format
            wav_audio = self._ensure_wav(audio_path)
            if not wav_audio:
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": f"Failed to convert audio to WAV: {audio_path}",
                }

            logger.info(
                f"Submitting to Wan2.2 S2V (resolution={self.config.resolution}, "
                f"steps={self.config.sample_steps}, cfg={self.config.cfg_scale})..."
            )
            start_time = time.time()

            # Wan2.2 S2V API: image + audio → video
            result = client.predict(
                image=handle_file(os.path.abspath(image_path)),
                audio=handle_file(os.path.abspath(wav_audio)),
                resolution=self.config.resolution,
                sample_steps=self.config.sample_steps,
                cfg_scale=self.config.cfg_scale,
                seed=self.config.seed,
                api_name="/generate",
            )

            elapsed = time.time() - start_time
            logger.info(f"Wan2.2 S2V generation completed in {elapsed:.0f}s")

            # Extract video path from result
            video_file = self._extract_video_path(result)
            if not video_file or not Path(video_file).exists():
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": f"Wan2.2 S2V returned no video. Result: {result}",
                }

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_file, output_path)

            duration = self._get_video_duration(output_path)
            logger.info(f"Wan2.2 S2V video saved: {output_path} ({duration:.1f}s)")

            return {
                "success": True,
                "video_path": output_path,
                "duration": duration,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Wan2.2 S2V generation failed: {e}", exc_info=True)
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": f"Wan2.2 S2V error: {e}",
            }
        finally:
            if 'wav_audio' in dir() and wav_audio != audio_path and Path(wav_audio).exists():
                try:
                    os.remove(wav_audio)
                except OSError:
                    pass

    # ──────────────────────────────────────────────────────────────────────
    #  Chunked generation for long audio
    # ──────────────────────────────────────────────────────────────────────

    def _generate_chunked(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        total_duration: float,
    ) -> dict:
        """Split long audio into chunks, generate video for each, concatenate."""
        chunk_dur = self.config.max_chunk_seconds
        n_chunks = math.ceil(total_duration / chunk_dur)

        logger.info(
            f"Wan2.2 S2V chunked generation: {total_duration:.1f}s audio -> "
            f"{n_chunks} chunks of {chunk_dur:.0f}s"
        )

        tmp_dir = Path(output_path).parent / "_wan2sv_chunks"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        chunk_videos = []
        failed_chunks = []

        try:
            for i in range(n_chunks):
                start = i * chunk_dur
                end = min(start + chunk_dur, total_duration)
                chunk_audio = str(tmp_dir / f"chunk_{i:04d}.wav")
                chunk_video = str(tmp_dir / f"chunk_{i:04d}.mp4")

                if not self._extract_audio_chunk(audio_path, chunk_audio, start, end):
                    logger.error(f"Failed to extract audio chunk {i}")
                    failed_chunks.append(i)
                    continue

                logger.info(f"Generating chunk {i + 1}/{n_chunks} ({start:.1f}s - {end:.1f}s)")

                result = self._generate_with_retry(chunk_audio, image_path, chunk_video)

                if result["success"]:
                    chunk_videos.append(chunk_video)
                    logger.info(f"Chunk {i + 1}/{n_chunks} complete")
                else:
                    logger.warning(f"Chunk {i + 1} failed: {result['error']}")
                    failed_chunks.append(i)

            if not chunk_videos:
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": "All Wan2.2 S2V chunks failed to generate",
                }

            if failed_chunks:
                logger.warning(
                    f"{len(failed_chunks)}/{n_chunks} chunks failed. "
                    f"Proceeding with {len(chunk_videos)} successful chunks."
                )

            logger.info(f"Concatenating {len(chunk_videos)} video chunks...")
            if not self._concatenate_videos(chunk_videos, output_path):
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": "Failed to concatenate video chunks",
                }

            # Re-attach original audio for perfect sync
            self._replace_audio(output_path, audio_path)

            duration = self._get_video_duration(output_path)
            logger.info(f"Wan2.2 S2V chunked video complete: {output_path} ({duration:.1f}s)")

            return {
                "success": True,
                "video_path": output_path,
                "duration": duration,
                "error": None,
            }

        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_video_path(result) -> Optional[str]:
        """Extract video file path from HF Space result (handles various formats)."""
        if isinstance(result, dict):
            return str(result.get("video", result.get("output", "")))
        elif isinstance(result, (list, tuple)):
            first = result[0]
            if isinstance(first, dict):
                return str(first.get("video", first.get("output", "")))
            return str(first)
        elif result:
            return str(result)
        return None

    def _ensure_wav(self, audio_path: str) -> Optional[str]:
        """Convert audio to WAV format if needed."""
        if audio_path.lower().endswith(".wav"):
            return audio_path
        try:
            wav_path = str(Path(audio_path).with_suffix("")) + "_wan2sv.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, check=True, timeout=120,
            )
            return wav_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Audio conversion to WAV failed: {e}")
            return None

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _extract_audio_chunk(
        self, audio_path: str, output_path: str, start: float, end: float
    ) -> bool:
        """Extract a chunk of audio using ffmpeg."""
        try:
            duration = end - start
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", audio_path,
                    "-ss", str(start),
                    "-t", str(duration),
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    output_path,
                ],
                capture_output=True, check=True, timeout=60,
            )
            return Path(output_path).exists()
        except Exception as e:
            logger.error(f"Audio chunk extraction failed: {e}")
            return False

    def _concatenate_videos(self, video_paths: list, output_path: str) -> bool:
        """Concatenate video files using ffmpeg concat demuxer."""
        try:
            list_file = Path(output_path).parent / "_wan2sv_concat_list.txt"
            with open(list_file, "w") as f:
                for vp in video_paths:
                    safe_path = str(Path(vp).resolve()).replace("\\", "/")
                    f.write(f"file '{safe_path}'\n")

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(list_file),
                    "-c", "copy",
                    output_path,
                ],
                capture_output=True, check=True, timeout=300,
            )

            list_file.unlink(missing_ok=True)
            return Path(output_path).exists()
        except Exception as e:
            logger.error(f"Video concatenation failed: {e}")
            return False

    def _replace_audio(self, video_path: str, audio_path: str) -> bool:
        """Replace video audio track with original audio for perfect sync."""
        try:
            tmp_out = video_path + ".tmp.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", audio_path,
                    "-c:v", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    tmp_out,
                ],
                capture_output=True, check=True, timeout=120,
            )
            shutil.move(tmp_out, video_path)
            return True
        except Exception as e:
            logger.error(f"Audio replacement failed: {e}")
            if Path(video_path + ".tmp.mp4").exists():
                try:
                    os.remove(video_path + ".tmp.mp4")
                except OSError:
                    pass
            return False

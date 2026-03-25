"""
SadTalker HF Space Avatar Engine

Generates talking head videos using SadTalker via free HuggingFace Space API.
Handles long audio by chunking into segments, generating clips, and concatenating.

HF Space: https://huggingface.co/spaces/vinthony/SadTalker
Max audio per call: 180s (we use 30s chunks for free ZeroGPU quota).
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
class SadTalkerHFConfig:
    """Configuration for SadTalker HF Space generation."""
    hf_space_id: str = "vinthony/SadTalker"

    # SadTalker settings
    preprocess: str = "crop"          # crop, resize, full, extcrop, extfull
    size_of_image: int = 512          # 256 or 512
    still_mode: bool = True           # Less head movement (good for teacher style)
    enhancer: bool = True             # GFPGAN face enhancement
    pose_style: int = 0               # 0-45
    exp_weight: float = 1.0           # Expression scale (0-3)
    batch_size: int = 1               # 1-10
    facerender: str = "facevid2vid"   # facevid2vid or pirender
    blink_every: bool = True          # Enable blinking

    # Chunking for long audio
    max_chunk_seconds: float = 30.0   # Free ZeroGPU quota safe limit


class SadTalkerHFEngine:
    """
    Generates talking head videos using SadTalker HF Space.

    For long audio (>30s), splits into chunks, generates each,
    and concatenates with ffmpeg.
    """

    def __init__(self, config: Optional[SadTalkerHFConfig] = None):
        self.config = config or SadTalkerHFConfig()

    def is_available(self) -> bool:
        """Check if SadTalker HF Space is usable (gradio_client installed)."""
        try:
            from gradio_client import Client
            return True
        except ImportError:
            logger.debug("gradio_client not installed. Run: pip install gradio_client")
            return False

    def generate(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
    ) -> dict:
        """
        Generate a talking head video using SadTalker HF Space.

        For long audio, chunks and concatenates automatically.

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

        # Short audio: generate directly
        if audio_duration <= self.config.max_chunk_seconds:
            return self._generate_single(audio_path, image_path, output_path)

        # Long audio: chunk, generate, concatenate
        return self._generate_chunked(audio_path, image_path, output_path, audio_duration)

    def _generate_single(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
    ) -> dict:
        """Generate a single video clip via the SadTalker HF Space API."""
        wav_audio = None
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

            logger.info("Submitting to SadTalker HF Space...")
            start_time = time.time()

            result = client.predict(
                source_image=handle_file(os.path.abspath(image_path)),
                driven_audio=handle_file(os.path.abspath(wav_audio)),
                preprocess_type=self.config.preprocess,
                is_still_mode=self.config.still_mode,
                enhancer=self.config.enhancer,
                batch_size=self.config.batch_size,
                size_of_image=self.config.size_of_image,
                pose_style=self.config.pose_style,
                facerender=self.config.facerender,
                exp_weight=self.config.exp_weight,
                use_ref_video=False,
                ref_info="pose",
                use_idle_mode=False,
                length_of_audio=5,
                blink_every=self.config.blink_every,
                api_name="/test",
            )

            elapsed = time.time() - start_time
            logger.info(f"SadTalker generation completed in {elapsed:.0f}s")

            # Result is a video file path
            video_file = str(result) if result else None

            if not video_file or not Path(video_file).exists():
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": f"SadTalker returned no video file. Result: {result}",
                }

            # Copy to expected output path
            shutil.copy2(video_file, output_path)

            duration = self._get_video_duration(output_path)
            logger.info(f"SadTalker video saved: {output_path} ({duration:.1f}s)")

            return {
                "success": True,
                "video_path": output_path,
                "duration": duration,
                "error": None,
            }

        except Exception as e:
            logger.error(f"SadTalker HF Space generation failed: {e}", exc_info=True)
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": f"SadTalker error: {e}",
            }
        finally:
            if wav_audio and wav_audio != audio_path and Path(wav_audio).exists():
                try:
                    os.remove(wav_audio)
                except OSError:
                    pass

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
            f"SadTalker chunked generation: {total_duration:.1f}s audio -> "
            f"{n_chunks} chunks of {chunk_dur:.0f}s"
        )

        tmp_dir = Path(output_path).parent / "_sadtalker_chunks"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        chunk_videos = []
        failed_chunks = []

        try:
            for i in range(n_chunks):
                start = i * chunk_dur
                end = min(start + chunk_dur, total_duration)
                chunk_audio = str(tmp_dir / f"chunk_{i:04d}.wav")
                chunk_video = str(tmp_dir / f"chunk_{i:04d}.mp4")

                # Extract audio chunk
                if not self._extract_audio_chunk(audio_path, chunk_audio, start, end):
                    logger.error(f"Failed to extract audio chunk {i}")
                    failed_chunks.append(i)
                    continue

                logger.info(
                    f"Generating chunk {i + 1}/{n_chunks} "
                    f"({start:.1f}s - {end:.1f}s)"
                )
                result = self._generate_single(chunk_audio, image_path, chunk_video)

                if result["success"]:
                    chunk_videos.append(chunk_video)
                    logger.info(f"Chunk {i + 1}/{n_chunks} complete")
                else:
                    logger.warning(f"Chunk {i + 1} failed: {result['error']}")
                    failed_chunks.append(i)

            if not chunk_videos:
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": "All chunks failed to generate",
                }

            if failed_chunks:
                logger.warning(
                    f"{len(failed_chunks)}/{n_chunks} chunks failed. "
                    f"Proceeding with {len(chunk_videos)} successful chunks."
                )

            # Concatenate all chunk videos
            logger.info(f"Concatenating {len(chunk_videos)} video chunks...")
            if not self._concatenate_videos(chunk_videos, output_path):
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": "Failed to concatenate video chunks",
                }

            # Re-attach original audio for perfect sync
            self._replace_audio(output_path, audio_path)

            duration = self._get_video_duration(output_path)
            logger.info(
                f"SadTalker chunked video complete: {output_path} ({duration:.1f}s)"
            )

            return {
                "success": True,
                "video_path": output_path,
                "duration": duration,
                "error": None,
            }

        finally:
            # Cleanup temp chunks
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────

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
            list_file = Path(output_path).parent / "_concat_list.txt"
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

    def _ensure_wav(self, audio_path: str) -> Optional[str]:
        """Convert audio to WAV format if needed."""
        if audio_path.lower().endswith(".wav"):
            return audio_path
        try:
            wav_path = str(Path(audio_path).with_suffix(".wav"))
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    wav_path,
                ],
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
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

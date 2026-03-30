"""
NVIDIA Audio2Face-3D Avatar Engine

Generates 3D digital human talking head videos using:
1. NVIDIA Audio2Face-3D Cloud API (audio → ARKit blendshapes via gRPC)
2. Blender headless rendering (blendshapes + 3D character → MP4 video)

Uses NVIDIA's built-in characters (Mark, Claire, James) — no external 3D model needed.
Cloud API is FREE with an NVIDIA API key from build.nvidia.com.

Pipeline:
    Audio (WAV) → A2F-3D gRPC API → Blendshapes CSV → Blender Render → PNG frames → FFmpeg → MP4
"""

import asyncio
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# NVIDIA character function IDs (with tongue animation)
CHARACTER_FUNCTION_IDS = {
    "mark": "8efc55f5-6f00-424e-afe9-26212cd2c630",
    "claire": "0961a6da-fb9e-4f2e-8491-247e5fd7bf8d",
    "james": "9327c39f-a361-4e02-bd72-e11b4c9b7b5e",
}

# Default face parameters per character
CHARACTER_FACE_PARAMS = {
    "mark": {
        "lowerFaceSmoothing": 0.0023,
        "upperFaceSmoothing": 0.003,
        "lowerFaceStrength": 1.3,
        "upperFaceStrength": 1.15,
        "faceMaskSoftness": 0.008,
        "faceMaskLevel": 0.8,
        "skinStrength": 1.1,
        "eyelidOpenOffset": 0.06,
        "lipOpenOffset": -0.03,
    },
    "claire": {
        "lowerFaceSmoothing": 0.006,
        "upperFaceSmoothing": 0.003,
        "lowerFaceStrength": 1.25,
        "upperFaceStrength": 1.15,
        "faceMaskSoftness": 0.008,
        "faceMaskLevel": 0.8,
        "skinStrength": 1.0,
        "eyelidOpenOffset": 0.0,
        "lipOpenOffset": 0.0,
    },
    "james": {
        "lowerFaceSmoothing": 0.0023,
        "upperFaceSmoothing": 0.003,
        "lowerFaceStrength": 1.3,
        "upperFaceStrength": 1.15,
        "faceMaskSoftness": 0.008,
        "faceMaskLevel": 0.8,
        "skinStrength": 1.1,
        "eyelidOpenOffset": 0.06,
        "lipOpenOffset": -0.03,
    },
}

# Default blendshape multipliers
DEFAULT_BS_MULTIPLIERS = {
    "EyeBlinkLeft": 1.0,
    "EyeBlinkRight": 1.0,
    "JawOpen": 1.0,
    "MouthClose": 0.2,
    "MouthFunnel": 1.0,
    "MouthPucker": 1.0,
    "MouthSmileLeft": 1.0,
    "MouthSmileRight": 1.0,
    "MouthFrownLeft": 1.0,
    "MouthFrownRight": 1.0,
    "BrowInnerUp": 1.0,
    "BrowDownLeft": 1.0,
    "BrowDownRight": 1.0,
}

# Default emotion config (neutral teacher style)
DEFAULT_EMOTIONS = {
    "emotion_1": {
        "time_code": 0.0,
        "emotions": {
            "amazement": 0.0,
            "anger": 0.0,
            "cheekiness": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "grief": 0.0,
            "joy": 0.3,
            "outofbreath": 0.0,
            "pain": 0.0,
            "sadness": 0.0,
        }
    }
}

DEFAULT_POST_PROCESSING = {
    "emotion_contrast": 1.0,
    "live_blend_coef": 0.85,
    "enable_preferred_emotion": True,
    "preferred_emotion_strength": 0.5,
    "emotion_strength": 0.7,
    "max_emotions": 3,
}


@dataclass
class Audio2Face3DConfig:
    """Configuration for Audio2Face-3D engine."""
    # NVIDIA API
    nvidia_api_key: str = ""
    character: str = "james"  # "mark", "claire", "james"
    grpc_uri: str = "grpc.nvcf.nvidia.com:443"

    # Blender
    blender_path: str = ""
    render_script: str = ""  # Path to render_avatar.py

    # Render settings
    render_resolution_x: int = 720
    render_resolution_y: int = 720
    render_fps: int = 30
    background_color: str = "0.12,0.15,0.22"  # Dark studio blue

    # Chunking for long audio
    max_chunk_seconds: float = 60.0  # A2F-3D cloud can handle longer audio
    max_retries: int = 3
    retry_delay: float = 10.0

    # Emotion
    emotion_joy: float = 0.3  # Base joy level for teacher character

    # Face parameters (override per character)
    face_params: dict = field(default_factory=dict)
    blendshape_multipliers: dict = field(default_factory=dict)


class Audio2Face3DEngine:
    """
    Generates 3D digital human talking head videos using NVIDIA Audio2Face-3D.

    Pipeline:
    1. Send audio to NVIDIA Cloud API via gRPC
    2. Receive ARKit blendshape animation data
    3. Render 3D character with Blender headless
    4. Compose final MP4 with FFmpeg (frames + audio)
    """

    def __init__(self, config: Optional[Audio2Face3DConfig] = None):
        self.config = config or Audio2Face3DConfig()

        # Resolve blender path
        if not self.config.blender_path:
            self.config.blender_path = os.getenv("BLENDER_PATH", "")

        # Resolve render script path
        if not self.config.render_script:
            self.config.render_script = str(
                Path(__file__).parent.parent.parent / "assets" / "blender" / "render_avatar.py"
            )

        # Resolve API key
        if not self.config.nvidia_api_key:
            self.config.nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")

    def is_available(self) -> bool:
        """Check if Audio2Face-3D engine is usable."""
        # Check API key
        if not self.config.nvidia_api_key:
            logger.debug("NVIDIA_API_KEY not set")
            return False

        # Check Blender
        blender = self.config.blender_path
        if not blender or not Path(blender).exists():
            logger.debug(f"Blender not found at: {blender}")
            return False

        # Check render script
        if not Path(self.config.render_script).exists():
            logger.debug(f"Render script not found: {self.config.render_script}")
            return False

        # Check gRPC dependencies
        try:
            import grpc
            from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
            return True
        except ImportError as e:
            logger.debug(f"gRPC/nvidia-ace dependencies missing: {e}")
            return False

    def generate(
        self,
        audio_path: str,
        output_path: str,
        image_path: str = None,  # Not used — we generate 3D character
    ) -> dict:
        """
        Generate a 3D digital human talking head video from audio.

        Args:
            audio_path: Path to input WAV audio file
            output_path: Path to save output MP4 video
            image_path: Ignored (3D character is generated by Blender)

        Returns:
            dict with keys: success, video_path, duration, error
        """
        if not self.is_available():
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": "Audio2Face-3D engine not available. Check NVIDIA_API_KEY and BLENDER_PATH.",
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        audio_duration = self._get_audio_duration(audio_path)
        if audio_duration <= 0:
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": f"Could not determine audio duration: {audio_path}",
            }

        logger.info(f"Audio2Face-3D: generating 3D avatar video ({audio_duration:.1f}s audio)")

        # Convert audio to PCM-16 WAV mono if needed
        wav_path = self._prepare_audio(audio_path)
        if not wav_path:
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": "Failed to convert audio to PCM-16 WAV",
            }

        try:
            # Step 1: Get blendshapes from NVIDIA Cloud API
            logger.info("Step 1/3: Sending audio to NVIDIA Audio2Face-3D API...")
            csv_path = self._get_blendshapes(wav_path)
            if not csv_path:
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": "Failed to get blendshapes from NVIDIA API",
                }

            # Step 2: Render with Blender
            logger.info("Step 2/3: Rendering 3D character with Blender...")
            frames_dir = self._render_with_blender(csv_path, wav_path, output_path)
            if not frames_dir:
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": "Blender rendering failed",
                }

            # Step 3: Compose final video (frames + audio → MP4)
            logger.info("Step 3/3: Composing final video with FFmpeg...")
            success = self._compose_video(frames_dir, wav_path, output_path)
            if not success:
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": "FFmpeg composition failed",
                }

            logger.info(f"Audio2Face-3D: 3D avatar video generated successfully: {output_path}")
            return {
                "success": True,
                "video_path": output_path,
                "duration": audio_duration,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Audio2Face-3D generation failed: {e}")
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": str(e),
            }

        finally:
            # Cleanup temp WAV if we created one
            if wav_path != audio_path and Path(wav_path).exists():
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    # ──────────────────────────────────────────────────────────────────────
    #  Step 1: NVIDIA Audio2Face-3D gRPC API
    # ──────────────────────────────────────────────────────────────────────

    def _get_blendshapes(self, wav_path: str) -> Optional[str]:
        """
        Send audio to NVIDIA Audio2Face-3D Cloud API and save blendshape CSV.
        Returns path to CSV file, or None on failure.
        """
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info(f"A2F-3D API attempt {attempt}/{self.config.max_retries}")
                csv_path = self._run_async(self._async_get_blendshapes(wav_path))
                if csv_path and Path(csv_path).exists():
                    return csv_path
            except Exception as e:
                logger.warning(f"A2F-3D API attempt {attempt} failed: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)

        return None

    def _run_async(self, coro):
        """Run an async coroutine, handling the case where an event loop is already running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Event loop already running — create a new thread to run the coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=300)
        else:
            return asyncio.run(coro)

    async def _async_get_blendshapes(self, wav_path: str) -> Optional[str]:
        """Async gRPC call to Audio2Face-3D API."""
        import grpc
        import numpy
        import scipy.io.wavfile
        import pandas
        from nvidia_ace.animation_data.v1_pb2 import AnimationData, AnimationDataStreamHeader
        from nvidia_ace.a2f.v1_pb2 import (
            AudioWithEmotion, EmotionPostProcessingParameters,
            FaceParameters, BlendShapeParameters,
        )
        from nvidia_ace.audio.v1_pb2 import AudioHeader
        from nvidia_ace.controller.v1_pb2 import AudioStream, AudioStreamHeader
        from nvidia_ace.emotion_with_timecode.v1_pb2 import EmotionWithTimeCode
        from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub

        character = self.config.character.lower()
        function_id = CHARACTER_FUNCTION_IDS.get(character, CHARACTER_FUNCTION_IDS["james"])
        face_params = self.config.face_params or CHARACTER_FACE_PARAMS.get(character, CHARACTER_FACE_PARAMS["james"])
        bs_multipliers = self.config.blendshape_multipliers or DEFAULT_BS_MULTIPLIERS

        # Build metadata for authentication
        metadata = [
            ("function-id", function_id),
            ("authorization", f"Bearer {self.config.nvidia_api_key}"),
        ]

        def metadata_callback(context, callback):
            callback(metadata, None)

        # Create secure gRPC channel
        ssl_creds = grpc.ssl_channel_credentials()
        auth_creds = grpc.metadata_call_credentials(metadata_callback)
        composite_creds = grpc.composite_channel_credentials(ssl_creds, auth_creds)
        channel = grpc.aio.secure_channel(self.config.grpc_uri, composite_creds)

        stub = A2FControllerServiceStub(channel)
        stream = stub.ProcessAudioStream()

        # Read audio
        samplerate, audio_data = scipy.io.wavfile.read(wav_path)
        logger.info(f"Audio: {len(audio_data)} samples, {samplerate}Hz, {len(audio_data)/samplerate:.1f}s")

        # Build emotion timecodes
        emotions_config = DEFAULT_EMOTIONS.copy()
        emotions_config["emotion_1"]["emotions"]["joy"] = self.config.emotion_joy
        emotion_tc_list = [
            EmotionWithTimeCode(
                emotion={**v["emotions"]},
                time_code=v["time_code"]
            ) for v in emotions_config.values()
        ]

        # Send header
        post_proc = DEFAULT_POST_PROCESSING.copy()
        header = AudioStream(
            audio_stream_header=AudioStreamHeader(
                audio_header=AudioHeader(
                    samples_per_second=samplerate,
                    bits_per_sample=16,
                    channel_count=1,
                    audio_format=AudioHeader.AUDIO_FORMAT_PCM,
                ),
                emotion_post_processing_params=EmotionPostProcessingParameters(**post_proc),
                face_params=FaceParameters(float_params=face_params),
                blendshape_params=BlendShapeParameters(
                    bs_weight_multipliers=bs_multipliers,
                    bs_weight_offsets={},
                ),
            )
        )
        await stream.write(header)

        # Send audio in 1-second chunks
        chunk_size = samplerate  # 1 second per chunk
        num_chunks = math.ceil(len(audio_data) / chunk_size)

        for i in range(num_chunks):
            chunk = audio_data[i * chunk_size: (i + 1) * chunk_size]
            if i == 0:
                # First chunk includes emotion timecodes
                await stream.write(
                    AudioStream(
                        audio_with_emotion=AudioWithEmotion(
                            audio_buffer=chunk.astype(numpy.int16).tobytes(),
                            emotions=emotion_tc_list,
                        )
                    )
                )
            else:
                await stream.write(
                    AudioStream(
                        audio_with_emotion=AudioWithEmotion(
                            audio_buffer=chunk.astype(numpy.int16).tobytes(),
                        )
                    )
                )

        # Signal end of audio
        await stream.write(AudioStream(end_of_audio=AudioStream.EndOfAudio()))

        # Read responses
        bs_names = []
        animation_frames = []

        while True:
            message = await stream.read()
            if message == grpc.aio.EOF:
                break

            if message.HasField("animation_data_stream_header"):
                logger.info("Receiving blendshape data from NVIDIA API...")
                header_resp = message.animation_data_stream_header
                bs_names = list(header_resp.skel_animation_header.blend_shapes)
                logger.info(f"  Blendshape count: {len(bs_names)}")

            elif message.HasField("animation_data"):
                anim_data = message.animation_data
                for bs_frame in anim_data.skel_animation.blend_shape_weights:
                    bs_values = dict(zip(bs_names, bs_frame.values))
                    animation_frames.append({
                        "timeCode": bs_frame.time_code,
                        "blendShapes": bs_values,
                    })

            elif message.HasField("status"):
                status = message.status
                status_labels = {0: "SUCCESS", 1: "INFO", 2: "WARNING", 3: "ERROR"}
                label = status_labels.get(status.code, "UNKNOWN")
                logger.info(f"A2F-3D Status: {label} — {status.message}")
                if status.code == 3:
                    raise Exception(f"A2F-3D API error: {status.message}")

        await channel.close()

        if not animation_frames:
            raise Exception("No animation frames received from NVIDIA API")

        logger.info(f"Received {len(animation_frames)} animation frames")

        # Save to CSV
        csv_path = wav_path.replace(".wav", "_blendshapes.csv")
        rows = []
        for frame in animation_frames:
            row = {"timeCode": frame["timeCode"]}
            for bs_name, bs_val in frame["blendShapes"].items():
                row[f"blendShapes.{bs_name}"] = bs_val
            rows.append(row)

        pandas = __import__("pandas")
        df = pandas.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Blendshapes saved to: {csv_path}")

        return csv_path

    # ──────────────────────────────────────────────────────────────────────
    #  Step 2: Blender Headless Rendering
    # ──────────────────────────────────────────────────────────────────────

    def _render_with_blender(self, csv_path: str, audio_path: str, output_path: str) -> Optional[str]:
        """
        Run Blender headless to render 3D character with blendshape animation.
        Returns path to rendered frames directory, or None on failure.
        """
        blender = self.config.blender_path
        script = self.config.render_script

        if not Path(blender).exists():
            logger.error(f"Blender not found: {blender}")
            return None

        if not Path(script).exists():
            logger.error(f"Render script not found: {script}")
            return None

        # Build command
        cmd = [
            blender,
            "--background",
            "--python", script,
            "--",
            "--csv", csv_path,
            "--audio", audio_path,
            "--output", output_path,
            "--fps", str(self.config.render_fps),
            "--resolution", str(self.config.render_resolution_x), str(self.config.render_resolution_y),
            "--character", self.config.character,
            "--background-color", self.config.background_color,
        ]

        logger.info(f"Running Blender: {' '.join(cmd[:4])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes max
            )

            if result.returncode != 0:
                logger.error(f"Blender stderr: {result.stderr[-500:]}")
                return None

            # Check for rendered frames
            frames_dir = output_path.replace(".mp4", "_frames")
            if not Path(frames_dir).exists():
                logger.error(f"Frames directory not created: {frames_dir}")
                return None

            frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
            if not frame_files:
                logger.error("No rendered frames found")
                return None

            logger.info(f"Blender rendered {len(frame_files)} frames to {frames_dir}")
            return frames_dir

        except subprocess.TimeoutExpired:
            logger.error("Blender rendering timed out (10 min)")
            return None
        except Exception as e:
            logger.error(f"Blender rendering failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    #  Step 3: FFmpeg Composition (frames + audio → MP4)
    # ──────────────────────────────────────────────────────────────────────

    def _compose_video(self, frames_dir: str, audio_path: str, output_path: str) -> bool:
        """
        Combine rendered PNG frames + audio into final MP4 using FFmpeg.
        """
        # Find frame pattern
        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
        if not frame_files:
            logger.error("No frame files to compose")
            return False

        # Determine frame numbering pattern
        # Blender outputs: frame_0001.png, frame_0002.png, etc.
        frame_pattern = os.path.join(frames_dir, "frame_%04d.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.config.render_fps),
            "-i", frame_pattern,
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr[-500:]}")
                return False

            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                logger.info(f"Final video: {output_path} ({Path(output_path).stat().st_size / 1024 / 1024:.1f} MB)")

                # Cleanup frames directory
                try:
                    shutil.rmtree(frames_dir)
                    logger.debug(f"Cleaned up frames dir: {frames_dir}")
                except OSError:
                    pass

                return True

            logger.error("FFmpeg produced empty output")
            return False

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out")
            return False
        except Exception as e:
            logger.error(f"FFmpeg failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    #  Utility Methods
    # ──────────────────────────────────────────────────────────────────────

    def _prepare_audio(self, audio_path: str) -> Optional[str]:
        """
        Ensure audio is PCM-16 bit mono WAV (required by Audio2Face-3D API).
        Returns path to converted WAV, or original if already correct format.
        """
        # Check if already WAV
        if audio_path.lower().endswith(".wav"):
            # Verify it's PCM-16 mono
            try:
                import scipy.io.wavfile
                sr, data = scipy.io.wavfile.read(audio_path)
                if data.dtype.name == "int16" and (data.ndim == 1 or data.shape[1] == 1):
                    return audio_path
            except Exception:
                pass

        # Convert with FFmpeg
        output_wav = audio_path.rsplit(".", 1)[0] + "_pcm16mono.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            output_wav,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and Path(output_wav).exists():
                logger.info(f"Audio converted to PCM-16 mono: {output_wav}")
                return output_wav
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")

        return None

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass

        # Fallback: try scipy
        try:
            import scipy.io.wavfile
            sr, data = scipy.io.wavfile.read(audio_path)
            return len(data) / sr
        except Exception:
            pass

        return 0.0

"""
MeiGen MultiTalk Avatar Engine

Generates realistic talking head videos using MeiGen-AI/MultiTalk.
Supports two modes:
  1. "hf_space" (default) - Free, no GPU needed. Uses the public HuggingFace
     Gradio Space API (fffiloni/Meigen-MultiTalk).
  2. "local"  - Runs locally via CLI. Requires CUDA GPU + model weights.

GitHub: https://github.com/MeiGen-AI/MultiTalk
HF Space: https://huggingface.co/spaces/fffiloni/Meigen-MultiTalk
"""

import os
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MultiTalkConfig:
    """Configuration for MultiTalk generation."""
    # Execution mode: "hf_space" (free, no GPU) or "local" (requires CUDA GPU)
    execution_mode: str = "hf_space"

    # HuggingFace Space settings
    hf_space_id: str = "pravatdey/Meigen-MultiTalk"
    hf_sample_steps: int = 12            # Default for the Space (fast)

    # Local mode settings
    multitalk_path: str = ""
    ckpt_dir: str = ""
    wav2vec_dir: str = ""
    size: str = "multitalk-480"
    mode: str = "streaming"
    sample_steps: int = 40
    use_teacache: bool = True
    teacache_thresh: float = 0.3
    use_apg: bool = False
    text_guide_scale: float = 5.0
    audio_guide_scale: float = 4.0
    low_vram: bool = False
    num_gpus: int = 1

    prompt_template: str = (
        "A professional Indian male teacher is speaking directly to the camera "
        "in a studio setting. He maintains eye contact, has natural head movements, "
        "and speaks with clear lip movements synchronized to the audio. "
        "The background is clean and professional."
    )


class MultiTalkEngine:
    """
    Generates talking head videos using MeiGen MultiTalk.

    Supports two execution modes:
      - hf_space: Calls the free HuggingFace Gradio Space API (no GPU needed)
      - local:    Runs generate_multitalk.py locally (needs CUDA GPU + weights)
    """

    def __init__(self, config: Optional[MultiTalkConfig] = None):
        self.config = config or MultiTalkConfig()

        # Resolve local paths from environment if not set
        if not self.config.multitalk_path:
            self.config.multitalk_path = os.getenv("MULTITALK_PATH", "")
        if not self.config.ckpt_dir:
            self.config.ckpt_dir = os.getenv(
                "MULTITALK_CKPT_DIR",
                os.path.join(self.config.multitalk_path, "weights", "Wan2.1-I2V-14B-480P")
                if self.config.multitalk_path else ""
            )
        if not self.config.wav2vec_dir:
            self.config.wav2vec_dir = os.getenv(
                "MULTITALK_WAV2VEC_DIR",
                os.path.join(self.config.multitalk_path, "weights", "chinese-wav2vec2-base")
                if self.config.multitalk_path else ""
            )

    def is_available(self) -> bool:
        """Check if MultiTalk is available in the configured mode."""
        if self.config.execution_mode == "hf_space":
            return self._check_hf_space()
        return self._check_local()

    def _check_hf_space(self) -> bool:
        """Check if the HuggingFace Space is reachable."""
        try:
            from gradio_client import Client
            return True
        except ImportError:
            logger.debug("gradio_client not installed. Run: pip install gradio_client")
            return False

    def _check_local(self) -> bool:
        """Check if local MultiTalk installation is available."""
        mt_path = Path(self.config.multitalk_path)
        if not mt_path.exists():
            return False
        if not (mt_path / "generate_multitalk.py").exists():
            return False
        if not Path(self.config.ckpt_dir).exists():
            return False
        if not Path(self.config.wav2vec_dir).exists():
            return False
        return True

    def generate(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        prompt: Optional[str] = None,
    ) -> dict:
        """
        Generate a talking head video using MultiTalk.

        Args:
            audio_path:  Path to input audio file.
            image_path:  Path to reference avatar image.
            output_path: Path to save the generated video.
            prompt:      Scene description prompt (uses default if None).

        Returns:
            dict with keys: success, video_path, duration, error
        """
        if not self.is_available():
            mode = self.config.execution_mode
            if mode == "hf_space":
                err = "gradio_client not installed. Run: pip install gradio_client"
            else:
                err = "MultiTalk local mode not available. Check paths and GPU."
            return {"success": False, "video_path": "", "duration": 0, "error": err}

        if self.config.execution_mode == "hf_space":
            return self._generate_via_hf_space(audio_path, image_path, output_path, prompt)
        return self._generate_local(audio_path, image_path, output_path, prompt)

    # ──────────────────────────────────────────────────────────────────────
    #  HuggingFace Space API mode (free, no GPU)
    # ──────────────────────────────────────────────────────────────────────

    def _generate_via_hf_space(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        prompt: Optional[str] = None,
    ) -> dict:
        """Generate video via the free HuggingFace Gradio Space API."""
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

            use_prompt = prompt or self.config.prompt_template

            logger.info(
                f"Submitting to MultiTalk HF Space "
                f"(steps={self.config.hf_sample_steps})..."
            )
            start_time = time.time()

            # The Space API expects: prompt, image, audio_spk1, audio_spk2, steps
            # For single-speaker, we pass the same audio for both speakers
            result = client.predict(
                prompt=use_prompt,
                cond_image_path=handle_file(os.path.abspath(image_path)),
                cond_audio_path_spk1=handle_file(os.path.abspath(wav_audio)),
                cond_audio_path_spk2=handle_file(os.path.abspath(wav_audio)),
                sample_steps=self.config.hf_sample_steps,
                api_name="/infer",
            )

            elapsed = time.time() - start_time
            logger.info(f"HF Space generation completed in {elapsed:.0f}s")

            # Result is dict with 'video' key containing the file path
            if isinstance(result, dict):
                video_file = result.get("video")
            elif isinstance(result, (list, tuple)):
                # Sometimes returns (video_dict,)
                video_file = result[0].get("video") if isinstance(result[0], dict) else result[0]
            else:
                video_file = str(result)

            if not video_file or not Path(video_file).exists():
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": f"HF Space returned no video file. Result: {result}",
                }

            # Move to expected output path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_file, output_path)

            duration = self._get_video_duration(output_path)

            logger.info(f"MultiTalk (HF Space) video saved: {output_path} ({duration:.1f}s)")
            return {
                "success": True,
                "video_path": output_path,
                "duration": duration,
                "error": None,
            }

        except Exception as e:
            logger.error(f"MultiTalk HF Space generation failed: {e}", exc_info=True)
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": f"HF Space error: {e}",
            }
        finally:
            if wav_audio != audio_path and Path(wav_audio).exists():
                try:
                    os.remove(wav_audio)
                except OSError:
                    pass

    # ──────────────────────────────────────────────────────────────────────
    #  Local GPU mode
    # ──────────────────────────────────────────────────────────────────────

    def _generate_local(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        prompt: Optional[str] = None,
    ) -> dict:
        """Generate video locally using generate_multitalk.py (requires CUDA GPU)."""
        wav_audio = self._ensure_wav(audio_path)
        if not wav_audio:
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": f"Failed to convert audio to WAV: {audio_path}",
            }

        try:
            input_json = self._create_input_json(
                audio_path=wav_audio,
                image_path=image_path,
                prompt=prompt or self.config.prompt_template,
            )

            cmd = self._build_command(input_json, output_path)
            logger.info(f"Running MultiTalk locally: {' '.join(cmd[:5])}...")

            result = subprocess.run(
                cmd,
                cwd=self.config.multitalk_path,
                capture_output=True,
                text=True,
                timeout=1800,
            )

            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                logger.error(f"MultiTalk failed: {error_msg}")
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": f"MultiTalk process failed: {error_msg}",
                }

            generated = self._find_generated_video(output_path)
            if not generated:
                return {
                    "success": False, "video_path": "", "duration": 0,
                    "error": "MultiTalk completed but output video not found.",
                }

            duration = self._get_video_duration(generated)
            logger.info(f"MultiTalk video generated: {generated} ({duration:.1f}s)")
            return {
                "success": True,
                "video_path": generated,
                "duration": duration,
                "error": None,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": "MultiTalk generation timed out (30 min limit).",
            }
        except Exception as e:
            logger.error(f"MultiTalk local generation error: {e}", exc_info=True)
            return {
                "success": False, "video_path": "", "duration": 0,
                "error": str(e),
            }
        finally:
            if wav_audio != audio_path and Path(wav_audio).exists():
                try:
                    os.remove(wav_audio)
                except OSError:
                    pass

    # ──────────────────────────────────────────────────────────────────────
    #  Shared helpers
    # ──────────────────────────────────────────────────────────────────────

    def _create_input_json(
        self, audio_path: str, image_path: str, prompt: str
    ) -> str:
        """Create a temporary JSON input file for local MultiTalk."""
        input_data = {
            "prompt": prompt,
            "cond_image": os.path.abspath(image_path),
            "cond_audio": {"person1": os.path.abspath(audio_path)},
        }

        tmp_dir = Path(self.config.multitalk_path) / "temp_inputs"
        tmp_dir.mkdir(exist_ok=True)

        json_path = str(tmp_dir / "history_avatar_input.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(input_data, f, indent=2)
        return json_path

    def _build_command(self, input_json: str, output_path: str) -> list:
        """Build the CLI command for local MultiTalk generation."""
        script = str(Path(self.config.multitalk_path) / "generate_multitalk.py")
        save_name = Path(output_path).stem

        if self.config.num_gpus > 1:
            cmd = [
                "torchrun",
                f"--nproc_per_node={self.config.num_gpus}",
                "--standalone", script,
                "--dit_fsdp", "--t5_fsdp",
                f"--ulysses_size={self.config.num_gpus}",
            ]
        else:
            cmd = ["python", script]

        cmd.extend([
            "--ckpt_dir", self.config.ckpt_dir,
            "--wav2vec_dir", self.config.wav2vec_dir,
            "--input_json", input_json,
            "--sample_steps", str(self.config.sample_steps),
            "--mode", self.config.mode,
            "--size", self.config.size,
            "--save_file", save_name,
            "--sample_text_guide_scale", str(self.config.text_guide_scale),
            "--sample_audio_guide_scale", str(self.config.audio_guide_scale),
        ])

        if self.config.use_teacache:
            cmd.extend(["--use_teacache", "--teacache_thresh", str(self.config.teacache_thresh)])
        if self.config.use_apg:
            cmd.append("--use_apg")
        if self.config.low_vram:
            cmd.extend(["--num_persistent_param_in_dit", "0"])

        return cmd

    def _ensure_wav(self, audio_path: str) -> Optional[str]:
        """Convert audio to WAV format if needed."""
        if audio_path.lower().endswith(".wav"):
            return audio_path
        try:
            wav_path = str(Path(audio_path).with_suffix(".wav"))
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, check=True, timeout=120,
            )
            return wav_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Audio conversion to WAV failed: {e}")
            return None

    def _find_generated_video(self, output_path: str) -> Optional[str]:
        """Find the generated video and move it to the expected output path."""
        output = Path(output_path)
        if output.exists():
            return str(output)

        mt_output_dir = Path(self.config.multitalk_path) / "output"
        save_name = output.stem

        for ext in [".mp4", ".avi"]:
            for candidate in mt_output_dir.rglob(f"*{save_name}*{ext}"):
                shutil.move(str(candidate), str(output))
                return str(output)

        for ext in [".mp4", ".avi"]:
            for candidate in Path(self.config.multitalk_path).glob(f"*{save_name}*{ext}"):
                shutil.move(str(candidate), str(output))
                return str(output)

        return None

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

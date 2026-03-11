"""
Avatar Video Generator - Creates talking head videos from image + audio

Supports:
- SadTalker (realistic lip sync + head movements)
- Wav2Lip (precise lip sync)
- Simple fallback (image + audio overlay)
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import tempfile

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AvatarResult:
    """Result of avatar video generation"""
    success: bool
    video_path: str
    duration: float
    method: str
    error: Optional[str] = None


class VisemeSpriteSheet:
    """
    Creates mouth-shape sprites by warping the ACTUAL avatar face pixels
    using cv2.remap displacement fields.  Each of 8 viseme positions is
    combined with 4 amplitude levels → 32 pre-rendered RGBA sprites that
    blend seamlessly onto the face via feathered alpha.

    Viseme IDs:
      0 CLOSED  — lips together, neutral
      1 AH      — mouth open, jaw drops  (a, e, i)
      2 OH      — lips rounded            (o, u)
      3 EE      — wide smile shape        (ee, y)
      4 FV      — lower lip tucked        (f, v)
      5 BMP     — lips pressed            (b, m, p)
      6 LDN     — small opening           (l, d, t, n, r, s)
      7 WQ      — lips pursed / forward   (w, ch, j, sh, k, g)
    """

    NUM_VISEMES = 8
    AMP_LEVELS = 4  # amplitude sub-levels per viseme

    def __init__(self, mouth_region: dict, face_image):
        import cv2
        import numpy as np
        from PIL import Image, ImageFilter

        self.region = mouth_region
        cx, cy = mouth_region['cx'], mouth_region['cy']
        mw, mh = mouth_region['w'], mouth_region['h']

        # Extended crop around mouth (cheeks + chin for natural blending)
        pad_w = int(mw * 1.3)
        pad_h = int(mh * 3.5)
        self.crop_x1 = max(0, cx - pad_w)
        self.crop_y1 = max(0, cy - pad_h // 2)
        self.crop_x2 = min(face_image.width, cx + pad_w)
        self.crop_y2 = min(face_image.height, cy + pad_h)
        self.crop_w = self.crop_x2 - self.crop_x1
        self.crop_h = self.crop_y2 - self.crop_y1

        # Relative mouth centre within the crop
        self.rel_cx = cx - self.crop_x1
        self.rel_cy = cy - self.crop_y1

        # Base crop as numpy BGR (for cv2) and RGB PIL
        base_pil = face_image.crop((self.crop_x1, self.crop_y1,
                                    self.crop_x2, self.crop_y2))
        self.base_np = np.array(base_pil)  # RGB, H×W×3

        # Sample a dark cavity colour from inside the mouth area
        self._cavity_color = self._sample_cavity_color()

        # Build feathered alpha mask (reused for every sprite)
        self._alpha_mask = self._build_feather_mask(self.crop_w, self.crop_h)

        # Pre-render all sprites: (viseme_id, amp_level) → RGBA PIL Image
        self.sprites = {}
        for v in range(self.NUM_VISEMES):
            for a in range(self.AMP_LEVELS):
                amp = a / max(self.AMP_LEVELS - 1, 1)
                self.sprites[(v, a)] = self._warp_for_viseme(v, amp, cv2, np, Image)

        logger.info(f"VisemeSpriteSheet: {len(self.sprites)} sprites "
                     f"({self.crop_w}×{self.crop_h} px crop)")

    # ── helpers ────────────────────────────────────────────────────────────

    def _sample_cavity_color(self):
        """Dark colour to paint inside open-mouth visemes."""
        import numpy as np
        h, w = self.base_np.shape[:2]
        cx, cy = self.rel_cx, self.rel_cy
        r = max(2, min(self.region['w'] // 6, w // 4))
        y1, y2 = max(0, cy - r), min(h, cy + r)
        x1, x2 = max(0, cx - r), min(w, cx + r)
        patch = self.base_np[y1:y2, x1:x2]
        if patch.size == 0:
            return np.array([30, 15, 15], dtype=np.uint8)
        avg = patch.mean(axis=(0, 1)).astype(np.float32)
        # Darken significantly — mouth interior is very dark
        dark = np.clip(avg * 0.15, 0, 40).astype(np.uint8)
        return dark

    def _build_feather_mask(self, w, h, feather=8):
        """Elliptical feathered alpha mask for smooth blending."""
        from PIL import Image, ImageDraw, ImageFilter
        mask = Image.new('L', (w, h), 0)
        d = ImageDraw.Draw(mask)
        pad = feather
        d.ellipse([pad, pad, w - pad, h - pad], fill=255)
        return mask.filter(ImageFilter.GaussianBlur(radius=feather))

    # ── per-viseme warp ────────────────────────────────────────────────────

    def _warp_for_viseme(self, viseme_id, amplitude, cv2, np, Image):
        """
        Warp the mouth crop for a specific viseme + amplitude using
        cv2.remap with a displacement field.

        The displacement is strongest at the mouth centre and fades
        smoothly via a gaussian mask, so surrounding skin stays natural.
        """
        h, w = self.base_np.shape[:2]
        mcx, mcy = float(self.rel_cx), float(self.rel_cy)

        # Base mapping (identity)
        map_x, map_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )

        # Normalised distance from mouth centre
        dx = (map_x - mcx) / max(w / 2, 1)
        dy = (map_y - mcy) / max(h / 2, 1)

        # Gaussian influence: strongest at mouth, fading out
        influence = np.exp(-(dx ** 2 + dy ** 2) * 2.5)

        # Mouth half-mask: below mouth centre
        below = np.clip((map_y - mcy) / max(h * 0.15, 1), 0, 1)
        above = np.clip((mcy - map_y) / max(h * 0.15, 1), 0, 1)

        # Scale factor based on amplitude (0 → no warp, 1 → max)
        amp = amplitude

        # ── Apply viseme-specific displacements ────────────────────────
        if viseme_id == 0:
            # CLOSED — no change
            pass

        elif viseme_id == 1:
            # AH — jaw drops, mouth opens wide vertically
            jaw_drop = amp * 10.0
            map_y = map_y - below * influence * jaw_drop
            # Slight horizontal widening
            h_widen = amp * 3.0
            map_x = map_x - dx * influence * h_widen * 0.3

        elif viseme_id == 2:
            # OH — lips round: horizontal compress + vertical stretch
            h_compress = amp * 5.0
            v_stretch = amp * 7.0
            map_x = map_x + dx * influence * h_compress
            map_y = map_y - below * influence * v_stretch

        elif viseme_id == 3:
            # EE — wide smile: horizontal stretch + slight vertical compress
            h_stretch = amp * 6.0
            v_compress = amp * 2.0
            map_x = map_x - dx * influence * h_stretch
            map_y = map_y + below * influence * v_compress * 0.5

        elif viseme_id == 4:
            # FV — lower lip tucks up toward upper teeth
            lip_tuck = amp * 5.0
            map_y = map_y + below * influence * lip_tuck * 0.6
            # Slight narrowing
            map_x = map_x + dx * influence * amp * 2.0

        elif viseme_id == 5:
            # BMP — lips press together (vertical compression at mouth)
            press = amp * 4.0
            map_y = map_y + dy * influence * press * 0.5

        elif viseme_id == 6:
            # LDN — small opening, slight jaw drop
            jaw_drop = amp * 5.0
            map_y = map_y - below * influence * jaw_drop

        elif viseme_id == 7:
            # WQ — lips purse forward: horizontal compress (rounded)
            h_compress = amp * 6.0
            v_slight = amp * 3.0
            map_x = map_x + dx * influence * h_compress
            map_y = map_y - below * influence * v_slight * 0.5

        # Remap
        warped = cv2.remap(
            self.base_np, map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Paint dark cavity for open-mouth visemes
        if viseme_id in (1, 2, 3, 6, 7) and amp > 0.15:
            self._paint_cavity(warped, np, viseme_id, amp)

        # Convert to RGBA PIL with feathered alpha
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = warped
        rgba[:, :, 3] = np.array(self._alpha_mask)
        return Image.fromarray(rgba, 'RGBA')

    def _paint_cavity(self, img_np, np, viseme_id, amplitude):
        """Paint a subtle dark ellipse at the mouth centre to simulate cavity."""
        h, w = img_np.shape[:2]
        mcx, mcy = self.rel_cx, self.rel_cy
        mw = self.region['w']

        # Cavity dimensions depend on viseme shape
        if viseme_id in (1,):  # AH — wide horizontal
            cw = int(mw * 0.5 * amplitude)
            ch = int(mw * 0.35 * amplitude)
        elif viseme_id in (2, 7):  # OH, WQ — narrow rounded
            cw = int(mw * 0.3 * amplitude)
            ch = int(mw * 0.35 * amplitude)
        elif viseme_id == 3:  # EE — wide thin
            cw = int(mw * 0.5 * amplitude)
            ch = int(mw * 0.15 * amplitude)
        else:  # LDN etc — small
            cw = int(mw * 0.3 * amplitude)
            ch = int(mw * 0.2 * amplitude)

        if cw < 2 or ch < 2:
            return

        # Create cavity elliptical mask
        Y, X = np.ogrid[:h, :w]
        ell = ((X - mcx) / max(cw, 1)) ** 2 + ((Y - mcy) / max(ch, 1)) ** 2
        mask = np.clip(1.0 - ell, 0, 1)
        mask = mask ** 1.5  # sharper edges

        # Blend dark cavity colour
        cavity = self._cavity_color.reshape(1, 1, 3).astype(np.float32)
        blended = img_np.astype(np.float32) * (1 - mask[:, :, None]) + cavity * mask[:, :, None]
        img_np[:] = np.clip(blended, 0, 255).astype(np.uint8)

    # ── public API ──────────────────────────────────────────────────────────

    def get_sprite(self, viseme_id: int, amplitude: float):
        """Return the pre-rendered sprite for a viseme + amplitude level."""
        v = max(0, min(viseme_id, self.NUM_VISEMES - 1))
        a = int(amplitude * (self.AMP_LEVELS - 1) + 0.5)
        a = max(0, min(a, self.AMP_LEVELS - 1))
        return self.sprites[(v, a)]

    def get_paste_position(self):
        """Return (x, y) top-left corner where this sprite pastes on the full frame."""
        return (self.crop_x1, self.crop_y1)


class AvatarGenerator:
    """
    Generates talking head videos from a still image and audio.

    Supports multiple methods:
    1. SadTalker - Best quality, realistic head movements
    2. Wav2Lip - Good lip sync, less head movement
    3. Simple - Static image with audio (fallback)
    """

    def __init__(
        self,
        method: str = "auto",
        avatar_image: str = None,
        sadtalker_path: str = None,
        wav2lip_path: str = None
    ):
        """
        Initialize avatar generator.

        Args:
            method: Generation method ("sadtalker", "wav2lip", "simple", "auto")
            avatar_image: Path to default avatar image
            sadtalker_path: Path to SadTalker installation
            wav2lip_path: Path to Wav2Lip installation
        """
        self.method = method
        self.avatar_image = avatar_image or "assets/avatars/news_anchor.png"
        self.sadtalker_path = sadtalker_path or os.getenv("SADTALKER_PATH", "")
        self.wav2lip_path = wav2lip_path or os.getenv("WAV2LIP_PATH", "")

        # Detect available methods
        self.available_methods = self._detect_methods()

        # Select best available method if auto
        if method == "auto":
            self.method = self._select_best_method()

        # Mouth region for the default avatar (set when avatar is created)
        self._default_mouth_region = None

        logger.info(f"AvatarGenerator initialized: method={self.method}, available={self.available_methods}")

    def _detect_methods(self) -> list:
        """Detect which generation methods are available"""
        available = ["simple"]  # Always available

        # Check for SadTalker
        if self._check_sadtalker():
            available.append("sadtalker")

        # Check for Wav2Lip
        if self._check_wav2lip():
            available.append("wav2lip")

        return available

    def _check_sadtalker(self) -> bool:
        """Check if SadTalker is available"""
        if self.sadtalker_path and Path(self.sadtalker_path).exists():
            inference_script = Path(self.sadtalker_path) / "inference.py"
            return inference_script.exists()

        # Try to import as package
        try:
            import sadtalker
            return True
        except ImportError:
            pass

        return False

    def _check_wav2lip(self) -> bool:
        """Check if Wav2Lip is available"""
        if self.wav2lip_path and Path(self.wav2lip_path).exists():
            inference_script = Path(self.wav2lip_path) / "inference.py"
            return inference_script.exists()

        return False

    def _select_best_method(self) -> str:
        """Select the best available method"""
        if "sadtalker" in self.available_methods:
            return "sadtalker"
        elif "wav2lip" in self.available_methods:
            return "wav2lip"
        return "simple"

    def generate(
        self,
        audio_path: str,
        output_path: str,
        avatar_image: str = None,
        method: str = None
    ) -> AvatarResult:
        """
        Generate talking head video from audio.

        Args:
            audio_path: Path to input audio file
            output_path: Path to save output video
            avatar_image: Path to avatar image (uses default if not provided)
            method: Override generation method

        Returns:
            AvatarResult object
        """
        method = method or self.method
        avatar_image = avatar_image or self.avatar_image

        # Validate inputs
        if not Path(audio_path).exists():
            return AvatarResult(
                success=False,
                video_path="",
                duration=0,
                method=method,
                error=f"Audio file not found: {audio_path}"
            )

        if not Path(avatar_image).exists():
            # Try to create a default avatar
            avatar_image = self._create_default_avatar()
            if not avatar_image:
                return AvatarResult(
                    success=False,
                    video_path="",
                    duration=0,
                    method=method,
                    error="Avatar image not found and couldn't create default"
                )

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating avatar video: method={method}")

        # Generate based on method
        if method == "sadtalker" and "sadtalker" in self.available_methods:
            return self._generate_sadtalker(audio_path, output_path, avatar_image)
        elif method == "wav2lip" and "wav2lip" in self.available_methods:
            return self._generate_wav2lip(audio_path, output_path, avatar_image)
        else:
            return self._generate_simple(audio_path, output_path, avatar_image)

    def _generate_sadtalker(
        self,
        audio_path: str,
        output_path: str,
        avatar_image: str
    ) -> AvatarResult:
        """Generate video using SadTalker"""
        try:
            # SadTalker command
            cmd = [
                "python",
                str(Path(self.sadtalker_path) / "inference.py"),
                "--driven_audio", audio_path,
                "--source_image", avatar_image,
                "--result_dir", str(Path(output_path).parent),
                "--still",  # Less head movement for news style
                "--preprocess", "crop",
                "--enhancer", "gfpgan"  # Face enhancement
            ]

            logger.info(f"Running SadTalker: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.sadtalker_path
            )

            if result.returncode != 0:
                raise Exception(f"SadTalker failed: {result.stderr}")

            # Find output file (SadTalker generates with timestamp)
            output_dir = Path(output_path).parent
            generated_files = list(output_dir.glob("*.mp4"))
            if generated_files:
                # Move to expected output path
                latest_file = max(generated_files, key=lambda p: p.stat().st_mtime)
                shutil.move(str(latest_file), output_path)

            duration = self._get_video_duration(output_path)

            return AvatarResult(
                success=True,
                video_path=output_path,
                duration=duration,
                method="sadtalker"
            )

        except Exception as e:
            logger.error(f"SadTalker generation failed: {e}")
            # Fallback to simple method
            logger.info("Falling back to simple method")
            return self._generate_simple(audio_path, output_path, avatar_image)

    def _generate_wav2lip(
        self,
        audio_path: str,
        output_path: str,
        avatar_image: str
    ) -> AvatarResult:
        """Generate video using Wav2Lip"""
        try:
            # First, create a static video from image
            temp_video = Path(output_path).parent / "temp_static.mp4"

            # Get audio duration
            audio_duration = self._get_audio_duration(audio_path)

            # Create static video
            self._create_static_video(avatar_image, str(temp_video), audio_duration)

            # Wav2Lip command
            cmd = [
                "python",
                str(Path(self.wav2lip_path) / "inference.py"),
                "--checkpoint_path", str(Path(self.wav2lip_path) / "checkpoints/wav2lip_gan.pth"),
                "--face", str(temp_video),
                "--audio", audio_path,
                "--outfile", output_path,
                "--resize_factor", "1",
                "--nosmooth"
            ]

            logger.info(f"Running Wav2Lip: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.wav2lip_path
            )

            # Cleanup temp file
            temp_video.unlink(missing_ok=True)

            if result.returncode != 0:
                raise Exception(f"Wav2Lip failed: {result.stderr}")

            duration = self._get_video_duration(output_path)

            return AvatarResult(
                success=True,
                video_path=output_path,
                duration=duration,
                method="wav2lip"
            )

        except Exception as e:
            logger.error(f"Wav2Lip generation failed: {e}")
            # Fallback to simple method
            return self._generate_simple(audio_path, output_path, avatar_image)

    def _generate_simple(
        self,
        audio_path: str,
        output_path: str,
        avatar_image: str
    ) -> AvatarResult:
        """
        Generate lip-sync animated avatar video with viseme-based mouth shapes.

        Uses phoneme-aware viseme mapping (from word-timing data) combined
        with audio amplitude to produce natural lip movement.
        Falls back to amplitude-only if word timing is unavailable.
        """
        try:
            from moviepy.video.VideoClip import VideoClip
            from moviepy.editor import AudioFileClip
            from PIL import Image, ImageDraw
            import numpy as np
            import json
            import math

            logger.info("Generating viseme lip-sync avatar video")

            # ── Audio ────────────────────────────────────────────────────────
            audio = AudioFileClip(audio_path)
            duration = audio.duration
            fps = 30

            # Per-frame amplitude values in [0, 1] — modulates viseme intensity
            amplitudes = self._extract_audio_amplitude(audio_path, fps, duration)
            logger.info(f"Lip-sync: extracted {len(amplitudes)} amplitude frames")

            # ── Viseme timeline (phoneme-aware) ──────────────────────────────
            viseme_timeline = None
            word_timing_path = str(Path(audio_path).with_suffix('.wordtiming.json'))
            if Path(word_timing_path).exists():
                try:
                    from src.avatar.viseme_mapper import generate_viseme_timeline
                    with open(word_timing_path, 'r', encoding='utf-8') as f:
                        word_boundaries = json.load(f)
                    viseme_timeline = generate_viseme_timeline(
                        word_boundaries, fps, duration
                    )
                    logger.info(f"Viseme timeline loaded: {len(viseme_timeline)} frames")
                except Exception as e:
                    logger.warning(f"Viseme mapping failed, using amplitude-only: {e}")
            else:
                logger.info("No word timing file — using amplitude-only lip-sync")

            # ── Avatar image ─────────────────────────────────────────────────
            pil_image = Image.open(avatar_image).convert('RGB')

            # Resize to 720p, preserving aspect ratio, even width
            target_height = 720
            aspect_ratio = pil_image.width / pil_image.height
            target_width = int(target_height * aspect_ratio)
            target_width = target_width if target_width % 2 == 0 else target_width + 1
            pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)

            # ── Mouth detection ──────────────────────────────────────────────
            mouth_region = self._get_mouth_region(pil_image, avatar_image)
            logger.info(f"Mouth region ({mouth_region['method']}): "
                        f"cx={mouth_region['cx']} cy={mouth_region['cy']} "
                        f"w={mouth_region['w']} h={mouth_region['h']}")

            # Frozen base frame (numpy) — we copy & paint over each frame
            base_array = np.array(pil_image)

            # ── Pre-render viseme sprite sheet (face-warped) ─────────────────
            try:
                sprite_sheet = VisemeSpriteSheet(mouth_region, pil_image)
                use_viseme = True
            except Exception as e:
                logger.warning(f"VisemeSpriteSheet failed, no mouth animation: {e}")
                sprite_sheet = None
                use_viseme = False

            # Paste position for viseme sprites
            paste_pos = sprite_sheet.get_paste_position() if use_viseme else None

            # ── Frame generator ──────────────────────────────────────────────
            def make_frame(t):
                frame_idx = min(int(t * fps), len(amplitudes) - 1)
                amplitude = float(amplitudes[frame_idx])

                # Subtle breathing brightness variation
                brightness = 1.0 + math.sin(t * 0.9) * 0.010 + math.cos(t * 1.7) * 0.005
                frame_array = np.clip(base_array * brightness, 0, 255).astype(np.uint8)

                if not use_viseme:
                    return frame_array

                frame_img = Image.fromarray(frame_array)

                # Get viseme ID for this frame
                if viseme_timeline is not None and frame_idx < len(viseme_timeline):
                    viseme_id = int(viseme_timeline[frame_idx])
                else:
                    # Amplitude-only fallback: map amplitude to AH(1) vs CLOSED(0)
                    viseme_id = 1 if amplitude > 0.15 else 0

                # Natural micro-jitter on amplitude
                jitter = (math.sin(t * 21.3) * 0.03 + math.cos(t * 13.7) * 0.02) * amplitude
                eff_amp = max(0.0, min(1.0, amplitude + jitter))

                # Get the warped sprite
                sprite = sprite_sheet.get_sprite(viseme_id, eff_amp)

                # Transition blending: blend with previous viseme for smoothness
                if frame_idx > 0 and viseme_timeline is not None:
                    prev_viseme = int(viseme_timeline[max(0, frame_idx - 1)])
                    if prev_viseme != viseme_id:
                        prev_sprite = sprite_sheet.get_sprite(prev_viseme, eff_amp)
                        sprite = Image.blend(prev_sprite, sprite, 0.7)

                # Composite onto frame
                frame_img.paste(sprite, paste_pos, sprite)

                return np.array(frame_img)

            # ── Build and export ─────────────────────────────────────────────
            video_clip = VideoClip(make_frame, duration=duration)
            video_clip = video_clip.set_fps(fps)
            video_clip = video_clip.set_audio(audio)

            logger.info(f"Writing viseme lip-sync avatar video: {output_path}")
            video_clip.write_videofile(
                output_path,
                fps=fps,
                codec='libx264',
                audio_codec='aac',
                bitrate='4000k',
                verbose=False,
                logger=None
            )

            audio.close()
            video_clip.close()

            method = "viseme_lip_sync" if viseme_timeline is not None else "amplitude_lip_sync"
            logger.info(f"Avatar video generated ({method}): {duration:.1f}s")
            return AvatarResult(
                success=True,
                video_path=output_path,
                duration=duration,
                method=method
            )

        except Exception as e:
            logger.error(f"Lip-sync avatar generation failed: {e}")
            return self._generate_static_fallback(audio_path, output_path, avatar_image)

    # ── Lip-sync helpers ──────────────────────────────────────────────────────

    def _extract_audio_amplitude(
        self, audio_path: str, fps: int, duration: float
    ):
        """
        Return a float32 numpy array of length int(duration*fps)+1 whose
        values are the normalised RMS amplitude of the audio at each video frame.
        0 = silence, 1 = peak loudness.

        Falls back gracefully: librosa → pydub → simulated speech rhythm.
        """
        import numpy as np

        n_frames = int(duration * fps) + 1

        # ── librosa (best quality) ────────────────────────────────────────────
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            hop = max(1, int(sr / fps))
            rms = librosa.feature.rms(y=y, hop_length=hop)[0].astype(np.float32)
            logger.debug("Lip-sync amplitude: librosa")
        except ImportError:
            rms = None

        # ── pydub ─────────────────────────────────────────────────────────────
        if rms is None:
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_file(audio_path).set_channels(1)
                sr = seg.frame_rate
                samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                frame_size = max(1, int(sr / fps))
                rms = np.array([
                    np.sqrt(np.mean(
                        samples[i * frame_size: min((i + 1) * frame_size, len(samples))] ** 2
                    ))
                    for i in range(n_frames)
                ], dtype=np.float32)
                logger.debug("Lip-sync amplitude: pydub")
            except Exception as exc:
                logger.debug(f"Lip-sync pydub failed: {exc}")
                rms = None

        # ── simulated speech rhythm (last resort) ─────────────────────────────
        if rms is None:
            logger.debug("Lip-sync amplitude: simulated rhythm")
            t = np.linspace(0, duration, n_frames, dtype=np.float32)
            rms = (
                np.abs(np.sin(t * 3.8)) * np.abs(np.sin(t * 7.1 + 0.9)) * 0.75
                + np.abs(np.sin(t * 2.3 + 0.4)) * 0.20
            )

        # Normalise
        rms = np.asarray(rms, dtype=np.float32)
        peak = rms.max()
        if peak > 1e-6:
            rms = rms / peak

        # Smooth (~80 ms window) for natural jaw movement
        window = max(3, fps // 12)
        kernel = np.ones(window, dtype=np.float32) / window
        rms = np.convolve(np.pad(rms, (window // 2, window // 2), mode='edge'),
                          kernel, mode='valid')

        # Guarantee exact frame count
        if len(rms) > n_frames:
            rms = rms[:n_frames]
        elif len(rms) < n_frames:
            rms = np.pad(rms, (0, n_frames - len(rms)), mode='edge')

        return rms

    def _get_mouth_region(self, pil_image, avatar_image_path: str = None) -> dict:
        """
        Locate the lip/mouth area in the (already-resized) avatar image.

        Detection order:
          1. MediaPipe face-mesh  (most accurate, optional dependency)
          2. OpenCV Haar cascade  (good, optional dependency)
          3. Default avatar known proportions
          4. Generic portrait heuristic
        Returns dict: {cx, cy, w, h, method}
        """
        import numpy as np

        img_w, img_h = pil_image.size

        # ── 1. MediaPipe ──────────────────────────────────────────────────────
        try:
            import mediapipe as mp
            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True
            ) as fm:
                res = fm.process(np.array(pil_image))
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    # Outer-lip landmark indices (MediaPipe 468-point mesh)
                    lip_ids = [
                        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                        146, 91, 181, 84, 17, 314, 405, 321, 375,
                    ]
                    lx = [lm[i].x * img_w for i in lip_ids]
                    ly = [lm[i].y * img_h for i in lip_ids]
                    cx = int(np.mean(lx))
                    cy = int(np.mean(ly))
                    lip_w = int((max(lx) - min(lx)) * 1.5)
                    lip_h = int((max(ly) - min(ly)) * 3.5)
                    return {
                        'cx': cx, 'cy': cy,
                        'w': max(lip_w, int(img_w * 0.10)),
                        'h': max(lip_h, int(img_h * 0.04)),
                        'method': 'mediapipe',
                    }
        except (ImportError, Exception):
            pass

        # ── 2. OpenCV Haar cascade ────────────────────────────────────────────
        try:
            import cv2
            gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                return {
                    'cx': fx + fw // 2,
                    'cy': fy + int(fh * 0.76),   # mouth ≈ 76 % down the face box
                    'w': int(fw * 0.48),
                    'h': int(fh * 0.20),
                    'method': 'opencv',
                }
        except (ImportError, Exception):
            pass

        # ── 3. Default avatar known proportions ───────────────────────────────
        # news_anchor.png (512 x 640): mouth centre ≈ 50 % x, 49 % y after resize.
        if avatar_image_path and Path(avatar_image_path).name == 'news_anchor.png':
            return {
                'cx': img_w // 2,
                'cy': int(img_h * 0.49),
                'w': int(img_w * 0.15),
                'h': int(img_h * 0.05),
                'method': 'default_avatar_known',
            }

        # ── 4. Generic portrait heuristic ─────────────────────────────────────
        return {
            'cx': img_w // 2,
            'cy': int(img_h * 0.66),
            'w': int(img_w * 0.22),
            'h': int(img_h * 0.09),
            'method': 'heuristic',
        }

    # _draw_mouth_frame removed — replaced by viseme sprite compositing in make_frame

    # ─────────────────────────────────────────────────────────────────────────

    def _generate_static_fallback(
        self,
        audio_path: str,
        output_path: str,
        avatar_image: str
    ) -> AvatarResult:
        """
        Ultimate fallback - simple static image with audio.
        """
        try:
            from moviepy.editor import ImageClip, AudioFileClip

            logger.info("Using static fallback for avatar video")

            # Load audio
            audio = AudioFileClip(audio_path)
            duration = audio.duration

            # Create image clip
            image_clip = ImageClip(avatar_image, duration=duration)

            # Resize to 1080p maintaining aspect ratio
            target_height = 1080
            image_clip = image_clip.resize(height=target_height)

            # Ensure width is even
            if image_clip.w % 2 != 0:
                image_clip = image_clip.resize(width=image_clip.w + 1)

            # Set audio
            video = image_clip.set_audio(audio)

            # Write output
            video.write_videofile(
                output_path,
                fps=30,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )

            # Cleanup
            audio.close()
            image_clip.close()

            return AvatarResult(
                success=True,
                video_path=output_path,
                duration=duration,
                method="simple_static"
            )

        except Exception as e:
            logger.error(f"Static fallback also failed: {e}")
            return AvatarResult(
                success=False,
                video_path="",
                duration=0,
                method="simple",
                error=str(e)
            )

    def _create_default_avatar(self) -> Optional[str]:
        """Create a professional, human-like AI avatar image with gradient shading."""
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter
            import numpy as np
            import math

            avatar_path = Path("assets/avatars/news_anchor.png")
            avatar_path.parent.mkdir(parents=True, exist_ok=True)

            # Draw at 2x resolution for anti-aliasing, downscale at the end
            S = 2  # supersampling factor
            width, height = 512 * S, 640 * S
            center_x = width // 2

            img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # ── Background gradient (dark navy, subtle radial glow) ──────
            bg_arr = np.zeros((height, width, 4), dtype=np.uint8)
            Y, X = np.ogrid[:height, :width]
            dist = np.sqrt(((X - center_x) / (width * 0.6)) ** 2 + ((Y - height * 0.35) / (height * 0.7)) ** 2)
            dist = np.clip(dist, 0, 1)
            # Core color: (18, 22, 40) -> edge: (10, 12, 24)
            for c, (c0, c1) in enumerate([(18, 10), (22, 12), (40, 24)]):
                bg_arr[:, :, c] = (c0 * (1 - dist) + c1 * dist).astype(np.uint8)
            bg_arr[:, :, 3] = 255
            img = Image.fromarray(bg_arr, 'RGBA')
            draw = ImageDraw.Draw(img)

            # ── Helper: radial gradient ellipse ──────────────────────────
            def draw_gradient_ellipse(base_img, bbox, color_center, color_edge, alpha=255):
                """Draw an ellipse filled with radial gradient."""
                x1, y1, x2, y2 = bbox
                ew, eh = x2 - x1, y2 - y1
                cx_e, cy_e = ew // 2, eh // 2
                layer = np.zeros((eh, ew, 4), dtype=np.uint8)
                Ye, Xe = np.ogrid[:eh, :ew]
                d = np.sqrt(((Xe - cx_e) / max(cx_e, 1)) ** 2 + ((Ye - cy_e) / max(cy_e, 1)) ** 2)
                d = np.clip(d, 0, 1)
                mask = d <= 1.0
                for c in range(3):
                    layer[:, :, c] = (color_center[c] * (1 - d) + color_edge[c] * d).astype(np.uint8)
                layer[:, :, 3] = np.where(mask, alpha, 0).astype(np.uint8)
                ellipse_img = Image.fromarray(layer, 'RGBA')
                base_img.paste(ellipse_img, (x1, y1), ellipse_img)

            # ── Helper: linear gradient rectangle ────────────────────────
            def draw_gradient_rect(base_img, bbox, color_top, color_bottom, alpha=255):
                x1, y1, x2, y2 = bbox
                rw, rh = x2 - x1, y2 - y1
                layer = np.zeros((rh, rw, 4), dtype=np.uint8)
                t = np.linspace(0, 1, rh).reshape(-1, 1)
                for c in range(3):
                    layer[:, :, c] = (color_top[c] * (1 - t) + color_bottom[c] * t).astype(np.uint8)
                layer[:, :, 3] = alpha
                rect_img = Image.fromarray(layer, 'RGBA')
                base_img.paste(rect_img, (x1, y1), rect_img)

            # ── Color palette ────────────────────────────────────────────
            skin_light = (245, 218, 190)
            skin_mid = (228, 190, 155)
            skin_shadow = (200, 160, 125)
            hair_dark = (25, 28, 35)
            hair_mid = (40, 44, 55)
            suit_main = (30, 75, 180)
            suit_light = (50, 100, 210)
            suit_dark = (20, 55, 140)
            lapel_dark = (22, 60, 155)
            shirt_white = (240, 242, 248)
            tie_red = (185, 35, 40)
            tie_dark = (145, 25, 30)

            # ── BODY / SUIT ──────────────────────────────────────────────
            suit_top = 760

            # Shoulders (curved) - using polygon with many points for smoothness
            shoulder_pts = []
            for angle in range(0, 181, 3):
                rad = math.radians(angle)
                sx = center_x + int(310 * math.cos(rad))
                sy = suit_top - int(50 * math.sin(rad))
                shoulder_pts.append((sx, sy))
            shoulder_pts.extend([
                (center_x + 360, height),
                (center_x - 360, height),
            ])
            draw.polygon(shoulder_pts, fill=suit_main)

            # Suit gradient overlay (lighter at top, darker at bottom)
            draw_gradient_rect(img, (center_x - 310, suit_top, center_x + 310, height),
                               suit_light, suit_dark, alpha=120)

            # Left lapel with gradient
            lapel_pts_l = [
                (center_x - 60, suit_top + 30),
                (center_x - 120, height),
                (center_x - 30, height),
                (center_x - 5, suit_top + 140),
            ]
            draw.polygon(lapel_pts_l, fill=lapel_dark)

            # Right lapel
            lapel_pts_r = [
                (center_x + 60, suit_top + 30),
                (center_x + 120, height),
                (center_x + 30, height),
                (center_x + 5, suit_top + 140),
            ]
            draw.polygon(lapel_pts_r, fill=lapel_dark)

            # White shirt / collar V-neck
            collar_pts = [
                (center_x - 55, suit_top + 15),
                (center_x, suit_top + 170),
                (center_x + 55, suit_top + 15),
            ]
            draw.polygon(collar_pts, fill=shirt_white)

            # Collar shadow
            draw.polygon([
                (center_x - 50, suit_top + 20),
                (center_x - 10, suit_top + 60),
                (center_x, suit_top + 50),
            ], fill=(220, 222, 230))
            draw.polygon([
                (center_x + 50, suit_top + 20),
                (center_x + 10, suit_top + 60),
                (center_x, suit_top + 50),
            ], fill=(220, 222, 230))

            # Tie with gradient shading
            tie_w = 22
            draw.polygon([
                (center_x - tie_w, suit_top + 55),
                (center_x + tie_w, suit_top + 55),
                (center_x + int(tie_w * 0.7), height - 180),
                (center_x - int(tie_w * 0.7), height - 180),
            ], fill=tie_red)
            # Tie center highlight
            draw.polygon([
                (center_x - 4, suit_top + 70),
                (center_x + 4, suit_top + 70),
                (center_x + 3, height - 200),
                (center_x - 3, height - 200),
            ], fill=(210, 55, 60))
            # Tie knot
            draw_gradient_ellipse(img, (center_x - 28, suit_top + 35, center_x + 28, suit_top + 80),
                                  (200, 45, 50), tie_dark, alpha=255)

            # ── NECK ─────────────────────────────────────────────────────
            neck_top = 680
            neck_w = 70
            draw_gradient_rect(img, (center_x - neck_w, neck_top, center_x + neck_w, suit_top + 50),
                               skin_light, skin_mid, alpha=255)
            # Neck shadow under chin
            draw_gradient_ellipse(img, (center_x - neck_w + 10, neck_top - 10,
                                        center_x + neck_w - 10, neck_top + 40),
                                  skin_shadow, skin_mid, alpha=140)

            # ── HEAD ─────────────────────────────────────────────────────
            head_cy = 430
            head_rx = 160
            head_ry = 195

            # Ears (behind head)
            ear_y = head_cy - 15
            # Left ear
            draw_gradient_ellipse(img, (center_x - head_rx - 10, ear_y - 35,
                                        center_x - head_rx + 35, ear_y + 50),
                                  skin_light, skin_shadow, alpha=255)
            # Inner ear shadow
            draw_gradient_ellipse(img, (center_x - head_rx + 2, ear_y - 15,
                                        center_x - head_rx + 25, ear_y + 30),
                                  skin_shadow, skin_mid, alpha=180)
            # Right ear
            draw_gradient_ellipse(img, (center_x + head_rx - 35, ear_y - 35,
                                        center_x + head_rx + 10, ear_y + 50),
                                  skin_light, skin_shadow, alpha=255)
            draw_gradient_ellipse(img, (center_x + head_rx - 25, ear_y - 15,
                                        center_x + head_rx - 2, ear_y + 30),
                                  skin_shadow, skin_mid, alpha=180)

            # Main face with gradient (lighter center, shadow at edges)
            draw_gradient_ellipse(img, (center_x - head_rx, head_cy - head_ry,
                                        center_x + head_rx, head_cy + head_ry),
                                  skin_light, skin_mid, alpha=255)

            # Jawline shadow (bottom of face)
            draw_gradient_ellipse(img, (center_x - head_rx + 20, head_cy + head_ry - 80,
                                        center_x + head_rx - 20, head_cy + head_ry + 5),
                                  skin_shadow, skin_mid, alpha=100)

            # Cheek blush (subtle pink)
            draw_gradient_ellipse(img, (center_x - 110, head_cy + 20,
                                        center_x - 40, head_cy + 70),
                                  (235, 180, 170), skin_light, alpha=60)
            draw_gradient_ellipse(img, (center_x + 40, head_cy + 20,
                                        center_x + 110, head_cy + 70),
                                  (235, 180, 170), skin_light, alpha=60)

            # ── HAIR ─────────────────────────────────────────────────────
            # Main hair (top of head) - layered for volume
            draw_gradient_ellipse(img, (center_x - head_rx - 12, head_cy - head_ry - 35,
                                        center_x + head_rx + 12, head_cy - 45),
                                  hair_mid, hair_dark, alpha=255)
            # Hair volume top layer
            draw_gradient_ellipse(img, (center_x - head_rx + 5, head_cy - head_ry - 25,
                                        center_x + head_rx - 5, head_cy - 60),
                                  (50, 55, 68), hair_dark, alpha=200)
            # Hair highlight (sheen)
            draw_gradient_ellipse(img, (center_x - 50, head_cy - head_ry - 15,
                                        center_x + 30, head_cy - head_ry + 35),
                                  (70, 75, 90), hair_dark, alpha=100)

            # Side hair (left) - slightly covering ear
            draw_gradient_ellipse(img, (center_x - head_rx - 15, head_cy - head_ry + 30,
                                        center_x - head_rx + 50, head_cy + 30),
                                  hair_mid, hair_dark, alpha=255)
            # Side hair (right)
            draw_gradient_ellipse(img, (center_x + head_rx - 50, head_cy - head_ry + 30,
                                        center_x + head_rx + 15, head_cy + 30),
                                  hair_mid, hair_dark, alpha=255)

            # Hairline blend into forehead
            draw_gradient_ellipse(img, (center_x - head_rx + 20, head_cy - head_ry + 5,
                                        center_x + head_rx - 20, head_cy - head_ry + 70),
                                  hair_dark, skin_light, alpha=80)

            # ── EYEBROWS (thicker, natural arcs) ────────────────────────
            draw = ImageDraw.Draw(img)
            eb_y = head_cy - 75
            # Left eyebrow
            for offset in range(-3, 4):
                draw.arc([center_x - 105, eb_y - 18 + offset, center_x - 25, eb_y + 18 + offset],
                         start=200, end=340, fill=(35, 38, 48, 220), width=3)
            # Right eyebrow
            for offset in range(-3, 4):
                draw.arc([center_x + 25, eb_y - 18 + offset, center_x + 105, eb_y + 18 + offset],
                         start=200, end=340, fill=(35, 38, 48, 220), width=3)

            # ── EYES (almond-shaped, layered) ────────────────────────────
            eye_y = head_cy - 28
            eye_spacing = 65

            for side in [-1, 1]:
                ex = center_x + side * eye_spacing

                # Eye socket shadow
                draw_gradient_ellipse(img, (ex - 48, eye_y - 32, ex + 48, eye_y + 32),
                                      skin_shadow, skin_light, alpha=50)

                # Eye white (sclera) - almond shape using ellipse
                draw_gradient_ellipse(img, (ex - 42, eye_y - 22, ex + 42, eye_y + 22),
                                      (255, 255, 255), (235, 235, 240), alpha=255)

                # Iris (colored ring)
                iris_x = ex + side * 5  # slight look direction
                iris_r = 18
                draw_gradient_ellipse(img, (iris_x - iris_r, eye_y - iris_r,
                                            iris_x + iris_r, eye_y + iris_r),
                                      (70, 50, 30), (45, 30, 18), alpha=255)

                # Pupil (black center)
                pupil_r = 10
                draw.ellipse([iris_x - pupil_r, eye_y - pupil_r,
                              iris_x + pupil_r, eye_y + pupil_r],
                             fill=(10, 10, 12, 255))

                # Eye shine (catchlight) - two spots
                shine_r = 6
                draw.ellipse([iris_x - shine_r + 8, eye_y - shine_r - 5,
                              iris_x + shine_r + 8, eye_y + shine_r - 5],
                             fill=(255, 255, 255, 240))
                draw.ellipse([iris_x - 3, eye_y + 3, iris_x + 3, eye_y + 9],
                             fill=(255, 255, 255, 140))

                # Upper eyelid line (defines almond shape)
                draw.arc([ex - 44, eye_y - 26, ex + 44, eye_y + 18],
                         start=190, end=350, fill=(50, 40, 35, 200), width=4)
                # Lower eyelid (subtle)
                draw.arc([ex - 38, eye_y - 10, ex + 38, eye_y + 26],
                         start=10, end=170, fill=(180, 155, 130, 100), width=2)

                # Eyelashes (subtle upper)
                for lash_angle in range(200, 345, 20):
                    rad = math.radians(lash_angle)
                    lx1 = ex + int(44 * math.cos(rad))
                    ly1 = eye_y - 4 + int(22 * math.sin(rad))
                    lx2 = ex + int(50 * math.cos(rad))
                    ly2 = eye_y - 6 + int(28 * math.sin(rad))
                    draw.line([(lx1, ly1), (lx2, ly2)], fill=(40, 35, 30, 160), width=2)

            # ── NOSE (soft, shaded) ──────────────────────────────────────
            nose_y = head_cy + 40

            # Nose bridge shadow (subtle vertical line)
            draw_gradient_rect(img, (center_x - 6, head_cy - 10, center_x + 6, nose_y),
                               skin_mid, skin_shadow, alpha=40)

            # Nose tip - soft ellipse
            draw_gradient_ellipse(img, (center_x - 20, nose_y - 10,
                                        center_x + 20, nose_y + 18),
                                  skin_mid, skin_shadow, alpha=80)

            # Nostrils (small dark ellipses)
            draw.ellipse([center_x - 16, nose_y + 2, center_x - 6, nose_y + 12],
                         fill=(180, 140, 110, 150))
            draw.ellipse([center_x + 6, nose_y + 2, center_x + 16, nose_y + 12],
                         fill=(180, 140, 110, 150))

            # Nose highlight
            draw_gradient_ellipse(img, (center_x - 8, nose_y - 25, center_x + 8, nose_y - 5),
                                  (255, 235, 215), skin_light, alpha=60)

            # ── MOUTH REGION (stored for lip-sync, not drawn) ────────────
            mouth_cy = head_cy + 100
            self._default_mouth_region = {
                'cx': center_x // S, 'cy': mouth_cy // S,
                'w': 160 // S, 'h': 85 // S,
                'method': 'default_avatar_known'
            }

            # Subtle closed lips hint (neutral expression)
            draw.arc([center_x - 45, mouth_cy - 8, center_x + 45, mouth_cy + 16],
                     start=10, end=170, fill=(195, 120, 110, 180), width=3)
            draw.arc([center_x - 40, mouth_cy - 12, center_x + 40, mouth_cy + 8],
                     start=195, end=345, fill=(195, 120, 110, 160), width=3)

            # ── MICROPHONE ───────────────────────────────────────────────
            mic_x = center_x - 190
            mic_y = suit_top + 100

            # Mic arm
            draw.line([(mic_x + 15, mic_y + 25), (mic_x - 20, suit_top + 200)],
                      fill=(55, 60, 70, 200), width=4)
            # Mic body (rounded rect)
            draw.rounded_rectangle([mic_x - 12, mic_y - 20, mic_x + 12, mic_y + 20],
                                   radius=8, fill=(60, 65, 75, 230))
            # Mic grill lines
            for gy in range(mic_y - 14, mic_y + 14, 5):
                draw.line([(mic_x - 8, gy), (mic_x + 8, gy)],
                          fill=(80, 85, 95, 180), width=1)
            # Mic highlight
            draw.line([(mic_x - 10, mic_y - 18), (mic_x - 10, mic_y + 18)],
                      fill=(100, 105, 120, 120), width=2)

            # ── AI BADGE ─────────────────────────────────────────────────
            badge_x = center_x + 155
            badge_y = suit_top + 85
            badge_r = 24

            # Badge glow
            draw_gradient_ellipse(img, (badge_x - badge_r - 6, badge_y - badge_r - 6,
                                        badge_x + badge_r + 6, badge_y + badge_r + 6),
                                  (255, 200, 50), (200, 150, 30), alpha=60)
            # Badge circle
            draw_gradient_ellipse(img, (badge_x - badge_r, badge_y - badge_r,
                                        badge_x + badge_r, badge_y + badge_r),
                                  (255, 200, 60), (210, 160, 30), alpha=255)

            draw = ImageDraw.Draw(img)
            try:
                badge_font = ImageFont.truetype("arial.ttf", 22)
            except Exception:
                badge_font = ImageFont.load_default()
            # Center "AI" text on badge
            bbox = draw.textbbox((0, 0), "AI", font=badge_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((badge_x - tw // 2, badge_y - th // 2 - 2), "AI",
                      fill=(30, 25, 20, 255), font=badge_font)

            # ── Subtle shoulder highlight ────────────────────────────────
            draw_gradient_ellipse(img, (center_x - 200, suit_top - 30,
                                        center_x + 200, suit_top + 40),
                                  suit_light, suit_main, alpha=50)

            # ── Downscale 2x → 1x with LANCZOS anti-aliasing ────────────
            final_w, final_h = 512, 640
            img = img.resize((final_w, final_h), Image.LANCZOS)

            # Convert to RGB for saving
            bg_rgb = Image.new('RGB', (final_w, final_h), (14, 16, 30))
            bg_rgb.paste(img, (0, 0), img)

            bg_rgb.save(str(avatar_path), quality=95)
            logger.info(f"Created professional avatar: {avatar_path}")

            return str(avatar_path)

        except Exception as e:
            logger.error(f"Failed to create professional avatar: {e}")
            return None

    def _create_static_video(
        self,
        image_path: str,
        output_path: str,
        duration: float
    ) -> None:
        """Create a static video from an image"""
        from moviepy.editor import ImageClip

        clip = ImageClip(image_path, duration=duration)
        clip.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            verbose=False,
            logger=None
        )
        clip.close()

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file"""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0
        except Exception:
            return 0.0

    def _get_video_duration(self, video_path: str) -> float:
        """Get duration of video file"""
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close()
            return duration
        except Exception:
            return 0.0

    def setup_instructions(self) -> str:
        """Get setup instructions for advanced avatar methods"""
        return """
=== Avatar Generator Setup Instructions ===

Currently using: {method}
Available methods: {available}

To enable advanced avatar generation:

1. SadTalker (Recommended - Best quality)
   - Clone: git clone https://github.com/OpenTalker/SadTalker
   - Install: pip install -r requirements.txt
   - Download models from the repo instructions
   - Set environment variable: SADTALKER_PATH=/path/to/SadTalker

2. Wav2Lip (Good lip sync)
   - Clone: git clone https://github.com/Rudrabha/Wav2Lip
   - Install: pip install -r requirements.txt
   - Download checkpoints from the repo
   - Set environment variable: WAV2LIP_PATH=/path/to/Wav2Lip

Note: Both methods require a GPU for reasonable performance.
Without GPU, video generation will be slow (10-20 minutes per video).

The 'simple' method (static image + audio) works without any setup.
""".format(method=self.method, available=self.available_methods)


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avatar Generator CLI")
    parser.add_argument("--test", action="store_true", help="Run test generation")
    parser.add_argument("--audio", type=str, help="Input audio file")
    parser.add_argument("--image", type=str, help="Avatar image")
    parser.add_argument("--output", type=str, default="output/test_avatar.mp4", help="Output path")
    parser.add_argument("--method", type=str, default="auto", help="Generation method")
    parser.add_argument("--setup", action="store_true", help="Show setup instructions")

    args = parser.parse_args()

    generator = AvatarGenerator(method=args.method)

    if args.setup:
        print(generator.setup_instructions())

    elif args.test or args.audio:
        if not args.audio:
            print("Error: --audio required for generation")
            print("Example: python -m src.avatar.avatar_generator --audio output/test.mp3 --output output/test.mp4")
        else:
            result = generator.generate(
                audio_path=args.audio,
                output_path=args.output,
                avatar_image=args.image
            )

            if result.success:
                print(f"\nSuccess! Video generated at: {result.video_path}")
                print(f"Duration: {result.duration:.1f} seconds")
                print(f"Method: {result.method}")
            else:
                print(f"\nError: {result.error}")

    else:
        parser.print_help()
        print("\n" + generator.setup_instructions())

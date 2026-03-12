"""
Video Composer - Composes educational history lesson videos.
Adapted for History Course: no ticker, no headline overlay, history branding.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

import yaml

# Configure moviepy to use ffmpeg from imageio-ffmpeg
try:
    import imageio_ffmpeg
    import os
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
except ImportError:
    pass

from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip,
    ColorClip, CompositeVideoClip, concatenate_videoclips,
    VideoClip, concatenate_audioclips
)
from moviepy.video.fx.all import fadein, fadeout

from .effects import VideoEffects
from .educational_effects import (
    EducationalEffects, KeyPointDisplay, FactCard,
    TopicHeader, ImageOverlay
)
from .presentation_slides import PresentationSlideGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EducationalContent:
    """Educational content to display in video"""
    key_points: List[Dict[str, Any]] = field(default_factory=list)
    topic_headers: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    statistics: List[Dict[str, Any]] = field(default_factory=list)
    timelines: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CompositionResult:
    """Result of video composition"""
    success: bool
    video_path: str
    duration: float
    resolution: tuple
    error: Optional[str] = None
    pdf_notes_path: Optional[str] = None
    timestamps: List[Dict[str, str]] = field(default_factory=list)


class VideoComposer:
    """
    Composes educational UPSC-focused videos from components:
    - Avatar video (talking head)
    - Background (solid, image, or video)
    - Text overlays (headlines, key points)
    - Educational elements (topic headers, fact cards, images)
    - News ticker with exam relevance tags
    - Intro/outro sequences
    - Background music
    - PDF notes generation integration
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize video composer with educational features.

        Args:
            config_path: Path to settings configuration
        """
        self.config = self._load_config(config_path)
        self.video_config = self.config.get("video", {})
        self.composition_config = self.config.get("composition", {})

        # Video settings
        self.resolution = (
            self.video_config.get("resolution", {}).get("width", 1920),
            self.video_config.get("resolution", {}).get("height", 1080)
        )
        self.fps = self.video_config.get("fps", 30)

        # Initialize effects modules
        self.effects = VideoEffects()
        self.edu_effects = EducationalEffects()
        slides_cfg = self.composition_config.get("presentation_slides", {})
        self.slide_generator = PresentationSlideGenerator(
            content_start_x_pct=slides_cfg.get("content_start_x_pct", 0.33),
            max_key_points=slides_cfg.get("max_key_points", 4),
            show_subject_badge=slides_cfg.get("show_subject_badge", True),
            show_terms_as_badges=slides_cfg.get("show_terms_as_badges", True),
            bullet_style=slides_cfg.get("bullet_style", "numbered"),
        )

        # UPSC mode settings
        self.upsc_mode = self.config.get("upsc_mode", True)

        logger.info(f"VideoComposer initialized: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps (UPSC Mode: {self.upsc_mode})")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}

    def compose(
        self,
        avatar_video_path: str,
        output_path: str,
        headlines: List[str] = None,
        title: str = None,
        date: str = None,
        background_music_path: str = None,
        add_intro: bool = True,
        add_outro: bool = True,
        add_ticker: bool = True,
        # New UPSC educational parameters
        educational_content: EducationalContent = None,
        script_data: Dict[str, Any] = None,
        generate_pdf_notes: bool = True
    ) -> CompositionResult:
        """
        Compose final educational video from components.

        Args:
            avatar_video_path: Path to avatar video
            output_path: Path to save final video
            headlines: List of headlines to show
            title: Video title
            date: Video date
            background_music_path: Path to background music
            add_intro: Whether to add intro
            add_outro: Whether to add outro
            add_ticker: Whether to add news ticker
            educational_content: Educational overlays (key points, images, etc.)
            script_data: Script data with timestamps and UPSC metadata
            generate_pdf_notes: Whether to generate PDF study notes

        Returns:
            CompositionResult object with video path and optional PDF path
        """
        try:
            logger.info("Starting UPSC educational video composition")

            # Load avatar video
            if not Path(avatar_video_path).exists():
                return CompositionResult(
                    success=False,
                    video_path="",
                    duration=0,
                    resolution=self.resolution,
                    error=f"Avatar video not found: {avatar_video_path}"
                )

            avatar_clip = VideoFileClip(avatar_video_path)
            duration = avatar_clip.duration

            # Prepare title and date
            title = title or "History Class"
            date = date or datetime.now().strftime("%B %d, %Y")

            # Initialize educational content if not provided
            if educational_content is None:
                educational_content = EducationalContent()

            clips = []
            timestamps = []

            # 1. Add UPSC-styled intro if enabled
            intro_config = self.composition_config.get("intro", {})
            if add_intro and intro_config.get("enabled", True):
                intro_duration = intro_config.get("duration", 4)  # Slightly longer for UPSC
                intro_clip = self._create_upsc_intro(
                    title=intro_config.get("text", "History Class"),
                    subtitle=date,
                    duration=intro_duration,
                    subjects=script_data.get('subjects_covered', []) if script_data else []
                )
                clips.append(intro_clip)
                timestamps.append({'time': '00:00', 'title': 'Introduction', 'subject': 'Overview'})

            # 2. Create main content with educational overlays
            main_clip = self._create_main_composition(
                avatar_clip=avatar_clip,
                headlines=headlines or [],
                add_ticker=add_ticker,
                educational_content=educational_content,
                script_data=script_data
            )
            clips.append(main_clip)

            # Extract timestamps from script data
            if script_data and 'timestamps' in script_data:
                timestamps.extend(script_data['timestamps'])

            # 3. Add UPSC-styled outro if enabled
            outro_config = self.composition_config.get("outro", {})
            if add_outro and outro_config.get("enabled", True):
                outro_duration = outro_config.get("duration", 6)  # Longer for UPSC call-to-action
                outro_clip = self._create_upsc_outro(
                    text="Best of luck with your preparation!",
                    subscribe_text=outro_config.get("text", "Subscribe for daily UPSC updates!"),
                    duration=outro_duration
                )
                clips.append(outro_clip)

            # 4. Concatenate all clips with proper audio handling
            # Note: Intro and outro don't have audio, only the main clip does
            # We need to ensure audio from main clip is properly preserved
            if len(clips) > 1:
                # Create silent audio for intro/outro to match the concatenation
                from moviepy.audio.AudioClip import AudioClip

                for i, clip in enumerate(clips):
                    if clip.audio is None:
                        # Create silent audio for clips without audio
                        silent_audio = AudioClip(make_frame=lambda t: 0, duration=clip.duration)
                        clips[i] = clip.set_audio(silent_audio)

            final_video = concatenate_videoclips(clips, method="compose")
            logger.info(f"Concatenated {len(clips)} clips, total duration: {final_video.duration:.1f}s")

            # 5. Add background music if provided
            music_config = self.composition_config.get("music", {})
            if background_music_path and music_config.get("enabled", True):
                final_video = self._add_background_music(
                    video=final_video,
                    music_path=background_music_path,
                    volume=music_config.get("volume", 0.1)
                )

            # 6. Export final video
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Exporting video to: {output_path}")

            # Build ffmpeg parameters for faster encoding
            preset = self.video_config.get("preset", "medium")
            threads = self.video_config.get("threads", 4)
            bitrate = self.video_config.get("bitrate", "5000k")

            ffmpeg_params = [
                '-preset', preset,
                '-threads', str(threads)
            ]

            logger.info(f"Export settings: preset={preset}, threads={threads}, bitrate={bitrate}, resolution={self.resolution}")

            final_video.write_videofile(
                str(output_path),
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                bitrate=bitrate,
                ffmpeg_params=ffmpeg_params,
                verbose=True,
                logger='bar'
            )

            final_duration = final_video.duration

            # Cleanup
            avatar_clip.close()
            final_video.close()

            logger.info(f"Video composition complete: {final_duration:.1f}s")

            # Generate PDF notes if enabled
            pdf_notes_path = None
            if generate_pdf_notes and script_data:
                pdf_notes_path = self._generate_pdf_notes(
                    script_data=script_data,
                    title=title,
                    date=date,
                    video_duration=final_duration
                )

            return CompositionResult(
                success=True,
                video_path=str(output_path),
                duration=final_duration,
                resolution=self.resolution,
                pdf_notes_path=pdf_notes_path,
                timestamps=timestamps
            )

        except Exception as e:
            logger.error(f"Video composition failed: {e}")
            return CompositionResult(
                success=False,
                video_path="",
                duration=0,
                resolution=self.resolution,
                error=str(e)
            )

    def _create_upsc_intro(
        self,
        title: str,
        subtitle: str,
        duration: float,
        subjects: List[str] = None
    ) -> VideoClip:
        """Create UPSC-styled intro with subject tags."""
        # Use educational effects for topic header style intro
        topic_header = TopicHeader(
            title=title,
            subtitle=subtitle,
            start_time=0,
            duration=duration,
            topic_number=0,
            exam_tag="UPSC CSE",
            subject=", ".join(subjects[:3]) if subjects else "History"
        )

        intro_clip = self.edu_effects.create_topic_header(
            topic=topic_header,
            video_size=self.resolution
        )

        return intro_clip

    def _create_upsc_outro(
        self,
        text: str,
        subscribe_text: str,
        duration: float
    ) -> VideoClip:
        """Create UPSC-styled outro with call-to-action."""
        return self.effects.create_outro(
            text=text,
            subscribe_text=subscribe_text + "\nPDF Notes in Description!",
            size=self.resolution,
            duration=duration
        )

    def _generate_pdf_notes(
        self,
        script_data: Dict[str, Any],
        title: str,
        date: str,
        video_duration: float
    ) -> Optional[str]:
        """
        Generate PDF study notes from script data.

        NOTE: For comprehensive 10-15 page notes with 20 MCQs + 5 Mains questions,
        use the main pipeline's _generate_comprehensive_notes() method instead.
        This method provides a basic fallback for direct composer usage.
        """
        try:
            from src.notes.pdf_generator import PDFNotesGenerator, StudyNote, TopicNote

            generator = PDFNotesGenerator()

            # Build topics from script segments
            segments = script_data.get('segments', [])
            era = script_data.get('era', 'History')
            exam_focus = script_data.get('exam_focus', 'BOTH')

            sub_sections = []
            all_terms = {}
            all_questions = []

            for segment in segments:
                content = segment.get('content', '')
                kps = segment.get('key_points', [])
                seg_title = segment.get('title', 'Content')
                terms = segment.get('important_terms', {})

                if content and len(content) > 50:
                    sub_sections.append({
                        'heading': seg_title,
                        'points': kps if kps else [],
                        'sub_points': {},
                    })
                all_terms.update(terms)

            topic_note = TopicNote(
                title=title,
                trigger_line=f"{title} - Study notes for UPSC preparation.",
                what_is_it=f"These notes cover {title} ({era}). "
                           f"Content is based on the video lesson script.",
                key_provisions=[],
                sub_sections=sub_sections,
                challenges=[],
                suggestions=[],
                important_terms=all_terms,
                practice_questions=all_questions,
                upsc_tags=f"History | {era} | {exam_focus}",
            )

            study_note = StudyNote(
                title=title,
                date=date,
                topics=[topic_note],
                video_duration=video_duration
            )

            pdf_path = generator.generate_notes(study_note)
            logger.info(f"PDF notes generated: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.warning(f"Failed to generate PDF notes: {e}")
            return None

    def _create_main_composition(
        self,
        avatar_clip: VideoFileClip,
        headlines: List[str],
        add_ticker: bool,
        educational_content: EducationalContent = None,
        script_data: Dict[str, Any] = None
    ) -> VideoClip:
        """Create main video composition with avatar and presentation slides background"""
        width, height = self.resolution
        duration = avatar_clip.duration

        # Store the original audio from avatar clip
        original_audio = avatar_clip.audio

        # Initialize educational content if not provided
        if educational_content is None:
            educational_content = EducationalContent()

        # ── Base background (always present, visible during gaps) ────────
        bg_config = self.composition_config.get("background", {})
        bg_color = bg_config.get("color", "#0f1419")
        base_background = ColorClip(
            size=self.resolution,
            color=self._hex_to_rgb(bg_color)
        ).set_duration(duration)

        # ── Presentation slides as background ────────────────────────────
        slides_config = self.composition_config.get("presentation_slides", {})
        use_slides = slides_config.get("enabled", True) and script_data

        slide_clips = []
        if use_slides:
            try:
                slide_clips = self.slide_generator.generate_slides(
                    script_data=script_data,
                    video_size=self.resolution,
                    total_duration=duration
                )
                logger.info(f"Generated {len(slide_clips)} presentation slides for background")
            except Exception as e:
                logger.warning(f"Failed to generate presentation slides: {e}")

        # ── Resize and position avatar ───────────────────────────────────
        avatar_config = self.composition_config.get("avatar", {})
        slides_config = self.composition_config.get("presentation_slides", {})
        position = avatar_config.get("position", "left")

        if position == "overlay_left":
            # Avatar overlays on top of full-screen slides, bottom-left aligned
            avatar_scale = avatar_config.get("scale", 0.55)
            avatar_height = int(height * avatar_scale)
            avatar_clip = avatar_clip.resize(height=avatar_height)

            # Cap width to prevent covering too much slide content
            max_avatar_width = int(width * 0.30)
            if avatar_clip.w > max_avatar_width:
                avatar_clip = avatar_clip.resize(width=max_avatar_width)

            x_offset = avatar_config.get("x_offset", 20)
            x_pos = x_offset
            y_pos = height - avatar_clip.h  # Bottom-aligned

        elif position == "left":
            # Legacy: Auto-fill left zone
            content_start_pct = slides_config.get("content_start_x_pct", 0.33)
            content_start_x = int(width * content_start_pct)

            avatar_clip = avatar_clip.resize(width=content_start_x)
            if avatar_clip.h > height:
                avatar_clip = avatar_clip.resize(height=height)

            x_pos = (content_start_x - avatar_clip.w) // 2
            y_pos = height - avatar_clip.h
        else:
            # Centre / right: use manual scale + offsets
            avatar_scale = avatar_config.get("scale", 0.55)
            avatar_height = int(height * avatar_scale)
            avatar_clip = avatar_clip.resize(height=avatar_height)
            x_offset = avatar_config.get("x_offset", 0)
            y_offset = avatar_config.get("y_offset", 0)

            if position == "right":
                x_pos = 4 * width // 5 - avatar_clip.w // 2 + x_offset
            else:
                x_pos = width // 2 - avatar_clip.w // 2 + x_offset
            y_pos = height // 2 - avatar_clip.h // 2 + y_offset

        avatar_clip = avatar_clip.set_position((x_pos, y_pos))

        # ── Layer clips: base_bg -> slides -> avatar -> overlays ─────────
        layers = [base_background] + slide_clips + [avatar_clip]

        # Add topic header transitions (shorter when slides are active)
        if script_data:
            topic_duration = 2.0 if slide_clips else 4.0
            layers.extend(self._create_topic_transitions(
                script_data, duration, topic_duration=topic_duration))

        # Only add key point / image / stats overlays if slides are NOT active
        # (slides already show the key points and terms)
        if not slide_clips:
            if educational_content.key_points:
                layers.extend(self._create_key_point_overlays(educational_content.key_points))

            if educational_content.images:
                layers.extend(self._create_image_overlays(educational_content.images))

            if educational_content.statistics:
                layers.extend(self._create_stats_overlays(educational_content.statistics))

        # Add headline overlay at bottom
        text_config = self.composition_config.get("text", {}).get("headline", {})
        if headlines and text_config.get("enabled", True):
            headline_text = " | ".join(headlines[:3])
            headline_overlay = self._create_headline_overlay(
                text=headline_text,
                duration=duration,
                config=text_config
            )
            layers.append(headline_overlay)

        # Add UPSC ticker with exam relevance
        ticker_config = self.composition_config.get("text", {}).get("ticker", {})
        if add_ticker and ticker_config.get("enabled", True):
            subjects = script_data.get('subjects_covered', []) if script_data else []
            sep = ' \u2022 '
            ticker_text = f"UPSC Current Affairs | Topics: {sep.join(subjects[:4])}" if subjects else ""
            if not ticker_text and headlines:
                ticker_text = "UPSC CURRENT AFFAIRS: " + " \u2022 ".join(headlines[:3])

            if ticker_text:
                ticker = self.effects.create_news_ticker(
                    text=ticker_text,
                    size=self.resolution,
                    duration=duration,
                    speed=ticker_config.get("speed", 80),
                    fontsize=ticker_config.get("size", 28),
                    color=ticker_config.get("color", "white"),
                    bg_color=ticker_config.get("background", "#1a365d")
                )
                layers.append(ticker)

        # Apply fade effects
        composition = CompositeVideoClip(layers, size=self.resolution)
        composition = fadein(composition, 0.5)
        composition = fadeout(composition, 0.5)

        # IMPORTANT: Preserve the original audio from avatar clip
        if original_audio is not None:
            composition = composition.set_audio(original_audio)
            logger.info(f"Audio preserved from avatar clip: {original_audio.duration:.1f}s")

        return composition

    def _create_topic_transitions(
        self,
        script_data: Dict[str, Any],
        total_duration: float,
        topic_duration: float = 4.0
    ) -> List[VideoClip]:
        """Create topic transition overlays from script data."""
        clips = []

        segments = script_data.get('segments', [])
        for segment in segments:
            if segment.get('type') == 'news':
                # Parse timestamp to seconds
                timestamp_str = segment.get('timestamp', '00:00')
                parts = timestamp_str.split(':')
                start_time = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else 0

                # Create brief topic indicator
                topic_header = TopicHeader(
                    title=segment.get('article_title', '')[:50],
                    subtitle=segment.get('subject_category', ''),
                    start_time=start_time,
                    duration=topic_duration,
                    topic_number=segments.index(segment),
                    exam_tag=segment.get('exam_relevance', ''),
                    subject=segment.get('subject_category', '')
                )

                try:
                    clip = self.edu_effects.create_topic_header(
                        topic=topic_header,
                        video_size=self.resolution
                    )
                    clips.append(clip)
                except Exception as e:
                    logger.warning(f"Failed to create topic header: {e}")

        return clips

    def _create_key_point_overlays(
        self,
        key_points: List[Dict[str, Any]]
    ) -> List[VideoClip]:
        """Create key point overlay clips."""
        clips = []

        for kp_data in key_points:
            try:
                kp = KeyPointDisplay(
                    text=kp_data.get('text', ''),
                    start_time=kp_data.get('start_time', 0),
                    duration=kp_data.get('duration', 5.0),
                    importance=kp_data.get('importance', 3),
                    category=kp_data.get('category', '')
                )

                clip = self.edu_effects.create_key_point_overlay(
                    key_point=kp,
                    video_size=self.resolution,
                    theme=kp_data.get('theme', 'blue')
                )
                clips.append(clip)
            except Exception as e:
                logger.warning(f"Failed to create key point overlay: {e}")

        return clips

    def _create_image_overlays(
        self,
        images: List[Dict[str, Any]]
    ) -> List[VideoClip]:
        """Create image overlay clips for maps, diagrams, etc."""
        clips = []

        for img_data in images:
            try:
                img_overlay = ImageOverlay(
                    image_path=img_data.get('path', ''),
                    start_time=img_data.get('start_time', 0),
                    duration=img_data.get('duration', 8.0),
                    position=img_data.get('position', 'right'),
                    scale=img_data.get('scale', 0.3),
                    caption=img_data.get('caption', '')
                )

                clip = self.edu_effects.create_image_overlay(
                    image_overlay=img_overlay,
                    video_size=self.resolution
                )
                if clip:
                    clips.append(clip)
            except Exception as e:
                logger.warning(f"Failed to create image overlay: {e}")

        return clips

    def _create_stats_overlays(
        self,
        statistics: List[Dict[str, Any]]
    ) -> List[VideoClip]:
        """Create statistics card overlays."""
        clips = []

        for stats_data in statistics:
            try:
                clip = self.edu_effects.create_stats_card(
                    stats=stats_data.get('stats', {}),
                    video_size=self.resolution,
                    start_time=stats_data.get('start_time', 0),
                    duration=stats_data.get('duration', 6.0),
                    title=stats_data.get('title', 'Key Statistics')
                )
                clips.append(clip)
            except Exception as e:
                logger.warning(f"Failed to create stats overlay: {e}")

        return clips

    def _create_headline_overlay(
        self,
        text: str,
        duration: float,
        config: Dict[str, Any]
    ) -> VideoClip:
        """Create headline text overlay"""
        return self.effects.create_text_overlay(
            text=text,
            size=self.resolution,
            duration=duration,
            fontsize=config.get("size", 48),
            color=config.get("color", "white"),
            bg_color=config.get("background", "#000000"),
            bg_opacity=0.7,
            position=config.get("position", "bottom")
        )

    def _add_background_music(
        self,
        video: VideoClip,
        music_path: str,
        volume: float
    ) -> VideoClip:
        """Add background music to video"""
        try:
            music = AudioFileClip(music_path)

            # Loop music if shorter than video
            if music.duration < video.duration:
                # Calculate how many loops needed
                loops_needed = int(video.duration / music.duration) + 1
                music = concatenate_audioclips([music] * loops_needed)

            # Trim to video duration
            music = music.subclip(0, video.duration)

            # Adjust volume
            music = music.volumex(volume)

            # Mix with original audio
            if video.audio:
                from moviepy.audio.AudioClip import CompositeAudioClip
                final_audio = CompositeAudioClip([video.audio, music])
            else:
                final_audio = music

            return video.set_audio(final_audio)

        except Exception as e:
            logger.warning(f"Failed to add background music: {e}")
            return video

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video Composer CLI")
    parser.add_argument("--avatar", type=str, required=True, help="Avatar video path")
    parser.add_argument("--output", type=str, default="output/composed_video.mp4", help="Output path")
    parser.add_argument("--title", type=str, default="History Class", help="Video title")
    parser.add_argument("--music", type=str, help="Background music path")
    parser.add_argument("--no-intro", action="store_true", help="Skip intro")
    parser.add_argument("--no-outro", action="store_true", help="Skip outro")

    args = parser.parse_args()

    composer = VideoComposer()

    result = composer.compose(
        avatar_video_path=args.avatar,
        output_path=args.output,
        title=args.title,
        headlines=["Ancient History", "Vedic Period"],
        background_music_path=args.music,
        add_intro=not args.no_intro,
        add_outro=not args.no_outro
    )

    if result.success:
        print(f"\nSuccess! Video saved to: {result.video_path}")
        print(f"Duration: {result.duration:.1f} seconds")
        print(f"Resolution: {result.resolution[0]}x{result.resolution[1]}")
    else:
        print(f"\nError: {result.error}")

"""
Video Effects - Transitions, overlays, and effects for video composition
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
from pathlib import Path

from moviepy.editor import (
    VideoClip, ImageClip, ColorClip,
    CompositeVideoClip, concatenate_videoclips
)
from moviepy.video.fx.all import fadein, fadeout

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Check if ImageMagick is available (for TextClip)
IMAGEMAGICK_AVAILABLE = False
try:
    from moviepy.editor import TextClip
    # Try to create a simple TextClip to test ImageMagick
    test_clip = TextClip("test", fontsize=20)
    test_clip.close()
    IMAGEMAGICK_AVAILABLE = True
except Exception:
    logger.warning("ImageMagick not available - using PIL-based text rendering")


def create_text_image(
    text: str,
    size: Tuple[int, int],
    fontsize: int = 48,
    color: str = "white",
    bg_color: str = None,
    font_path: str = None
) -> Image.Image:
    """Create a PIL image with text using PIL instead of ImageMagick"""
    width, height = size

    # Parse colors
    if isinstance(color, str):
        if color.startswith('#'):
            color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        elif color == "white":
            color = (255, 255, 255)
        elif color == "black":
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)

    # Create image
    if bg_color:
        if isinstance(bg_color, str) and bg_color.startswith('#'):
            bg_color = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        img = Image.new('RGBA', (width, height), bg_color + (255,))
    else:
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    draw = ImageDraw.Draw(img)

    # Try to use a system font
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
        except:
            font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center text
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=color + (255,) if len(color) == 3 else color, font=font)

    return img


class VideoEffects:
    """Collection of video effects and transitions"""

    @staticmethod
    def fade_in(clip: VideoClip, duration: float = 1.0) -> VideoClip:
        """Apply fade in effect"""
        return fadein(clip, duration)

    @staticmethod
    def fade_out(clip: VideoClip, duration: float = 1.0) -> VideoClip:
        """Apply fade out effect"""
        return fadeout(clip, duration)

    @staticmethod
    def fade_transition(clip1: VideoClip, clip2: VideoClip, duration: float = 0.5) -> VideoClip:
        """Create fade transition between two clips"""
        clip1_faded = fadeout(clip1, duration)
        clip2_faded = fadein(clip2, duration)

        # Overlap the clips
        clip2_start = clip1.duration - duration
        clip2_positioned = clip2_faded.set_start(clip2_start)

        return CompositeVideoClip([clip1_faded, clip2_positioned])

    @staticmethod
    def create_text_overlay(
        text: str,
        size: Tuple[int, int],
        duration: float,
        font: str = "Arial-Bold",
        fontsize: int = 48,
        color: str = "white",
        bg_color: str = "black",
        bg_opacity: float = 0.7,
        position: str = "bottom"
    ) -> VideoClip:
        """
        Create a text overlay with background.

        Args:
            text: Text to display
            size: Video size (width, height)
            duration: Duration of overlay
            font: Font name
            fontsize: Font size
            color: Text color
            bg_color: Background color
            bg_opacity: Background opacity (0-1)
            position: Position ("top", "center", "bottom")

        Returns:
            Composite video clip with text overlay
        """
        width, height = size
        bg_height = fontsize + 60  # Estimated height with padding

        # Position
        if position == "top":
            y_pos = 0
        elif position == "center":
            y_pos = (height - bg_height) // 2
        else:  # bottom
            y_pos = height - bg_height

        # Use PIL-based rendering (works without ImageMagick)
        # Create text image using PIL
        text_img = create_text_image(
            text=text,
            size=(width, bg_height),
            fontsize=fontsize,
            color=color,
            bg_color=bg_color
        )

        # Save to temp file and load as ImageClip
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            text_img.save(f.name)
            txt_clip = ImageClip(f.name).set_duration(duration)
            txt_clip = txt_clip.set_opacity(bg_opacity)
            txt_clip = txt_clip.set_position((0, y_pos))

        return CompositeVideoClip([txt_clip], size=size)

    @staticmethod
    def create_news_ticker(
        text: str,
        size: Tuple[int, int],
        duration: float,
        speed: int = 100,
        fontsize: int = 32,
        color: str = "white",
        bg_color: str = "#c41e3a"
    ) -> VideoClip:
        """
        Create a static news ticker bar (scrolling disabled for simplicity).

        Args:
            text: Ticker text
            size: Video size (width, height)
            duration: Duration
            speed: Scroll speed (not used in simplified version)
            fontsize: Font size
            color: Text color
            bg_color: Background color

        Returns:
            Video clip with ticker bar
        """
        width, height = size
        ticker_height = fontsize + 20

        # Create ticker image using PIL
        ticker_img = create_text_image(
            text=text[:80] + "..." if len(text) > 80 else text,  # Truncate long text
            size=(width, ticker_height),
            fontsize=fontsize,
            color=color,
            bg_color=bg_color
        )

        # Save to temp file and load as ImageClip
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            ticker_img.save(f.name)
            ticker_clip = ImageClip(f.name).set_duration(duration)
            ticker_clip = ticker_clip.set_position((0, height - ticker_height))

        return CompositeVideoClip([ticker_clip], size=size)

    @staticmethod
    def create_intro(
        title: str,
        subtitle: str,
        size: Tuple[int, int],
        duration: float = 3.0,
        bg_color: str = "#1a1a2e"
    ) -> VideoClip:
        """
        Create intro sequence using PIL-based text rendering.

        Args:
            title: Main title
            subtitle: Subtitle text
            size: Video size
            duration: Intro duration
            bg_color: Background color

        Returns:
            Intro video clip
        """
        width, height = size

        # Create intro image with PIL
        bg_rgb = VideoEffects._hex_to_rgb(bg_color)
        img = Image.new('RGB', (width, height), bg_rgb)
        draw = ImageDraw.Draw(img)

        # Load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 72)
            subtitle_font = ImageFont.truetype("arial.ttf", 36)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()

        # Draw title (centered)
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
        title_x = (width - title_w) // 2
        title_y = height // 2 - title_h

        draw.text((title_x, title_y), title, fill=(255, 255, 255), font=title_font)

        # Draw subtitle
        sub_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        sub_x = (width - sub_w) // 2
        sub_y = height // 2 + 40

        draw.text((sub_x, sub_y), subtitle, fill=(200, 200, 200), font=subtitle_font)

        # Save and create clip
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            intro_clip = ImageClip(f.name).set_duration(duration)

        intro_clip = fadein(intro_clip, 0.5)
        return fadeout(intro_clip, 0.5)

    @staticmethod
    def create_outro(
        text: str,
        subscribe_text: str,
        size: Tuple[int, int],
        duration: float = 5.0,
        bg_color: str = "#1a1a2e"
    ) -> VideoClip:
        """
        Create outro sequence using PIL-based text rendering.

        Args:
            text: Main outro text
            subscribe_text: Call to action text
            size: Video size
            duration: Outro duration
            bg_color: Background color

        Returns:
            Outro video clip
        """
        width, height = size

        # Create outro image with PIL
        bg_rgb = VideoEffects._hex_to_rgb(bg_color)
        img = Image.new('RGB', (width, height), bg_rgb)
        draw = ImageDraw.Draw(img)

        # Load fonts
        try:
            main_font = ImageFont.truetype("arial.ttf", 48)
            sub_font = ImageFont.truetype("arial.ttf", 32)
        except:
            main_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()

        # Draw main text (centered)
        main_bbox = draw.textbbox((0, 0), text, font=main_font)
        main_w = main_bbox[2] - main_bbox[0]
        main_h = main_bbox[3] - main_bbox[1]
        main_x = (width - main_w) // 2
        main_y = height // 2 - main_h

        draw.text((main_x, main_y), text, fill=(255, 255, 255), font=main_font)

        # Draw subscribe text
        sub_bbox = draw.textbbox((0, 0), subscribe_text, font=sub_font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        sub_x = (width - sub_w) // 2
        sub_y = height // 2 + 60

        draw.text((sub_x, sub_y), subscribe_text, fill=(255, 107, 107), font=sub_font)

        # Save and create clip
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            outro_clip = ImageClip(f.name).set_duration(duration)

        outro_clip = fadein(outro_clip, 0.5)
        return fadeout(outro_clip, 1.0)

    @staticmethod
    def add_logo_watermark(
        clip: VideoClip,
        logo_path: str,
        position: str = "top-right",
        scale: float = 0.1,
        opacity: float = 0.7
    ) -> VideoClip:
        """
        Add logo watermark to video.

        Args:
            clip: Source video clip
            logo_path: Path to logo image
            position: Position ("top-left", "top-right", "bottom-left", "bottom-right")
            scale: Logo scale relative to video width
            opacity: Logo opacity

        Returns:
            Video with watermark
        """
        try:
            logo = ImageClip(logo_path)
            logo = logo.resize(width=int(clip.w * scale))
            logo = logo.set_opacity(opacity).set_duration(clip.duration)

            # Calculate position
            padding = 20
            if position == "top-left":
                pos = (padding, padding)
            elif position == "top-right":
                pos = (clip.w - logo.w - padding, padding)
            elif position == "bottom-left":
                pos = (padding, clip.h - logo.h - padding)
            else:  # bottom-right
                pos = (clip.w - logo.w - padding, clip.h - logo.h - padding)

            logo = logo.set_position(pos)

            return CompositeVideoClip([clip, logo])

        except Exception as e:
            logger.warning(f"Failed to add watermark: {e}")
            return clip

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

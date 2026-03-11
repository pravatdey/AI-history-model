"""
Educational Effects - Text overlays, key points, and visual elements for educational videos
Optimized for UPSC/competitive exam preparation content
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import requests
from io import BytesIO

from moviepy.editor import (
    VideoClip, ImageClip, ColorClip, CompositeVideoClip,
    concatenate_videoclips, AudioFileClip
)
from moviepy.video.fx.all import fadein, fadeout

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KeyPointDisplay:
    """A key point to display in video"""
    text: str
    start_time: float
    duration: float
    importance: int = 3  # 1-5, affects styling
    category: str = ""
    icon: str = ""  # emoji or icon indicator


@dataclass
class FactCard:
    """A fact card to display"""
    title: str
    facts: List[str]
    start_time: float
    duration: float
    color_theme: str = "blue"  # blue, green, orange, purple


@dataclass
class ImageOverlay:
    """An image overlay configuration"""
    image_path: str
    start_time: float
    duration: float
    position: str = "right"  # left, right, center, fullscreen
    scale: float = 0.3
    caption: str = ""


@dataclass
class TopicHeader:
    """Topic header/transition card"""
    title: str
    subtitle: str
    start_time: float
    duration: float
    topic_number: int = 1
    exam_tag: str = ""  # "PRELIMS", "MAINS", "BOTH"
    subject: str = ""


class EducationalEffects:
    """
    Creates educational video overlays optimized for UPSC preparation.
    Includes key points, fact cards, images, topic headers, and more.
    """

    # Color themes for different content types
    THEMES = {
        'blue': {
            'primary': (26, 54, 93),      # Dark blue
            'secondary': (44, 82, 130),   # Medium blue
            'accent': (66, 153, 225),     # Light blue
            'text': (255, 255, 255),
            'bg': (26, 32, 44)
        },
        'green': {
            'primary': (34, 84, 61),
            'secondary': (56, 161, 105),
            'accent': (104, 211, 145),
            'text': (255, 255, 255),
            'bg': (28, 35, 33)
        },
        'orange': {
            'primary': (124, 45, 18),
            'secondary': (237, 137, 54),
            'accent': (251, 211, 141),
            'text': (255, 255, 255),
            'bg': (44, 33, 26)
        },
        'purple': {
            'primary': (68, 51, 122),
            'secondary': (128, 90, 213),
            'accent': (183, 148, 244),
            'text': (255, 255, 255),
            'bg': (35, 30, 45)
        },
        'prelims': {
            'primary': (49, 130, 206),
            'secondary': (66, 153, 225),
            'accent': (144, 205, 244),
            'text': (255, 255, 255),
            'bg': (26, 32, 44)
        },
        'mains': {
            'primary': (128, 90, 213),
            'secondary': (159, 122, 234),
            'accent': (214, 188, 250),
            'text': (255, 255, 255),
            'bg': (35, 30, 45)
        }
    }

    # Exam relevance icons
    EXAM_ICONS = {
        'PRELIMS': '📝',
        'MAINS': '📖',
        'BOTH': '⭐',
        'IMPORTANT': '🔴'
    }

    def __init__(self, assets_dir: str = "assets"):
        """Initialize educational effects generator."""
        self.assets_dir = Path(assets_dir)
        self.fonts_dir = self.assets_dir / "fonts"
        self._load_fonts()
        logger.info("EducationalEffects initialized")

    def _load_fonts(self):
        """Load or set default fonts."""
        self.fonts = {}

        # Try to load custom fonts, fall back to system fonts
        font_sizes = {
            'title': 48,
            'heading': 36,
            'subheading': 28,
            'body': 24,
            'small': 18,
            'tiny': 14
        }

        for name, size in font_sizes.items():
            try:
                font_path = self.fonts_dir / "NotoSans-Bold.ttf"
                if font_path.exists():
                    self.fonts[name] = ImageFont.truetype(str(font_path), size)
                else:
                    # Try system fonts
                    self.fonts[name] = ImageFont.truetype("arial.ttf", size)
            except Exception:
                try:
                    self.fonts[name] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
                except Exception:
                    self.fonts[name] = ImageFont.load_default()

    def create_key_point_overlay(
        self,
        key_point: KeyPointDisplay,
        video_size: Tuple[int, int],
        theme: str = "blue"
    ) -> VideoClip:
        """
        Create an animated key point overlay.

        Args:
            key_point: KeyPoint data
            video_size: Video dimensions (width, height)
            theme: Color theme name

        Returns:
            VideoClip with the key point animation
        """
        width, height = video_size
        colors = self.THEMES.get(theme, self.THEMES['blue'])

        # Create key point card
        card_width = int(width * 0.45)
        card_height = 120
        padding = 20

        # Create image
        img = Image.new('RGBA', (card_width, card_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw rounded rectangle background
        self._draw_rounded_rect(
            draw,
            (0, 0, card_width, card_height),
            radius=15,
            fill=colors['primary'] + (230,)  # Semi-transparent
        )

        # Draw accent bar on left
        draw.rectangle(
            (0, 10, 6, card_height - 10),
            fill=colors['accent'] + (255,)
        )

        # Draw importance indicator
        if key_point.importance >= 4:
            indicator_color = (237, 100, 100)  # Red for important
        elif key_point.importance >= 3:
            indicator_color = colors['accent']
        else:
            indicator_color = colors['secondary']

        draw.ellipse(
            (card_width - 30, 10, card_width - 10, 30),
            fill=indicator_color + (255,)
        )

        # Draw category tag if present
        y_offset = 15
        if key_point.category:
            category_font = self.fonts.get('tiny', ImageFont.load_default())
            draw.text(
                (padding + 10, y_offset),
                key_point.category.upper(),
                fill=colors['accent'] + (255,),
                font=category_font
            )
            y_offset += 22

        # Draw main text (with word wrap)
        text_font = self.fonts.get('body', ImageFont.load_default())
        wrapped_text = self._wrap_text(key_point.text, card_width - 2 * padding - 20, text_font, draw)

        draw.text(
            (padding + 10, y_offset),
            wrapped_text,
            fill=colors['text'] + (255,),
            font=text_font
        )

        # Save to temp file and create clip
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            clip = ImageClip(f.name).set_duration(key_point.duration)

        # Position on right side of video
        x_pos = width - card_width - 30
        y_pos = height // 2 - card_height // 2

        clip = clip.set_position((x_pos, y_pos))
        clip = clip.set_start(key_point.start_time)

        # Add fade effects
        clip = fadein(clip, 0.3)
        clip = fadeout(clip, 0.3)

        return clip

    def create_fact_card(
        self,
        fact_card: FactCard,
        video_size: Tuple[int, int]
    ) -> VideoClip:
        """
        Create an animated fact card showing multiple facts.

        Args:
            fact_card: FactCard data
            video_size: Video dimensions

        Returns:
            VideoClip with the fact card
        """
        width, height = video_size
        colors = self.THEMES.get(fact_card.color_theme, self.THEMES['blue'])

        # Card dimensions
        card_width = int(width * 0.4)
        card_height = min(60 + len(fact_card.facts) * 40, int(height * 0.6))
        padding = 15

        # Create image
        img = Image.new('RGBA', (card_width, card_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw background
        self._draw_rounded_rect(
            draw,
            (0, 0, card_width, card_height),
            radius=12,
            fill=colors['bg'] + (240,)
        )

        # Draw header bar
        draw.rectangle(
            (0, 0, card_width, 45),
            fill=colors['primary'] + (255,)
        )

        # Draw title
        title_font = self.fonts.get('subheading', ImageFont.load_default())
        draw.text(
            (padding, 10),
            fact_card.title,
            fill=colors['text'] + (255,),
            font=title_font
        )

        # Draw facts
        fact_font = self.fonts.get('small', ImageFont.load_default())
        y = 55

        for i, fact in enumerate(fact_card.facts[:6]):  # Max 6 facts
            # Draw bullet point
            draw.ellipse(
                (padding, y + 5, padding + 8, y + 13),
                fill=colors['accent'] + (255,)
            )

            # Draw fact text
            wrapped_fact = self._wrap_text(fact, card_width - 2 * padding - 20, fact_font, draw)
            draw.text(
                (padding + 15, y),
                wrapped_fact,
                fill=colors['text'] + (200,),
                font=fact_font
            )
            y += 35

        # Save and create clip
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            clip = ImageClip(f.name).set_duration(fact_card.duration)

        # Position
        clip = clip.set_position(('right', 'center'))
        clip = clip.set_start(fact_card.start_time)

        clip = fadein(clip, 0.4)
        clip = fadeout(clip, 0.4)

        return clip

    def create_topic_header(
        self,
        topic: TopicHeader,
        video_size: Tuple[int, int]
    ) -> VideoClip:
        """
        Create a topic transition header/title card.

        Args:
            topic: TopicHeader data
            video_size: Video dimensions

        Returns:
            VideoClip with the topic header
        """
        width, height = video_size

        # Determine theme based on exam tag
        if topic.exam_tag == 'PRELIMS':
            colors = self.THEMES['prelims']
        elif topic.exam_tag == 'MAINS':
            colors = self.THEMES['mains']
        else:
            colors = self.THEMES['blue']

        # Create full-screen image
        img = Image.new('RGB', (width, height), colors['bg'])
        draw = ImageDraw.Draw(img)

        # Draw decorative elements
        # Top gradient bar
        for i in range(100):
            alpha = int(255 * (1 - i/100))
            draw.rectangle(
                (0, i, width, i + 1),
                fill=colors['primary']
            )

        # Bottom gradient bar
        for i in range(100):
            alpha = int(255 * (i/100))
            draw.rectangle(
                (0, height - 100 + i, width, height - 99 + i),
                fill=colors['primary']
            )

        # Draw topic number
        topic_num_font = self.fonts.get('title', ImageFont.load_default())
        topic_num_text = f"TOPIC {topic.topic_number}"
        num_bbox = draw.textbbox((0, 0), topic_num_text, font=topic_num_font)
        num_x = (width - (num_bbox[2] - num_bbox[0])) // 2
        draw.text(
            (num_x, height // 2 - 100),
            topic_num_text,
            fill=colors['accent'],
            font=topic_num_font
        )

        # Draw main title
        title_font = self.fonts.get('title', ImageFont.load_default())
        wrapped_title = self._wrap_text(topic.title, width - 100, title_font, draw)
        title_bbox = draw.textbbox((0, 0), wrapped_title, font=title_font)
        title_x = (width - (title_bbox[2] - title_bbox[0])) // 2
        draw.text(
            (title_x, height // 2 - 30),
            wrapped_title,
            fill=colors['text'],
            font=title_font
        )

        # Draw subtitle
        if topic.subtitle:
            sub_font = self.fonts.get('subheading', ImageFont.load_default())
            sub_bbox = draw.textbbox((0, 0), topic.subtitle, font=sub_font)
            sub_x = (width - (sub_bbox[2] - sub_bbox[0])) // 2
            draw.text(
                (sub_x, height // 2 + 50),
                topic.subtitle,
                fill=colors['secondary'],
                font=sub_font
            )

        # Draw exam tag
        if topic.exam_tag:
            tag_font = self.fonts.get('small', ImageFont.load_default())
            icon = self.EXAM_ICONS.get(topic.exam_tag, '')
            tag_text = f"{icon} {topic.exam_tag}"

            # Tag background
            tag_bbox = draw.textbbox((0, 0), tag_text, font=tag_font)
            tag_width = tag_bbox[2] - tag_bbox[0] + 30
            tag_x = (width - tag_width) // 2
            tag_y = height // 2 + 100

            draw.rounded_rectangle(
                (tag_x, tag_y, tag_x + tag_width, tag_y + 35),
                radius=5,
                fill=colors['accent']
            )
            draw.text(
                (tag_x + 15, tag_y + 7),
                tag_text,
                fill=colors['bg'],
                font=tag_font
            )

        # Draw subject tag if present
        if topic.subject:
            subject_font = self.fonts.get('tiny', ImageFont.load_default())
            draw.text(
                (50, height - 50),
                f"Subject: {topic.subject}",
                fill=colors['secondary'],
                font=subject_font
            )

        # Save and create clip
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            clip = ImageClip(f.name).set_duration(topic.duration)

        clip = clip.set_start(topic.start_time)
        clip = fadein(clip, 0.5)
        clip = fadeout(clip, 0.5)

        return clip

    def create_image_overlay(
        self,
        image_overlay: ImageOverlay,
        video_size: Tuple[int, int]
    ) -> Optional[VideoClip]:
        """
        Create an image overlay (for maps, diagrams, etc.).

        Args:
            image_overlay: ImageOverlay configuration
            video_size: Video dimensions

        Returns:
            VideoClip with the image overlay, or None if image not found
        """
        width, height = video_size

        try:
            # Load image
            if image_overlay.image_path.startswith(('http://', 'https://')):
                response = requests.get(image_overlay.image_path, timeout=10)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_overlay.image_path)

            # Convert to RGBA
            img = img.convert('RGBA')

            # Calculate size
            target_width = int(width * image_overlay.scale)
            aspect_ratio = img.height / img.width
            target_height = int(target_width * aspect_ratio)

            # Resize
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # Add border/frame
            framed = Image.new('RGBA', (target_width + 10, target_height + 10), (255, 255, 255, 255))
            framed.paste(img, (5, 5))

            # Add caption if present
            if image_overlay.caption:
                draw = ImageDraw.Draw(framed)
                caption_font = self.fonts.get('tiny', ImageFont.load_default())

                # Caption background
                caption_height = 25
                new_height = framed.height + caption_height
                captioned = Image.new('RGBA', (framed.width, new_height), (0, 0, 0, 0))
                captioned.paste(framed, (0, 0))

                draw = ImageDraw.Draw(captioned)
                draw.rectangle(
                    (0, framed.height, framed.width, new_height),
                    fill=(40, 40, 40, 200)
                )
                draw.text(
                    (10, framed.height + 5),
                    image_overlay.caption,
                    fill=(255, 255, 255, 255),
                    font=caption_font
                )
                framed = captioned

            # Save and create clip
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                framed.save(f.name)
                clip = ImageClip(f.name).set_duration(image_overlay.duration)

            # Position
            if image_overlay.position == 'left':
                x_pos = 30
            elif image_overlay.position == 'right':
                x_pos = width - framed.width - 30
            elif image_overlay.position == 'center':
                x_pos = (width - framed.width) // 2
            else:  # fullscreen
                clip = clip.resize((width, height))
                x_pos = 0

            y_pos = (height - framed.height) // 2 if image_overlay.position != 'fullscreen' else 0

            clip = clip.set_position((x_pos, y_pos))
            clip = clip.set_start(image_overlay.start_time)
            clip = fadein(clip, 0.3)
            clip = fadeout(clip, 0.3)

            return clip

        except Exception as e:
            logger.warning(f"Failed to create image overlay: {e}")
            return None

    def create_timeline_bar(
        self,
        events: List[Dict[str, str]],
        video_size: Tuple[int, int],
        start_time: float,
        duration: float
    ) -> VideoClip:
        """
        Create a timeline visualization for historical events.

        Args:
            events: List of {date, event} dictionaries
            video_size: Video dimensions
            start_time: When to show the timeline
            duration: How long to show

        Returns:
            VideoClip with timeline
        """
        width, height = video_size
        colors = self.THEMES['blue']

        # Timeline dimensions
        timeline_width = int(width * 0.8)
        timeline_height = 150
        padding = 30

        img = Image.new('RGBA', (timeline_width, timeline_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Background
        self._draw_rounded_rect(
            draw,
            (0, 0, timeline_width, timeline_height),
            radius=10,
            fill=colors['bg'] + (230,)
        )

        # Title
        title_font = self.fonts.get('small', ImageFont.load_default())
        draw.text((padding, 10), "TIMELINE", fill=colors['accent'], font=title_font)

        # Draw timeline line
        line_y = 60
        draw.line(
            (padding, line_y, timeline_width - padding, line_y),
            fill=colors['secondary'],
            width=3
        )

        # Draw events
        if events:
            event_spacing = (timeline_width - 2 * padding) // min(len(events), 5)
            event_font = self.fonts.get('tiny', ImageFont.load_default())

            for i, event in enumerate(events[:5]):
                x = padding + i * event_spacing + event_spacing // 2

                # Event dot
                draw.ellipse(
                    (x - 6, line_y - 6, x + 6, line_y + 6),
                    fill=colors['accent']
                )

                # Date
                date_text = event.get('date', '')[:15]
                draw.text((x - 20, line_y - 25), date_text, fill=colors['text'], font=event_font)

                # Event (truncated)
                event_text = event.get('event', '')[:20]
                draw.text((x - 30, line_y + 15), event_text, fill=colors['secondary'], font=event_font)

        # Save and create clip
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            clip = ImageClip(f.name).set_duration(duration)

        clip = clip.set_position(('center', 'bottom'))
        clip = clip.set_start(start_time)
        clip = fadein(clip, 0.4)
        clip = fadeout(clip, 0.4)

        return clip

    def create_stats_card(
        self,
        stats: Dict[str, str],
        video_size: Tuple[int, int],
        start_time: float,
        duration: float,
        title: str = "Key Statistics"
    ) -> VideoClip:
        """
        Create a statistics card showing important numbers.

        Args:
            stats: Dictionary of stat_name: value
            video_size: Video dimensions
            start_time: When to show
            duration: How long to show
            title: Card title

        Returns:
            VideoClip with stats card
        """
        width, height = video_size
        colors = self.THEMES['green']

        # Card dimensions
        card_width = int(width * 0.35)
        card_height = min(80 + len(stats) * 50, int(height * 0.5))

        img = Image.new('RGBA', (card_width, card_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Background
        self._draw_rounded_rect(
            draw,
            (0, 0, card_width, card_height),
            radius=12,
            fill=colors['bg'] + (235,)
        )

        # Header
        draw.rectangle((0, 0, card_width, 50), fill=colors['primary'])
        title_font = self.fonts.get('subheading', ImageFont.load_default())
        draw.text((15, 12), title, fill=colors['text'], font=title_font)

        # Stats
        stat_font = self.fonts.get('small', ImageFont.load_default())
        value_font = self.fonts.get('heading', ImageFont.load_default())
        y = 60

        for name, value in list(stats.items())[:5]:
            # Value (large)
            draw.text((15, y), str(value), fill=colors['accent'], font=value_font)

            # Name (small, below)
            draw.text((15, y + 30), name, fill=colors['secondary'], font=stat_font)
            y += 55

        # Save and create clip
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            clip = ImageClip(f.name).set_duration(duration)

        clip = clip.set_position((width - card_width - 30, 100))
        clip = clip.set_start(start_time)
        clip = fadein(clip, 0.3)
        clip = fadeout(clip, 0.3)

        return clip

    def _draw_rounded_rect(
        self,
        draw: ImageDraw.ImageDraw,
        coords: Tuple[int, int, int, int],
        radius: int,
        fill: Tuple
    ):
        """Draw a rounded rectangle."""
        x1, y1, x2, y2 = coords
        draw.rounded_rectangle(coords, radius=radius, fill=fill)

    def _wrap_text(
        self,
        text: str,
        max_width: int,
        font: ImageFont.FreeTypeFont,
        draw: ImageDraw.ImageDraw
    ) -> str:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines[:3])  # Max 3 lines


# Utility function to download and cache images
def download_image(url: str, cache_dir: str = "assets/cache") -> Optional[str]:
    """Download an image and cache it locally."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create filename from URL hash
    import hashlib
    filename = hashlib.md5(url.encode()).hexdigest() + ".png"
    local_path = cache_path / filename

    if local_path.exists():
        return str(local_path)

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img.save(local_path)

        return str(local_path)
    except Exception as e:
        logger.warning(f"Failed to download image {url}: {e}")
        return None

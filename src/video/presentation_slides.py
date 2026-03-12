"""
Presentation Slides Generator - Creates timed presentation slides from script data
for use as video background behind the avatar.

Adapted for History Video Course. Each lesson segment gets a professional slide showing:
- Topic title and part number
- Key bullet points
- Important terms / abbreviations (as pill badges)
- Era category (Ancient/Medieval/Modern) and exam relevance tags
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import ImageClip, CompositeVideoClip, ColorClip
from moviepy.video.fx.all import fadein, fadeout

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SlideContent:
    """Data for one presentation slide."""
    title: str
    subtitle: str                              # subject category
    bullet_points: List[str] = field(default_factory=list)
    important_terms: Dict[str, str] = field(default_factory=dict)
    exam_tag: str = ""                         # PRELIMS / MAINS / BOTH
    topic_number: int = 1
    start_time: float = 0.0
    duration: float = 10.0
    table_data: Optional[List[List[str]]] = None
    content: str = ""                          # script narration text (for fallback notes)


class PresentationSlideGenerator:
    """
    Generates presentation-style slide images from script data.
    Slides are designed so the left ~35 % can be overlaid by the avatar
    while key content remains readable on the right.
    """

    # Color themes – each subject has a unique, vivid palette for instant recognition
    THEMES = {
        'Polity': {
            'primary': (26, 54, 93), 'accent': (66, 153, 225), 'header': (20, 45, 80),
            'badge_bg': (37, 99, 186), 'bullet_highlight': (100, 180, 255),
            'card_bg': (22, 38, 62), 'gradient_start': (15, 35, 70), 'gradient_end': (26, 54, 93),
        },
        'Economy': {
            'primary': (20, 70, 45), 'accent': (46, 204, 113), 'header': (15, 55, 35),
            'badge_bg': (34, 139, 84), 'bullet_highlight': (80, 220, 140),
            'card_bg': (18, 42, 32), 'gradient_start': (12, 35, 25), 'gradient_end': (20, 70, 45),
        },
        'International Relations': {
            'primary': (68, 51, 122), 'accent': (155, 89, 255), 'header': (55, 40, 100),
            'badge_bg': (98, 71, 170), 'bullet_highlight': (190, 150, 255),
            'card_bg': (40, 32, 68), 'gradient_start': (30, 24, 55), 'gradient_end': (68, 51, 122),
        },
        'Environment': {
            'primary': (15, 80, 70), 'accent': (0, 206, 180), 'header': (10, 65, 55),
            'badge_bg': (20, 130, 110), 'bullet_highlight': (60, 230, 200),
            'card_bg': (12, 45, 40), 'gradient_start': (8, 38, 32), 'gradient_end': (15, 80, 70),
        },
        'Science & Technology': {
            'primary': (20, 50, 80), 'accent': (0, 180, 255), 'header': (15, 40, 65),
            'badge_bg': (30, 120, 200), 'bullet_highlight': (80, 210, 255),
            'card_bg': (16, 32, 52), 'gradient_start': (10, 25, 45), 'gradient_end': (20, 50, 80),
        },
        'Social Issues': {
            'primary': (124, 45, 18), 'accent': (255, 165, 50), 'header': (100, 35, 14),
            'badge_bg': (180, 80, 30), 'bullet_highlight': (255, 195, 100),
            'card_bg': (55, 28, 15), 'gradient_start': (45, 22, 10), 'gradient_end': (124, 45, 18),
        },
        'Security': {
            'primary': (120, 30, 30), 'accent': (240, 80, 80), 'header': (95, 22, 22),
            'badge_bg': (180, 45, 45), 'bullet_highlight': (255, 120, 120),
            'card_bg': (55, 20, 20), 'gradient_start': (42, 15, 15), 'gradient_end': (120, 30, 30),
        },
        'Geography': {
            'primary': (55, 85, 30), 'accent': (140, 200, 60), 'header': (42, 68, 22),
            'badge_bg': (80, 130, 40), 'bullet_highlight': (170, 225, 90),
            'card_bg': (30, 45, 18), 'gradient_start': (24, 38, 12), 'gradient_end': (55, 85, 30),
        },
        'History': {
            'primary': (100, 60, 20), 'accent': (218, 165, 32), 'header': (80, 48, 15),
            'badge_bg': (160, 100, 30), 'bullet_highlight': (240, 200, 80),
            'card_bg': (50, 35, 15), 'gradient_start': (40, 28, 10), 'gradient_end': (100, 60, 20),
        },
        'Current Affairs': {
            'primary': (26, 54, 93), 'accent': (66, 153, 225), 'header': (20, 45, 80),
            'badge_bg': (37, 99, 186), 'bullet_highlight': (100, 180, 255),
            'card_bg': (22, 38, 62), 'gradient_start': (15, 35, 70), 'gradient_end': (26, 54, 93),
        },
        # History era-specific themes
        'Ancient History': {
            'primary': (139, 69, 19), 'accent': (218, 165, 32), 'header': (100, 50, 10),
            'badge_bg': (180, 100, 25), 'bullet_highlight': (240, 200, 70),
            'card_bg': (60, 38, 15), 'gradient_start': (80, 40, 10), 'gradient_end': (139, 69, 19),
        },
        'Medieval History': {
            'primary': (92, 56, 16), 'accent': (184, 134, 11), 'header': (72, 42, 10),
            'badge_bg': (140, 90, 20), 'bullet_highlight': (220, 180, 50),
            'card_bg': (48, 32, 12), 'gradient_start': (60, 36, 8), 'gradient_end': (92, 56, 16),
        },
        'Modern History': {
            'primary': (107, 26, 26), 'accent': (205, 133, 63), 'header': (85, 18, 18),
            'badge_bg': (160, 50, 50), 'bullet_highlight': (230, 160, 90),
            'card_bg': (55, 20, 20), 'gradient_start': (70, 15, 15), 'gradient_end': (107, 26, 26),
        },
    }

    EXAM_TAG_COLORS = {
        'PRELIMS': (49, 130, 206),
        'MAINS': (128, 90, 213),
        'BOTH': (56, 161, 105),
    }

    def __init__(
        self,
        content_start_x_pct: float = 0.33,
        max_key_points: int = 4,
        show_subject_badge: bool = True,
        show_terms_as_badges: bool = True,
        bullet_style: str = "numbered",
    ):
        """
        Args:
            content_start_x_pct: Fraction of width where slide content starts
                                 (left of this is avatar territory).
            max_key_points: Max bullet points to show (4 fits best with larger fonts).
            show_subject_badge: Show prominent subject badge bar on right panel.
            show_terms_as_badges: Show terms as pill badges instead of a table.
            bullet_style: 'numbered' for circled numbers, 'dots' for classic dots.
        """
        self.content_start_x_pct = content_start_x_pct
        self.max_key_points = max_key_points
        self.show_subject_badge = show_subject_badge
        self.show_terms_as_badges = show_terms_as_badges
        self.bullet_style = bullet_style
        self._load_fonts()
        logger.info("PresentationSlideGenerator initialized")

    def _load_fonts(self):
        """Load fonts with fallback chain."""
        self.fonts = {}
        sizes = {
            'title': 42,
            'heading': 32,
            'section_label': 22,
            'body': 30,
            'sub_body': 24,
            'small': 20,
            'tiny': 16,
            'tag': 18,
            'number': 34,
        }
        for name, size in sizes.items():
            bold = name in ('heading', 'number')
            self.fonts[name] = self._try_load_font(size, bold=bold)

    @staticmethod
    def _try_load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        if bold:
            paths = ["arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        else:
            paths = ["arial.ttf", "arialbd.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        for path in paths:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    # ── Public API ────────────────────────────────────────────────────────

    def generate_slides(
        self,
        script_data: Dict[str, Any],
        video_size: Tuple[int, int],
        total_duration: float
    ) -> List:
        """
        Generate all presentation slide clips from script_data.

        Returns a list of moviepy ImageClip objects, each with
        set_start / set_duration / fade applied.
        """
        segments = script_data.get('segments', [])
        # History lessons use main_content, subtopic, exam_corner segment types
        content_segments = [
            s for s in segments
            if s.get('type') in ('news', 'main_content', 'subtopic', 'exam_corner')
        ]

        if not content_segments:
            return []

        # Get part number and era from script_data for display
        part_number = script_data.get('part_number', 0)
        era = script_data.get('era', 'History')

        clips = []
        for i, segment in enumerate(content_segments):
            start = self._parse_timestamp(segment.get('timestamp', '00:00'))

            # Duration = gap until next segment (or until total_duration)
            if i + 1 < len(content_segments):
                next_start = self._parse_timestamp(content_segments[i + 1].get('timestamp', '00:00'))
                duration = max(next_start - start, 5.0)
            else:
                duration = max(total_duration - start, 5.0)

            bullet_points = segment.get('key_points', [])
            content = segment.get('content', '')

            # Fallback: extract key sentences from content when key_points is empty
            if not bullet_points and content:
                bullet_points = self._extract_points_from_content(content)

            # Use segment title or fall back to topic title
            slide_title = segment.get('title', '') or segment.get('article_title', 'Topic')

            # Use era as subtitle (maps to color theme)
            subtitle = segment.get('subject_category', era)

            slide = SlideContent(
                title=slide_title,
                subtitle=subtitle,
                bullet_points=bullet_points,
                important_terms=segment.get('important_terms', {}),
                exam_tag=segment.get('exam_relevance', ''),
                topic_number=i + 1,
                start_time=start,
                duration=duration,
                table_data=self._extract_table_data(segment),
                content=content,
            )

            try:
                clip = self._slide_to_clip(slide, video_size)
                clips.append(clip)
            except Exception as e:
                logger.warning(f"Failed to create slide {i + 1}: {e}")

        logger.info(f"Generated {len(clips)} presentation slides")
        return clips

    # ── Internal ──────────────────────────────────────────────────────────

    def _slide_to_clip(self, slide: SlideContent, video_size: Tuple[int, int]):
        """Render one slide image and wrap as a timed ImageClip."""
        img = self._create_slide_image(slide, video_size)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            clip = ImageClip(f.name).set_duration(slide.duration)

        clip = clip.set_start(slide.start_time)
        clip = fadein(clip, 0.5)
        clip = fadeout(clip, 0.5)
        return clip

    def _create_slide_image(
        self,
        slide: SlideContent,
        video_size: Tuple[int, int]
    ) -> Image.Image:
        """Create a single presentation slide as a PIL image (full-screen layout)."""
        width, height = video_size
        colors = self.THEMES.get(slide.subtitle, self.THEMES['Current Affairs'])

        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Full-screen: content starts from left edge (content_x=0)
        content_x = 0
        header_h = 90
        footer_h = 55
        footer_y = height - footer_h

        # 1. Header bar (full width)
        self._draw_header(draw, img, slide, width, header_h, colors, content_x)

        # 2. Full-screen gradient background
        self._draw_right_panel_gradient(draw, content_x, header_h, footer_y, width, colors)

        # 3. Subject badge bar (full width)
        y_cursor = header_h + 8
        if self.show_subject_badge:
            y_cursor = self._draw_subject_badge_bar(draw, slide, content_x, width, y_cursor, colors)

        # 4. Numbered key points (full width, with card backgrounds)
        y_cursor = self._draw_key_points_enhanced(draw, slide, content_x, width, y_cursor, colors)

        # 5. Terms as pill badges (or fallback to table)
        if y_cursor < footer_y - 60:
            if slide.important_terms and self.show_terms_as_badges:
                y_cursor = self._draw_terms_as_badges(draw, slide, content_x, width, y_cursor, footer_y, colors)
            elif slide.table_data:
                y_cursor = self._draw_table(draw, slide.table_data, y_cursor,
                                            content_x + 20, width - 30, colors)
            elif slide.important_terms:
                y_cursor = self._draw_terms(draw, slide.important_terms, y_cursor,
                                            content_x + 20, width - 30, colors)

        # 6. Footer bar (full width)
        self._draw_footer(draw, slide, width, height, footer_h, colors, content_x)

        return img

    # ── Drawing helpers ───────────────────────────────────────────────────

    def _draw_header(self, draw, img, slide, width, header_h, colors, content_x=0):
        """Draw the slide header bar with topic number, title, exam tag."""
        # Header background (right panel only to avoid avatar overlap)
        draw.rectangle([(content_x, 0), (width, header_h)], fill=colors['header'])

        # Bottom border
        draw.line([(content_x, header_h), (width, header_h)],
                  fill=colors['accent'], width=3)

        # Topic number badge
        badge_text = f"#{slide.topic_number}"
        badge_font = self.fonts['heading']
        badge_bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
        badge_w = badge_bbox[2] - badge_bbox[0] + 24
        badge_h = badge_bbox[3] - badge_bbox[1] + 14

        badge_x = content_x + 20
        badge_y = (header_h - badge_h) // 2
        draw.rounded_rectangle(
            [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
            radius=8, fill=colors['accent']
        )
        draw.text((badge_x + 12, badge_y + 5), badge_text,
                  fill=(20, 25, 40), font=badge_font)

        # Title text
        title_x = badge_x + badge_w + 20
        max_title_w = width - title_x - 160  # leave room for exam tag
        title_text = self._truncate_text(slide.title, max_title_w,
                                         self.fonts['title'], draw)
        title_y = (header_h - 42) // 2
        draw.text((title_x, title_y), title_text,
                  fill=(240, 245, 255), font=self.fonts['title'])

        # Exam tag badge (right side)
        if slide.exam_tag:
            tag_color = self.EXAM_TAG_COLORS.get(slide.exam_tag, (100, 100, 100))
            tag_text = slide.exam_tag
            tag_font = self.fonts['tag']
            tag_bbox = draw.textbbox((0, 0), tag_text, font=tag_font)
            tag_w = tag_bbox[2] - tag_bbox[0] + 20
            tag_h = tag_bbox[3] - tag_bbox[1] + 10

            tag_x = width - tag_w - 20
            tag_y = (header_h - tag_h) // 2
            draw.rounded_rectangle(
                [tag_x, tag_y, tag_x + tag_w, tag_y + tag_h],
                radius=6, fill=tag_color
            )
            draw.text((tag_x + 10, tag_y + 4), tag_text,
                      fill=(255, 255, 255), font=tag_font)

    def _draw_right_panel_gradient(self, draw, content_x, header_h, footer_y, width, colors):
        """Draw a subtle subject-colored gradient on the right panel area."""
        panel_h = footer_y - header_h
        gs = colors.get('gradient_start', colors['primary'])
        ge = colors.get('gradient_end', colors['header'])
        for y_offset in range(panel_h):
            t = y_offset / max(panel_h, 1)
            r = int(gs[0] * (1 - t) + ge[0] * t)
            g = int(gs[1] * (1 - t) + ge[1] * t)
            b = int(gs[2] * (1 - t) + ge[2] * t)
            draw.line(
                [(content_x, header_h + y_offset), (width, header_h + y_offset)],
                fill=(r, g, b)
            )

    def _draw_avatar_area(self, draw, content_x, header_h, footer_y, colors):
        """Draw minimal decoration for the left avatar zone (legacy, no-op for full-screen)."""
        if content_x <= 0:
            return  # Full-screen mode — no avatar area decoration
        accent_x = content_x - 6
        draw.line(
            [(accent_x, header_h + 10), (accent_x, footer_y - 10)],
            fill=colors['accent'], width=3
        )

    def _draw_subject_badge_bar(self, draw, slide, content_x, width, y_start, colors):
        """Draw a prominent subject category banner across the right panel."""
        bar_height = 42
        bar_x_start = content_x + 5
        bar_x_end = width - 15
        bar_y_end = y_start + bar_height

        badge_bg = colors.get('badge_bg', colors['primary'])
        draw.rounded_rectangle(
            [bar_x_start, y_start, bar_x_end, bar_y_end],
            radius=8, fill=badge_bg
        )

        subject_text = slide.subtitle.upper()
        draw.text(
            (bar_x_start + 18, y_start + 6),
            subject_text,
            fill=(255, 255, 255), font=self.fonts['heading']
        )

        if slide.exam_tag:
            tag_color = self.EXAM_TAG_COLORS.get(slide.exam_tag, (100, 100, 100))
            tag_text = slide.exam_tag
            tag_bbox = draw.textbbox((0, 0), tag_text, font=self.fonts['tag'])
            tag_w = tag_bbox[2] - tag_bbox[0] + 16
            tag_h = tag_bbox[3] - tag_bbox[1] + 8
            tag_x = bar_x_end - tag_w - 10
            tag_y = y_start + (bar_height - tag_h) // 2
            draw.rounded_rectangle(
                [tag_x, tag_y, tag_x + tag_w, tag_y + tag_h],
                radius=4, fill=tag_color
            )
            draw.text((tag_x + 8, tag_y + 3), tag_text,
                      fill=(255, 255, 255), font=self.fonts['tag'])

        return bar_y_end + 10

    def _draw_key_points_enhanced(self, draw, slide, content_x, width, y_start, colors):
        """Draw numbered key points with card backgrounds and large readable text."""
        padding = 20
        right_margin = 30
        max_points = self.max_key_points

        if not slide.bullet_points:
            return y_start

        label_x = content_x + padding
        draw.text((label_x, y_start), "KEY POINTS",
                  fill=colors['accent'], font=self.fonts['section_label'])
        y_start += 28
        draw.line([(label_x, y_start), (label_x + 120, y_start)],
                  fill=colors['accent'], width=2)
        y_start += 12

        circle_size = 30  # diameter of number circle
        text_x_offset = circle_size + 18  # gap after circle
        max_text_w = width - content_x - padding - right_margin - text_x_offset

        bullet_hl = colors.get('bullet_highlight', colors['accent'])
        card_bg = colors.get('card_bg', (22, 28, 42))

        for j, point in enumerate(slide.bullet_points[:max_points]):
            card_y = y_start + 2

            # Measure text to determine card height
            text_x = content_x + padding + text_x_offset
            wrapped = self._wrap_text(point, max_text_w, self.fonts['body'], draw, max_lines=2)
            lines = wrapped.count('\n') + 1
            text_h = lines * 36 + 10  # 36px line height for 30px font
            card_h = max(text_h + 16, 56)

            card_x_start = content_x + padding - 5
            card_x_end = width - right_margin + 5

            # Card background
            draw.rounded_rectangle(
                [card_x_start, card_y, card_x_end, card_y + card_h],
                radius=6, fill=card_bg
            )

            # Left accent stripe
            draw.rectangle(
                [card_x_start, card_y + 4, card_x_start + 4, card_y + card_h - 4],
                fill=colors['accent']
            )

            # Numbered circle or dot
            circle_x = content_x + padding + 8
            circle_y = card_y + (card_h - circle_size) // 2

            if self.bullet_style == "numbered":
                draw.ellipse(
                    [circle_x, circle_y, circle_x + circle_size, circle_y + circle_size],
                    fill=bullet_hl
                )
                num_text = str(j + 1)
                num_bbox = draw.textbbox((0, 0), num_text, font=self.fonts['small'])
                num_w = num_bbox[2] - num_bbox[0]
                num_h = num_bbox[3] - num_bbox[1]
                draw.text(
                    (circle_x + circle_size // 2 - num_w // 2,
                     circle_y + circle_size // 2 - num_h // 2 - 1),
                    num_text, fill=(20, 25, 40), font=self.fonts['small']
                )
            else:
                dot_r = 6
                dot_cx = circle_x + circle_size // 2
                dot_cy = circle_y + circle_size // 2
                draw.ellipse(
                    [dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r],
                    fill=bullet_hl
                )

            # Point text
            text_y = card_y + (card_h - text_h) // 2 + 5
            draw.text((text_x, text_y), wrapped,
                      fill=(235, 240, 250), font=self.fonts['body'])

            y_start = card_y + card_h + 8

        return y_start

    def _draw_terms_as_badges(self, draw, slide, content_x, width, y_start, footer_y, colors):
        """Draw important terms as colored pill badges in a flow layout."""
        terms = slide.important_terms
        if not terms:
            return y_start

        padding = 20
        right_margin = 30

        label_x = content_x + padding
        draw.text((label_x, y_start), "KEY TERMS",
                  fill=colors['accent'], font=self.fonts['section_label'])
        y_start += 30

        badge_x = content_x + padding
        badge_y = y_start
        max_x = width - right_margin
        badge_h = 30
        badge_gap_x = 10
        badge_gap_y = 8
        badge_font = self.fonts['small']
        badge_bg = colors.get('badge_bg', colors['primary'])

        for term, _definition in list(terms.items())[:6]:
            text_bbox = draw.textbbox((0, 0), term, font=badge_font)
            text_w = text_bbox[2] - text_bbox[0]
            badge_w = text_w + 20

            if badge_x + badge_w > max_x:
                badge_x = content_x + padding
                badge_y += badge_h + badge_gap_y

            if badge_y + badge_h > footer_y - 10:
                break

            draw.rounded_rectangle(
                [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                radius=badge_h // 2,
                fill=badge_bg, outline=colors['accent'], width=1
            )
            draw.text(
                (badge_x + 10, badge_y + 5),
                term, fill=(255, 255, 255), font=badge_font
            )

            badge_x += badge_w + badge_gap_x

        return badge_y + badge_h + 15

    def _draw_footer(self, draw, slide, width, height, footer_h, colors, content_x=0):
        """Draw the bottom footer bar."""
        footer_y = height - footer_h
        draw.rectangle([(content_x, footer_y), (width, height)], fill=colors['header'])
        draw.line([(content_x, footer_y), (width, footer_y)], fill=colors['accent'], width=3)

        draw.text((content_x + 20, footer_y + 15), "UPSC History Course",
                  fill=(200, 210, 230), font=self.fonts['small'])

        sub_text = slide.subtitle
        sub_bbox = draw.textbbox((0, 0), sub_text, font=self.fonts['small'])
        sub_w = sub_bbox[2] - sub_bbox[0]
        draw.text(((width - sub_w) // 2, footer_y + 15), sub_text,
                  fill=colors['accent'], font=self.fonts['small'])

        topic_text = f"Topic {slide.topic_number}"
        t_bbox = draw.textbbox((0, 0), topic_text, font=self.fonts['small'])
        draw.text((width - (t_bbox[2] - t_bbox[0]) - 20, footer_y + 15),
                  topic_text, fill=(200, 210, 230), font=self.fonts['small'])

    def _draw_table(self, draw, table_data, y_start, x_start, x_end,
                    colors) -> int:
        """Draw a simple table. Returns the y position after the table."""
        if not table_data or len(table_data) < 2:
            return y_start

        col_count = len(table_data[0])
        table_w = x_end - x_start
        col_w = table_w // col_count
        row_h = 36
        font = self.fonts['small']

        # Table label
        draw.text((x_start, y_start), "IMPORTANT DATA",
                  fill=colors['accent'], font=self.fonts['tag'])
        y_start += 28

        for row_idx, row in enumerate(table_data[:7]):  # max 7 rows
            y = y_start + row_idx * row_h

            # Header row background
            if row_idx == 0:
                draw.rectangle(
                    [x_start, y, x_end, y + row_h],
                    fill=colors['primary']
                )
            else:
                # Alternating row colors
                bg = (22, 28, 42) if row_idx % 2 == 0 else (18, 22, 35)
                draw.rectangle([x_start, y, x_end, y + row_h], fill=bg)

            # Row border
            draw.line([(x_start, y + row_h), (x_end, y + row_h)],
                      fill=(50, 60, 80), width=1)

            # Cell text
            for col_idx, cell in enumerate(row[:col_count]):
                cx = x_start + col_idx * col_w + 10
                text = str(cell)[:30]  # truncate long text
                text_color = colors['accent'] if row_idx == 0 else (210, 215, 230)
                draw.text((cx, y + 8), text, fill=text_color, font=font)

            # Column dividers
            for col_idx in range(1, col_count):
                div_x = x_start + col_idx * col_w
                draw.line([(div_x, y), (div_x, y + row_h)],
                          fill=(50, 60, 80), width=1)

        # Outer border
        total_h = len(table_data[:7]) * row_h
        draw.rectangle(
            [x_start, y_start, x_end, y_start + total_h],
            outline=colors['accent'], width=2
        )

        return y_start + total_h + 15

    def _draw_terms(self, draw, terms, y_start, x_start, x_end,
                    colors) -> int:
        """Draw important terms as a compact key-value section."""
        if not terms:
            return y_start

        draw.text((x_start, y_start), "KEY TERMS",
                  fill=colors['accent'], font=self.fonts['tag'])
        y_start += 28

        draw.line([(x_start, y_start), (x_start + 80, y_start)],
                  fill=colors['accent'], width=2)
        y_start += 8

        font = self.fonts['small']
        max_w = x_end - x_start

        for i, (term, definition) in enumerate(list(terms.items())[:5]):
            text = f"{term}: {definition}"
            text = self._truncate_text(text, max_w, font, draw)

            # Term in accent color, definition in white
            draw.text((x_start + 5, y_start), term + ":",
                      fill=colors['accent'], font=font)

            term_bbox = draw.textbbox((0, 0), term + ": ", font=font)
            term_w = term_bbox[2] - term_bbox[0]

            draw.text((x_start + 5 + term_w, y_start), definition,
                      fill=(210, 215, 230), font=font)

            y_start += 30

        return y_start + 10

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_points_from_content(content: str, max_points: int = 4) -> List[str]:
        """
        Extract important sentences from narration content as fallback bullet points.
        Picks the most informative sentences (containing facts, numbers, names).
        """
        import re

        # Split content into sentences
        sentences = re.split(r'[.!?]\s+', content.strip())
        sentences = [s.strip().rstrip('.!?') for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return []

        # Score sentences by informativeness (numbers, proper nouns, key phrases)
        scored = []
        for sent in sentences:
            score = 0
            # Contains numbers / dates / percentages
            if re.search(r'\d', sent):
                score += 3
            # Contains proper nouns (capitalized words not at start)
            words = sent.split()
            proper_nouns = sum(1 for w in words[1:] if w[0:1].isupper()) if len(words) > 1 else 0
            score += proper_nouns
            # Contains key indicator phrases
            indicators = ['important', 'significant', 'launched', 'announced',
                          'crore', 'billion', 'million', 'percent', '%',
                          'first', 'new', 'key', 'major', 'historic']
            score += sum(1 for ind in indicators if ind in sent.lower())
            # Penalize very short or very long sentences
            if len(sent) < 30:
                score -= 2
            if len(sent) > 200:
                score -= 1
            scored.append((score, sent))

        # Sort by score descending and pick top sentences
        scored.sort(key=lambda x: x[0], reverse=True)
        points = []
        for _score, sent in scored[:max_points]:
            # Truncate long sentences
            if len(sent) > 100:
                sent = sent[:97] + "..."
            points.append(sent)

        return points

    @staticmethod
    def _parse_timestamp(ts: str) -> float:
        """Convert 'MM:SS' to seconds."""
        try:
            parts = ts.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return 0.0
        except (ValueError, AttributeError):
            return 0.0

    @staticmethod
    def _extract_table_data(segment: Dict[str, Any]) -> Optional[List[List[str]]]:
        """
        Extract table data from a segment.
        Uses important_terms as a two-column table if there are enough entries.
        """
        terms = segment.get('important_terms', {})
        if terms and len(terms) >= 2:
            rows = [["Term", "Definition"]]
            for term, defn in list(terms.items())[:6]:
                rows.append([str(term), str(defn)])
            return rows
        return None

    def _wrap_text(self, text: str, max_width: int,
                   font: ImageFont.FreeTypeFont,
                   draw: ImageDraw.ImageDraw,
                   max_lines: int = 3) -> str:
        """Wrap text to fit within max_width pixels."""
        words = text.split()
        lines = []
        current = []

        for word in words:
            test = ' '.join(current + [word])
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current.append(word)
            else:
                if current:
                    lines.append(' '.join(current))
                current = [word]

        if current:
            lines.append(' '.join(current))

        return '\n'.join(lines[:max_lines])

    def _truncate_text(self, text: str, max_width: int,
                       font: ImageFont.FreeTypeFont,
                       draw: ImageDraw.ImageDraw) -> str:
        """Truncate text with ellipsis if it exceeds max_width."""
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return text

        while len(text) > 3:
            text = text[:-1]
            bbox = draw.textbbox((0, 0), text + "...", font=font)
            if bbox[2] - bbox[0] <= max_width:
                return text + "..."

        return text

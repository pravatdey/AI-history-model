"""
Thumbnail Generator - Creates UPSC-style YouTube thumbnails for videos.

Produces thumbnails matching the "Current Affairs UPSC" channel aesthetic:
- Dark navy blue background with subtle tech-grid overlay
- Bold yellow/orange headline text with glow outline
- Category badge (e.g. "DAILY DOSE", "MCQ CHALLENGE", "WEEKLY ROUNDUP")
- Decorative accent bar and bottom branding strip
- Optional clock/newspaper icon element
"""

from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass
import textwrap
import math

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Colour palette (matches screenshot style)
# ---------------------------------------------------------------------------
NAVY_DARK   = (10,  18,  40)   # deepest background
NAVY_MID    = (15,  30,  70)   # mid background
NAVY_LIGHT  = (20,  45, 100)   # lighter layer / grid lines
GOLD        = (255, 200,  30)   # primary accent – yellow/gold
ORANGE      = (255, 140,  20)   # secondary accent – orange
CYAN        = ( 30, 220, 255)   # highlight / glow
WHITE       = (255, 255, 255)
RED_BADGE   = (200,  30,  50)   # badge background
DARK_BADGE  = ( 20,  20,  50)   # dark badge variant


@dataclass
class ThumbnailResult:
    """Result of thumbnail generation."""
    success: bool
    thumbnail_path: str
    resolution: Tuple[int, int]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: font loader
# ---------------------------------------------------------------------------
_FONT_FALLBACKS_BOLD = [
    "arialbd.ttf", "Arial Bold", "Arial-Bold",
    "DejaVuSans-Bold.ttf", "DejaVuSans-Bold",
    "FreeSansBold.ttf", "FreeSansBold",
]
_FONT_FALLBACKS_REG = [
    "arial.ttf", "Arial", "DejaVuSans.ttf", "DejaVuSans", "FreeSans.ttf",
]

def _load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    fallbacks = _FONT_FALLBACKS_BOLD if bold else _FONT_FALLBACKS_REG
    for name in fallbacks:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _draw_text_outlined(
    draw: ImageDraw.ImageDraw,
    pos: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple,
    outline: Tuple = (0, 0, 0),
    outline_width: int = 4,
):
    """Draw text with a solid outline for legibility."""
    x, y = pos
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _centered_x(draw, text, font, canvas_width):
    w, _ = _text_size(draw, text, font)
    return (canvas_width - w) // 2


# ---------------------------------------------------------------------------
# Background creators
# ---------------------------------------------------------------------------
def _make_navy_gradient(size: Tuple[int, int]) -> Image.Image:
    """Create dark navy gradient background."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        r = int(NAVY_DARK[0] + (NAVY_MID[0] - NAVY_DARK[0]) * t)
        g = int(NAVY_DARK[1] + (NAVY_MID[1] - NAVY_DARK[1]) * t)
        b = int(NAVY_DARK[2] + (NAVY_MID[2] - NAVY_DARK[2]) * t)
        arr[y, :] = [r, g, b]
    return Image.fromarray(arr, "RGB")


def _add_tech_grid(img: Image.Image, alpha: int = 25) -> Image.Image:
    """Overlay a subtle circuit/grid pattern (like the MCQ Challenge panel)."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    grid_color = (*NAVY_LIGHT, alpha)
    step = 60
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=grid_color, width=1)
    for y in range(0, h, step):
        d.line([(0, y), (w, y)], fill=grid_color, width=1)

    # Dot nodes at intersections
    node_color = (*CYAN, alpha)
    for x in range(0, w, step):
        for y in range(0, h, step):
            d.ellipse([x-2, y-2, x+2, y+2], fill=node_color)

    base = img.convert("RGBA")
    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


def _add_radial_glow(img: Image.Image, center_x_frac: float = 0.5, color: Tuple = CYAN) -> Image.Image:
    """Add a subtle radial glow at a given horizontal position."""
    w, h = img.size
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    cx = int(w * center_x_frac)
    cy = h // 2
    for radius in range(300, 0, -20):
        a = max(0, int(18 * (1 - radius / 300)))
        d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                  fill=(*color, a))
    base = img.convert("RGBA")
    result = Image.alpha_composite(base, glow)
    return result.convert("RGB")


# ---------------------------------------------------------------------------
# Decorative elements
# ---------------------------------------------------------------------------
def _draw_gold_border(draw: ImageDraw.ImageDraw, size: Tuple[int, int], width: int = 8):
    """Draw a gold/orange rectangular border (like the screenshot frame)."""
    w, h = size
    for i in range(width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=GOLD)


def _draw_accent_bar(draw: ImageDraw.ImageDraw, y: int, width: int, color: Tuple, height: int = 5):
    """Horizontal coloured accent line."""
    draw.rectangle([0, y, width, y + height], fill=color)


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    x: int, y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    bg: Tuple = RED_BADGE,
    fg: Tuple = WHITE,
    padding_x: int = 18,
    padding_y: int = 10,
    radius: int = 8,
) -> Tuple[int, int]:
    """Draw a rounded-rectangle badge and return its (width, height)."""
    tw, th = _text_size(draw, text, font)
    bw = tw + padding_x * 2
    bh = th + padding_y * 2

    # Rounded rect via successive rectangles
    draw.rectangle([x + radius, y, x + bw - radius, y + bh], fill=bg)
    draw.rectangle([x, y + radius, x + bw, y + bh - radius], fill=bg)
    draw.ellipse([x, y, x + radius * 2, y + radius * 2], fill=bg)
    draw.ellipse([x + bw - radius * 2, y, x + bw, y + radius * 2], fill=bg)
    draw.ellipse([x, y + bh - radius * 2, x + radius * 2, y + bh], fill=bg)
    draw.ellipse([x + bw - radius * 2, y + bh - radius * 2, x + bw, y + bh], fill=bg)

    draw.text((x + padding_x, y + padding_y), text, font=font, fill=fg)
    return bw, bh


def _draw_clock_icon(draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int = 55):
    """Draw a simple analog clock (as in the 'Daily Dose 12 AM' panel)."""
    # Outer ring
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 outline=CYAN, width=4)
    draw.ellipse([cx - radius + 6, cy - radius + 6,
                  cx + radius - 6, cy + radius - 6],
                 outline=(*CYAN, 80), width=2)
    # Hour hand (pointing up – 12 o'clock)
    draw.line([cx, cy, cx, cy - int(radius * 0.6)], fill=WHITE, width=4)
    # Minute hand (pointing right)
    draw.line([cx, cy, cx + int(radius * 0.75), cy], fill=GOLD, width=3)
    # Center dot
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=WHITE)


def _draw_newspaper_stack(draw: ImageDraw.ImageDraw, x: int, y: int, w: int = 140, h: int = 100):
    """Draw a simplified stacked-newspaper graphic."""
    for i in range(3, -1, -1):
        ox, oy = i * 5, i * (-4)
        draw.rectangle([x + ox, y + oy, x + ox + w, y + oy + h],
                        fill=(30 + i * 10, 35 + i * 10, 55 + i * 10),
                        outline=(80, 90, 120), width=1)
        # Simulated text lines
        for row in range(3):
            lw = w - 20 - row * 10
            ly = y + oy + 15 + row * 18
            draw.rectangle([x + ox + 10, ly, x + ox + 10 + lw, ly + 5],
                            fill=(120, 130, 160))


def _draw_bottom_branding(draw: ImageDraw.ImageDraw, size: Tuple[int, int], channel_name: str = "HISTORY UPSC & STATE PSC"):
    """Draw the bottom branding bar (matches screenshot bottom strip)."""
    w, h = size
    bar_h = 52
    bar_y = h - bar_h
    # Dark background
    draw.rectangle([0, bar_y, w, h], fill=(8, 14, 35))
    # Gold top edge line
    draw.rectangle([0, bar_y, w, bar_y + 3], fill=GOLD)

    font = _load_font(26, bold=True)
    text_w, text_h = _text_size(draw, channel_name, font)
    tx = (w - text_w) // 2
    ty = bar_y + (bar_h - text_h) // 2 - 2
    draw.text((tx, ty), channel_name, font=font, fill=GOLD)


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------
STYLE_PRESETS = {
    "history_class": {
        "badge_text": "HISTORY CLASS",
        "badge_bg": (139, 69, 19),
        "subtitle": "UPSC & STATE PSC",
        "subtitle_color": GOLD,
        "icon": "newspaper",
        "glow_x": 0.25,
    },
    "ancient_history": {
        "badge_text": "ANCIENT HISTORY",
        "badge_bg": (160, 82, 16),
        "subtitle": "UPSC & STATE PSC",
        "subtitle_color": GOLD,
        "icon": "newspaper",
        "glow_x": 0.25,
    },
    "medieval_history": {
        "badge_text": "MEDIEVAL HISTORY",
        "badge_bg": (92, 56, 16),
        "subtitle": "UPSC & STATE PSC",
        "subtitle_color": GOLD,
        "icon": "newspaper",
        "glow_x": 0.5,
    },
    "modern_history": {
        "badge_text": "MODERN HISTORY",
        "badge_bg": (107, 26, 26),
        "subtitle": "UPSC & STATE PSC",
        "subtitle_color": GOLD,
        "icon": "newspaper",
        "glow_x": 0.75,
    },
    "revision": {
        "badge_text": "REVISION CLASS",
        "badge_bg": (30, 100, 200),
        "subtitle": "QUICK REVISION",
        "subtitle_color": CYAN,
        "icon": "grid",
        "glow_x": 0.5,
    },
    "default": {
        "badge_text": "HISTORY CLASS",
        "badge_bg": (139, 69, 19),
        "subtitle": "UPSC & STATE PSC",
        "subtitle_color": GOLD,
        "icon": "newspaper",
        "glow_x": 0.5,
    },
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class ThumbnailGenerator:
    """
    Generates UPSC-style YouTube thumbnails for History Course:
      - Dark navy/brown background + tech grid
      - Gold border frame
      - Bold yellow/orange headline with Part number
      - Era badge (Ancient/Medieval/Modern) and subtitle
      - Decorative icon
      - Bottom branding bar
    """

    DEFAULT_SIZE = (1280, 720)

    def __init__(
        self,
        size: Tuple[int, int] = None,
        channel_name: str = "HISTORY UPSC & STATE PSC",
    ):
        self.size = size or self.DEFAULT_SIZE
        self.channel_name = channel_name
        logger.info(f"ThumbnailGenerator initialized: {self.size[0]}x{self.size[1]}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        output_path: str,
        title: str,
        date: str = None,
        style: str = "default",
        background_image: str = None,
        video_path: str = None,
    ) -> ThumbnailResult:
        """
        Generate a UPSC-style thumbnail.

        Args:
            output_path:      Where to save the PNG.
            title:            Main headline text.
            date:             Optional date string shown below headline.
            style:            One of the STYLE_PRESETS keys.
            background_image: Optional custom BG image path (overrides generated BG).
            video_path:       Optional video for frame extraction (fallback BG).

        Returns:
            ThumbnailResult
        """
        try:
            preset = STYLE_PRESETS.get(style, STYLE_PRESETS["default"])
            img = self._build_background(background_image, video_path)
            img = self._composite_content(img, title, date, preset)
            self._save(img, output_path)
            logger.info(f"Thumbnail saved: {output_path}")
            return ThumbnailResult(True, str(output_path), self.size)
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            return ThumbnailResult(False, "", self.size, str(e))

    def generate_from_headlines(
        self,
        output_path: str,
        headlines: List[str],
        date: str = None,
        style: str = None,
    ) -> ThumbnailResult:
        """
        Auto-select style from headlines and generate thumbnail.

        Args:
            output_path: Save path.
            headlines:   List of article headlines (first is used as title).
            date:        Date string.
            style:       Force a specific preset; auto-detected if None.

        Returns:
            ThumbnailResult
        """
        title = headlines[0] if headlines else "Today's Top UPSC Current Affairs"
        if len(title) > 55:
            title = title[:52] + "…"

        if style is None:
            style = self._detect_style(title)

        return self.generate(output_path=output_path, title=title, date=date, style=style)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _detect_style(self, title: str) -> str:
        """Pick a preset based on keywords in the title (history-specific)."""
        t = title.lower()
        if any(k in t for k in ("ancient", "vedic", "maurya", "gupta", "indus", "stone age", "sangam")):
            return "ancient_history"
        if any(k in t for k in ("medieval", "sultanate", "mughal", "delhi", "bhakti", "sufi", "maratha")):
            return "medieval_history"
        if any(k in t for k in ("modern", "british", "gandhi", "congress", "revolt", "independence", "freedom")):
            return "modern_history"
        if any(k in t for k in ("revision", "recap", "summary", "pyq", "mock", "test")):
            return "revision"
        return "history_class"

    def _build_background(self, bg_image_path: str = None, video_path: str = None) -> Image.Image:
        """Build the base background image."""
        w, h = self.size

        if bg_image_path and Path(bg_image_path).exists():
            img = Image.open(bg_image_path).resize(self.size)
            # Darken to match navy feel
            img = ImageEnhance.Brightness(img).enhance(0.35)
        elif video_path:
            img = self._extract_video_frame(video_path)
            img = ImageEnhance.Brightness(img).enhance(0.35)
        else:
            img = _make_navy_gradient(self.size)

        img = _add_tech_grid(img, alpha=30)
        img = _add_radial_glow(img, center_x_frac=0.5, color=CYAN)
        return img

    def _composite_content(
        self,
        img: Image.Image,
        title: str,
        date: str,
        preset: dict,
    ) -> Image.Image:
        """Composite all text/graphic elements onto the background."""
        w, h = self.size
        draw = ImageDraw.Draw(img)

        # 1. Gold border
        _draw_gold_border(draw, self.size, width=10)

        # 2. Top accent bar (under border)
        _draw_accent_bar(draw, y=10, width=w, color=ORANGE, height=4)

        # 3. Decorative icon (left side)
        icon_type = preset.get("icon", "newspaper")
        if icon_type == "clock":
            _draw_clock_icon(draw, cx=130, cy=h // 2 - 20, radius=65)
        elif icon_type == "newspaper":
            _draw_newspaper_stack(draw, x=40, y=h // 2 - 80, w=130, h=110)
        # "grid" icon = rely on tech grid background

        # 4. Badge (top-left area, right of icon)
        badge_font = _load_font(28, bold=True)
        badge_x = 240 if icon_type != "grid" else 40
        badge_y = 35
        bw, bh = _draw_badge(
            draw, badge_x, badge_y,
            preset["badge_text"], badge_font,
            bg=preset["badge_bg"], fg=WHITE,
        )

        # 5. Subtitle (below badge, same x)
        sub_font = _load_font(38, bold=True)
        sub_text = preset["subtitle"]
        sub_x = badge_x
        sub_y = badge_y + bh + 12
        _draw_text_outlined(
            draw, (sub_x, sub_y), sub_text, sub_font,
            fill=preset["subtitle_color"], outline=(0, 0, 20), outline_width=3,
        )

        # 6. Horizontal divider below subtitle
        div_y = sub_y + 55
        draw.rectangle([badge_x, div_y, w - 50, div_y + 3], fill=GOLD)

        # 7. Main headline (centred, large, bold yellow)
        self._draw_headline(draw, title, top_y=div_y + 18)

        # 8. Date (bottom-right, small)
        if date:
            date_font = _load_font(30, bold=False)
            _draw_text_outlined(
                draw,
                (w - 320, h - 90),
                date, date_font,
                fill=GOLD, outline=(0, 0, 0), outline_width=2,
            )

        # 9. Bottom branding bar
        _draw_bottom_branding(draw, self.size, self.channel_name)

        return img

    def _draw_headline(self, draw: ImageDraw.ImageDraw, title: str, top_y: int):
        """Draw the main headline with large bold yellow text, centered."""
        w, h = self.size
        branding_h = 52
        available_h = h - branding_h - top_y - 20

        # Determine font size and wrapping
        for font_size in (72, 62, 54, 46, 40):
            font = _load_font(font_size, bold=True)
            # Wrap to fit width (account for left icon area)
            max_line_w = w - 260  # leave margin for icon + border
            words = title.split()
            lines = []
            current = ""
            for word in words:
                test = (current + " " + word).strip()
                tw, _ = _text_size(draw, test, font)
                if tw <= max_line_w:
                    current = test
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current:
                lines.append(current)

            line_h = font_size + 10
            total_h = len(lines) * line_h
            if total_h <= available_h or font_size == 40:
                break

        # Vertical centering within available space
        text_start_y = top_y + (available_h - total_h) // 2

        for i, line in enumerate(lines):
            tw, _ = _text_size(draw, line, font)
            # Centre among the right portion of canvas
            right_area_x = 220
            tx = right_area_x + (w - right_area_x - 40 - tw) // 2
            ty = text_start_y + i * line_h

            # Soft glow behind text
            glow_img = Image.new("RGBA", self.size, (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_img)
            for blur_offset in range(8, 0, -2):
                glow_draw.text(
                    (tx, ty), line, font=font,
                    fill=(*GOLD, 40)
                )
            glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=6))
            base = draw._image.convert("RGBA")
            composite = Image.alpha_composite(base, glow_img)
            draw._image.paste(composite.convert("RGB"))
            draw = ImageDraw.Draw(draw._image)

            # Actual text with outline
            _draw_text_outlined(
                draw, (tx, ty), line, font,
                fill=GOLD, outline=(0, 0, 0), outline_width=4,
            )

        return draw

    def _extract_video_frame(self, video_path: str) -> Image.Image:
        """Extract a frame from video as fallback background."""
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            frame = clip.get_frame(clip.duration * 0.3)
            clip.close()
            return Image.fromarray(frame).resize(self.size)
        except Exception as e:
            logger.warning(f"Frame extraction failed: {e}")
            return _make_navy_gradient(self.size)

    def _save(self, img: Image.Image, output_path: str):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(str(out), "PNG", optimize=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UPSC Thumbnail Generator CLI")
    parser.add_argument("--title", default="India Signs Historic Trade Deal with EU", help="Headline text")
    parser.add_argument("--date", default=None, help="Date text")
    parser.add_argument("--style", default="history_class",
                        choices=list(STYLE_PRESETS.keys()), help="Thumbnail style preset")
    parser.add_argument("--output", default="output/thumbnails/test_thumbnail.png")
    parser.add_argument("--background", default=None, help="Custom background image")
    parser.add_argument("--video", default=None, help="Video for frame extraction")
    args = parser.parse_args()

    gen = ThumbnailGenerator()
    result = gen.generate(
        output_path=args.output,
        title=args.title,
        date=args.date,
        style=args.style,
        background_image=args.background,
        video_path=args.video,
    )

    if result.success:
        print(f"\nSuccess! Thumbnail saved: {result.thumbnail_path}")
        print(f"Resolution: {result.resolution[0]}x{result.resolution[1]}")
    else:
        print(f"\nError: {result.error}")

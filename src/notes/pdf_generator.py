"""
PDF Notes Generator - History study notes
History Academy - UPSC & State PSC

Layout matches the reference Drishti IAS Daily Current Affairs format:
- Page header with academy name + date + page number
- Contextual trigger box at top of each topic
- Structured two-column layouts for Challenges vs Suggestions
- Hierarchical bullets: main (a) → sub (♦) → deeper (→)
- Comparison/contrast tables (e.g. Freebies vs Welfare)
- Key judgements / key facts boxes
- Important terms glossary table
- Practice questions section
- Dense, information-rich layout on A4
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, Frame, PageTemplate
)
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.colors import HexColor
from reportlab.platypus.flowables import HRFlowable

from .content_extractor import ExtractedContent, KeyPoint, UPSCRelevance
from src.utils.logger import get_logger

logger = get_logger(__name__)

ACADEMY_NAME = "History Academy"
ACADEMY_TAGLINE = "Your Daily Dose of Knowledge | UPSC | State PSC | Banking | SSC"
WEBSITE = "www.currentaffairsacademy.in"

# ---------------------------------------------------------------------------
# Colour palette  (Drishti IAS aesthetic – navy / white / grey)
# ---------------------------------------------------------------------------
C = {
    'navy':       HexColor('#0A2E5C'),
    'navy_light': HexColor('#1A4A8A'),
    'red':        HexColor('#C8372D'),
    'gold':       HexColor('#D4A017'),
    'dark':       HexColor('#1A202C'),
    'body':       HexColor('#2D3748'),
    'muted':      HexColor('#718096'),
    'white':      colors.white,
    'bg_light':   HexColor('#F7FAFC'),
    'bg_blue':    HexColor('#EBF4FF'),
    'bg_yellow':  HexColor('#FFFBEB'),
    'bg_green':   HexColor('#F0FFF4'),
    'bg_red':     HexColor('#FFF5F5'),
    'prelims':    HexColor('#2B6CB0'),
    'mains':      HexColor('#553C9A'),
    'border':     HexColor('#CBD5E0'),
    'orange':     HexColor('#C05621'),
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TopicNote:
    """One topic's complete structured notes."""
    title: str
    trigger_line: str           # 1-sentence news trigger (like Drishti IAS)
    what_is_it: str             # "About:" section — definition / overview
    key_provisions: List[str]   # Constitutional / legal / structural points
    sub_sections: List[Dict]    # [{'heading': str, 'points': [str], 'sub_points': {str: [str]}}]
    challenges: List[str]       # Challenge bullet points (left column)
    suggestions: List[str]      # Suggestion / way-forward bullet points (right column)
    comparison_table: Optional[Dict] = None   # {'headers': [], 'rows': [[]], 'col_widths': []}
    key_judgements: List[str] = field(default_factory=list)
    key_facts_box: List[str] = field(default_factory=list)   # sidebar facts
    important_terms: Dict[str, str] = field(default_factory=dict)
    practice_questions: List[str] = field(default_factory=list)
    upsc_tags: str = ""         # e.g. "GS2 | Polity | Prelims + Mains"
    timestamp: str = ""


@dataclass
class StudyNote:
    """Full day's notes."""
    title: str
    date: str
    topics: List[TopicNote]
    video_duration: float = 0.0
    language: str = "English"
    additional_resources: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Style factory
# ---------------------------------------------------------------------------

def _make_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    def S(name, parent='Normal', **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        # ── Header / Titles ──────────────────────────────────────────────
        'PageTitle': S('PageTitle', 'Normal',
                       fontSize=9, textColor=C['muted'], alignment=TA_RIGHT),

        'MainTitle': S('MainTitle', 'Normal',
                       fontSize=20, textColor=C['navy'],
                       alignment=TA_CENTER, fontName='Helvetica-Bold',
                       spaceAfter=4, leading=24),

        'SubTitle': S('SubTitle', 'Normal',
                      fontSize=11, textColor=C['navy_light'],
                      alignment=TA_CENTER, spaceAfter=10),

        # ── Topic heading ────────────────────────────────────────────────
        'TopicTitle': S('TopicTitle', 'Normal',
                        fontSize=13, textColor=C['white'],
                        fontName='Helvetica-Bold',
                        alignment=TA_LEFT,
                        spaceBefore=4, spaceAfter=0,
                        leading=17),

        # ── Trigger line ─────────────────────────────────────────────────
        'Trigger': S('Trigger', 'Normal',
                     fontSize=9, textColor=C['dark'],
                     alignment=TA_JUSTIFY, leading=13,
                     spaceBefore=2, spaceAfter=4,
                     fontName='Helvetica-Oblique'),

        # ── Section headers ───────────────────────────────────────────────
        'SecHeader': S('SecHeader', 'Normal',
                       fontSize=10, textColor=C['navy'],
                       fontName='Helvetica-Bold',
                       spaceBefore=7, spaceAfter=3),

        'SecHeaderRed': S('SecHeaderRed', 'Normal',
                          fontSize=10, textColor=C['red'],
                          fontName='Helvetica-Bold',
                          spaceBefore=7, spaceAfter=3),

        # ── Body text ────────────────────────────────────────────────────
        'Body': S('Body', 'Normal',
                  fontSize=9, textColor=C['body'],
                  alignment=TA_JUSTIFY, leading=13,
                  spaceBefore=2, spaceAfter=2),

        # ── Bullet levels ────────────────────────────────────────────────
        # Level 1  • ▸  (main bullet)
        'Bullet1': S('Bullet1', 'Normal',
                     fontSize=9, textColor=C['body'],
                     leftIndent=10, firstLineIndent=-10,
                     leading=13, spaceBefore=2, spaceAfter=2),

        # Level 2  ♦  (sub-bullet)
        'Bullet2': S('Bullet2', 'Normal',
                     fontSize=9, textColor=C['body'],
                     leftIndent=22, firstLineIndent=-10,
                     leading=13, spaceBefore=1, spaceAfter=1),

        # Level 3  →  (deep sub-bullet)
        'Bullet3': S('Bullet3', 'Normal',
                     fontSize=8.5, textColor=C['muted'],
                     leftIndent=34, firstLineIndent=-10,
                     leading=12, spaceBefore=1, spaceAfter=1),

        # ── Two-column bullets ────────────────────────────────────────────
        'ColBullet': S('ColBullet', 'Normal',
                       fontSize=9, textColor=C['body'],
                       leftIndent=10, firstLineIndent=-10,
                       leading=13, spaceBefore=2, spaceAfter=2),

        # ── Table styles ─────────────────────────────────────────────────
        'TH': S('TH', 'Normal',
                fontSize=9, textColor=C['white'],
                fontName='Helvetica-Bold', alignment=TA_CENTER, leading=12),

        'TD': S('TD', 'Normal',
                fontSize=8.5, textColor=C['body'],
                alignment=TA_JUSTIFY, leading=12,
                spaceBefore=2, spaceAfter=2),

        'TDLeft': S('TDLeft', 'Normal',
                    fontSize=8.5, textColor=C['navy'],
                    fontName='Helvetica-Bold',
                    alignment=TA_LEFT, leading=12),

        # ── Tags ─────────────────────────────────────────────────────────
        'Tag': S('Tag', 'Normal',
                 fontSize=8, textColor=C['white'],
                 fontName='Helvetica-Bold', alignment=TA_CENTER),

        # ── Questions ────────────────────────────────────────────────────
        'QNum': S('QNum', 'Normal',
                  fontSize=9, textColor=C['red'],
                  fontName='Helvetica-Bold',
                  spaceBefore=8, spaceAfter=2),

        'QText': S('QText', 'Normal',
                   fontSize=9, textColor=C['body'],
                   leftIndent=8, leading=13,
                   spaceBefore=0, spaceAfter=4),

        'QAns': S('QAns', 'Normal',
                  fontSize=8.5, textColor=C['muted'],
                  leftIndent=16, fontName='Helvetica-Oblique',
                  spaceBefore=1, spaceAfter=6),

        # ── Footer ───────────────────────────────────────────────────────
        'Footer': S('Footer', 'Normal',
                    fontSize=7.5, textColor=C['muted'],
                    alignment=TA_CENTER),

        # ── Key facts box ────────────────────────────────────────────────
        'FactItem': S('FactItem', 'Normal',
                      fontSize=8.5, textColor=C['body'],
                      leftIndent=8, firstLineIndent=-8,
                      leading=12, spaceBefore=1, spaceAfter=1),

        # ── Judgement item ────────────────────────────────────────────────
        'JudgeItem': S('JudgeItem', 'Normal',
                       fontSize=9, textColor=C['body'],
                       leftIndent=14, firstLineIndent=-14,
                       leading=13, spaceBefore=2, spaceAfter=2),
    }


# ---------------------------------------------------------------------------
# Page-level callback for running header + footer
# ---------------------------------------------------------------------------

class _HeaderFooterCanvas:
    """Mixin – we inject header/footer via onPage callback."""

    def __init__(self, academy: str, date: str, website: str):
        self.academy = academy
        self.date = date
        self.website = website

    def draw(self, canvas, doc):
        canvas.saveState()
        w, h = A4

        # ── Top header bar ──
        canvas.setFillColor(C['navy'])
        canvas.rect(1.5*cm, h - 1.2*cm, w - 3*cm, 0.7*cm, fill=1, stroke=0)

        canvas.setFont('Helvetica-Bold', 8.5)
        canvas.setFillColor(colors.white)
        canvas.drawString(1.7*cm, h - 0.85*cm, self.academy.upper())

        canvas.setFont('Helvetica', 8)
        canvas.drawRightString(w - 1.7*cm, h - 0.85*cm, f"{self.date}  |  {self.website}")

        # Gold underline
        canvas.setStrokeColor(C['gold'])
        canvas.setLineWidth(1.5)
        canvas.line(1.5*cm, h - 1.25*cm, w - 1.5*cm, h - 1.25*cm)

        # ── Bottom footer bar ──
        canvas.setStrokeColor(C['border'])
        canvas.setLineWidth(0.5)
        canvas.line(1.5*cm, 1.6*cm, w - 1.5*cm, 1.6*cm)

        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(C['muted'])
        canvas.drawCentredString(w / 2, 1.2*cm, self.academy + "  •  " + self.website)
        canvas.drawRightString(w - 1.5*cm, 1.2*cm, f"Page {doc.page}")

        canvas.restoreState()


# ---------------------------------------------------------------------------
# Element builders
# ---------------------------------------------------------------------------

def _horizontal_rule(color=C['border'], thickness=0.5) -> HRFlowable:
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceBefore=4, spaceAfter=4)


def _topic_header_table(title: str, tags: str, styles) -> Table:
    """Dark navy header band for topic (full page width)."""
    title_para = Paragraph(title.upper(), styles['TopicTitle'])
    tag_para = Paragraph(
        f"<font size='8' color='#FFD700'>{tags}</font>",
        styles['TopicTitle']
    )
    data = [[title_para, tag_para]]
    t = Table(data, colWidths=[11.5*cm, 5.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C['navy']),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING',  (0, 0), (0, 0), 8),
        ('RIGHTPADDING', (1, 0), (1, 0), 8),
        ('ALIGN',        (1, 0), (1, 0), 'RIGHT'),
    ]))
    return t


def _trigger_box(text: str, styles) -> Table:
    """Italic context sentence in a light-blue box."""
    para = Paragraph(text, styles['Trigger'])
    t = Table([[para]], colWidths=[17*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C['bg_blue']),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BOX',           (0, 0), (-1, -1), 0.8, C['navy_light']),
    ]))
    return t


def _section_header(text: str, styles, color_key='SecHeader') -> Paragraph:
    return Paragraph(text, styles[color_key])


def _bullet(text: str, styles, level=1) -> Paragraph:
    """Return a Paragraph with proper bullet prefix for the given level."""
    prefixes = {1: '▸ ', 2: '♦ ', 3: '→ '}
    prefix = prefixes.get(level, '• ')
    return Paragraph(f"{prefix}{text}", styles[f'Bullet{level}'])


def _two_column_section(
    left_heading: str,
    left_items: List[str],
    right_heading: str,
    right_items: List[str],
    styles,
    left_color=C['red'],
    right_color=C['navy'],
) -> Table:
    """
    Render challenges (left) and suggestions (right) in two equal columns,
    matching the Drishti IAS format.
    """

    def build_col(heading, items, hdr_color):
        elems = []
        # Column header band
        hdr = Table([[Paragraph(f"<b>{heading}</b>", styles['TH'])]],
                    colWidths=[8.2*cm])
        hdr.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), hdr_color),
            ('TOPPADDING',    (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elems.append(hdr)
        # Items
        for item in items:
            # Check for sub-structure: items may be "Main: sub" or plain strings
            elems.append(Paragraph(f"• {item}", styles['ColBullet']))
        return elems

    left_col  = build_col(left_heading,  left_items,  left_color)
    right_col = build_col(right_heading, right_items, right_color)

    # Pad to equal length
    max_len = max(len(left_col), len(right_col))
    while len(left_col)  < max_len: left_col.append(Spacer(1, 2))
    while len(right_col) < max_len: right_col.append(Spacer(1, 2))

    rows = [[left_col[i], right_col[i]] for i in range(max_len)]
    t = Table(rows, colWidths=[8.3*cm, 8.7*cm])
    t.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
        ('TOPPADDING',    (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LINEAFTER',     (0, 0), (0, -1), 0.5, C['border']),
    ]))
    return t


def _comparison_table(data: Dict, styles) -> Table:
    """
    data = {
        'headers': ['Aspect', 'Freebies', 'Welfare'],
        'rows': [['Definition', '...', '...'], ...],
        'col_widths': [3*cm, 7*cm, 7*cm]  # optional
    }
    """
    headers = data.get('headers', [])
    rows    = data.get('rows', [])
    widths  = data.get('col_widths', None)

    if not headers or not rows:
        return Spacer(1, 1)

    n_cols = len(headers)
    if widths is None:
        total = 17 * cm
        widths = [total / n_cols] * n_cols

    table_data = [[Paragraph(h, styles['TH']) for h in headers]]
    for row in rows:
        table_data.append([Paragraph(str(cell), styles['TD']) for cell in row])

    t = Table(table_data, colWidths=widths)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), C['navy']),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, C['bg_light']]),
        ('GRID',          (0, 0), (-1, -1), 0.4, C['border']),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 5),
        ('FONTNAME',      (0, 1), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (0, 1), (0, -1), C['navy']),
    ]))
    return t


def _key_judgements_box(judgements: List[str], styles) -> Table:
    """Yellow box listing key SC judgements."""
    elems = [Paragraph("<b>Key SC / Court Judgements:</b>", styles['SecHeader'])]
    for j in judgements:
        elems.append(Paragraph(f"♦ {j}", styles['JudgeItem']))
    t = Table([[elems]], colWidths=[17*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C['bg_yellow']),
        ('BOX',           (0, 0), (-1, -1), 0.8, C['gold']),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
    ]))
    return t


def _key_facts_box(facts: List[str], label: str, styles) -> Table:
    """Green sidebar box for key facts / news hooks."""
    elems = [Paragraph(f"<b>{label}</b>", styles['SecHeader'])]
    for f in facts:
        elems.append(Paragraph(f"ª {f}", styles['FactItem']))
    t = Table([[elems]], colWidths=[17*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C['bg_green']),
        ('BOX',           (0, 0), (-1, -1), 0.8, HexColor('#38A169')),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
    ]))
    return t


def _terms_table(terms: Dict[str, str], styles) -> Table:
    """Two-column term ↔ definition table."""
    if not terms:
        return Spacer(1, 1)
    rows = [[Paragraph('<b>Term / Concept</b>', styles['TH']),
             Paragraph('<b>Definition / Explanation</b>', styles['TH'])]]
    for term, defn in terms.items():
        rows.append([
            Paragraph(term, styles['TDLeft']),
            Paragraph(defn, styles['TD'])
        ])
    t = Table(rows, colWidths=[4.5*cm, 12.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), C['navy_light']),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, C['bg_light']]),
        ('GRID',          (0, 0), (-1, -1), 0.4, C['border']),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
        ('BACKGROUND',    (0, 1), (0, -1), C['bg_blue']),
    ]))
    return t


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class PDFNotesGenerator:
    """
    Generates Drishti IAS SARAANSH-style PDF study notes.

    Structure per topic:
      ┌─────────────────────────────────────────────────────────────┐
      │  [TOPIC TITLE — navy band]          [UPSC Tags — gold]      │
      ├─────────────────────────────────────────────────────────────┤
      │  [Trigger line — light blue box]                            │
      │  About / What is it                                         │
      │    ▸ bullet …   ♦ sub-bullet …                             │
      │  Key Provisions / Legal Framework                           │
      │    ▸ …                                                      │
      │  [Sub-sections as needed]                                   │
      │  [Key SC Judgements — yellow box]                           │
      │  ┌──────────────────┬───────────────────────────────────┐  │
      │  │  Challenges       │  Suggestions / Way Forward        │  │
      │  └──────────────────┴───────────────────────────────────┘  │
      │  [Comparison table if any]                                  │
      │  [Key facts green box]                                      │
      │  [Important Terms table]                                    │
      │  [Practice Questions]                                       │
      └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, output_dir: str = "output/notes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = _make_styles()
        logger.info(f"PDFNotesGenerator (Drishti style) ready → {self.output_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_notes(
        self,
        study_note: StudyNote,
        include_images: bool = True,
        include_questions: bool = True
    ) -> str:
        date_str  = datetime.now().strftime("%Y%m%d")
        safe      = "".join(c for c in study_note.title if c.isalnum() or c in ' -_')[:50]
        filename  = f"{date_str}_{safe.replace(' ', '_')}_Notes.pdf"
        out_path  = self.output_dir / filename

        logger.info(f"Generating Drishti-style PDF: {filename}")

        hf = _HeaderFooterCanvas(ACADEMY_NAME, study_note.date, WEBSITE)

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=1.8*cm,     # room for header
            bottomMargin=2.0*cm,  # room for footer
        )

        story = []

        # ── Cover / title page ──
        story.extend(self._cover_page(study_note))

        # ── Table of contents ──
        story.extend(self._toc(study_note))

        # ── Each topic ──
        for i, topic in enumerate(study_note.topics, 1):
            story.extend(self._topic_section(topic, i, include_questions))

        # ── Quick revision / key facts ──
        story.extend(self._quick_revision(study_note))

        # ── Consolidated practice questions ──
        if include_questions:
            story.extend(self._questions_section(study_note))

        # ── Back page ──
        story.extend(self._back_page(study_note))

        doc.build(story, onFirstPage=hf.draw, onLaterPages=hf.draw)
        logger.info(f"PDF saved: {out_path}")
        return str(out_path)

    # ------------------------------------------------------------------
    # Cover page
    # ------------------------------------------------------------------

    def _cover_page(self, note: StudyNote) -> List:
        S = self.styles
        elems = []

        elems.append(Spacer(1, 0.4*inch))

        # Academy banner
        banner = Table([[Paragraph(ACADEMY_NAME.upper(), S['MainTitle'])]],
                       colWidths=[17*cm])
        banner.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), C['navy']),
            ('TEXTCOLOR',     (0, 0), (-1, -1), colors.white),
            ('TOPPADDING',    (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ]))
        elems.append(banner)

        # Tagline strip
        tag_row = Table(
            [[Paragraph(ACADEMY_TAGLINE, S['SubTitle'])]],
            colWidths=[17*cm]
        )
        tag_row.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), C['red']),
            ('TEXTCOLOR',     (0, 0), (-1, -1), C['gold']),
            ('TOPPADDING',    (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elems.append(tag_row)

        elems.append(Spacer(1, 0.25*inch))

        # Doc-type label
        dtype = Table(
            [[Paragraph("HISTORY CLASS — COMPLETE STUDY NOTES",
                        S['MainTitle'])]],
            colWidths=[17*cm]
        )
        dtype.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), C['navy_light']),
            ('TEXTCOLOR',     (0, 0), (-1, -1), colors.white),
            ('TOPPADDING',    (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ]))
        elems.append(dtype)
        elems.append(Spacer(1, 0.15*inch))

        # Date + title
        elems.append(Paragraph(note.title, S['MainTitle']))
        elems.append(Paragraph(note.date, S['SubTitle']))
        elems.append(Spacer(1, 0.2*inch))

        # Info grid
        total_q = sum(len(t.practice_questions) for t in note.topics)
        info = [
            ['Topics Covered',     str(len(note.topics))],
            ['Language',           note.language],
            ['Practice Questions', f"{max(20, total_q)}+ MCQ & Descriptive"],
            ['Content Type',       'Detailed — More than video'],
        ]
        info_t = Table(info, colWidths=[6*cm, 11*cm])
        info_t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (0, -1), C['navy']),
            ('TEXTCOLOR',     (0, 0), (0, -1), colors.white),
            ('FONTNAME',      (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BACKGROUND',    (1, 0), (1, -1), C['bg_blue']),
            ('TEXTCOLOR',     (1, 0), (1, -1), C['dark']),
            ('FONTSIZE',      (0, 0), (-1, -1), 9.5),
            ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
            ('GRID',          (0, 0), (-1, -1), 0.5, C['border']),
        ]))
        elems.append(info_t)
        elems.append(Spacer(1, 0.2*inch))

        # What's inside box
        inside = (
            "<b>What's inside this PDF:</b><br/>"
            "✔ <b>Structured topic-wise notes</b> in Drishti IAS style<br/>"
            "✔ <b>Trigger context</b> for each topic — why it's in the news<br/>"
            "✔ <b>About / Definition</b> → Key Provisions → Sub-sections<br/>"
            "✔ <b>Challenges ↔ Suggestions</b> in two-column layout<br/>"
            "✔ <b>Key SC Judgements</b> and landmark case references<br/>"
            "✔ <b>Comparison Tables</b> (where applicable)<br/>"
            "✔ <b>Important Terms</b> glossary for each topic<br/>"
            "✔ <b>20+ Practice Questions</b> (MCQ + Descriptive)"
        )
        inside_t = Table(
            [[Paragraph(inside, S['Body'])]],
            colWidths=[17*cm]
        )
        inside_t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), C['bg_light']),
            ('BOX',           (0, 0), (-1, -1), 1.2, C['navy']),
            ('TOPPADDING',    (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING',   (0, 0), (-1, -1), 12),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
        ]))
        elems.append(inside_t)

        elems.append(PageBreak())
        return elems

    # ------------------------------------------------------------------
    # Table of contents
    # ------------------------------------------------------------------

    def _toc(self, note: StudyNote) -> List:
        S = self.styles
        elems = []

        elems.append(
            _topic_header_table("TABLE OF CONTENTS", "", S)
        )
        elems.append(Spacer(1, 0.15*inch))

        rows = [
            [Paragraph('<b>#</b>', S['TH']),
             Paragraph('<b>Topic</b>', S['TH']),
             Paragraph('<b>Subject</b>', S['TH']),
             Paragraph('<b>Exam</b>', S['TH'])]
        ]
        for i, t in enumerate(note.topics, 1):
            tag_parts = t.upsc_tags.split('|') if t.upsc_tags else ['', '', '']
            subject = tag_parts[1].strip() if len(tag_parts) > 1 else ''
            exam    = tag_parts[2].strip() if len(tag_parts) > 2 else 'Both'
            title   = t.title[:60] + ('…' if len(t.title) > 60 else '')
            rows.append([
                Paragraph(str(i), S['Body']),
                Paragraph(title,  S['Body']),
                Paragraph(subject, S['Body']),
                Paragraph(exam,   S['Body']),
            ])

        # Special sections
        for label in ['Quick Revision', '20 Practice Questions']:
            rows.append([Paragraph('', S['Body']),
                         Paragraph(f"<b>{label}</b>", S['SecHeader']),
                         Paragraph('', S['Body']),
                         Paragraph('', S['Body'])])

        toc_t = Table(rows, colWidths=[1*cm, 10*cm, 4*cm, 2*cm])
        toc_t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), C['navy']),
            ('ROWBACKGROUNDS',(0, 1), (-1, -3), [colors.white, C['bg_light']]),
            ('BACKGROUND',    (0, -2), (-1, -1), C['bg_yellow']),
            ('GRID',          (0, 0), (-1, -1), 0.4, C['border']),
            ('TOPPADDING',    (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ('ALIGN',         (0, 0), (0, -1), 'CENTER'),
            ('ALIGN',         (3, 0), (3, -1), 'CENTER'),
        ]))
        elems.append(toc_t)
        elems.append(PageBreak())
        return elems

    # ------------------------------------------------------------------
    # Topic section — the heart of the Drishti style
    # ------------------------------------------------------------------

    def _topic_section(self, topic: TopicNote, num: int, include_qs: bool) -> List:
        S  = self.styles
        el = []

        # ── 1. Header band ──────────────────────────────────────────────
        el.append(_topic_header_table(
            f"Topic {num}: {topic.title}",
            topic.upsc_tags,
            S
        ))

        # ── 2. Trigger line ─────────────────────────────────────────────
        if topic.trigger_line:
            el.append(Spacer(1, 0.06*inch))
            el.append(_trigger_box(topic.trigger_line, S))

        el.append(Spacer(1, 0.08*inch))

        # ── 3. What is it / About ────────────────────────────────────────
        if topic.what_is_it:
            el.append(_section_header("What is it / About:", S))
            el.append(Paragraph(topic.what_is_it, S['Body']))
            el.append(Spacer(1, 0.04*inch))

        # ── 4. Key Provisions ────────────────────────────────────────────
        if topic.key_provisions:
            el.append(_section_header("Key Provisions / Legal Framework:", S))
            for prov in topic.key_provisions:
                # Support nested: "Main point: sub a, sub b"
                if '\n' in prov:
                    lines = prov.split('\n')
                    el.append(_bullet(lines[0], S, level=1))
                    for sub in lines[1:]:
                        if sub.strip():
                            el.append(_bullet(sub.strip(), S, level=2))
                else:
                    el.append(_bullet(prov, S, level=1))
            el.append(Spacer(1, 0.04*inch))

        # ── 5. Sub-sections (About / Context / Types / Impact …) ─────────
        for sec in topic.sub_sections:
            heading = sec.get('heading', '')
            points  = sec.get('points', [])
            subs    = sec.get('sub_points', {})  # {point_text: [sub-bullets]}

            if heading:
                el.append(_section_header(heading + ":", S))

            for pt in points:
                el.append(_bullet(pt, S, level=1))
                for sub in subs.get(pt, []):
                    el.append(_bullet(sub, S, level=2))

            el.append(Spacer(1, 0.04*inch))

        # ── 6. Key Judgements box ────────────────────────────────────────
        if topic.key_judgements:
            el.append(_key_judgements_box(topic.key_judgements, S))
            el.append(Spacer(1, 0.08*inch))

        # ── 7. Challenges ↔ Suggestions two-column ───────────────────────
        if topic.challenges or topic.suggestions:
            el.append(_section_header("Challenges & Way Forward:", S, 'SecHeaderRed'))
            el.append(Spacer(1, 0.04*inch))
            el.append(_two_column_section(
                "Challenges",   topic.challenges,
                "Suggestions / Way Forward", topic.suggestions,
                S
            ))
            el.append(Spacer(1, 0.08*inch))

        # ── 8. Comparison table ──────────────────────────────────────────
        if topic.comparison_table:
            el.append(_section_header("Comparison:", S))
            el.append(Spacer(1, 0.04*inch))
            el.append(_comparison_table(topic.comparison_table, S))
            el.append(Spacer(1, 0.08*inch))

        # ── 9. Key facts green box ───────────────────────────────────────
        if topic.key_facts_box:
            el.append(_key_facts_box(topic.key_facts_box, "Key Facts & Data Points:", S))
            el.append(Spacer(1, 0.08*inch))

        # ── 10. Important Terms table ─────────────────────────────────────
        if topic.important_terms:
            el.append(_section_header("Important Terms & Definitions:", S))
            el.append(_terms_table(topic.important_terms, S))
            el.append(Spacer(1, 0.08*inch))

        # ── 11. Practice Questions (per-topic mini set) ───────────────────
        if include_qs and topic.practice_questions:
            el.append(_section_header("Practice Questions:", S, 'SecHeaderRed'))
            for q in topic.practice_questions[:3]:
                el.append(Paragraph(f"Q. {q}", S['QText']))
                el.append(Paragraph(
                    "Answer: " + "_" * 70,
                    S['QAns']
                ))

        el.append(Spacer(1, 0.1*inch))
        el.append(_horizontal_rule(C['navy'], 1.0))
        el.append(PageBreak())
        return el

    # ------------------------------------------------------------------
    # Quick revision
    # ------------------------------------------------------------------

    def _quick_revision(self, note: StudyNote) -> List:
        S  = self.styles
        el = []

        el.append(_topic_header_table("QUICK REVISION — KEY FACTS", "", S))
        el.append(Paragraph(
            "<i>Use this section for last-minute revision before exams</i>",
            S['Trigger']
        ))
        el.append(Spacer(1, 0.1*inch))

        n = 1
        for topic in note.topics:
            el.append(_section_header(topic.title, S))
            # Pull key facts from key_facts_box + first few challenge/suggestion bullets
            facts = list(topic.key_facts_box)[:4]
            if not facts:
                # Fallback: first provision
                facts = topic.key_provisions[:3] if topic.key_provisions else []
            for f in facts:
                el.append(Paragraph(f"<b>{n}.</b> {f}", S['Bullet1']))
                n += 1
            el.append(Spacer(1, 0.04*inch))

        el.append(PageBreak())
        return el

    # ------------------------------------------------------------------
    # Consolidated 20 questions
    # ------------------------------------------------------------------

    def _questions_section(self, note: StudyNote) -> List:
        S  = self.styles
        el = []

        el.append(_topic_header_table("PRACTICE QUESTIONS", "20 Questions | MCQ + Descriptive", S))
        el.append(Spacer(1, 0.1*inch))

        # Gather all questions
        all_q: List[tuple] = []  # (question, topic_title)
        for t in note.topics:
            for q in t.practice_questions:
                all_q.append((q, t.title))

        # Top-up to 20 with generic templates
        all_q = self._topup_questions(all_q, note, 20)
        mcq   = all_q[:10]
        desc  = all_q[10:20]
        bonus = all_q[20:]

        # Part A
        pa = Table(
            [[Paragraph("PART A — Objective / MCQ Questions (Q1–Q10)", S['TH'])]],
            colWidths=[17*cm]
        )
        pa.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), C['navy']),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ]))
        el.append(pa)
        el.append(Spacer(1, 0.06*inch))

        for i, (q, topic) in enumerate(mcq, 1):
            el.append(Paragraph(
                f"Q{i}.  <font color='#666666' size='8'>[{topic[:35]}]</font>",
                S['QNum']
            ))
            el.append(Paragraph(q, S['QText']))
            el.append(Paragraph("Answer: " + "_" * 65, S['QAns']))

        el.append(Spacer(1, 0.15*inch))

        # Part B
        pb = Table(
            [[Paragraph("PART B — Descriptive / Analytical Questions (Q11–Q20)", S['TH'])]],
            colWidths=[17*cm]
        )
        pb.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), C['mains']),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ]))
        el.append(pb)
        el.append(Spacer(1, 0.06*inch))

        for i, (q, topic) in enumerate(desc, 11):
            el.append(Paragraph(
                f"Q{i}.  <font color='#666666' size='8'>[{topic[:35]}]</font>",
                S['QNum']
            ))
            el.append(Paragraph(q, S['QText']))
            for _ in range(3):
                el.append(Paragraph("_" * 90, S['QAns']))
            el.append(Spacer(1, 0.04*inch))

        # Bonus
        if bonus:
            bh = Table(
                [[Paragraph(f"BONUS QUESTIONS (Q21–Q{20+len(bonus)})", S['TH'])]],
                colWidths=[17*cm]
            )
            bh.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, -1), HexColor('#38A169')),
                ('TOPPADDING',    (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING',   (0, 0), (-1, -1), 10),
            ]))
            el.append(Spacer(1, 0.1*inch))
            el.append(bh)
            el.append(Spacer(1, 0.06*inch))
            for i, (q, topic) in enumerate(bonus, 21):
                el.append(Paragraph(
                    f"Q{i}.  <font color='#666666' size='8'>[{topic[:35]}]</font>",
                    S['QNum']
                ))
                el.append(Paragraph(q, S['QText']))
                el.append(Paragraph("_" * 90, S['QAns']))

        el.append(PageBreak())
        return el

    # ------------------------------------------------------------------
    # Back page
    # ------------------------------------------------------------------

    def _back_page(self, note: StudyNote) -> List:
        S = self.styles
        msg = (
            f"<b>Thank you for studying with {ACADEMY_NAME}!</b><br/><br/>"
            f"This PDF covers all current affairs from <b>{note.date}</b> in Drishti IAS style.<br/>"
            "The video covers key highlights — this PDF gives you the complete picture.<br/><br/>"
            "<b>Keep Learning. Keep Growing.</b><br/><br/>"
            f"<font size='8' color='#718096'>{ACADEMY_TAGLINE}</font><br/>"
            f"<font size='7.5' color='#718096'>Generated: {datetime.now().strftime('%B %d, %Y')}</font>"
        )
        t = Table([[Paragraph(msg, S['SubTitle'])]], colWidths=[17*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), C['bg_light']),
            ('BOX',           (0, 0), (-1, -1), 2, C['navy']),
            ('TOPPADDING',    (0, 0), (-1, -1), 25),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 25),
            ('LEFTPADDING',   (0, 0), (-1, -1), 20),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 20),
        ]))
        return [t]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _topup_questions(existing: List, note: StudyNote, target: int) -> List:
        """Add generic UPSC-style questions until we reach `target`."""
        templates = [
            "Consider the statements about {t}:\n1. Statement A  2. Statement B\n"
            "Which are correct? (a) 1 only  (b) 2 only  (c) Both  (d) Neither",
            "Discuss the significance of {t} in India's development agenda.",
            "What are the constitutional provisions relevant to {t}?",
            "Examine the government's approach towards {t}. (150 words)",
            "How does {t} affect India's socio-economic growth?",
            "What are the challenges and opportunities in {t}?",
            "Trace the evolution of policy on {t} in post-independent India.",
            "Critically analyse India's stand on {t} in the international arena.",
            "What are the suggestions for improving the situation regarding {t}?",
            "With reference to recent events, explain the relevance of {t}.",
        ]
        topics = [tp.title for tp in note.topics]
        needed = max(0, target - len(existing))
        extra  = []
        for i in range(needed):
            topic = topics[i % len(topics)]
            tmpl  = templates[i % len(templates)]
            extra.append((tmpl.format(t=topic[:40]), topic))
        return existing + extra

    # ------------------------------------------------------------------
    # Compatibility: generate from video composer's script_data
    # ------------------------------------------------------------------

    def generate_from_extracted_content(
        self,
        extracted_contents: List[ExtractedContent],
        title: str,
        date: str = None,
        video_duration: float = 0.0
    ) -> str:
        """Convert ExtractedContent list → StudyNote → PDF."""
        date = date or datetime.now().strftime("%B %d, %Y")
        topics = []

        for i, ec in enumerate(extracted_contents):
            kp_texts = [kp.text for kp in (ec.key_points or [])]
            terms    = dict(ec.important_terms or {})
            subject  = ec.upsc_relevance.subject.value if ec.upsc_relevance else "Current Affairs"
            exam     = ec.upsc_relevance.exam_relevance.value.upper() if ec.upsc_relevance else "BOTH"
            paper    = ec.upsc_relevance.mains_paper if ec.upsc_relevance else "GS3"
            ts       = (f"{(i * video_duration / max(len(extracted_contents),1) / 60):.0f}:00"
                        if video_duration else "")

            # Build a basic TopicNote from ExtractedContent
            topic = TopicNote(
                title=ec.article.title,
                trigger_line=ec.summary[:200] if ec.summary else "",
                what_is_it=ec.summary or "",
                key_provisions=[],
                sub_sections=[{
                    'heading': 'Key Points',
                    'points': kp_texts,
                    'sub_points': {}
                }] if kp_texts else [],
                challenges=[],
                suggestions=[],
                key_facts_box=[kp.text for kp in (ec.key_points or [])[:4]],
                important_terms=terms,
                practice_questions=list(ec.practice_questions or []),
                upsc_tags=f"Current Affairs | {subject} | {exam} | {paper}",
                timestamp=ts,
            )
            topics.append(topic)

        note = StudyNote(
            title=title,
            date=date,
            topics=topics,
            video_duration=video_duration
        )
        return self.generate_notes(note)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Current Affairs Academy PDF Generator — Drishti IAS Style ===\n")

    test_topic = TopicNote(
        title="Hate Speech and Hate Crime",
        trigger_line=(
            "SC raised concerns over hate crimes & speech, urging restraint while "
            "hearing a plea for a legal framework to recognise hate-based offences."
        ),
        what_is_it=(
            "Hate speech refers to words/actions intended to incite hatred against "
            "groups based on race, ethnicity, gender, religion, sexual orientation, etc., "
            "including speech or visuals that provoke fear or violence "
            "(267th Law Commission Report, 2017).\n\n"
            "Hate crime is a crime motivated by bias against race, colour, religion, "
            "national origin, sexual orientation, gender, gender identity, or disability."
        ),
        key_provisions=[
            "Art. 19(1)(a): Guarantees free speech; Art. 19(2): Permits reasonable restrictions "
            "(public order, dignity, sovereignty, incitement of offences).",
            "BNS 2023: Penalises promoting enmity between groups.",
            "RPA 1951: Disqualifies candidates convicted of promoting communal disharmony.",
            "SC/ST (Prevention of Atrocities) Act 1989: Punishes insults/humiliation of SC/ST members.",
            "Protection of Civil Rights Act 1955: Penalises acts promoting untouchability.",
        ],
        sub_sections=[
            {
                'heading': 'Hate Crime — Legal Status',
                'points': [
                    "No specific legal definition in India.",
                    "Provisions under BNS 2023 & SC/ST Act 1989 address mob lynching, caste-based violence.",
                ],
                'sub_points': {}
            }
        ],
        key_judgements=[
            "Shaheen Abdulla v. UoI (2022): Directed police to take suo motu action against hate speech.",
            "Tehseen S. Poonawalla v. UoI (2018): Issued guidelines to curb mob lynching.",
            "Shreya Singhal v. UoI (2015): Struck down Sec 66A, IT Act as vague; upheld Art. 19(1)(a).",
            "Pravasi Bhalai Sangathan v. UoI (2014): Urged Law Commission to define hate speech.",
        ],
        challenges=[
            "Legal Challenge: No standalone hate crime law; vague definitions under BNS 2023.",
            "Proving intent: Conviction requires proof of malicious intent — major evidentiary hurdle.",
            "Enforcement Gap: Weak suo motu action; low conviction rates due to political pressure.",
            "Digital Dilemma: Algorithmic amplification; anonymity via VPNs; transnational content.",
            "Societal: Fear-mongering used for political mobilisation; deep-rooted caste prejudices.",
            "Statistical blind spot: NCRB lacks specific data on lynchings/religious killings.",
        ],
        suggestions=[
            "Codify definition: Enact standalone law defining Hate Speech/Crime.",
            "Constitutional tort liability: Treat hate speech by public officials as civil wrong.",
            "Service rule enforcement: Classify failure to prevent hate speech as major misconduct.",
            "Suo motu FIR mandate: Strictly enforce SC (2022) directive; treat delay as contempt.",
            "24-hour digital takedown: Priority channel under IT Rules 2026 for DNOs.",
            "Fast-track 'Hate Courts': Complete trials within 6 months.",
            "Media literacy: Integrate critical thinking in NCERT curriculum.",
        ],
        key_facts_box=[
            "267th Law Commission Report (2017) — first formal attempt to define hate speech.",
            "SC/ST Atrocities Act 1989 & BNS 2023 — primary legal shields currently.",
            "NCRB lacks hate-crime specific data — policy gap.",
        ],
        important_terms={
            "Hate Speech":        "Expression that incites hatred/discrimination against a group (267th LCR 2017).",
            "Hate Crime":         "Bias-motivated criminal act targeting a protected characteristic.",
            "Suo Motu":           "Action taken by a court/authority on its own initiative without a petition.",
            "Art. 19(1)(a)":     "Fundamental Right to Freedom of Speech and Expression.",
            "Art. 19(2)":        "Permits 'reasonable restrictions' on free speech for public order, etc.",
            "Algorithmic Amplification": "Social media algorithms prioritising sensationalist/hateful content.",
        },
        practice_questions=[
            "Discuss the constitutional framework for regulating hate speech in India. (150 words)",
            "Consider the following: 1) India has a standalone Hate Crime law  2) SC has directed suo motu FIRs for hate speech. Which is/are correct? (a)1 only (b)2 only (c)Both (d)Neither",
            "What are the challenges in curbing hate speech in the digital age? Suggest a framework.",
        ],
        upsc_tags="GS2 | Polity & Governance | Prelims + Mains",
    )

    test_note = StudyNote(
        title="History Study Notes",
        date="February 20, 2026",
        topics=[test_topic],
    )

    gen = PDFNotesGenerator(output_dir="output/notes")
    path = gen.generate_notes(test_note)
    print(f"PDF generated: {path}")

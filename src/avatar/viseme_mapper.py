"""
Viseme Mapper — converts word-timing data from edge_tts into a per-frame
viseme timeline that drives phoneme-aware lip-sync animation.

8 viseme groups based on standard phoneme-to-viseme mapping:
  0  CLOSED  — silence / pause (lips together, neutral)
  1  AH      — a, e, i  (mouth open, jaw drops)
  2  OH      — o, u     (lips rounded)
  3  EE      — ee, y    (wide smile shape)
  4  FV      — f, v     (lower lip tucked under upper teeth)
  5  BMP     — b, m, p  (lips pressed together)
  6  LDN     — l, d, t, n, r, s, z  (tongue up, small opening)
  7  WQ      — w, ch, j, sh, k, g   (lips pursed / teeth together)

No external NLP dependencies — uses a fast rule-based English
letter-to-phoneme mapper that covers common patterns.
"""

import re
import numpy as np
from typing import List, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Viseme IDs ──────────────────────────────────────────────────────────────
CLOSED = 0
AH = 1
OH = 2
EE = 3
FV = 4
BMP = 5
LDN = 6
WQ = 7

VISEME_NAMES = ["CLOSED", "AH", "OH", "EE", "FV", "BMP", "LDN", "WQ"]

# ── Digraph / trigraph → viseme (checked FIRST, longest match wins) ─────────
_DIGRAPH_MAP: List[tuple] = [
    # Trigraphs
    ("tch", [LDN, WQ]),
    ("sch", [WQ]),
    ("igh", [AH]),
    ("ous", [AH, LDN]),
    ("tion", [WQ, AH, LDN]),
    ("sion", [WQ, AH, LDN]),
    # Digraphs — consonants
    ("th", [LDN]),
    ("sh", [WQ]),
    ("ch", [WQ]),
    ("ph", [FV]),
    ("wh", [WQ]),
    ("ck", [WQ]),
    ("ng", [WQ]),
    ("qu", [WQ, OH]),
    ("gh", []),       # silent in "light", aspirated elsewhere — skip
    ("kn", [LDN]),
    ("wr", [LDN]),
    ("gn", [LDN]),
    # Digraphs — vowels
    ("ee", [EE]),
    ("ea", [EE]),
    ("oo", [OH]),
    ("ou", [AH, OH]),
    ("ow", [AH, OH]),
    ("oi", [OH, EE]),
    ("oy", [OH, EE]),
    ("ai", [AH, EE]),
    ("ay", [AH, EE]),
    ("ei", [AH, EE]),
    ("ey", [AH, EE]),
    ("ie", [EE]),
    ("au", [AH, OH]),
    ("aw", [AH]),
    ("oa", [OH]),
    ("ue", [OH]),
    ("ui", [OH, EE]),
    ("er", [AH]),
    ("ir", [AH]),
    ("ur", [AH]),
    ("or", [OH]),
    ("ar", [AH]),
]

# ── Single-letter → viseme ──────────────────────────────────────────────────
_LETTER_MAP: Dict[str, int] = {
    # Vowels
    "a": AH,
    "e": EE,
    "i": EE,
    "o": OH,
    "u": OH,
    "y": EE,
    # Labial stops / nasals (lips close)
    "b": BMP,
    "m": BMP,
    "p": BMP,
    # Labiodental fricatives (lip tuck)
    "f": FV,
    "v": FV,
    # Alveolar / dental (tongue up, small opening)
    "d": LDN,
    "l": LDN,
    "n": LDN,
    "r": LDN,
    "s": LDN,
    "t": LDN,
    "z": LDN,
    # Velar / palatal / glottal (teeth together / lips forward)
    "c": WQ,
    "g": WQ,
    "h": AH,
    "j": WQ,
    "k": WQ,
    "q": WQ,
    "w": WQ,
    "x": WQ,
}


def word_to_visemes(word: str) -> List[int]:
    """
    Convert an English word to a sequence of viseme IDs using
    rule-based letter pattern matching.

    Handles digraphs/trigraphs first, then falls back to single letters.
    """
    word = word.lower().strip()
    # Remove non-alpha characters
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return [CLOSED]

    visemes: List[int] = []
    i = 0
    while i < len(word):
        matched = False
        # Try longest patterns first (trigraphs, then digraphs)
        for pattern, vids in _DIGRAPH_MAP:
            plen = len(pattern)
            if word[i: i + plen] == pattern:
                visemes.extend(vids)
                i += plen
                matched = True
                break
        if not matched:
            ch = word[i]
            vid = _LETTER_MAP.get(ch, LDN)  # default to small opening
            visemes.append(vid)
            i += 1

    # Filter out empty sequences (silent digraphs like "gh")
    if not visemes:
        visemes = [CLOSED]

    return visemes


def generate_viseme_timeline(
    word_boundaries: List[Dict],
    fps: int,
    total_duration: float,
) -> np.ndarray:
    """
    Generate a per-frame viseme timeline from edge_tts word boundary data.

    Parameters
    ----------
    word_boundaries : list of dicts with keys: text, offset_us, duration_us
    fps : video frame rate (typically 30)
    total_duration : total audio duration in seconds

    Returns
    -------
    numpy int8 array of shape (n_frames,) — viseme ID per frame
    """
    n_frames = int(total_duration * fps) + 1
    timeline = np.zeros(n_frames, dtype=np.int8)  # default CLOSED

    for wb in word_boundaries:
        word = wb.get("text", "")
        offset_us = wb.get("offset_us", 0)
        duration_us = wb.get("duration_us", 0)

        start_sec = offset_us / 1_000_000
        dur_sec = max(duration_us / 1_000_000, 0.01)  # min 10ms

        visemes = word_to_visemes(word)
        if not visemes:
            continue

        # Distribute visemes evenly across the word's duration
        phoneme_dur = dur_sec / len(visemes)

        for j, vid in enumerate(visemes):
            ph_start = start_sec + j * phoneme_dur
            ph_end = ph_start + phoneme_dur

            frame_start = max(0, min(int(ph_start * fps), n_frames - 1))
            frame_end = max(0, min(int(ph_end * fps), n_frames - 1))

            timeline[frame_start: frame_end + 1] = vid

    unique = len(set(timeline.tolist()))
    logger.info(f"Viseme timeline: {n_frames} frames, {unique} unique visemes used")
    return timeline

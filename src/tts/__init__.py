"""Text-to-Speech Module - Converts scripts to audio"""

from .base_tts import BaseTTS
from .edge_tts_engine import EdgeTTSEngine
from .indic_parler_tts_engine import IndicParlerTTSEngine
from .tts_manager import TTSManager

__all__ = ["BaseTTS", "EdgeTTSEngine", "IndicParlerTTSEngine", "TTSManager"]

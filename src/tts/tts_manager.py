"""
TTS Manager - Orchestrates text-to-speech generation for History lessons.
Adapted for Hinglish content with history-specific preprocessing.
"""

import re
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import yaml

from .edge_tts_engine import EdgeTTSEngine
from .base_tts import TTSResult, TTSVoice
from src.utils.logger import get_logger

logger = get_logger(__name__)


# History-specific abbreviations to expand for TTS clarity
HISTORY_ABBREVIATIONS = {
    "BCE": "Before Common Era",
    "CE": "Common Era",
    "BC": "Before Christ",
    "AD": "Anno Domini",
    "IVC": "Indus Valley Civilization",
    "INC": "Indian National Congress",
    "EIC": "East India Company",
    "INA": "Indian National Army",
    "NCM": "Non Cooperation Movement",
    "CDM": "Civil Disobedience Movement",
    "QIM": "Quit India Movement",
    "RTC": "Round Table Conference",
    "GoI": "Government of India",
    "UPSC": "U P S C",
    "PSC": "P S C",
    "MCQ": "M C Q",
    "PYQ": "Previous Year Question",
    "GS": "General Studies",
    "NBPW": "Northern Black Polished Ware",
    "PGW": "Painted Grey Ware",
    "OCP": "Ochre Coloured Pottery",
    "AIWC": "All India Womens Conference",
    "AITUC": "All India Trade Union Congress",
    "HSRA": "Hindustan Socialist Republican Association",
    "NAM": "Non Aligned Movement",
    "DPSP": "Directive Principles of State Policy",
}


class TTSManager:
    """
    Manages TTS generation for history lessons.
    Optimized for Hinglish (Hindi-English mix) content.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize TTS Manager.

        Args:
            config_path: Path to settings configuration.
        """
        self.config = self._load_config(config_path)
        self.engine = EdgeTTSEngine()

        # Load language configurations
        self.languages = self._load_languages()

        logger.info(f"TTSManager initialized with {len(self.languages)} languages")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}

    def _load_languages(self) -> Dict[str, Dict[str, Any]]:
        """Load language configurations."""
        languages = {}

        lang_config = self.config.get("languages", {}).get("supported", [])

        for lang in lang_config:
            code = lang.get("code", "")
            if code:
                languages[code] = {
                    "name": lang.get("name", code),
                    "voice": lang.get("tts_voice", ""),
                    "rate": lang.get("tts_rate", "-5%"),
                    "pitch": lang.get("tts_pitch", "+0Hz")
                }

        # Ensure Hindi/Hinglish is always available
        if "hi" not in languages:
            languages["hi"] = {
                "name": "Hinglish",
                "voice": "hi-IN-MadhurNeural",
                "rate": "-5%",
                "pitch": "+0Hz"
            }

        return languages

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for TTS compatibility.
        Handles Hinglish content and history-specific terms.

        Args:
            text: Raw script text.

        Returns:
            Preprocessed text suitable for TTS engine.
        """
        # Expand history abbreviations (word boundaries)
        for abbr, expansion in HISTORY_ABBREVIATIONS.items():
            text = re.sub(rf'\b{re.escape(abbr)}\b', expansion, text)

        # Handle year ranges properly: "1526-1530" → "1526 to 1530"
        text = re.sub(r'(\d{3,4})\s*[-–]\s*(\d{3,4})', r'\1 to \2', text)

        # Handle century notation: "18th century" stays as is (TTS handles it)

        # Expand common symbols
        text = text.replace("₹", "rupees ")
        text = text.replace("%", " percent")
        text = text.replace("&", " and ")
        text = text.replace("vs", "versus")
        text = text.replace("vs.", "versus")

        # Add natural pauses at section breaks
        text = re.sub(r'\.\s*\n', '.\n\n', text)

        # Remove any remaining markdown/formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'#{1,6}\s*', '', text)

        # Clean up extra whitespace
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Add slight pauses with commas for better pacing
        # After "Toh" (common Hinglish transition)
        text = re.sub(r'\bToh\b', 'Toh,', text)
        text = re.sub(r'\bToh,,', 'Toh,', text)  # Fix double comma

        return text.strip()

    async def generate_audio(
        self,
        text: str,
        output_path: str,
        language: str = "hi",
        voice: str = None,
        rate: str = None,
        pitch: str = None
    ) -> TTSResult:
        """
        Generate audio from text.

        Args:
            text: Text to synthesize.
            output_path: Path to save audio file.
            language: Language code (default: "hi" for Hinglish).
            voice: Override voice ID.
            rate: Override speaking rate.
            pitch: Override pitch.

        Returns:
            TTSResult object.
        """
        # Get language settings
        lang_config = self.languages.get(language, self.languages.get("hi", {}))

        voice = voice or lang_config.get("voice", "hi-IN-MadhurNeural")
        rate = rate or lang_config.get("rate", "-5%")
        pitch = pitch or lang_config.get("pitch", "+0Hz")

        # Preprocess text for TTS
        text = self.preprocess_text(text)

        logger.info(f"Generating audio: {len(text)} chars, voice={voice}, rate={rate}")

        # Use long text synthesis for longer content (history lessons are typically long)
        if len(text) > 5000:
            result = await self.engine.synthesize_long_text(
                text=text,
                output_path=output_path,
                voice=voice,
                rate=rate,
                pitch=pitch
            )
        else:
            result = await self.engine.synthesize(
                text=text,
                output_path=output_path,
                voice=voice,
                rate=rate,
                pitch=pitch
            )

        return result

    def generate_audio_sync(
        self,
        text: str,
        output_path: str,
        language: str = "hi",
        voice: str = None,
        rate: str = None,
        pitch: str = None
    ) -> TTSResult:
        """
        Synchronous wrapper for audio generation.

        Args:
            text: Text to synthesize.
            output_path: Path to save audio file.
            language: Language code.
            voice: Override voice ID.
            rate: Override speaking rate.
            pitch: Override pitch.

        Returns:
            TTSResult object.
        """
        return asyncio.run(self.generate_audio(
            text=text,
            output_path=output_path,
            language=language,
            voice=voice,
            rate=rate,
            pitch=pitch
        ))

    async def generate_lesson_audio(
        self,
        script_text: str,
        output_dir: str,
        part_number: int = None,
        language: str = "hi"
    ) -> Dict[str, Any]:
        """
        Generate audio for a complete history lesson.

        Args:
            script_text: Full lesson script text.
            output_dir: Directory to save audio files.
            part_number: Part number for file naming.
            language: Language code.

        Returns:
            Dictionary with audio info.
        """
        file_id = f"part_{part_number:03d}" if part_number else datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"{file_id}_audio.mp3"

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        result = await self.generate_audio(
            text=script_text,
            output_path=str(output_path),
            language=language
        )

        if result.success:
            logger.info(f"Generated lesson audio: {result.duration:.1f}s ({result.duration/60:.1f} min)")
            return {
                "success": True,
                "audio_path": result.audio_path,
                "duration": result.duration,
                "voice": result.voice.id if result.voice else "unknown",
                "language": language
            }
        else:
            logger.error(f"Failed to generate lesson audio: {result.error}")
            return {
                "success": False,
                "error": result.error
            }

    async def list_available_voices(self, language: str = None) -> List[TTSVoice]:
        """List available voices."""
        return await self.engine.list_voices(language)

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages."""
        return [
            {"code": code, "name": config["name"]}
            for code, config in self.languages.items()
        ]


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="History TTS Manager CLI")
    parser.add_argument("--test", action="store_true", help="Run TTS test")
    parser.add_argument("--lang", type=str, default="hi", help="Language code")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output/test_audio.mp3", help="Output path")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")

    args = parser.parse_args()

    manager = TTSManager()

    if args.list_voices:
        print(f"\n=== Available Voices for {args.lang} ===\n")
        voices = asyncio.run(manager.list_available_voices(args.lang))
        for voice in voices[:20]:
            print(f"  {voice.id}: {voice.name} ({voice.gender})")

    elif args.test or args.text:
        text = args.text or "Namaste friends! Aaj ki class mein hum padhenge Indus Valley Civilization ke baare mein. Yeh ek bahut important topic hai UPSC ke liye."

        print(f"\n=== History TTS Test ===")
        print(f"Language: {args.lang}")
        print(f"Text: {text[:80]}...")
        print(f"Output: {args.output}\n")

        result = manager.generate_audio_sync(
            text=text,
            output_path=args.output,
            language=args.lang
        )

        if result.success:
            print(f"Success! Audio saved to: {result.audio_path}")
            print(f"Duration: {result.duration:.1f} seconds")
        else:
            print(f"Error: {result.error}")

    else:
        parser.print_help()

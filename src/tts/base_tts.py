"""
Base TTS class - Abstract interface for text-to-speech engines
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class TTSVoice:
    """Represents a TTS voice"""
    id: str
    name: str
    language: str
    language_code: str
    gender: str
    provider: str


@dataclass
class TTSResult:
    """Result of TTS generation"""
    audio_path: str
    duration: float  # in seconds
    text: str
    voice: TTSVoice
    success: bool
    error: Optional[str] = None


class BaseTTS(ABC):
    """Abstract base class for TTS engines"""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%"
    ) -> TTSResult:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            voice: Voice ID to use
            rate: Speaking rate adjustment
            pitch: Pitch adjustment
            volume: Volume adjustment

        Returns:
            TTSResult object
        """
        pass

    @abstractmethod
    async def list_voices(self, language: str = None) -> List[TTSVoice]:
        """
        List available voices.

        Args:
            language: Filter by language code (optional)

        Returns:
            List of TTSVoice objects
        """
        pass

    @abstractmethod
    def get_default_voice(self, language: str) -> str:
        """
        Get default voice for a language.

        Args:
            language: Language code (e.g., 'en', 'hi')

        Returns:
            Voice ID
        """
        pass

"""
Indic Parler-TTS Engine - Natural Indian voice synthesis via HuggingFace Space.

Uses ai4bharat/indic-parler-tts for high-quality, natural-sounding Indian voices.
Supports 22 Indian languages with 69 unique voices.
Free via HuggingFace Spaces (ZeroGPU).

Pipeline: Text → HF Space API → Audio chunks → Merge → Post-process → MP3
"""

import asyncio
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional

# Configure pydub to use ffmpeg from imageio-ffmpeg
try:
    import imageio_ffmpeg
    import pydub
    pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    pydub.AudioSegment.ffprobe = imageio_ffmpeg.get_ffmpeg_exe().replace('ffmpeg', 'ffprobe')
except ImportError:
    pass

from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range

from .base_tts import BaseTTS, TTSVoice, TTSResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Available speakers per language for Indic Parler-TTS
INDIC_SPEAKERS = {
    "hi": ["Rohit", "Divya", "Aman", "Rani"],
    "en": ["Thoma", "Mary", "Swapna", "Dinesh", "Meera", "Jatin", "Aakash", "Sneha",
           "Kabir", "Tisha", "Priya", "Tarun", "Gauri", "Nisha", "Raghav", "Kavya",
           "Ravi", "Vikas", "Riya"],
    "bn": ["Arjun", "Aditi", "Tapan", "Rashmi", "Arnav", "Riya"],
    "ta": ["Kavitha", "Jaya"],
    "te": ["Prakash", "Lalitha", "Kiran"],
    "mr": ["Sanjay", "Sunita", "Nikhil", "Radha", "Varun", "Isha"],
    "gu": ["Yash", "Neha"],
    "kn": ["Suresh", "Anu", "Chetan", "Vidya"],
    "ml": ["Anjali", "Anju", "Harish"],
    "as": ["Amit", "Sita", "Poonam", "Rakesh"],
    "or": ["Manas", "Debjani"],
    "pa": ["Divjot", "Gurpreet"],
    "sa": ["Aryan"],
    "ne": ["Amrita"],
}

# Default male speakers per language
DEFAULT_MALE_SPEAKERS = {
    "hi": "Rohit",
    "en": "Kabir",
    "bn": "Arjun",
    "ta": "Kavitha",
    "te": "Prakash",
    "mr": "Sanjay",
    "gu": "Yash",
    "kn": "Suresh",
    "ml": "Harish",
    "as": "Amit",
    "or": "Manas",
    "pa": "Divjot",
    "sa": "Aryan",
    "ne": "Amrita",
}


class IndicParlerTTSEngine(BaseTTS):
    """
    Indic Parler-TTS engine using ai4bharat HuggingFace Space.
    Natural-sounding Indian voices with prompt-based voice control.
    """

    HF_SPACE_ID = "ai4bharat/indic-parler-tts"
    MAX_CHUNK_CHARS = 500  # Parler-TTS works best with shorter text
    MAX_RETRIES = 3
    RETRY_DELAY = 10

    def __init__(
        self,
        speaker: str = None,
        language: str = "hi",
        voice_description: str = None,
        hf_token: str = None,
    ):
        self.language = language
        self.speaker = speaker or DEFAULT_MALE_SPEAKERS.get(language, "Rohit")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", None)
        self.voice_description = voice_description or self._default_description()
        self._client = None

        logger.info(
            f"IndicParlerTTS initialized: speaker={self.speaker}, lang={self.language}"
        )

    def _default_description(self) -> str:
        """Generate default voice description for a teacher voice."""
        return (
            f"{self.speaker} speaks with a moderate-pitched voice delivering words "
            f"at a moderate pace in a close-sounding environment with very clear audio "
            f"and no background noise. The tone is confident and engaging, like a "
            f"professional teacher explaining concepts clearly."
        )

    def _get_client(self):
        """Lazy-initialize gradio client with fallback for API schema bugs."""
        if self._client is None:
            from gradio_client import Client
            # The HF Space has a Gradio version with a bug in json_schema_to_python_type
            # that causes TypeError during API info fetch. Try multiple approaches.
            for attempt, kwargs in enumerate([
                {"hf_token": self.hf_token},
                {},  # without token
            ], 1):
                try:
                    self._client = Client(self.HF_SPACE_ID, **kwargs)
                    logger.info(f"Connected to HF Space: {self.HF_SPACE_ID}")
                    return self._client
                except TypeError as e:
                    if "not iterable" in str(e):
                        # Gradio schema parsing bug — client may still work for predict
                        logger.warning(f"API schema parse error (attempt {attempt}), trying raw client")
                        try:
                            # Force client creation by skipping API info validation
                            client = Client.__new__(Client)
                            client.src = f"https://{self.HF_SPACE_ID.replace('/', '-')}.hf.space"
                            client.hf_token = kwargs.get("hf_token")
                            self._client = Client(
                                self.HF_SPACE_ID,
                                download_files=False,
                                **kwargs,
                            )
                            logger.info(f"Connected to HF Space (download_files=False)")
                            return self._client
                        except Exception:
                            pass
                    else:
                        raise
                except Exception as e:
                    logger.warning(f"Client init attempt {attempt} failed: {e}")

            # Final fallback: use httpx directly
            logger.warning("All gradio_client attempts failed, will use direct HTTP API")
            self._client = "httpx_fallback"
        return self._client

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks suitable for Parler-TTS."""
        sentences = []
        # Split on sentence boundaries
        for part in text.replace('\n\n', '\n').split('\n'):
            part = part.strip()
            if not part:
                continue
            # Further split long paragraphs on sentence endings
            import re
            sent_parts = re.split(r'(?<=[.!?।])\s+', part)
            sentences.extend([s.strip() for s in sent_parts if s.strip()])

        # Merge short sentences, split long ones
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.MAX_CHUNK_CHARS:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If single sentence is too long, split at commas
                if len(sentence) > self.MAX_CHUNK_CHARS:
                    comma_parts = sentence.split(', ')
                    sub_chunk = ""
                    for cp in comma_parts:
                        if len(sub_chunk) + len(cp) + 2 <= self.MAX_CHUNK_CHARS:
                            sub_chunk = f"{sub_chunk}, {cp}".strip(', ')
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = cp
                    if sub_chunk:
                        current_chunk = sub_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _generate_chunk(self, text: str, output_path: str) -> Optional[str]:
        """Generate audio for a single text chunk via HF Space."""
        client = self._get_client()

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Parler-TTS chunk ({len(text)} chars), attempt {attempt}/{self.MAX_RETRIES}"
                )

                if client == "httpx_fallback":
                    result = self._generate_via_httpx(text, output_path)
                else:
                    result = client.predict(
                        text=text,
                        description=self.voice_description,
                        api_name="/generate_finetuned",
                    )
                    if result and Path(result).exists():
                        shutil.copy2(result, output_path)
                        result = output_path

                if result and Path(result).exists():
                    return result

            except Exception as e:
                logger.warning(f"Parler-TTS attempt {attempt} failed: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)
                    # Reset client on failure
                    self._client = None
                    client = self._get_client()

        return None

    def _generate_via_httpx(self, text: str, output_path: str) -> Optional[str]:
        """Fallback: call HF Space Gradio API directly via HTTP."""
        import httpx
        import json
        import hashlib

        base_url = f"https://{self.HF_SPACE_ID.replace('/', '-')}.hf.space"
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        # Step 1: Get a session hash
        session_hash = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12]

        # Step 2: Call the predict endpoint
        payload = {
            "data": [text, self.voice_description],
            "fn_index": 1,  # generate_finetuned
            "session_hash": session_hash,
        }

        with httpx.Client(timeout=300, headers=headers) as http:
            # Queue the request
            resp = http.post(f"{base_url}/queue/push", json=payload)
            resp.raise_for_status()
            event_id = resp.json().get("event_id")

            if not event_id:
                logger.error("No event_id from queue/push")
                return None

            # Poll for result
            for _ in range(120):  # max 10 min polling
                time.sleep(5)
                status_resp = http.get(
                    f"{base_url}/queue/status/{event_id}"
                )
                if status_resp.status_code != 200:
                    continue
                status_data = status_resp.json()

                if status_data.get("status") == "COMPLETE":
                    # Get the audio file URL
                    output = status_data.get("data", {}).get("data", [])
                    if output and isinstance(output[0], dict):
                        file_url = output[0].get("url") or output[0].get("path")
                        if file_url:
                            if not file_url.startswith("http"):
                                file_url = f"{base_url}/file={file_url}"
                            audio_resp = http.get(file_url)
                            audio_resp.raise_for_status()
                            with open(output_path, "wb") as f:
                                f.write(audio_resp.content)
                            return output_path
                    return None
                elif status_data.get("status") == "FAILED":
                    logger.error(f"HF Space returned FAILED: {status_data}")
                    return None

        return None

    def _postprocess_audio(self, audio_path: str) -> str:
        """Apply post-processing for broadcast quality."""
        try:
            audio = AudioSegment.from_file(audio_path)

            # High-pass filter at 80Hz (remove low rumble)
            audio = audio.high_pass_filter(80)

            # Normalize loudness
            audio = normalize(audio)

            # Light compression for consistent volume
            audio = compress_dynamic_range(
                audio,
                threshold=-22.0,
                ratio=2.5,
                attack=10.0,
                release=100.0,
            )

            # Target loudness: -14 LUFS for YouTube
            target_dbfs = -14.0
            change_in_dbfs = target_dbfs - audio.dBFS
            audio = audio.apply_gain(change_in_dbfs)

            # Export as MP3
            audio.export(audio_path, format="mp3", bitrate="192k")
            return audio_path

        except Exception as e:
            logger.warning(f"Post-processing failed (using raw audio): {e}")
            return audio_path

    async def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> TTSResult:
        """Synthesize text to speech using Indic Parler-TTS."""
        # Update speaker if voice provided
        if voice and voice in [s for speakers in INDIC_SPEAKERS.values() for s in speakers]:
            self.speaker = voice
            self.voice_description = self._default_description()

        voice_obj = TTSVoice(
            id=f"indic-parler-{self.speaker.lower()}",
            name=self.speaker,
            language=self.language,
            language_code=f"{self.language}-IN",
            gender="Male",
            provider="indic-parler-tts",
        )

        try:
            chunks = self._split_text(text)
            logger.info(f"Indic Parler-TTS: {len(text)} chars → {len(chunks)} chunks")

            if len(chunks) == 0:
                return TTSResult(
                    audio_path=output_path,
                    duration=0,
                    text=text,
                    voice=voice_obj,
                    success=False,
                    error="No text to synthesize",
                )

            # Generate audio for each chunk
            temp_dir = tempfile.mkdtemp(prefix="parler_tts_")
            chunk_paths = []

            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(temp_dir, f"chunk_{i:04d}.mp3")
                logger.info(f"Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")

                # Run in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                result_path = await loop.run_in_executor(
                    None, self._generate_chunk, chunk, chunk_path
                )

                if result_path:
                    chunk_paths.append(result_path)
                else:
                    logger.error(f"Failed to generate chunk {i+1}")
                    # Continue with remaining chunks

            if not chunk_paths:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return TTSResult(
                    audio_path=output_path,
                    duration=0,
                    text=text,
                    voice=voice_obj,
                    success=False,
                    error="All chunks failed to generate",
                )

            # Merge all chunks
            combined = AudioSegment.empty()
            pause = AudioSegment.silent(duration=300)  # 300ms pause between chunks

            for cp in chunk_paths:
                try:
                    chunk_audio = AudioSegment.from_file(cp)
                    combined += chunk_audio + pause
                except Exception as e:
                    logger.warning(f"Failed to load chunk {cp}: {e}")

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Export merged audio
            combined.export(output_path, format="mp3", bitrate="192k")

            # Post-process
            self._postprocess_audio(output_path)

            # Calculate duration
            final_audio = AudioSegment.from_file(output_path)
            duration = len(final_audio) / 1000.0

            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

            logger.info(f"Indic Parler-TTS: generated {duration:.1f}s audio")

            return TTSResult(
                audio_path=output_path,
                duration=duration,
                text=text,
                voice=voice_obj,
                success=True,
            )

        except Exception as e:
            logger.error(f"Indic Parler-TTS synthesis failed: {e}")
            return TTSResult(
                audio_path=output_path,
                duration=0,
                text=text,
                voice=voice_obj,
                success=False,
                error=str(e),
            )

    async def synthesize_long_text(
        self,
        text: str,
        output_path: str,
        voice: str = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> TTSResult:
        """Long text synthesis — same as synthesize since we chunk internally."""
        return await self.synthesize(text, output_path, voice, rate, pitch, volume)

    async def list_voices(self, language: str = None) -> List[TTSVoice]:
        """List available Indic Parler-TTS voices."""
        voices = []
        languages = [language] if language else INDIC_SPEAKERS.keys()

        for lang in languages:
            speakers = INDIC_SPEAKERS.get(lang, [])
            for speaker in speakers:
                voices.append(
                    TTSVoice(
                        id=f"indic-parler-{speaker.lower()}",
                        name=speaker,
                        language=lang,
                        language_code=f"{lang}-IN",
                        gender="Male" if speaker in [
                            DEFAULT_MALE_SPEAKERS.get(l) for l in INDIC_SPEAKERS
                        ] else "Unknown",
                        provider="indic-parler-tts",
                    )
                )
        return voices

    def get_default_voice(self, language: str) -> str:
        """Get default voice for a language."""
        return DEFAULT_MALE_SPEAKERS.get(language, "Rohit")

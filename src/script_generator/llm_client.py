"""
LLM Client - Interface for Groq and Ollama language models
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """Generate text from prompt"""
        pass


class GroqClient(BaseLLMClient):
    """Client for Groq API (free tier)"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "llama-3.3-70b-versatile"
    ):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter. Get free key at: https://console.groq.com/"
            )

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Initialized Groq client with model: {model}")
        except ImportError:
            raise ImportError("Please install groq: pip install groq")

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using Groq API.

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            max_tokens: Maximum tokens to generate
            temperature: Creativity level (0-1)

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            generated_text = response.choices[0].message.content
            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise


class OllamaClient(BaseLLMClient):
    """Client for local Ollama models"""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama2"
    ):
        """
        Initialize Ollama client.

        Args:
            host: Ollama server URL
            model: Model to use (must be pulled first)
        """
        self.host = host
        self.model = model

        try:
            import ollama
            self.client = ollama.Client(host=host)

            # Test connection
            self.client.list()
            logger.info(f"Initialized Ollama client with model: {model}")

        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}. Make sure Ollama is running.")

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using local Ollama model.

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            max_tokens: Maximum tokens to generate
            temperature: Creativity level (0-1)

        Returns:
            Generated text
        """
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        full_prompt += prompt

        try:
            response = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )

            generated_text = response.get("response", "")
            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise


class LLMClient:
    """
    Unified LLM client that supports multiple providers.
    Falls back gracefully between providers.
    """

    def __init__(self, provider: str = "groq", **kwargs):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider ("groq" or "ollama")
            **kwargs: Provider-specific arguments
        """
        self.provider = provider
        self.client = None

        # Try to initialize the specified provider
        if provider == "groq":
            try:
                self.client = GroqClient(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")

        elif provider == "ollama":
            try:
                self.client = OllamaClient(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {e}")

        # Fallback logic
        if self.client is None:
            logger.warning(f"Primary provider {provider} failed, trying fallback...")

            if provider == "groq":
                # Try Ollama as fallback
                try:
                    self.client = OllamaClient()
                    self.provider = "ollama"
                    logger.info("Falling back to Ollama")
                except Exception:
                    pass
            else:
                # Try Groq as fallback
                try:
                    self.client = GroqClient()
                    self.provider = "groq"
                    logger.info("Falling back to Groq")
                except Exception:
                    pass

        if self.client is None:
            raise RuntimeError(
                "No LLM provider available. Please either:\n"
                "1. Set GROQ_API_KEY environment variable (get free key at https://console.groq.com/)\n"
                "2. Install and run Ollama locally (https://ollama.ai/)"
            )

        logger.info(f"LLM Client initialized with provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            max_tokens: Maximum tokens to generate
            temperature: Creativity level (0-1)

        Returns:
            Generated text
        """
        return self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: str = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of user prompts
            system_prompt: System prompt for context
            max_tokens: Maximum tokens per generation
            temperature: Creativity level (0-1)

        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Generation failed for prompt: {e}")
                results.append("")

        return results

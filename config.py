"""
config.py
─────────
Central configuration for Smart PDF Chat.
All tuneable knobs live here — read from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM provider selection ────────────────────────────────────
    # "gemini" or "groq"  (groq recommended — free tier, no billing needed)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()

    # ── Gemini settings ───────────────────────────────────────────
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # ── Groq settings ─────────────────────────────────────────────
    # Free tier: 14,400 req/day — https://console.groq.com/keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # ── Embedding model (local, no API key needed) ────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Chunking ──────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))

    # ── Retrieval ─────────────────────────────────────────────────
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", 5))

    # ── UI ────────────────────────────────────────────────────────
    APP_TITLE: str = "Smart PDF Chat"
    APP_SUBTITLE: str = "Upload PDFs · Ask questions · Get grounded answers"

    @classmethod
    def validate(cls) -> None:
        """Fail fast if the selected provider has no API key."""
        if cls.LLM_PROVIDER == "gemini":
            if not cls.GOOGLE_API_KEY:
                raise EnvironmentError(
                    "GOOGLE_API_KEY is not set. Add it to your .env file.\n"
                    "Get a key at: https://aistudio.google.com/app/apikey\n"
                    "Or switch to Groq: set LLM_PROVIDER=groq in .env"
                )
        elif cls.LLM_PROVIDER == "groq":
            if not cls.GROQ_API_KEY:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Add it to your .env file.\n"
                    "Get a free key at: https://console.groq.com/keys"
                )
        else:
            raise EnvironmentError(
                f"Unknown LLM_PROVIDER='{cls.LLM_PROVIDER}'. "
                "Must be 'gemini' or 'groq'."
            )


cfg = Config()
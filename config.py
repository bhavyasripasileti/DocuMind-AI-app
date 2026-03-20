"""
config.py
─────────
Central configuration for Smart PDF Chat.
All tuneable knobs live here — read from environment variables.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: str = ""):
    return os.getenv(key) or st.secrets.get(key, default)


class Config:
    # ── LLM Provider ──
    LLM_PROVIDER = get_env("LLM_PROVIDER", "groq").lower()

    # ── Groq (PRIMARY) ──
    GROQ_API_KEY = get_env("GROQ_API_KEY")
    LLM_MODEL = get_env("LLM_MODEL", "llama-3.1-8b-instant")

    # ── Embeddings ──
    EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Chunking ──
    CHUNK_SIZE = int(get_env("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(get_env("CHUNK_OVERLAP", "150"))

    # ── Retrieval ──
    TOP_K_CHUNKS = int(get_env("TOP_K_CHUNKS", "5"))

    # ── UI ──
    APP_TITLE = "DocuMind AI"
    APP_SUBTITLE = "Chat with your documents using AI"

    @classmethod
    def validate(cls):
        if cls.LLM_PROVIDER == "groq":
            if not cls.GROQ_API_KEY:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set.\n"
                    "Add it in .env (local) or Streamlit Secrets (cloud).\n"
                    "Get key: https://console.groq.com/keys"
                )
        else:
            raise EnvironmentError("Only 'groq' is supported in this version.")


cfg = Config()

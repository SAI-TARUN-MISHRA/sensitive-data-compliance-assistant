"""
llm_utils.py
------------
Shared LLM client initialization and utilities for Groq and OpenAI.
"""

import os

def get_llm_client():
    """
    Returns (client, model, provider) using whichever API key is configured.
    Priority: GROQ_API_KEY → OPENAI_API_KEY → None
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            return Groq(api_key=groq_key), "llama-3.1-8b-instant", "groq"
        except Exception:
            pass

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            return OpenAI(api_key=openai_key), "gpt-4o-mini", "openai"
        except Exception:
            pass

    return None, None, None

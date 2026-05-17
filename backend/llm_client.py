"""
LLM client for MedLex RAG.
Gemini is used for cloud/local deployment without Ollama.
"""

from typing import AsyncGenerator

from google import genai
from google.genai import types

from config import get_settings


def _settings():
    return get_settings()


def _gemini_client():
    s = _settings()
    if not s.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in .env")
    return genai.Client(api_key=s.gemini_api_key)


async def call_llm_chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
) -> str:
    s = _settings()
    provider = (s.llm_provider or "gemini").lower()

    if provider != "gemini":
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER='{s.llm_provider}'. "
            "Set LLM_PROVIDER=gemini for this deployment."
        )

    temp = float(temperature if temperature is not None else s.llm_temperature)

    client = _gemini_client()

    response = client.models.generate_content(
        model=s.gemini_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temp,
            top_p=float(getattr(s, "llm_top_p", 0.9)),
            max_output_tokens=int(getattr(s, "llm_max_tokens", 1200)),
        ),
    )

    return (response.text or "").strip()


async def stream_llm_chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
) -> AsyncGenerator[str, None]:
    s = _settings()
    provider = (s.llm_provider or "gemini").lower()

    if provider != "gemini":
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER='{s.llm_provider}'. "
            "Set LLM_PROVIDER=gemini for this deployment."
        )

    temp = float(temperature if temperature is not None else s.llm_temperature)

    client = _gemini_client()

    stream = client.models.generate_content_stream(
        model=s.gemini_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temp,
            top_p=float(getattr(s, "llm_top_p", 0.9)),
            max_output_tokens=int(getattr(s, "llm_max_tokens", 1200)),
        ),
    )

    for chunk in stream:
        if chunk.text:
            yield chunk.text
"""
llm.py — Single gateway to Claude for ProcureAI
===============================================

Claude is used ONLY to read documents and to write / simulate messages. Every function in
the rest of the system works without it: when no credentials are configured (or a call
fails) callers receive `None` and fall back to deterministic offline logic, so a demo never
dies on bad Wi-Fi.

Credentials are resolved by the Anthropic SDK (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
`ant auth login` profile). Set PROCUREAI_OFFLINE=1 to force offline mode.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)

_client = None
_last_error: Optional[str] = None


def llm_configured() -> bool:
    """True when Claude credentials appear to be available and offline mode is not forced."""
    if os.environ.get("PROCUREAI_OFFLINE", "").strip() in {"1", "true", "yes"}:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    profile_dir = os.path.expanduser("~/.config/anthropic")
    return os.path.isdir(profile_dir) and bool(os.listdir(profile_dir))


def last_error() -> Optional[str]:
    return _last_error


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(max_retries=2, timeout=120.0)
    return _client


def structured_call(
    system: str,
    content: List[dict] | str,
    output_model: Type[T],
    effort: str = "medium",
    max_tokens: int = 16000,
) -> Optional[T]:
    """Ask Claude for a response validated against `output_model`.
    Returns None (and records `last_error()`) on any failure so callers can fall back offline."""
    global _last_error
    _last_error = None
    if not llm_configured():
        _last_error = "Claude is not configured (offline mode)."
        return None
    try:
        import anthropic
    except ImportError:
        _last_error = "The 'anthropic' package is not installed."
        return None

    messages = [{"role": "user", "content": content}]
    try:
        client = _get_client()
        # Server-side refusal fallback is enabled by default for claude-opus-5-5.
        response = client.beta.messages.parse(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": effort},
            output_format=output_model,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            _last_error = "Claude declined this request."
            return None
        if response.stop_reason == "max_tokens":
            _last_error = "Claude's response was cut off (max_tokens)."
            return None
        parsed = response.parsed_output
        if parsed is None:
            _last_error = "Claude returned no structured output."
        return parsed
    except anthropic.AuthenticationError:
        _last_error = "Claude credentials were rejected (authentication error)."
    except anthropic.PermissionDeniedError:
        _last_error = "Claude credentials lack permission for this model."
    except anthropic.RateLimitError:
        _last_error = "Claude rate limit reached — using offline logic."
    except anthropic.BadRequestError as exc:
        _last_error = f"Claude rejected the request: {exc.message}"
    except anthropic.APIStatusError as exc:
        _last_error = f"Claude API error ({exc.status_code})."
    except anthropic.APIConnectionError:
        _last_error = "Could not reach Claude (network error) — using offline logic."
    except Exception as exc:  # noqa: BLE001 — never let an AI failure break the app
        _last_error = f"Unexpected Claude error: {exc}"
    return None


def pdf_document_block(file_bytes: bytes) -> dict:
    import base64

    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf",
                   "data": base64.standard_b64encode(file_bytes).decode("utf-8")},
    }


def status_line() -> Tuple[bool, str]:
    if llm_configured():
        return True, f"Claude AI connected ({MODEL})"
    return False, "Deterministic engine — runs fully offline"


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}

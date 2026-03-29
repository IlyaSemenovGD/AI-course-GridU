"""
Langfuse observability helpers.

The module is designed to degrade gracefully: if Langfuse credentials are not
configured, every function becomes a no-op so the rest of the application
continues to work without any changes.
"""

from __future__ import annotations

import os
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Client factory (cached singleton)
# ---------------------------------------------------------------------------

_langfuse_client = None
_initialised = False


def get_langfuse():
    """
    Return a Langfuse client instance, or None if credentials are absent /
    the package is not installed.
    """
    global _langfuse_client, _initialised
    if _initialised:
        return _langfuse_client

    _initialised = True
    try:
        from langfuse import Langfuse  # noqa: PLC0415

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        if not public_key or not secret_key:
            _langfuse_client = None
            return None

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        _langfuse_client = None

    return _langfuse_client


# ---------------------------------------------------------------------------
# Tracing helpers
# ---------------------------------------------------------------------------

def start_trace(name: str, metadata: Optional[dict] = None, user_id: str = ""):
    """Create and return a Langfuse trace, or a no-op stub."""
    lf = get_langfuse()
    if lf is None:
        return _NullTrace()
    try:
        return lf.trace(
            name=name,
            metadata=metadata or {},
            user_id=user_id or None,
        )
    except Exception:
        return _NullTrace()


def start_generation(trace, name: str, model: str, prompt: str, params: dict):
    """Create a generation span on *trace*."""
    if isinstance(trace, _NullTrace):
        return _NullTrace()
    try:
        return trace.generation(
            name=name,
            model=model,
            input=prompt,
            model_parameters=params,
        )
    except Exception:
        return _NullTrace()


def end_generation(generation, output: str, usage: Optional[dict] = None) -> None:
    if isinstance(generation, _NullTrace):
        return
    try:
        generation.end(output=output, usage=usage or {})
    except Exception:
        pass


def end_trace(trace, output: Any = None) -> None:
    if isinstance(trace, _NullTrace):
        return
    try:
        trace.update(output=output)
        lf = get_langfuse()
        if lf:
            lf.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Null-object pattern (no-op stub)
# ---------------------------------------------------------------------------

class _NullTrace:
    """Drop-in replacement when Langfuse is unavailable."""

    def generation(self, **_):
        return _NullTrace()

    def update(self, **_):
        pass

    def end(self, **_):
        pass

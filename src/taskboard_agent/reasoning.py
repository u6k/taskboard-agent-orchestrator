"""Explicit server-specific reasoning controls shared by both LLM paths."""
from __future__ import annotations

from typing import Any

import httpx

from taskboard_agent.config import ConfigError


def validate_reasoning_config(model: str, backend: str | None, effort: str | None) -> None:
    if backend is None and effort is None:
        return
    if effort not in ("none", "low", "medium", "high"):
        raise ConfigError("llm_reasoning_effort must be none, low, medium or high")
    if backend not in ("strata", "ollama", "openai_compatible"):
        raise ConfigError("llm_reasoning_backend is required: strata, ollama or openai_compatible")
    if model.startswith("lm_studio/"):
        raise ConfigError("LM Studio Chat Completions reasoning control is not supported")
    providers = ("openai/", "ollama/", "ollama_chat/") if backend == "ollama" else ("openai/",)
    if not model.startswith(providers):
        raise ConfigError(f"llm_reasoning_backend={backend} requires model prefix {providers}")


def reasoning_kwargs(model: str, backend: str | None, effort: str | None) -> dict[str, Any]:
    validate_reasoning_config(model, backend, effort)
    if effort is None:
        return {}
    if backend == "strata":
        return {"extra_body": {"reasoning_effort": effort}}
    if backend == "ollama" and model.startswith(("ollama/", "ollama_chat/")):
        return {"think": False if effort == "none" else effort}
    if backend == "ollama":
        # Server capabilities were checked at startup; OpenAI model metadata
        # does not describe local Ollama model names.
        return {"extra_body": {"reasoning_effort": effort}}
    return {"reasoning_effort": effort, "drop_params": False}


def validate_ollama_capabilities(
    model: str, api_base: str | None, effort: str, *, timeout: int | None = None,
    api_key: str | None = None,
) -> None:
    if not api_base:
        raise ConfigError("Ollama reasoning requires an explicit llm_api_base")
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.post(
            f"{base}/api/show", json={"model": model.split("/", 1)[1]},
            headers=headers, timeout=timeout or 30,
        )
        response.raise_for_status()
        values = response.json()["thinking"]["values"]
        requested = False if effort == "none" else effort
        if not isinstance(values, list) or not any(
            type(value) is type(requested) and value == requested for value in values
        ):
            raise ConfigError(f"Ollama model does not support reasoning effort {effort}")
    except ConfigError:
        raise
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError("Unable to verify Ollama thinking capabilities via /api/show") from exc

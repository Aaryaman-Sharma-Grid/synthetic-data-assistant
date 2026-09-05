"""Langfuse tracing helpers with safe no-op behavior when unconfigured."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from typing import Any

from src.config import Settings


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)
_MAX_TRACED_STRING_LENGTH = 20_000
_UNSET = object()


def langfuse_configured(settings: Settings) -> bool:
    """Return whether both project credentials are available."""
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def mask_sensitive_data(
    value: Any = None,
    *,
    data: Any = _UNSET,
    **_: Any,
) -> Any:
    """Redact credential-like fields and cap unexpectedly large trace strings."""
    if data is not _UNSET:
        value = data
    if isinstance(value, Mapping):
        masked: dict[Any, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
                masked[key] = "[REDACTED]"
            else:
                masked[key] = mask_sensitive_data(item)
        return masked
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [mask_sensitive_data(item) for item in value]
    if isinstance(value, str) and len(value) > _MAX_TRACED_STRING_LENGTH:
        return f"{value[:_MAX_TRACED_STRING_LENGTH]}… [TRUNCATED]"
    return value


@lru_cache(maxsize=4)
def _create_client(
    public_key: str,
    secret_key: str,
    base_url: str,
    environment: str,
):
    """Create one reusable SDK client per configured Langfuse project."""
    from langfuse import Langfuse

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=environment,
        mask=mask_sensitive_data,
    )


def get_langfuse_client(settings: Settings):
    """Return an explicitly configured client, or ``None`` when disabled."""
    if not langfuse_configured(settings):
        return None
    return _create_client(
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_base_url,
        settings.langfuse_tracing_environment,
    )


def update_observation(observation: Any | None, **attributes: Any) -> None:
    """Update a span without allowing telemetry failures to break the app."""
    if observation is None:
        return
    try:
        observation.update(**mask_sensitive_data(attributes))
    except Exception:
        # Observability must never make the data-generation workflow fail.
        return


@contextmanager
def observe_operation(
    settings: Settings,
    name: str,
    *,
    as_type: str = "span",
    input_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    trace_name: str | None = None,
    version: str = "1.0",
) -> Iterator[Any | None]:
    """Create a nested observation, falling back to a no-op if setup fails."""
    try:
        client = get_langfuse_client(settings)
    except Exception:
        client = None
    if client is None:
        yield None
        return

    from langfuse import propagate_attributes

    stack = ExitStack()
    try:
        stack.enter_context(
            propagate_attributes(
                tags=tags,
                trace_name=trace_name,
                environment=settings.langfuse_tracing_environment,
            )
        )
        observation = stack.enter_context(
            client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=mask_sensitive_data(input_data),
                metadata=mask_sensitive_data(metadata),
                model=model,
                model_parameters=model_parameters,
                version=version,
            )
        )
    except Exception:
        stack.close()
        yield None
        return

    try:
        yield observation
    except Exception as exc:
        update_observation(
            observation,
            level="ERROR",
            status_message=f"{type(exc).__name__}: {exc}"[:500],
        )
        raise
    finally:
        stack.close()


def flush_langfuse(settings: Settings) -> None:
    """Synchronously send buffered observations when Langfuse is configured."""
    try:
        client = get_langfuse_client(settings)
        if client is not None:
            client.flush()
    except Exception:
        return

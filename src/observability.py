"""Optional Langfuse tracing that remains a no-op until keys are configured."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from src.config import Settings


def langfuse_configured(settings: Settings) -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


@contextmanager
def observe_operation(
    settings: Settings,
    name: str,
    *,
    as_type: str = "span",
    input_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> Iterator[Any | None]:
    """Create a Langfuse observation when configured, otherwise yield None."""
    if not langfuse_configured(settings):
        yield None
        return

    from langfuse import get_client

    client = get_client()
    with client.start_as_current_observation(
        name=name,
        as_type=as_type,
        input=input_data,
        metadata=metadata,
        model=model,
    ) as observation:
        yield observation


"""Verify Langfuse credentials and emit a small diagnostic trace."""

from __future__ import annotations

from src.config import get_settings
from src.observability import (
    flush_langfuse,
    get_langfuse_client,
    langfuse_configured,
    observe_operation,
    update_observation,
)


def main() -> None:
    settings = get_settings()
    if not langfuse_configured(settings):
        raise SystemExit("Langfuse credentials are not configured in .env.")

    client = get_langfuse_client(settings)
    if client is None or not client.auth_check():
        raise SystemExit(
            "Langfuse authentication failed. Check the keys and LANGFUSE_BASE_URL in .env."
        )

    trace_id = None
    with observe_operation(
        settings,
        "verify-observability-setup",
        input_data="Verify the Synthetic Data Assistant tracing connection.",
        metadata={"check": "credentials-and-ingestion"},
        tags=["synthetic-data", "diagnostic"],
        trace_name="verify-observability-setup",
    ) as root_observation:
        trace_id = client.get_current_trace_id()
        with observe_operation(
            settings,
            "check-langfuse-authentication",
            as_type="evaluator",
            input_data={"base_url": settings.langfuse_base_url},
        ) as check_observation:
            update_observation(check_observation, output={"authenticated": True})
        update_observation(
            root_observation,
            output={"authenticated": True, "trace_ingested": True},
        )

    flush_langfuse(settings)
    print("Langfuse authentication: successful")
    print(f"Trace ID: {trace_id}")
    print(f"Trace URL: {client.get_trace_url(trace_id=trace_id)}")


if __name__ == "__main__":
    main()

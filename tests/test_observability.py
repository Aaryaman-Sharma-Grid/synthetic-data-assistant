from types import SimpleNamespace

from src.config import Settings
from src.llm import _usage_details
from src.observability import mask_sensitive_data, observe_operation, update_observation


def test_mask_sensitive_data_redacts_nested_credentials_and_caps_text() -> None:
    payload = {
        "prompt": "generate rows",
        "authorization": "Bearer sensitive",
        "nested": {"api-key": "sensitive", "value": 7},
        "long_text": "x" * 20_100,
    }

    masked = mask_sensitive_data(payload)

    assert masked["prompt"] == "generate rows"
    assert masked["authorization"] == "[REDACTED]"
    assert masked["nested"] == {"api-key": "[REDACTED]", "value": 7}
    assert masked["long_text"].endswith("… [TRUNCATED]")


def test_observe_operation_is_noop_without_credentials() -> None:
    settings = Settings(
        _env_file=None,
        LANGFUSE_PUBLIC_KEY="",
        LANGFUSE_SECRET_KEY="",
    )

    with observe_operation(settings, "test-operation") as observation:
        assert observation is None


def test_update_observation_masks_attributes() -> None:
    class FakeObservation:
        attributes = None

        def update(self, **attributes):
            self.attributes = attributes

    observation = FakeObservation()
    update_observation(observation, input={"password": "sensitive", "safe": "value"})

    assert observation.attributes == {
        "input": {"password": "[REDACTED]", "safe": "value"}
    }


def test_gemini_usage_is_mapped_to_exclusive_langfuse_categories() -> None:
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=120,
            cached_content_token_count=20,
            candidates_token_count=35,
            thoughts_token_count=8,
        )
    )

    assert _usage_details(response) == {
        "input": 100,
        "output": 35,
        "cache_read_input_tokens": 20,
        "reasoning": 8,
    }

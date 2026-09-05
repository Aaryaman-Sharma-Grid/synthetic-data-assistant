"""Gemini helpers for streaming chat and schema-constrained JSON planning."""

from collections.abc import Iterator
import json
from typing import Any

from google import genai
from google.genai import types

from src.config import Settings
from src.models import GenerationPlan, ParsedSchema, TableEditPlan


def create_gemini_client(settings: Settings) -> genai.Client:
    """Create the official Google Gen AI client using Vertex AI configuration."""
    if not settings.google_cloud_project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured.")
    return genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )


def stream_text(client: genai.Client, model: str, prompt: str) -> Iterator[str]:
    """Yield Gemini chat text chunks using the SDK's recommended streaming API."""
    chat = client.chats.create(model=model)
    response_stream = chat.send_message_stream(prompt)
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


def generate_structured_json(
    client: genai.Client,
    model: str,
    prompt: str,
    response_schema: Any,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
) -> Any:
    """Generate validated JSON using Gemini's structured-output mode."""
    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    response = chat.send_message(prompt)
    if response.parsed is not None:
        return response.parsed
    if not response.text:
        raise RuntimeError("Gemini returned an empty structured response.")
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
        raise RuntimeError(
            f"Gemini returned incomplete structured JSON (finish reason: {finish_reason}). "
            "Increase Max output tokens or simplify the prompt."
        ) from exc


def create_generation_plan(
    client: genai.Client,
    settings: Settings,
    schema: ParsedSchema,
    instructions: str,
    temperature: float,
    max_output_tokens: int,
) -> GenerationPlan:
    """Ask Gemini for compact semantic rules; row creation remains deterministic."""
    compact_schema = [
        {
            "table": table.name,
            "columns": [
                {
                    "name": column.name,
                    "type": column.data_type,
                    "nullable": column.nullable,
                    "enum_values": column.enum_values,
                }
                for column in table.columns
            ],
        }
        for table in schema.tables
    ]
    prompt = f"""
You are planning realistic synthetic data generation for a relational database.
Return only a compact generation plan matching the supplied JSON schema.

DDL structure:
{json.dumps(compact_schema, indent=2)}

User instructions:
{instructions or "Create realistic, internally consistent business data."}

Rules:
- Include only column rules that improve semantic realism or implement the user instructions.
- Never create rules for primary-key or foreign-key columns.
- Use exact table and column names from the schema.
- Keep values compatible with the declared type, ENUM, nullability, and size constraints.
- Use minimum/maximum for numeric ranges and choices for controlled categories.
- Keep null_probability between 0 and 1.
"""
    parsed = generate_structured_json(
        client,
        settings.gemini_model,
        prompt,
        GenerationPlan,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return GenerationPlan.model_validate(parsed)


def create_table_edit_plan(
    client: genai.Client,
    settings: Settings,
    table_name: str,
    columns: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    feedback: str,
    temperature: float = 0.1,
) -> TableEditPlan:
    """Translate natural-language feedback into safe, scalable table operations."""
    prompt = f"""
Convert a user's table-edit request into structured operations.

Table: {table_name}
Columns: {json.dumps(columns, default=str)}
Sample rows: {json.dumps(sample_rows, default=str)}
User feedback: {feedback}

Rules:
- Use exact column names.
- Never modify primary keys or foreign keys.
- Prefer scalable operations rather than returning rows.
- For percentage increases, use multiply (for example 10% increase means 1.1).
- For minimum/maximum requirements use clamp_min/clamp_max.
- Use regenerate with a semantic_type for fresh realistic values.
- Return no more than five operations.
"""
    parsed = generate_structured_json(
        client,
        settings.gemini_model,
        prompt,
        TableEditPlan,
        temperature=temperature,
        max_output_tokens=2048,
    )
    return TableEditPlan.model_validate(parsed)

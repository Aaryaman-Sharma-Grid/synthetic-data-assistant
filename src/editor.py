"""Apply Gemini-planned natural-language changes to one generated table."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from src.generator import SyntheticDataGenerator
from src.models import ParsedSchema, TableEditPlan


def apply_edit_plan(
    schema: ParsedSchema,
    data: dict[str, pd.DataFrame],
    table_name: str,
    plan: TableEditPlan,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Apply safe column operations while protecting keys and constraints."""
    table = schema.table(table_name)
    updated = {name: frame.copy(deep=True) for name, frame in data.items()}
    frame = updated[table.name]
    protected = {column.name.lower() for column in table.primary_keys}
    protected.update(foreign_key.column.lower() for foreign_key in table.foreign_keys)
    notes: list[str] = []
    generator = SyntheticDataGenerator(schema, len(frame), seed=seed)

    for operation in plan.operations:
        try:
            column = table.column(operation.column)
        except KeyError:
            notes.append(f"Skipped unknown column: {operation.column}")
            continue
        if column.name.lower() in protected:
            notes.append(f"Protected key column was not modified: {column.name}")
            continue
        series = frame[column.name]
        action = operation.action
        value: Any = operation.value
        if action == "set":
            frame[column.name] = value
        elif action in {"multiply", "add", "clamp_min", "clamp_max"}:
            numeric = pd.to_numeric(series, errors="coerce")
            number = float(value)
            if action == "multiply":
                result = numeric * number
            elif action == "add":
                result = numeric + number
            elif action == "clamp_min":
                result = numeric.clip(lower=number)
            else:
                result = numeric.clip(upper=number)
            if column.base_type in {"DECIMAL", "NUMERIC"}:
                scale = column.scale or 2
                frame[column.name] = [None if pd.isna(item) else Decimal(str(round(float(item), scale))) for item in result]
            elif column.base_type in {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT"}:
                frame[column.name] = result.round().astype("Int64")
            else:
                frame[column.name] = result
        elif action == "uppercase":
            frame[column.name] = series.map(lambda item: item.upper() if isinstance(item, str) else item)
        elif action == "lowercase":
            frame[column.name] = series.map(lambda item: item.lower() if isinstance(item, str) else item)
        elif action == "regenerate":
            frame[column.name] = generator.regenerate_column(table, column, len(frame), operation.semantic_type)
        elif action == "fill_null":
            replacement = value
            if replacement is None:
                generated = generator.regenerate_column(table, column, len(frame), operation.semantic_type)
                frame[column.name] = [generated[index] if pd.isna(item) else item for index, item in enumerate(series)]
            else:
                frame[column.name] = series.fillna(replacement)
        notes.append(f"Applied {action} to {column.name}")

    return updated, notes


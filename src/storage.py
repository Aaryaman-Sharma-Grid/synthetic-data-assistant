"""Persist generated datasets locally and into isolated PostgreSQL schemas."""

from __future__ import annotations

import io
import json
import re
import uuid
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
    text,
)

from src.config import Settings
from src.models import ColumnSchema, DatasetArtifact, ParsedSchema


GENERATED_ROOT = Path(__file__).resolve().parents[1] / "data" / "generated"


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", value).lower().strip("_")
    return cleaned[:55] or "dataset"


def _sql_type(column: ColumnSchema) -> Any:
    if column.base_type in {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT"}:
        return Integer()
    if column.base_type in {"DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL"}:
        return Numeric(precision=column.precision or 18, scale=column.scale or 2)
    if column.base_type == "DATE":
        return Date()
    if column.base_type in {"DATETIME", "TIMESTAMP"}:
        return DateTime()
    if column.base_type in {"BOOL", "BOOLEAN"}:
        return Boolean()
    if column.base_type in {"TEXT", "LONGTEXT", "MEDIUMTEXT"}:
        return Text()
    return String(length=column.max_length or 255)


def _python_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (date, datetime)) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):
        return value.item()
    return value


def _build_tables(schema: ParsedSchema, postgres_schema: str) -> tuple[MetaData, dict[str, Table]]:
    metadata = MetaData(schema=postgres_schema)
    table_map: dict[str, Table] = {}
    table_names = {table.name.lower(): _safe_identifier(table.name) for table in schema.tables}
    for table_schema in schema.tables:
        foreign_keys = {foreign_key.column.lower(): foreign_key for foreign_key in table_schema.foreign_keys}
        columns: list[Column[Any]] = []
        for column_schema in table_schema.columns:
            foreign_key = foreign_keys.get(column_schema.name.lower())
            foreign = None
            if foreign_key:
                target_table = table_names[foreign_key.referenced_table.lower()]
                target_column = _safe_identifier(foreign_key.referenced_column)
                foreign = ForeignKey(
                    f"{postgres_schema}.{target_table}.{target_column}",
                    deferrable=True,
                    initially="DEFERRED",
                )
            args = [_sql_type(column_schema)]
            if foreign is not None:
                args.append(foreign)
            columns.append(
                Column(
                    _safe_identifier(column_schema.name),
                    *args,
                    primary_key=column_schema.primary_key,
                    nullable=column_schema.nullable and not column_schema.primary_key,
                    unique=column_schema.unique,
                )
            )
        table_map[table_schema.name] = Table(_safe_identifier(table_schema.name), metadata, *columns)
    return metadata, table_map


def persist_to_postgres(
    settings: Settings,
    schema: ParsedSchema,
    data: dict[str, pd.DataFrame],
    artifact: DatasetArtifact,
    ddl: str,
    instructions: str,
) -> None:
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    metadata, table_map = _build_tables(schema, artifact.postgres_schema)
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{artifact.postgres_schema}"'))
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS public.synthetic_datasets (
                    dataset_id VARCHAR(80) PRIMARY KEY,
                    display_name VARCHAR(255) NOT NULL,
                    postgres_schema VARCHAR(80) NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    source_ddl TEXT NOT NULL,
                    instructions TEXT
                )
                """
            )
        )
        metadata.create_all(connection)
        for table_schema in schema.tables:
            table = table_map[table_schema.name]
            records = [
                {_safe_identifier(key): _python_value(value) for key, value in record.items()}
                for record in data[table_schema.name].to_dict(orient="records")
            ]
            if records:
                connection.execute(table.insert(), records)
        connection.execute(
            text(
                """
                INSERT INTO public.synthetic_datasets
                    (dataset_id, display_name, postgres_schema, created_at, source_ddl, instructions)
                VALUES
                    (:dataset_id, :display_name, :postgres_schema, :created_at, :source_ddl, :instructions)
                ON CONFLICT (dataset_id) DO UPDATE SET
                    display_name = EXCLUDED.display_name,
                    source_ddl = EXCLUDED.source_ddl,
                    instructions = EXCLUDED.instructions
                """
            ),
            {
                "dataset_id": artifact.dataset_id,
                "display_name": artifact.display_name,
                "postgres_schema": artifact.postgres_schema,
                "created_at": artifact.created_at,
                "source_ddl": ddl,
                "instructions": instructions,
            },
        )
    engine.dispose()


def save_dataset(
    settings: Settings,
    schema: ParsedSchema,
    data: dict[str, pd.DataFrame],
    ddl: str,
    instructions: str,
    display_name: str,
) -> DatasetArtifact:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = f"dataset_{timestamp}_{uuid.uuid4().hex[:8]}"
    postgres_schema = _safe_identifier(dataset_id)
    local_path = GENERATED_ROOT / dataset_id
    local_path.mkdir(parents=True, exist_ok=True)
    (local_path / "schema.ddl").write_text(ddl, encoding="utf-8")
    for table_name, frame in data.items():
        frame.to_csv(local_path / f"{_safe_identifier(table_name)}.csv", index=False)

    artifact = DatasetArtifact(
        dataset_id=dataset_id,
        display_name=display_name,
        postgres_schema=postgres_schema,
        created_at=datetime.now(),
        local_path=str(local_path),
        rows_per_table={name: len(frame) for name, frame in data.items()},
    )
    try:
        persist_to_postgres(settings, schema, data, artifact, ddl, instructions)
        artifact.database_saved = True
    except Exception as exc:
        artifact.warnings.append(f"PostgreSQL persistence failed: {exc}")

    manifest = {
        "artifact": artifact.model_dump(mode="json"),
        "schema": schema.model_dump(mode="json"),
        "instructions": instructions,
    }
    (local_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return artifact


def build_zip_archive(
    artifact: DatasetArtifact,
    schema: ParsedSchema,
    data: dict[str, pd.DataFrame],
    ddl: str,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("schema.ddl", ddl)
        archive.writestr("manifest.json", json.dumps({"artifact": artifact.model_dump(mode="json"), "schema": schema.model_dump(mode="json")}, indent=2))
        for table_name, frame in data.items():
            archive.writestr(f"csv/{_safe_identifier(table_name)}.csv", frame.to_csv(index=False))
    return buffer.getvalue()


def list_saved_datasets() -> list[DatasetArtifact]:
    if not GENERATED_ROOT.exists():
        return []
    artifacts: list[DatasetArtifact] = []
    for manifest_path in GENERATED_ROOT.glob("*/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts.append(DatasetArtifact.model_validate(payload["artifact"]))
        except Exception:
            continue
    return sorted(artifacts, key=lambda item: item.created_at, reverse=True)


def load_saved_dataset(
    artifact: DatasetArtifact,
) -> tuple[ParsedSchema, dict[str, pd.DataFrame], str]:
    local_path = Path(artifact.local_path)
    payload = json.loads((local_path / "manifest.json").read_text(encoding="utf-8"))
    schema = ParsedSchema.model_validate(payload["schema"])
    ddl = (local_path / "schema.ddl").read_text(encoding="utf-8")
    data = {
        table.name: pd.read_csv(local_path / f"{_safe_identifier(table.name)}.csv")
        for table in schema.tables
    }
    return schema, data, ddl

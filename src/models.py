"""Domain models shared by schema parsing, generation, editing, and storage."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ForeignKeySchema(BaseModel):
    column: str
    referenced_table: str
    referenced_column: str


class ColumnSchema(BaseModel):
    name: str
    data_type: str
    base_type: str
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False
    auto_increment: bool = False
    default: Any | None = None
    enum_values: list[str] = Field(default_factory=list)
    max_length: int | None = None
    precision: int | None = None
    scale: int | None = None
    checks: list[str] = Field(default_factory=list)


class TableSchema(BaseModel):
    name: str
    columns: list[ColumnSchema]
    foreign_keys: list[ForeignKeySchema] = Field(default_factory=list)

    def column(self, name: str) -> ColumnSchema:
        for column in self.columns:
            if column.name.lower() == name.lower():
                return column
        raise KeyError(f"Unknown column {self.name}.{name}")

    @property
    def primary_keys(self) -> list[ColumnSchema]:
        return [column for column in self.columns if column.primary_key]


class ParsedSchema(BaseModel):
    tables: list[TableSchema]
    warnings: list[str] = Field(default_factory=list)

    def table(self, name: str) -> TableSchema:
        for table in self.tables:
            if table.name.lower() == name.lower():
                return table
        raise KeyError(f"Unknown table {name}")


SemanticType = Literal[
    "auto",
    "first_name",
    "last_name",
    "full_name",
    "email",
    "phone",
    "address",
    "city",
    "state",
    "postal_code",
    "company",
    "department",
    "job_title",
    "industry",
    "website",
    "date",
    "datetime",
    "currency",
    "integer",
    "number",
    "text",
    "isbn",
    "genre",
    "book_title",
    "restaurant_name",
    "menu_item",
    "boolean",
]


class ColumnGenerationRule(BaseModel):
    table: str
    column: str
    semantic_type: SemanticType = "auto"
    choices: list[str] = Field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    fixed_value: str | int | float | bool | None = None
    null_probability: float | None = Field(default=None, ge=0, le=1)


class GenerationPlan(BaseModel):
    summary: str = "Deterministic schema-aware generation"
    locale: str = "en_US"
    rules: list[ColumnGenerationRule] = Field(default_factory=list)


EditAction = Literal[
    "set",
    "multiply",
    "add",
    "clamp_min",
    "clamp_max",
    "uppercase",
    "lowercase",
    "regenerate",
    "fill_null",
]


class EditOperation(BaseModel):
    column: str
    action: EditAction
    value: str | int | float | bool | None = None
    semantic_type: SemanticType = "auto"


class TableEditPlan(BaseModel):
    summary: str
    operations: list[EditOperation] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    level: Literal["error", "warning"]
    table: str
    column: str | None = None
    message: str


class DatasetArtifact(BaseModel):
    dataset_id: str
    display_name: str
    postgres_schema: str
    created_at: datetime
    local_path: str
    rows_per_table: dict[str, int]
    database_saved: bool = False
    warnings: list[str] = Field(default_factory=list)

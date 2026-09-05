"""DDL parsing for the MySQL-flavoured sample schemas used by the course."""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from src.models import ColumnSchema, ForeignKeySchema, ParsedSchema, TableSchema


class SchemaParseError(ValueError):
    """Raised when uploaded DDL cannot be interpreted safely."""


def _identifier_name(value: Any) -> str:
    if hasattr(value, "name") and value.name:
        return str(value.name)
    return str(value)


def _literal_value(expression: exp.Expression | None) -> Any | None:
    if expression is None:
        return None
    if isinstance(expression, exp.Literal):
        if expression.is_string:
            return expression.this
        raw = str(expression.this)
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    if isinstance(expression, exp.Boolean):
        return bool(expression.this)
    if isinstance(expression, exp.Null):
        return None
    return expression.sql(dialect="mysql")


def _data_type_details(kind: exp.DataType) -> tuple[str, str, int | None, int | None, int | None, list[str]]:
    data_type = kind.sql(dialect="mysql")
    base_type = str(kind.this.value if hasattr(kind.this, "value") else kind.this).upper()
    params: list[str] = []
    for parameter in kind.expressions or []:
        inner = parameter.this if isinstance(parameter, exp.DataTypeParam) else parameter
        params.append(str(getattr(inner, "this", inner)))

    max_length = None
    precision = None
    scale = None
    enum_values: list[str] = []
    if base_type in {"VARCHAR", "CHAR"} and params:
        max_length = int(params[0])
    elif base_type in {"DECIMAL", "NUMERIC"} and params:
        precision = int(params[0])
        scale = int(params[1]) if len(params) > 1 else 0
    elif base_type == "ENUM":
        enum_values = [str(getattr(item, "this", item)) for item in kind.expressions or []]
    return data_type, base_type, max_length, precision, scale, enum_values


def _parse_column(column_def: exp.ColumnDef) -> ColumnSchema:
    kind = column_def.args.get("kind")
    if not isinstance(kind, exp.DataType):
        raise SchemaParseError(f"Column {column_def.name} has no supported data type")
    data_type, base_type, max_length, precision, scale, enum_values = _data_type_details(kind)

    primary_key = False
    unique = False
    auto_increment = False
    nullable = True
    default: Any | None = None
    checks: list[str] = []
    for wrapper in column_def.args.get("constraints") or []:
        constraint = wrapper.args.get("kind")
        if isinstance(constraint, exp.PrimaryKeyColumnConstraint):
            primary_key = True
            nullable = False
        elif isinstance(constraint, exp.UniqueColumnConstraint):
            unique = True
        elif isinstance(constraint, exp.AutoIncrementColumnConstraint):
            auto_increment = True
        elif isinstance(constraint, exp.NotNullColumnConstraint):
            nullable = False
        elif isinstance(constraint, exp.DefaultColumnConstraint):
            default = _literal_value(constraint.this)
        elif isinstance(constraint, exp.CheckColumnConstraint):
            checks.append(constraint.this.sql(dialect="mysql"))

    return ColumnSchema(
        name=column_def.name,
        data_type=data_type,
        base_type=base_type,
        nullable=nullable,
        primary_key=primary_key,
        unique=unique,
        auto_increment=auto_increment,
        default=default,
        enum_values=enum_values,
        max_length=max_length,
        precision=precision,
        scale=scale,
        checks=checks,
    )


def _parse_foreign_key(foreign_key: exp.ForeignKey) -> list[ForeignKeySchema]:
    local_columns = [_identifier_name(item) for item in foreign_key.expressions]
    reference = foreign_key.args.get("reference")
    reference_schema = reference.this if isinstance(reference, exp.Reference) else None
    if not isinstance(reference_schema, exp.Schema):
        return []
    referenced_table = _identifier_name(reference_schema.this)
    referenced_columns = [_identifier_name(item) for item in reference_schema.expressions]
    return [
        ForeignKeySchema(
            column=local,
            referenced_table=referenced_table,
            referenced_column=remote,
        )
        for local, remote in zip(local_columns, referenced_columns, strict=False)
    ]


def parse_ddl(ddl: str) -> ParsedSchema:
    """Parse uploaded DDL into a compact, validation-friendly schema model."""
    if not ddl.strip():
        raise SchemaParseError("The uploaded DDL file is empty.")
    try:
        statements = sqlglot.parse(ddl, read="mysql")
    except ParseError as exc:
        raise SchemaParseError(f"Could not parse DDL: {exc}") from exc

    tables: list[TableSchema] = []
    warnings: list[str] = []
    for statement in statements:
        if not isinstance(statement, exp.Create) or str(statement.args.get("kind", "")).upper() != "TABLE":
            continue
        schema_expression = statement.this
        if not isinstance(schema_expression, exp.Schema):
            continue
        table_name = _identifier_name(schema_expression.this)
        columns = [
            _parse_column(item)
            for item in schema_expression.expressions
            if isinstance(item, exp.ColumnDef)
        ]
        foreign_keys: list[ForeignKeySchema] = []
        for item in schema_expression.expressions:
            if isinstance(item, exp.ForeignKey):
                foreign_keys.extend(_parse_foreign_key(item))
        if not columns:
            warnings.append(f"Table {table_name} contains no parsed columns.")
        tables.append(TableSchema(name=table_name, columns=columns, foreign_keys=foreign_keys))

    table_by_name = {table.name.lower(): table for table in tables}
    for statement in statements:
        if not isinstance(statement, exp.Alter):
            continue
        table_name = _identifier_name(statement.this)
        target = table_by_name.get(table_name.lower())
        if target is None:
            warnings.append(f"ALTER TABLE references unknown table {table_name}.")
            continue
        existing = {(fk.column.lower(), fk.referenced_table.lower(), fk.referenced_column.lower()) for fk in target.foreign_keys}
        for foreign_key in statement.find_all(exp.ForeignKey):
            for parsed_fk in _parse_foreign_key(foreign_key):
                key = (
                    parsed_fk.column.lower(),
                    parsed_fk.referenced_table.lower(),
                    parsed_fk.referenced_column.lower(),
                )
                if key not in existing:
                    target.foreign_keys.append(parsed_fk)
                    existing.add(key)

    if not tables:
        raise SchemaParseError("No CREATE TABLE statements were found in the uploaded file.")

    for table in tables:
        column_names = {column.name.lower() for column in table.columns}
        for foreign_key in table.foreign_keys:
            if foreign_key.column.lower() not in column_names:
                raise SchemaParseError(f"Foreign key column {table.name}.{foreign_key.column} does not exist.")
            try:
                referenced = ParsedSchema(tables=tables).table(foreign_key.referenced_table)
                referenced.column(foreign_key.referenced_column)
            except KeyError as exc:
                raise SchemaParseError(f"Invalid foreign key on {table.name}.{foreign_key.column}: {exc}") from exc

    return ParsedSchema(tables=tables, warnings=warnings)


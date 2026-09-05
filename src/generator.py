"""Deterministic, scalable synthetic data generation with relational integrity."""

from __future__ import annotations

import random
import re
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
from faker import Faker

from src.models import (
    ColumnGenerationRule,
    ColumnSchema,
    GenerationPlan,
    ParsedSchema,
    TableSchema,
    ValidationIssue,
)


INTEGER_TYPES = {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT"}
DECIMAL_TYPES = {"DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL"}
TEXT_TYPES = {"VARCHAR", "CHAR", "TEXT", "LONGTEXT", "MEDIUMTEXT"}
DATETIME_TYPES = {"DATETIME", "TIMESTAMP"}


def infer_semantic_type(table: str, column: str, base_type: str) -> str:
    name = column.lower()
    table_name = table.lower()
    if name in {"first_name", "firstname"}:
        return "first_name"
    if name in {"last_name", "lastname", "surname"}:
        return "last_name"
    if name in {"name", "full_name"}:
        if "compan" in table_name:
            return "company"
        if "restaurant" in table_name:
            return "restaurant_name"
        if "department" in table_name:
            return "department"
        return "full_name"
    if "email" in name:
        return "email"
    if "phone" in name:
        return "phone"
    if name == "address" or name.endswith("_address"):
        return "address"
    if name == "city":
        return "city"
    if name == "state":
        return "state"
    if "zip" in name or "postal" in name:
        return "postal_code"
    if "company" in name and not name.endswith("_id"):
        return "company"
    if "department" in name and not name.endswith("_id"):
        return "department"
    if "job_title" in name or name == "role":
        return "job_title"
    if "industry" in name:
        return "industry"
    if "website" in name or name == "url":
        return "website"
    if "isbn" in name:
        return "isbn"
    if "genre" in name:
        return "genre"
    if name == "title" and "book" in table_name:
        return "book_title"
    if name == "item_name" or ("menu" in table_name and "item" in name):
        return "menu_item"
    if base_type == "DATE":
        return "date"
    if base_type in DATETIME_TYPES:
        return "datetime"
    if base_type in INTEGER_TYPES:
        return "integer"
    if base_type in DECIMAL_TYPES:
        return "currency" if any(token in name for token in ("salary", "amount", "price", "budget", "subtotal")) else "number"
    if base_type in {"BOOL", "BOOLEAN"}:
        return "boolean"
    if base_type in TEXT_TYPES and (base_type == "TEXT" or any(token in name for token in ("description", "comments", "biography", "goals", "instructions"))):
        return "text"
    return "auto"


def _rule_map(plan: GenerationPlan | None) -> dict[tuple[str, str], ColumnGenerationRule]:
    if plan is None:
        return {}
    return {(rule.table.lower(), rule.column.lower()): rule for rule in plan.rules}


def _numeric_bounds(column: ColumnSchema) -> tuple[float | None, float | None]:
    minimum: float | None = None
    maximum: float | None = None
    for check in column.checks:
        for operator, raw in re.findall(rf"\b{re.escape(column.name)}\s*(>=|>|<=|<)\s*(-?\d+(?:\.\d+)?)", check, flags=re.I):
            value = float(raw)
            if operator in {">=", ">"}:
                minimum = max(minimum if minimum is not None else value, value)
            else:
                maximum = min(maximum if maximum is not None else value, value)
    return minimum, maximum


class SyntheticDataGenerator:
    def __init__(
        self,
        schema: ParsedSchema,
        rows_per_table: int,
        seed: int = 42,
        plan: GenerationPlan | None = None,
        null_probability: float = 0.08,
    ) -> None:
        self.schema = schema
        self.rows_per_table = rows_per_table
        self.seed = seed
        self.random = random.Random(seed)
        locale = plan.locale if plan else "en_US"
        try:
            self.faker = Faker(locale)
        except Exception:
            self.faker = Faker("en_US")
        self.faker.seed_instance(seed)
        self.rules = _rule_map(plan)
        self.null_probability = null_probability

    def generate(self) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        for table in self.schema.tables:
            rows: list[dict[str, Any]] = []
            for row_index in range(self.rows_per_table):
                row: dict[str, Any] = {}
                for column in table.columns:
                    if column.primary_key:
                        row[column.name] = row_index + 1
                    else:
                        row[column.name] = self._generate_value(table, column, row_index)
                rows.append(row)
            data[table.name] = pd.DataFrame(rows, columns=[column.name for column in table.columns])

        self._populate_foreign_keys(data)
        self._apply_cross_column_rules(data)
        return data

    def regenerate_column(
        self,
        table: TableSchema,
        column: ColumnSchema,
        row_count: int,
        semantic_type: str = "auto",
    ) -> list[Any]:
        temporary_rule = ColumnGenerationRule(
            table=table.name,
            column=column.name,
            semantic_type=semantic_type,
            null_probability=0,
        )
        previous = self.rules.get((table.name.lower(), column.name.lower()))
        self.rules[(table.name.lower(), column.name.lower())] = temporary_rule
        values = [self._generate_value(table, column, index) for index in range(row_count)]
        if previous is None:
            self.rules.pop((table.name.lower(), column.name.lower()), None)
        else:
            self.rules[(table.name.lower(), column.name.lower())] = previous
        return values

    def _generate_value(self, table: TableSchema, column: ColumnSchema, row_index: int) -> Any:
        rule = self.rules.get((table.name.lower(), column.name.lower()))
        if rule and rule.fixed_value is not None:
            return self._fit(column, rule.fixed_value)

        null_probability = rule.null_probability if rule and rule.null_probability is not None else self.null_probability
        if column.nullable and column.default is None and self.random.random() < null_probability:
            return None

        if rule and rule.choices:
            return self._fit(column, self.random.choice(rule.choices))
        if column.enum_values:
            return self.random.choice(column.enum_values)

        semantic = rule.semantic_type if rule and rule.semantic_type != "auto" else infer_semantic_type(table.name, column.name, column.base_type)
        value = self._semantic_value(semantic, table, column, row_index, rule)
        return self._fit(column, value)

    def _semantic_value(
        self,
        semantic: str,
        table: TableSchema,
        column: ColumnSchema,
        row_index: int,
        rule: ColumnGenerationRule | None,
    ) -> Any:
        name = column.name.lower()
        if semantic == "first_name":
            return self.faker.first_name()
        if semantic == "last_name":
            return self.faker.last_name()
        if semantic == "full_name":
            return self.faker.name()
        if semantic == "email":
            return f"user{row_index + 1}.{self.faker.user_name()}@{self.faker.free_email_domain()}"
        if semantic == "phone":
            return self.faker.numerify("+1-###-###-####")
        if semantic == "address":
            return self.faker.street_address()
        if semantic == "city":
            return self.faker.city()
        if semantic == "state":
            if hasattr(self.faker, "state_abbr"):
                return self.faker.state_abbr()
            if hasattr(self.faker, "state"):
                return self.faker.state()
            return self.random.choice(["CA", "NY", "TX", "MH", "KA", "DL"])
        if semantic == "postal_code":
            return self.faker.postcode()
        if semantic == "company":
            return self.faker.company()
        if semantic == "department":
            return self.random.choice(["Engineering", "Sales", "Finance", "Operations", "People", "Marketing"])
        if semantic == "job_title":
            return self.faker.job()
        if semantic == "industry":
            return self.random.choice(["Technology", "Healthcare", "Finance", "Retail", "Education", "Manufacturing"])
        if semantic == "website":
            return self.faker.url()
        if semantic == "isbn":
            return self.faker.isbn13(separator="")
        if semantic == "genre":
            return self.random.choice(["Fiction", "Mystery", "History", "Science", "Biography", "Fantasy"])
        if semantic == "book_title":
            return self.faker.sentence(nb_words=4).rstrip(".").title()
        if semantic == "restaurant_name":
            return f"{self.faker.last_name()} {self.random.choice(['Kitchen', 'Bistro', 'Cafe', 'Grill'])}"
        if semantic == "menu_item":
            return self.random.choice(["Roasted Vegetable Bowl", "Classic Burger", "Herb Pasta", "Mango Tart", "Iced Tea"])
        if semantic == "text":
            return self.faker.paragraph(nb_sentences=2)
        if semantic == "boolean":
            return self.random.choice([True, False])
        if semantic == "date":
            if "birth" in name:
                return self.faker.date_between(start_date="-85y", end_date="-18y")
            return self.faker.date_between(start_date="-8y", end_date="today")
        if semantic == "datetime":
            return self.faker.date_time_between(start_date="-3y", end_date="now").replace(microsecond=0)
        if semantic in {"integer", "number", "currency"}:
            check_min, check_max = _numeric_bounds(column)
            minimum = rule.minimum if rule and rule.minimum is not None else check_min
            maximum = rule.maximum if rule and rule.maximum is not None else check_max
            if minimum is None or maximum is None:
                if "rating" in name:
                    minimum, maximum = minimum or 1, maximum or 5
                elif "salary" in name:
                    minimum, maximum = minimum or 45_000, maximum or 180_000
                elif "budget" in name:
                    minimum, maximum = minimum or 50_000, maximum or 2_000_000
                elif any(token in name for token in ("quantity", "pages")):
                    minimum, maximum = minimum or 1, maximum or (800 if "pages" in name else 25)
                elif any(token in name for token in ("amount", "price", "subtotal", "coverage")):
                    minimum, maximum = minimum or 1, maximum or 5_000
                else:
                    minimum, maximum = minimum or 1, maximum or 1_000
            if column.base_type in INTEGER_TYPES or semantic == "integer":
                return self.random.randint(int(minimum), int(maximum))
            scale = column.scale if column.scale is not None else 2
            return Decimal(str(round(self.random.uniform(float(minimum), float(maximum)), scale)))

        if column.default is not None:
            if isinstance(column.default, str) and "CURRENT_TIMESTAMP" in column.default.upper():
                return datetime.now().replace(microsecond=0)
            return column.default
        if column.base_type in INTEGER_TYPES:
            return self.random.randint(1, 1_000)
        if column.base_type in DECIMAL_TYPES:
            return Decimal(str(round(self.random.uniform(1, 1_000), column.scale or 2)))
        if column.base_type == "DATE":
            return self.faker.date_between(start_date="-5y", end_date="today")
        if column.base_type in DATETIME_TYPES:
            return self.faker.date_time_between(start_date="-2y", end_date="now").replace(microsecond=0)
        if column.base_type in {"BOOL", "BOOLEAN"}:
            return self.random.choice([True, False])
        return f"{column.name.replace('_', ' ').title()} {row_index + 1}"

    @staticmethod
    def _fit(column: ColumnSchema, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and column.max_length:
            return value[: column.max_length]
        return value

    def _populate_foreign_keys(self, data: dict[str, pd.DataFrame]) -> None:
        table_lookup = {name.lower(): name for name in data}
        for table in self.schema.tables:
            frame = data[table.name]
            for foreign_key in table.foreign_keys:
                referenced_name = table_lookup[foreign_key.referenced_table.lower()]
                candidates = data[referenced_name][foreign_key.referenced_column].dropna().tolist()
                column = table.column(foreign_key.column)
                if not candidates:
                    continue
                values: list[Any] = []
                for _ in range(len(frame)):
                    if column.nullable and self.random.random() < self.null_probability:
                        values.append(None)
                    else:
                        values.append(self.random.choice(candidates))
                frame[foreign_key.column] = values

    def _apply_cross_column_rules(self, data: dict[str, pd.DataFrame]) -> None:
        lookup = {name.lower(): name for name in data}
        for table_name, frame in data.items():
            table_schema = self.schema.table(table_name)
            lower_columns = {column.lower(): column for column in frame.columns}
            for start_name, end_name in [
                ("start_date", "end_date"),
                ("join_date", "termination_date"),
                ("birth_date", "death_date"),
                ("loan_date", "due_date"),
                ("loan_date", "return_date"),
            ]:
                if start_name in lower_columns and end_name in lower_columns:
                    start_column, end_column = lower_columns[start_name], lower_columns[end_name]
                    for index in frame.index:
                        start = frame.at[index, start_column]
                        end = frame.at[index, end_column]
                        if start is None or pd.isna(start):
                            continue
                        start_timestamp = pd.Timestamp(start)
                        end_missing = end is None or pd.isna(end)
                        end_before_start = False if end_missing else pd.Timestamp(end) < start_timestamp
                        if end_missing or end_before_start:
                            days = 14 if end_name == "due_date" else self.random.randint(10, 900)
                            adjusted = start_timestamp + timedelta(days=days)
                            target_schema = table_schema.column(end_column)
                            frame.at[index, end_column] = (
                                adjusted.date()
                                if target_schema.base_type == "DATE"
                                else adjusted.to_pydatetime().replace(microsecond=0)
                            )

            if {"quantity", "available_quantity"}.issubset(lower_columns):
                quantity = lower_columns["quantity"]
                available = lower_columns["available_quantity"]
                frame[available] = [self.random.randint(0, max(0, int(value))) for value in frame[quantity]]

        if "order_items" in lookup and "menu" in lookup:
            items = data[lookup["order_items"]]
            menu = data[lookup["menu"]]
            if {"menu_id", "price"}.issubset(menu.columns) and {"menu_id", "quantity", "subtotal"}.issubset(items.columns):
                prices = dict(zip(menu["menu_id"], menu["price"], strict=False))
                items["subtotal"] = [
                    Decimal(str(prices.get(menu_id, 0))) * Decimal(str(quantity))
                    for menu_id, quantity in zip(items["menu_id"], items["quantity"], strict=False)
                ]
        if "orders" in lookup and "order_items" in lookup:
            orders = data[lookup["orders"]]
            items = data[lookup["order_items"]]
            if {"order_id", "subtotal"}.issubset(items.columns) and {"order_id", "total_amount"}.issubset(orders.columns):
                totals = items.groupby("order_id")["subtotal"].sum().to_dict()
                orders["total_amount"] = [Decimal(str(totals.get(order_id, 0))) for order_id in orders["order_id"]]


def validate_dataset(schema: ParsedSchema, data: dict[str, pd.DataFrame]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    table_lookup = {name.lower(): name for name in data}
    for table in schema.tables:
        if table.name not in data:
            issues.append(ValidationIssue(level="error", table=table.name, message="Generated table is missing."))
            continue
        frame = data[table.name]
        for column in table.columns:
            if column.name not in frame.columns:
                issues.append(ValidationIssue(level="error", table=table.name, column=column.name, message="Generated column is missing."))
                continue
            series = frame[column.name]
            if not column.nullable and series.isna().any():
                issues.append(ValidationIssue(level="error", table=table.name, column=column.name, message="NOT NULL constraint violated."))
            if (column.primary_key or column.unique) and series.dropna().duplicated().any():
                issues.append(ValidationIssue(level="error", table=table.name, column=column.name, message="Uniqueness constraint violated."))
            if column.enum_values:
                invalid = set(series.dropna().astype(str)) - set(column.enum_values)
                if invalid:
                    issues.append(ValidationIssue(level="error", table=table.name, column=column.name, message=f"Invalid ENUM values: {sorted(invalid)}"))
            if column.max_length and series.dropna().astype(str).str.len().gt(column.max_length).any():
                issues.append(ValidationIssue(level="error", table=table.name, column=column.name, message=f"VARCHAR({column.max_length}) length exceeded."))
            minimum, maximum = _numeric_bounds(column)
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if minimum is not None and not numeric.empty and (numeric < minimum).any():
                issues.append(ValidationIssue(level="error", table=table.name, column=column.name, message=f"CHECK minimum {minimum} violated."))
            if maximum is not None and not numeric.empty and (numeric > maximum).any():
                issues.append(ValidationIssue(level="error", table=table.name, column=column.name, message=f"CHECK maximum {maximum} violated."))

        for foreign_key in table.foreign_keys:
            referenced_name = table_lookup.get(foreign_key.referenced_table.lower())
            if referenced_name is None:
                continue
            allowed = set(data[referenced_name][foreign_key.referenced_column].dropna())
            actual = set(frame[foreign_key.column].dropna())
            missing = actual - allowed
            if missing:
                issues.append(ValidationIssue(level="error", table=table.name, column=foreign_key.column, message=f"Foreign key contains {len(missing)} missing referenced value(s)."))

        columns = {column.lower(): column for column in frame.columns}
        if {"quantity", "available_quantity"}.issubset(columns):
            if (frame[columns["available_quantity"]] > frame[columns["quantity"]]).any():
                issues.append(ValidationIssue(level="error", table=table.name, column=columns["available_quantity"], message="Available quantity exceeds total quantity."))
    return issues

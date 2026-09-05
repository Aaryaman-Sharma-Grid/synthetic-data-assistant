from pathlib import Path

import pandas as pd
import pytest

from src.editor import apply_edit_plan
from src.generator import SyntheticDataGenerator, validate_dataset
from src.models import DatasetArtifact, EditOperation, GenerationPlan, TableEditPlan
from src.schema_parser import parse_ddl
from src.storage import build_zip_archive


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "data" / "schemas"


@pytest.mark.parametrize(
    ("filename", "expected_tables"),
    [
        ("company_employee_schema.ddl", 7),
        ("restaurants_schema.ddl", 7),
        ("library_mgm_schema.ddl", 9),
    ],
)
def test_sample_ddl_parsing_and_generation(filename: str, expected_tables: int) -> None:
    ddl = (SCHEMAS / filename).read_text(encoding="utf-8")
    schema = parse_ddl(ddl)
    assert len(schema.tables) == expected_tables

    data = SyntheticDataGenerator(schema, rows_per_table=20, seed=123).generate()
    assert all(len(frame) == 20 for frame in data.values())
    assert validate_dataset(schema, data) == []


def test_restaurant_totals_and_inventory_relationships() -> None:
    ddl = (SCHEMAS / "restaurants_schema.ddl").read_text(encoding="utf-8")
    schema = parse_ddl(ddl)
    data = SyntheticDataGenerator(schema, rows_per_table=30, seed=8).generate()

    items = data["Order_Items"]
    menu_prices = dict(zip(data["Menu"]["menu_id"], data["Menu"]["price"], strict=False))
    for row in items.itertuples(index=False):
        assert row.subtotal == menu_prices[row.menu_id] * row.quantity


def test_generation_supports_gemini_selected_indian_locale() -> None:
    ddl = (SCHEMAS / "restaurants_schema.ddl").read_text(encoding="utf-8")
    schema = parse_ddl(ddl)
    data = SyntheticDataGenerator(
        schema,
        rows_per_table=8,
        seed=8,
        plan=GenerationPlan(locale="en_IN"),
    ).generate()
    assert validate_dataset(schema, data) == []


def test_edit_plan_protects_keys_and_applies_scalable_changes() -> None:
    ddl = (SCHEMAS / "company_employee_schema.ddl").read_text(encoding="utf-8")
    schema = parse_ddl(ddl)
    data = SyntheticDataGenerator(schema, rows_per_table=10, seed=9).generate()
    original_ids = data["Employees"]["employee_id"].copy()
    original_salary = pd.to_numeric(data["Employees"]["salary"])
    plan = TableEditPlan(
        summary="Raise salary and attempt protected edit",
        operations=[
            EditOperation(column="salary", action="multiply", value=1.1),
            EditOperation(column="employee_id", action="set", value=999),
        ],
    )

    updated, notes = apply_edit_plan(schema, data, "Employees", plan, seed=9)
    pd.testing.assert_series_equal(updated["Employees"]["employee_id"], original_ids)
    assert all(pd.to_numeric(updated["Employees"]["salary"]) > original_salary)
    assert any("Protected key" in note for note in notes)
    assert validate_dataset(schema, updated) == []


def test_zip_contains_schema_manifest_and_every_csv() -> None:
    ddl = (SCHEMAS / "company_employee_schema.ddl").read_text(encoding="utf-8")
    schema = parse_ddl(ddl)
    data = SyntheticDataGenerator(schema, rows_per_table=4, seed=1).generate()
    artifact = DatasetArtifact(
        dataset_id="dataset_test",
        display_name="Test",
        postgres_schema="dataset_test",
        created_at="2026-09-05T00:00:00",
        local_path="/tmp/dataset_test",
        rows_per_table={name: len(frame) for name, frame in data.items()},
    )
    archive = build_zip_archive(artifact, schema, data, ddl)
    assert archive.startswith(b"PK")
    assert len(archive) > 1000

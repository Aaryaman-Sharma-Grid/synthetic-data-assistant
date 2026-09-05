# Data Assistant — Synthetic Data Copilot

ByGrid University practice project for generating valid relational synthetic data and querying it conversationally.

## Implemented Phase 1

- Upload `.sql`, `.txt`, or `.ddl` schemas, or use any of the three supplied samples.
- Parse tables, columns, MySQL-style `ENUM`/`AUTO_INCREMENT`, defaults, checks, primary keys, unique keys, and foreign keys.
- Use Gemini 2.5 Flash structured output to convert user instructions into a compact semantic generation plan.
- Generate 3–1,000 rows per table with Faker while preserving relational integrity, including circular foreign keys.
- Validate nullability, uniqueness, enums, checks, lengths, and foreign-key references before saving.
- Apply natural-language edits to individual tables through Gemini-generated safe operations.
- Preview every table and download one CSV or a ZIP containing all CSV files, the DDL, and a manifest.
- Persist each successful dataset locally and into an isolated PostgreSQL schema for Phases 2–3.
- Emit Langfuse observations when Langfuse credentials are configured.

## Run locally

```bash
cd "/Users/aaryamansharma/Documents/ChatGPT/ai_training_prompt_engineering course"
source .venv/bin/activate
docker compose up -d
streamlit run app.py
```

Open <http://localhost:8501>. The project uses Application Default Credentials from `gcloud auth application-default login`.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Project structure

```text
app.py                    Streamlit UI and workflow
src/schema_parser.py      DDL parser
src/generator.py          Scalable deterministic generation + validation
src/llm.py                Gemini streaming and structured-output helpers
src/editor.py             Natural-language table changes
src/storage.py            PostgreSQL/local persistence and ZIP export
src/observability.py      Optional Langfuse tracing
data/schemas/             Course sample DDL files
tests/test_phase1.py      Phase 1 automated checks
```

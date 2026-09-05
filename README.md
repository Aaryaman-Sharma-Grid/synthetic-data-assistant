# Data Assistant — Synthetic Data Copilot

ByGrid University practice project for generating valid relational synthetic data and querying it conversationally.

## Features

- Upload `.sql`, `.txt`, or `.ddl` schemas, or use any of the three supplied samples.
- Parse tables, columns, MySQL-style `ENUM`/`AUTO_INCREMENT`, defaults, checks, primary keys, unique keys, and foreign keys.
- Use Gemini 2.5 Flash structured output to convert user instructions into a compact semantic generation plan.
- Generate 3–1,000 rows per table with Faker while preserving relational integrity, including circular foreign keys.
- Validate nullability, uniqueness, enums, checks, lengths, and foreign-key references before saving.
- Apply natural-language edits to individual tables through Gemini-generated safe operations.
- Preview every table and download one CSV or a ZIP containing all CSV files, the DDL, and a manifest.
- Persist each successful dataset locally and into an isolated PostgreSQL schema for later conversational querying.
- Trace complete generation and refinement workflows in Langfuse with nested spans, Gemini token usage, stable feature tags, explicit development environments, and credential masking.

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
python -m pytest -q
```

## Verify Langfuse

Add the Langfuse public key, secret key, and matching cloud-region URL to `.env`, then run:

```bash
source .venv/bin/activate
python -m scripts.verify_langfuse
```

The command authenticates without printing credentials, emits a diagnostic trace, flushes it, and prints the private Langfuse trace URL. Application traces use one root observation per generation or table-refinement request, with nested observations for Gemini, DDL parsing, row generation, validation, and persistence.

## Project structure

```text
app.py                    Streamlit UI and workflow
src/schema_parser.py      DDL parser
src/generator.py          Scalable deterministic generation + validation
src/llm.py                Gemini streaming and structured-output helpers
src/editor.py             Natural-language table changes
src/storage.py            PostgreSQL/local persistence and ZIP export
src/observability.py      Optional Langfuse tracing
scripts/verify_langfuse.py  Credential and ingestion check
data/schemas/             Course sample DDL files
tests/test_synthetic_data.py  Automated generation checks
```

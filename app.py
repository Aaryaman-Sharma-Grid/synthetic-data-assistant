"""Streamlit application for relational synthetic-data generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.db import check_database_connection
from src.editor import apply_edit_plan
from src.generator import SyntheticDataGenerator, validate_dataset
from src.llm import create_gemini_client, create_generation_plan, create_table_edit_plan
from src.models import GenerationPlan
from src.observability import langfuse_configured, observe_operation
from src.schema_parser import SchemaParseError, parse_ddl
from src.storage import build_zip_archive, list_saved_datasets, load_saved_dataset, save_dataset


ROOT = Path(__file__).resolve().parent
SAMPLE_SCHEMAS = {
    "Company & Employees (7 tables)": ROOT / "data" / "schemas" / "company_employee_schema.ddl",
    "Restaurant Operations (7 tables)": ROOT / "data" / "schemas" / "restaurants_schema.ddl",
    "Library Management (9 tables)": ROOT / "data" / "schemas" / "library_mgm_schema.ddl",
}


st.set_page_config(
    page_title="Data Assistant",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #172033; --muted: #667085; --surface: #ffffff; --canvas: #f4f6f8; }
    .stApp { background: var(--canvas); }
    [data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        color: var(--ink); font-size: 1.65rem; letter-spacing: -0.03em; margin-bottom: 1.4rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .72rem .8rem; border-radius: .65rem; margin-bottom: .22rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: #eef1f5; }
    .block-container { padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1420px; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface); border: 1px solid #e2e6eb; border-radius: .8rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, .03);
    }
    .stButton > button, .stDownloadButton > button {
        background: var(--ink); color: white; border: 1px solid var(--ink); border-radius: .55rem;
        font-weight: 650;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #25324a; color: white; border-color: #25324a;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
    .app-kicker { color: #2563eb; font-size: .78rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
    .app-subtitle { color: var(--muted); margin-top: -.55rem; margin-bottom: 1.25rem; }
    .status-dot { display:inline-block; width:.55rem; height:.55rem; border-radius:50%; margin-right:.4rem; background:#12b76a; }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()


def _init_state() -> None:
    defaults = {
        "generated_data": None,
        "parsed_schema": None,
        "source_ddl": "",
        "artifact": None,
        "generation_plan": None,
        "generation_notes": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _schema_description(schema) -> pd.DataFrame:
    records = []
    for table in schema.tables:
        foreign_keys = {fk.column.lower(): fk for fk in table.foreign_keys}
        for column in table.columns:
            constraints = []
            if column.primary_key:
                constraints.append("PK")
            if not column.nullable:
                constraints.append("NOT NULL")
            if column.unique:
                constraints.append("UNIQUE")
            if column.name.lower() in foreign_keys:
                foreign_key = foreign_keys[column.name.lower()]
                constraints.append(f"FK → {foreign_key.referenced_table}.{foreign_key.referenced_column}")
            records.append(
                {
                    "Table": table.name,
                    "Column": column.name,
                    "Type": column.data_type,
                    "Constraints": ", ".join(constraints) or "—",
                }
            )
    return pd.DataFrame(records)


def _generation_page() -> None:
    st.markdown('<div class="app-kicker">Phase 1</div>', unsafe_allow_html=True)
    st.title("Synthetic Data Generation")
    st.markdown('<p class="app-subtitle">Upload a DDL schema, describe the data you need, and generate validated relational datasets.</p>', unsafe_allow_html=True)

    with st.container(border=True):
        prompt = st.text_area(
            "Prompt",
            placeholder="Example: Generate realistic US technology-company data with salaries between 60,000 and 180,000.",
            height=86,
        )
        source_col, upload_col = st.columns([1, 1.15], vertical_alignment="bottom")
        with source_col:
            sample_name = st.selectbox("Sample DDL schema", list(SAMPLE_SCHEMAS), index=0)
        with upload_col:
            uploaded = st.file_uploader("Upload your DDL schema", type=["sql", "txt", "ddl"])
        st.caption("Upload overrides the selected sample. Supported formats: SQL, TXT, DDL.")
        st.divider()
        st.markdown("#### Advanced parameters")
        temperature_col, rows_col, tokens_col, seed_col = st.columns([1.5, 1, 1, 1])
        with temperature_col:
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        with rows_col:
            rows_per_table = st.number_input("Rows per table", min_value=3, max_value=1000, value=25, step=1)
        with tokens_col:
            max_tokens = st.number_input("Max output tokens", min_value=512, max_value=16384, value=8192, step=256)
        with seed_col:
            seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)
        use_gemini = st.toggle("Use Gemini to interpret the prompt and improve semantic realism", value=True)

        if st.button("✦  Generate data", type="primary", width="content"):
            if uploaded is not None:
                try:
                    ddl = uploaded.getvalue().decode("utf-8-sig")
                except UnicodeDecodeError:
                    st.error("The uploaded file must be UTF-8 text.")
                    return
                display_name = Path(uploaded.name).stem.replace("_", " ").title()
            else:
                ddl = SAMPLE_SCHEMAS[sample_name].read_text(encoding="utf-8")
                display_name = sample_name.split(" (")[0]

            try:
                with st.status("Building your dataset…", expanded=True) as status:
                    st.write("Parsing tables, columns, and constraints")
                    schema = parse_ddl(ddl)
                    st.write(f"Found {len(schema.tables)} tables and {sum(len(table.columns) for table in schema.tables)} columns")

                    plan = GenerationPlan()
                    notes: list[str] = []
                    if use_gemini:
                        st.write("Asking Gemini for a structured generation plan")
                        try:
                            client = create_gemini_client(settings)
                            with observe_operation(
                                settings,
                                "create-generation-plan",
                                as_type="generation",
                                input_data={"instructions": prompt, "tables": [table.name for table in schema.tables]},
                                model=settings.gemini_model,
                            ) as observation:
                                plan = create_generation_plan(
                                    client,
                                    settings,
                                    schema,
                                    prompt,
                                    float(temperature),
                                    int(max_tokens),
                                )
                                if observation is not None:
                                    observation.update(output=plan.model_dump(mode="json"))
                            client.close()
                        except Exception as exc:
                            notes.append(f"Gemini planning was unavailable; deterministic inference was used ({exc}).")

                    st.write(f"Generating {int(rows_per_table):,} rows for each table")
                    generator = SyntheticDataGenerator(
                        schema,
                        rows_per_table=int(rows_per_table),
                        seed=int(seed),
                        plan=plan,
                    )
                    data = generator.generate()
                    st.write("Validating nullability, uniqueness, checks, and foreign keys")
                    issues = validate_dataset(schema, data)
                    errors = [issue for issue in issues if issue.level == "error"]
                    if errors:
                        raise RuntimeError("; ".join(issue.message for issue in errors[:5]))

                    st.write("Saving CSV files and loading tables into PostgreSQL")
                    artifact = save_dataset(settings, schema, data, ddl, prompt, display_name)
                    notes.extend(schema.warnings)
                    notes.extend(artifact.warnings)
                    st.session_state.generated_data = data
                    st.session_state.parsed_schema = schema
                    st.session_state.source_ddl = ddl
                    st.session_state.artifact = artifact
                    st.session_state.generation_plan = plan
                    st.session_state.generation_notes = notes
                    status.update(label="Dataset generated and validated", state="complete", expanded=False)
            except SchemaParseError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

    data = st.session_state.generated_data
    schema = st.session_state.parsed_schema
    artifact = st.session_state.artifact
    if data is None or schema is None or artifact is None:
        with st.container(border=True):
            st.subheader("Data Preview")
            st.info("Choose a sample or upload a schema, then click **Generate data** to preview each table.")
        return

    for note in st.session_state.generation_notes:
        st.warning(note)

    with st.container(border=True):
        header_col, table_col = st.columns([3, 1], vertical_alignment="center")
        with header_col:
            st.subheader("Data Preview")
            st.caption(f"{artifact.display_name} · {len(schema.tables)} tables · PostgreSQL schema `{artifact.postgres_schema}`")
        with table_col:
            selected_table = st.selectbox("Preview table", list(data), label_visibility="collapsed")

        frame = data[selected_table]
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Rows", f"{len(frame):,}")
        metric_b.metric("Columns", len(frame.columns))
        metric_c.metric("Validation", "Passed")
        st.dataframe(frame, width="stretch", hide_index=True, height=min(480, 86 + len(frame.head(12)) * 35))

        with st.expander("View parsed schema and constraints"):
            st.dataframe(_schema_description(schema), width="stretch", hide_index=True)

        st.markdown("#### Refine this table")
        with st.form("table_feedback_form", clear_on_submit=True):
            feedback_col, submit_col = st.columns([5, 1], vertical_alignment="bottom")
            with feedback_col:
                feedback = st.text_input(
                    "Quick edit instructions",
                    placeholder="Example: Increase salary by 10% and regenerate all job titles.",
                    label_visibility="collapsed",
                )
            with submit_col:
                edit_submitted = st.form_submit_button("✦  Apply changes", width="stretch")

        if edit_submitted:
            if not feedback.strip():
                st.warning("Enter an edit instruction first.")
            else:
                try:
                    table_schema = schema.table(selected_table)
                    client = create_gemini_client(settings)
                    columns = [
                        {
                            "name": column.name,
                            "type": column.data_type,
                            "primary_key": column.primary_key,
                            "foreign_key": any(fk.column.lower() == column.name.lower() for fk in table_schema.foreign_keys),
                        }
                        for column in table_schema.columns
                    ]
                    with st.spinner("Gemini is translating your feedback into safe edits…"):
                        edit_plan = create_table_edit_plan(
                            client,
                            settings,
                            selected_table,
                            columns,
                            frame.head(5).to_dict(orient="records"),
                            feedback,
                        )
                        client.close()
                        updated, edit_notes = apply_edit_plan(schema, data, selected_table, edit_plan, int(seed))
                        issues = validate_dataset(schema, updated)
                        errors = [issue for issue in issues if issue.level == "error"]
                        if errors:
                            raise RuntimeError("; ".join(issue.message for issue in errors[:5]))
                        new_artifact = save_dataset(
                            settings,
                            schema,
                            updated,
                            st.session_state.source_ddl,
                            f"{prompt}\nRevision: {feedback}".strip(),
                            f"{artifact.display_name} – revised",
                        )
                        st.session_state.generated_data = updated
                        st.session_state.artifact = new_artifact
                        st.session_state.generation_notes = edit_notes + new_artifact.warnings
                    st.success(edit_plan.summary)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not apply the requested changes: {exc}")

        st.divider()
        download_zip_col, download_csv_col, saved_col = st.columns([1.25, 1.25, 2])
        with download_zip_col:
            archive = build_zip_archive(artifact, schema, data, st.session_state.source_ddl)
            st.download_button(
                "↓  Download all tables (.zip)",
                data=archive,
                file_name=f"{artifact.dataset_id}.zip",
                mime="application/zip",
                width="stretch",
            )
        with download_csv_col:
            st.download_button(
                "↓  Download selected CSV",
                data=frame.to_csv(index=False),
                file_name=f"{selected_table.lower()}.csv",
                mime="text/csv",
                width="stretch",
            )
        with saved_col:
            storage_label = "PostgreSQL + local archive" if artifact.database_saved else "Local archive"
            st.caption(f"Saved as `{artifact.dataset_id}` · {storage_label}")


def _talk_to_data_page() -> None:
    st.markdown('<div class="app-kicker">Phases 2 & 3</div>', unsafe_allow_html=True)
    st.title("Talk to your data")
    st.markdown('<p class="app-subtitle">Your generated datasets are available here and ready for natural-language querying.</p>', unsafe_allow_html=True)
    artifacts = list_saved_datasets()
    if not artifacts:
        st.info("Generate a dataset first. It will appear here automatically.")
        return
    with st.container(border=True):
        selected_id = st.selectbox(
            "Dataset",
            [artifact.dataset_id for artifact in artifacts],
            format_func=lambda value: next(
                f"{item.display_name} · {item.created_at:%d %b %Y, %H:%M}"
                for item in artifacts
                if item.dataset_id == value
            ),
        )
        selected_artifact = next(item for item in artifacts if item.dataset_id == selected_id)
        saved_schema, saved_data, _ = load_saved_dataset(selected_artifact)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Tables", len(saved_schema.tables))
        col_b.metric("Total rows", f"{sum(len(frame) for frame in saved_data.values()):,}")
        col_c.metric("PostgreSQL", "Ready" if selected_artifact.database_saved else "Local only")
        preview_table = st.selectbox("Inspect table", list(saved_data))
        st.dataframe(saved_data[preview_table].head(20), width="stretch", hide_index=True)
        st.chat_input("Natural-language querying will be enabled in Phase 2", disabled=True)


_init_state()

with st.sidebar:
    st.title("Data Assistant")
    navigation = st.radio(
        "Navigation",
        ["▣  Data Generation", "☁  Talk to your data"],
        label_visibility="collapsed",
    )
    st.divider()
    database_ok = check_database_connection(settings)
    status_color = "#12b76a" if database_ok else "#f04438"
    status_text = "connected" if database_ok else "offline"
    st.markdown(
        f'<div style="font-size:.82rem;color:#667085"><span class="status-dot" style="background:{status_color}"></span>'
        f'PostgreSQL {status_text}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Gemini · {settings.gemini_model}")
    st.caption(f"Langfuse · {'configured' if langfuse_configured(settings) else 'awaiting keys'}")

if "Data Generation" in navigation:
    _generation_page()
else:
    _talk_to_data_page()

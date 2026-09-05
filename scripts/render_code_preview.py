"""Render a presentation-friendly HTML excerpt of the generation workflow."""

from __future__ import annotations

import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app.py"
OUTPUT = ROOT / "docs" / "screenshots" / "code.html"

lines = SOURCE.read_text(encoding="utf-8").splitlines()
start = next(index for index, line in enumerate(lines) if 'if st.button("✦  Generate data"' in line)
end = next(index for index, line in enumerate(lines[start:], start=start) if line.strip() == "data = st.session_state.generated_data")
excerpt = lines[start:end]

rendered_lines = "\n".join(
    f'<span class="line"><span class="number">{line_number:>3}</span><span class="code">{html.escape(line)}</span></span>'
    for line_number, line in enumerate(excerpt, start=start + 1)
)

document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Assistant — generation workflow code</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #0b1020; color: #d8dee9; font-family: Inter, ui-sans-serif, system-ui; }}
.shell {{ height: 100vh; display: grid; grid-template-columns: 270px 1fr; }}
.sidebar {{ background: #11182a; border-right: 1px solid #263149; padding: 30px 22px; }}
.brand {{ font-weight: 750; color: white; font-size: 22px; margin-bottom: 32px; }}
.label {{ color: #7180a0; font-size: 11px; font-weight: 750; text-transform: uppercase; letter-spacing: .12em; margin: 24px 8px 10px; }}
.file {{ padding: 10px 12px; border-radius: 8px; color: #aeb9cf; font-family: ui-monospace, monospace; font-size: 13px; }}
.file.active {{ background: #1d2942; color: #f5f7fb; }}
.main {{ overflow: hidden; padding: 24px 30px; }}
.header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }}
.title {{ font-size: 20px; font-weight: 700; color: white; }}
.meta {{ color: #7180a0; font-size: 12px; }}
.editor {{ height: calc(100vh - 84px); overflow: hidden; background: #11182a; border: 1px solid #263149; border-radius: 12px; padding: 15px 0; box-shadow: 0 18px 50px rgba(0,0,0,.25); }}
pre {{ margin: 0; font: 12.6px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
.line {{ display: block; min-height: 18px; white-space: pre; }}
.line:hover {{ background: #18223a; }}
.number {{ display: inline-block; width: 58px; padding-right: 16px; color: #50607e; text-align: right; user-select: none; }}
.code {{ color: #d8dee9; }}
.pill {{ background: #103b32; color: #6ee7b7; padding: 6px 10px; border-radius: 99px; font-size: 11px; font-weight: 700; }}
</style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand">Data Assistant</div>
    <div class="label">Application</div>
    <div class="file active">app.py</div>
    <div class="file">schema_parser.py</div>
    <div class="file">generator.py</div>
    <div class="file">llm.py</div>
    <div class="file">editor.py</div>
    <div class="file">storage.py</div>
    <div class="label">Tests</div>
    <div class="file">test_synthetic_data.py</div>
  </aside>
  <main class="main">
    <div class="header">
      <div><div class="title">Synthetic data generation workflow</div><div class="meta">app.py · Streamlit + Gemini + PostgreSQL</div></div>
      <div class="pill">7 tests passing</div>
    </div>
    <div class="editor"><pre>{rendered_lines}</pre></div>
  </main>
</div>
</body>
</html>"""

OUTPUT.write_text(document, encoding="utf-8")
print(OUTPUT)

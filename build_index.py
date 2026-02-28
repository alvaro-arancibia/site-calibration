import json

INDEX_HTML = "docs/index.html"

# Map of HTML key → source file path
EMBEDDED_FILES = {
    "app.py": "src/sitecal/ui/app.py",
    "sitecal/core/calibration_engine.py": "src/sitecal/core/calibration_engine.py",
    "sitecal/core/math_engine.py": "src/sitecal/core/math_engine.py",
    "sitecal/core/projections.py": "src/sitecal/core/projections.py",
    "sitecal/domain/schemas.py": "src/sitecal/domain/schemas.py",
    "sitecal/infrastructure/reports.py": "src/sitecal/infrastructure/reports.py",
}

with open(INDEX_HTML, "r", encoding="utf-8") as f:
    html = f.read()

decoder = json.JSONDecoder()

for key, src_path in EMBEDDED_FILES.items():
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    # ensure_ascii=True converts non-ASCII chars to \uXXXX — safe for JS string literals
    escaped = json.dumps(source, ensure_ascii=True)

    search_key = f'"{key}":'
    idx = html.find(search_key)
    if idx == -1:
        print(f"WARNING: key {search_key!r} not found in HTML, skipping")
        continue

    val_start = idx + len(search_key)
    while html[val_start] in ' \t\n':
        val_start += 1

    if html[val_start] != '"':
        print(f"WARNING: value for {key!r} is not a string, skipping")
        continue

    # raw_decode returns (value, end_index) where end_index is absolute in html
    _, val_end = decoder.raw_decode(html, val_start)

    html = html[:val_start] + escaped + html[val_end:]
    print(f"Updated: {key}")

with open(INDEX_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("docs/index.html updated successfully")

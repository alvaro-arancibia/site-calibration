import json

APP_PY = "src/sitecal/ui/app.py"
INDEX_HTML = "docs/index.html"

with open(APP_PY, "r", encoding="utf-8") as f:
    app_source = f.read()

# ensure_ascii=True converts non-ASCII chars to \uXXXX escapes that JS handles correctly
escaped = json.dumps(app_source, ensure_ascii=True)

with open(INDEX_HTML, "r", encoding="utf-8") as f:
    html = f.read()

key = '"app.py":'
idx = html.find(key)
val_start = idx + len(key)
while html[val_start] in ' \t\n':
    val_start += 1

decoder = json.JSONDecoder()
old_value, val_len = decoder.raw_decode(html, val_start)
val_end = val_start + val_len

updated = html[:val_start] + escaped + html[val_end:]

with open(INDEX_HTML, "w", encoding="utf-8") as f:
    f.write(updated)

print("docs/index.html updated successfully")

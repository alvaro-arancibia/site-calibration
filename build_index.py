import json
import re

APP_PY = "src/sitecal/ui/app.py"
INDEX_HTML = "index.html"

with open(APP_PY, "r", encoding="utf-8") as f:
    app_source = f.read()

escaped = json.dumps(app_source)

with open(INDEX_HTML, "r", encoding="utf-8") as f:
    html = f.read()

replacement = f'"app.py": {escaped}'
updated = re.sub(r'"app\.py":\s*".*?"', lambda m: replacement, html, flags=re.DOTALL)

with open(INDEX_HTML, "w", encoding="utf-8") as f:
    f.write(updated)

print("index.html updated successfully")

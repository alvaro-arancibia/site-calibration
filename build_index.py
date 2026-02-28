"""Rebuild docs/index.html — regenerate the files: {...} block from source."""
import json
from pathlib import Path

INDEX_HTML = Path("docs/index.html")
HEAD_LINES = 19  # lines 1-19 are the head (0-indexed: 0..18)

MANIFEST = {
    "sitecal/__init__.py": None,
    "sitecal/core/__init__.py": None,
    "sitecal/domain/__init__.py": None,
    "sitecal/infrastructure/__init__.py": None,
    "app.py": Path("src/sitecal/ui/app.py"),
    "sitecal/core/calibration_engine.py": Path("src/sitecal/core/calibration_engine.py"),
    "sitecal/core/math_engine.py": Path("src/sitecal/core/math_engine.py"),
    "sitecal/core/projections.py": Path("src/sitecal/core/projections.py"),
    "sitecal/domain/schemas.py": Path("src/sitecal/domain/schemas.py"),
    "sitecal/infrastructure/reports.py": Path("src/sitecal/infrastructure/reports.py"),
}

lines = INDEX_HTML.read_text(encoding="utf-8").splitlines(keepends=True)
head = lines[:HEAD_LINES]
tail = lines[HEAD_LINES + 1:]  # skip line 20 (index 19)

pairs = []
for vfs_path, src_path in MANIFEST.items():
    if src_path is None:
        pairs.append(f'"{vfs_path}": ""')
    else:
        source = src_path.read_text(encoding="utf-8")
        pairs.append(f'"{vfs_path}": {json.dumps(source, ensure_ascii=True)}')

files_line = "        files: { " + ", ".join(pairs) + " }\n"

INDEX_HTML.write_text("".join(head) + files_line + "".join(tail), encoding="utf-8")
print("docs/index.html rebuilt successfully")

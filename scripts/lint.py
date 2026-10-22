"""T0 lint: mechanical checks of the JAX discipline rules (11 §2.6) that can be enforced statically.

- no float literals / float dtypes in mechanics modules (vgcjax/core, vgcjax/battle, vgcjax/effects);
- no hand edits to generated data (vgcjax/data has a GENERATED header);
- every handler-decorated function carries a `ps=` citation.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MECH_DIRS = [ROOT / "vgcjax" / d for d in ("core", "battle", "effects")]
FLOAT_NAMES = {"float16", "float32", "float64", "bfloat16"}
ALLOW_FLOAT = re.compile(r"#\s*lint:\s*allow-float")


def check_file(path: pathlib.Path) -> list[str]:
    errs: list[str] = []
    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src, filename=str(path))
    for node in ast.walk(tree):
        line = lines[node.lineno - 1] if hasattr(node, "lineno") else ""
        if ALLOW_FLOAT.search(line):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            errs.append(f"{path}:{node.lineno}: float literal in mechanics code")
        if isinstance(node, ast.Attribute) and node.attr in FLOAT_NAMES:
            errs.append(f"{path}:{node.lineno}: float dtype in mechanics code")
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and getattr(d.func, "id", None) == "handler":
                    if not any(k.arg == "ps" for k in d.keywords):
                        errs.append(f"{path}:{node.lineno}: handler {node.name} lacks ps= citation")
    return errs


def main() -> int:
    errs: list[str] = []
    for d in MECH_DIRS:
        for p in sorted(d.rglob("*.py")):
            errs += check_file(p)
    for p in sorted((ROOT / "vgcjax" / "data").glob("*.py")):
        if p.name != "__init__.py" and "GENERATED" not in p.read_text()[:400]:
            errs.append(f"{p}: generated module lacks GENERATED header")
    for e in errs:
        print(e)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())

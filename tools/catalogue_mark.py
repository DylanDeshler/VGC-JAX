"""Set docs/catalogue.csv `status` for implemented entities (11 §3: the catalogue moves with the code).

Every entity of data/export/stages/impl.json that the catalogue lists (moves, abilities, items, conditions) plus the given
SYSTEM / ENGINE_DELTA ids becomes `impl` (never downgrades `done`). Usage:
    python -I tools/catalogue_mark.py [--system id ...] [--tests tests/...]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", nargs="*", default=[])
    ap.add_argument("--tests", default="")
    a = ap.parse_args()
    impl = json.loads((ROOT / "data/export/stages/impl.json").read_text())
    want = {("MOVE", x) for x in impl["moves"]} | {("ABILITY", x) for x in impl["abilities"]} | {
        ("ITEM", x) for x in impl["items"]} | {("CONDITION", x) for x in impl["implemented_conditions"]} | {
        (k, x) for x in a.system for k in ("SYSTEM", "ENGINE_DELTA")}
    path = ROOT / "docs/catalogue.csv"
    rows = list(csv.DictReader(path.open()))
    n = 0
    for r in rows:
        if (r["kind"], r["id"]) in want and r["status"] != "done":
            if r["status"] != "impl":
                n += 1
            r["status"] = "impl"
            r["ps_commit"] = "d601443"
            if a.tests and a.tests not in (r["scenario_tests"] or ""):
                r["scenario_tests"] = ";".join(x for x in (r["scenario_tests"], a.tests) if x)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"catalogue: {n} entities newly impl")


if __name__ == "__main__":
    main()

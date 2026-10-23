"""Gate evaluation (07 §12, ADR-027/029).

    uv run python -m harness.gates G-core reports/conformance/<run>/summary.json [--seed S]

Prints the sha256 of every protected contract file. A gate PASS counts only if those hashes match
docs/decisions/contract-hashes.md, which only the human edits (ADR-029); otherwise the verdict is
"PASS (UNVERIFIED CONTRACT)". Writes gate_report.md next to the summary.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROTECTED = sorted([*map(str, (ROOT / "harness" / "contract").glob("*.json")), str(ROOT / "harness" / "gates.yaml"),
                    str(ROOT / "docs" / "coverage" / "waivers.json"), str(ROOT / "docs" / "design" / "contract" / "drawkey.md")])
HASHES = ROOT / "docs" / "decisions" / "contract-hashes.md"


def contract_hashes() -> dict[str, str]:
    return {str(pathlib.Path(p).relative_to(ROOT)): hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
            for p in PROTECTED}


def recorded_hashes() -> dict[str, str] | None:
    if not HASHES.exists():
        return None
    out = {}
    for line in HASHES.read_text().splitlines():
        m = re.search(r"`?([\w./-]+)`?\s*[|:]\s*`?([0-9a-f]{64})`?", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def evaluate(gate: str, summary: dict) -> dict:
    cfg = yaml.safe_load((ROOT / "harness" / "gates.yaml").read_text())[gate]
    th = cfg["thresholds"]
    checks = []
    want_n = sum(c.get("battles", 0) for c in cfg.get("corpus", []))
    checks.append(("battles", summary["battles"], f">= {want_n}", summary["battles"] >= want_n))
    for k, v in th.items():
        got = summary.get(k)
        if k == "error_rate":
            checks.append((k, got, f"<= {v}", got is not None and got <= v))
        elif k == "DPR":
            dpr = 1 - summary.get("draw_failures", 0) / max(summary["battles"], 1)
            checks.append((k, round(dpr, 6), f">= {v}", dpr >= v))
        elif got is not None or k in ("CBR", "MPR"):
            checks.append((k, got, f">= {v}", got is not None and got >= v))
    waivers = json.loads((ROOT / "docs" / "coverage" / "waivers.json").read_text())["waivers"]
    mine = {w["target"]: w for w in waivers if w.get("gate") == gate}
    cov = summary.get("coverage", {})
    need = cfg.get("coverage", {}).get("min_hits", 0)
    for kind in cfg.get("coverage", {}).get("kinds", []):
        n = cov.get(kind, 0)
        w = mine.get(kind)
        if w and "covered_by" in w:            # crafted-scenario corpus counts instead (ADR-102)
            ok, note = n >= need or w.get("hits", 0) >= need, f" (crafted: {w.get('hits', 0)} hits, {w['covered_by']})"
        elif w:
            ok, note = True, " (waived)"
        else:
            ok, note = n >= need, ""
        checks.append((f"coverage:{kind}", n, f">= {need}{note}", ok))
    hashes = contract_hashes()
    rec = recorded_hashes()
    verified = rec is not None and all(rec.get(k) == v for k, v in hashes.items())
    passed = all(c[3] for c in checks)
    verdict = ("PASS" if verified else "PASS (UNVERIFIED CONTRACT)") if passed else "FAIL"
    return dict(gate=gate, verdict=verdict, checks=checks, hashes=hashes, contract_verified=verified, waivers=waivers)


def report_md(res: dict, summary: dict, seed: str | None) -> str:
    L = [f"# Gate report: {res['gate']} — {res['verdict']}", ""]
    if seed:
        L.append(f"- run seed: `{seed}`")
    L += [f"- trace dir: `{summary.get('trace_dir')}`", f"- replay seconds: {summary.get('seconds')}", "",
          "| check | value | requirement | ok |", "|---|---|---|---|"]
    for name, v, req, ok in res["checks"]:
        L.append(f"| {name} | {v} | {req} | {'yes' if ok else '**no**'} |")
    L += ["", "## Top buckets", ""]
    for k, v in (summary.get("buckets") or {}).items():
        L.append(f"- {k}: {v}")
    if not summary.get("buckets"):
        L.append("- none")
    L += ["", "## Waivers", "", json.dumps(res["waivers"]), "", "## Contract hashes (sha256)",
          f"verified against docs/decisions/contract-hashes.md: {res['contract_verified']}", ""]
    for k, v in res["hashes"].items():
        L.append(f"- `{k}`: `{v}`")
    return "\n".join(L) + "\n"


def main():
    gate, path = sys.argv[1], pathlib.Path(sys.argv[2])
    seed = sys.argv[sys.argv.index("--seed") + 1] if "--seed" in sys.argv else None
    summary = json.loads(path.read_text())
    res = evaluate(gate, summary)
    md = report_md(res, summary, seed)
    (path.parent / "gate_report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

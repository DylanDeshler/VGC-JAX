"""Attribute bench/hlo_kernels.py's kernel-like ops to source locations (ADR-106 profiling).

Uses the same count rules (while bodies x trip count, conditionals = max branch) and charges each counted
instruction to its HLO metadata source_file:source_line, then aggregates by file, by file:line, and by the
enclosing Python function.

    PYTHONPATH=. uv run python bench/hlo_attrib.py [--stage core] [--no-events] [--top 40] [--dump out.hlo.txt]
"""
import argparse
import ast
import collections
import pathlib
import re

from bench import hlo_kernels as HK

ROOT = pathlib.Path(__file__).resolve().parents[1]


_T = {}


def tables(text):
    """The optimised module's FileNames / FunctionNames / FileLocations / StackFrames tables."""
    sec, out = None, {"FileNames": {}, "FunctionNames": {}, "FileLocations": {}, "StackFrames": {}}
    for line in text.splitlines():
        if line in out:
            sec = line
            continue
        if sec is None:
            continue
        m = re.match(r"(\d+) (.*)$", line)
        if not m:
            if line and not line[0].isdigit():
                sec = None
            continue
        k, v = int(m.group(1)), m.group(2)
        if sec in ("FileNames", "FunctionNames"):
            out[sec][k] = v.strip('"')
        else:
            out[sec][k] = {a: int(b) for a, b in re.findall(r"(\w+)=(\d+)", v)}
    return out


def chain(fid):
    """[(file, function, line)] innermost first for a stack frame id."""
    t = _T["t"]
    out, seen = [], set()
    while fid and fid not in seen:
        seen.add(fid)
        fr = t["StackFrames"].get(fid)
        if not fr:
            break
        loc = t["FileLocations"].get(fr["file_location_id"], {})
        f = t["FileNames"].get(loc.get("file_name_id"), "?")
        if str(ROOT) in f:
            f = f.split(str(ROOT) + "/")[-1]
        out.append((f, t["FunctionNames"].get(loc.get("function_name_id"), "?"), loc.get("line", 0)))
        fid = fr.get("parent_frame_id", 0) - 1     # parent ids are offset by one from the table's numbering
    return out


def where(line):
    m = re.search(r"stack_frame_id=(\d+)", line)
    return int(m.group(1)) if m else 0


def attribute(comps, name, acc, mult=1):
    for op, line in comps[name]:
        if op == "while":
            body = re.search(r"body=%?([\w.\-]+)", line).group(1)
            cond = re.search(r"condition=%?([\w.\-]+)", line).group(1)
            n = re.search(r'"known_trip_count":\{"n":"(\d+)"', line)
            n = int(n.group(1)) if n else 1
            attribute(comps, body, acc, mult * n)
            attribute(comps, cond, acc, mult * n)
        elif op == "conditional":
            brs = re.findall(r"%?([\w.\-]+)", re.search(r"branch_computations=\{([^}]*)\}", line).group(1))
            acc[where(line)] += mult
            best, best_acc = None, None
            for b in brs:
                a = collections.Counter()
                attribute(comps, b, a, mult)
                if best is None or sum(a.values()) > sum(best_acc.values()):
                    best, best_acc = b, a
            acc.update(best_acc)
        elif op == "call":
            attribute(comps, re.search(r"to_apply=%?([\w.\-]+)", line).group(1), acc, mult)
        elif op not in HK.FREE:
            acc[where(line)] += mult


_FUNCS = {}


def func_of(path, line):
    """Innermost enclosing def of a source line (module scope: '<module>')."""
    if path not in _FUNCS:
        spans = []
        try:
            tree = ast.parse((ROOT / path).read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    spans.append((node.lineno, node.end_lineno, node.name))
        except Exception:
            pass
        _FUNCS[path] = spans
    best = None
    for a, b, n in _FUNCS[path]:
        if a <= line <= b and (best is None or a > best[0]):
            best = (a, n)
    return best[1] if best else "<module>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="core")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bank", default="data/teams/bank_core.npz")
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--micro-steps", type=int, default=None)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--hlo", default=None, help="read a dumped HLO text instead of compiling")
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    if a.hlo:
        text = pathlib.Path(a.hlo).read_text()
    else:
        import jax
        from vgcjax.config import EngineConfig
        from vgcjax.core.dex import load_dex
        from vgcjax.env import api, teambank as TB
        from vgcjax.env.autoreset import AutoResetEnv
        cfg = EngineConfig(stage=a.stage, max_turns=40, events=not a.no_events)
        if a.micro_steps:
            cfg = cfg.replace(micro_steps=a.micro_steps)
        dex = load_dex(a.stage)
        env = AutoResetEnv(cfg, dex, TB.load(a.bank))
        B = a.batch
        es, _ = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), B))

        def one(es, k):
            ks = jax.random.split(k, B * 2).reshape(B, 2, -1)
            masks = jax.vmap(lambda s: api.legal_masks(s, cfg, dex))(es.state)
            acts = jax.vmap(api.random_actions)(ks[:, 0], masks, es.state.meta.phase)
            es, _, stats = jax.vmap(env.step)(es, acts, ks[:, 1])
            return es, stats.done

        text = jax.jit(one).lower(es, jax.random.PRNGKey(1)).compile().as_text()
        if a.dump:
            pathlib.Path(a.dump).write_text(text)
    comps, entry = HK.parse(text)
    _T["t"] = tables(text)
    acc = collections.Counter()
    attribute(comps, entry, acc)
    total = sum(acc.values())
    by_file, by_func, by_line, incl = (collections.Counter() for _ in range(4))
    for fid, n in acc.items():
        ch = [x for x in chain(fid) if x[0].startswith("vgcjax/")]
        if not ch:
            by_file["<other>"] += n
            continue
        f, fn, ln = ch[0]                                   # innermost engine frame
        by_file[f] += n
        by_func[f"{f}:{fn}"] += n
        by_line[f"{f}:{ln} ({fn})"] += n
        for key in {f"{x[0]}:{x[1]}" for x in ch}:        # inclusive: every engine function on the stack
            incl[key] += n
    print(f"total {total}")
    for title, c in (("by file (innermost engine frame)", by_file), ("by function (self)", by_func),
                     ("by function (inclusive)", incl), ("by line", by_line)):
        print(f"\n== {title}")
        for k, n in c.most_common(a.top):
            print(f"{n:8d} {100 * n / total:5.1f}%  {k}")


if __name__ == "__main__":
    main()

"""Print markdown tables from results/*.json (used to build research/jax-patterns.md)."""
import json
import os
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, spec):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return format(v, spec)
    return str(v)


def table(rows, cols):
    out = ["| " + " | ".join(c[0] for c in cols) + " |", "|" + "|".join("---:" if c[2] else "---" for c in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(f(r.get(c[1], ""), c[2]) if c[2] else str(r.get(c[1], "")) for c in cols) + " |")
    return "\n".join(out)


def pivot(rows, rowkey, colkey, val, spec):
    rk = list(dict.fromkeys(r[rowkey] for r in rows))
    ck = list(dict.fromkeys(r[colkey] for r in rows))
    d = {(r[rowkey], r[colkey]): r.get(val, "") for r in rows}
    out = [f"| {rowkey} \\ {colkey} | " + " | ".join(map(str, ck)) + " |", "|---|" + "|".join("---:" for _ in ck) + "|"]
    for a in rk:
        out.append(f"| {a} | " + " | ".join(f(d.get((a, b), ""), spec) for b in ck) + " |")
    return "\n".join(out)


def main():
    b1 = load("b1_dispatch_AB_B65536.json")
    b1b = load("b1_dispatch_B_B65536.json")
    b1c = load("b1_dispatch_C_B16384.json")
    if b1:
        rows = [r for r in b1["1A"] if r["N"] > 0]
        print("### 1A runtime ns/elem (B=65536)\n" + pivot(rows, "variant", "N", "ns_per_elem", ".1f"))
        print("\n### 1A compile s\n" + pivot(rows, "variant", "N", "compile_s", ".2f"))
        print("\n### 1A temp MB\n" + pivot(rows, "variant", "N", "temp_MB", ".0f"))
        base = [r for r in b1["1A"] if r["N"] == 0]
        if base:
            print(f"\nbaseline copy: {base[0]['ns_per_elem']:.2f} ns/elem")
    if b1b:
        rows = b1b["1B"]
        print("\n### 1B runtime ns/elem (B=65536)\n" + pivot(rows, "variant", "N", "ns_per_elem", ".1f"))
        print("\n### 1B compile s\n" + pivot(rows, "variant", "N", "compile_s", ".2f"))
    if b1c:
        rows = b1c["1C"]
        for r in rows:
            r["cfg"] = r["loops"] + (" +barrier" if r["barrier"] else "")
        print("\n### 1C compile s\n" + pivot(rows, "cfg", "N", "compile_s", ".2f"))
        print("\n### 1C runtime ns/elem (B=4096)\n" + pivot(rows, "cfg", "N", "ns_per_elem", ".0f"))
    b2 = load("b2_control_AB.json")
    if b2:
        print("\n### 2A\n" + table(b2["2A"], [("outer", "outer", None), ("inner", "inner", None), ("index", "index", None),
                                              ("compile s", "compile_s", ".2f"), ("ns/elem", "ns_per_elem", ".0f")]))
        print("\n### 2B\n" + table(b2["2B"], [("queue", "queue_loop", None), ("targets", "targets", None), ("index", "index", None),
                                              ("lower s", "lower_s", ".2f"), ("compile s", "compile_s", ".2f"),
                                              ("ns/step", "ns_per_elem", ".0f")]))
    b3 = load("b3_state_ABCDEF.json")
    if b3:
        rows = b3["rows"]
        for t in dict.fromkeys(r["test"] for r in rows):
            sub = [r for r in rows if r["test"] == t]
            keys = [k for k in sub[0] if k not in ("test", "temp_bytes", "arg_bytes", "out_bytes", "run_s")]
            print(f"\n### {t}\n" + table(sub, [(k, k, ".3g" if isinstance(sub[0][k], float) else None) for k in keys]))
    b4 = load("b4_rng.json")
    if b4:
        rows = [r for r in b4["rows"] if "ns_per_elem" in r or "error" in r]
        print("\n### 4\n" + table(rows, [("impl", "impl", None), ("method", "method", None), ("compile s", "compile_s", ".2f"),
                                         ("ns/battle-turn", "ns_per_elem", ".0f"), ("ns/draw", "ns_per_draw", ".2f"),
                                         ("error", "error", None)]))
        print([r for r in b4["rows"] if "vmap_exact" in r])
    b5 = load("b5_rollout.json")
    if b5:
        print("\n### 5\n" + table(b5, [("B", "B", None), ("tag", "tag", None), ("compile s", "compile_s", ".1f"),
                                       ("lower s", "lower_s", ".1f"), ("steps/s", "steps_per_s", ",.0f"),
                                       ("us/step call", "us_per_step_call", ".0f"), ("temp MB", "temp_MB", ".0f"),
                                       ("peak RSS MB", "peak_rss_MB", ".0f"), ("GFLOP/rollout", "flops", ".3g"),
                                       ("GB acc", "bytes_accessed", ".3g"), ("fusions", "n_fusion_instrs", None),
                                       ("err", "error", None)]))
    b6 = load("b6_ergonomics.json")
    if b6:
        print("\n### 6")
        for r in b6["rows"]:
            print({k: v for k, v in r.items() if k not in ("temp_bytes", "arg_bytes", "out_bytes")})


if __name__ == "__main__":
    sys.exit(main())

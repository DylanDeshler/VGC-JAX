"""Run the b5 rollout grid (one subprocess per config) and collect results/b5_rollout.json.

Also covers the section-6 compile-time questions: persistent compilation cache cold vs warm, and
dev-mode XLA flags.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CACHE = "/tmp/jaxcache_b5"

CONFIGS = [
    dict(args=["--B", "1024"]),
    dict(args=["--B", "8192"]),
    dict(args=["--B", "32768"]),
    dict(args=["--B", "8192", "--reset", "init"], tag="reset=init every step"),
    dict(args=["--B", "8192", "--loop", "unroll"], tag="queue unrolled"),
    dict(args=["--B", "8192", "--index", "onehot"], tag="one-hot indexing"),
    dict(args=["--B", "8192", "--targets", "unroll"], tag="targets unrolled"),
    dict(args=["--B", "1024", "--store-obs"], tag="store obs (bf16)"),
    dict(args=["--B", "8192", "--store-obs"], tag="store obs (bf16)"),
    dict(args=["--B", "32768", "--devices", "10"], tag="10 CPU devices (sharded batch)"),
    dict(args=["--B", "32768", "--devices", "14"], tag="14 CPU devices (sharded batch)"),
    dict(args=["--B", "8192", "--cache-dir", CACHE], tag="persistent cache: cold", pre="clear_cache"),
    dict(args=["--B", "8192", "--cache-dir", CACHE], tag="persistent cache: warm"),
    dict(args=["--B", "8192"], env={"XLA_FLAGS": "--xla_backend_optimization_level=0"}, tag="XLA opt level 0"),
    dict(args=["--B", "8192"], env={"XLA_FLAGS": "--xla_llvm_disable_expensive_passes=true"}, tag="llvm cheap passes"),
    dict(args=["--B", "8192"], env={"JAX_DISABLE_MOST_OPTIMIZATIONS": "1"}, tag="jax_disable_most_optimizations"),
]


EXTRAS = [
    dict(args=["--B", "32768"], tag="1 device (repeat)"),
    dict(args=["--B", "32768", "--devices", "8"], tag="8 CPU devices (sharded batch)"),
    dict(args=["--B", "32000", "--devices", "10"], tag="10 CPU devices (sharded batch)"),
    dict(args=["--B", "32760", "--devices", "14"], tag="14 CPU devices (sharded batch)"),
    dict(args=["--B", "65536"], tag="B=64k"),
    dict(args=["--B", "8192", "--extra-hooks", "4"], tag="+4 hooks x 64 handlers"),
    dict(args=["--B", "8192", "--extra-hooks", "4", "--barrier", "1"], tag="+4 hooks x 64, barrier"),
    dict(args=["--B", "8192", "--extra-hooks", "8"], tag="+8 hooks x 64 handlers"),
    dict(args=["--B", "8192", "--extra-hooks", "8", "--barrier", "1"], tag="+8 hooks x 64, barrier"),
    dict(args=["--B", "8192", "--extra-hooks", "16", "--barrier", "1"], tag="+16 hooks x 64, barrier"),
    dict(args=["--B", "8192", "--dump-hlo", os.path.join(HERE, "results", "rollout_B8192.hlo.txt")], tag="hlo dump"),
]


def main():
    global CONFIGS
    if sys.argv[1:2] == ["extras"]:
        CONFIGS = EXTRAS
        sys.argv = sys.argv[:1] + ["extras"]
    only = [a for a in sys.argv[1:] if a != "extras"]
    out = []
    for i, c in enumerate(CONFIGS):
        if only and str(i) not in only:
            continue
        if c.get("pre") == "clear_cache":
            shutil.rmtree(CACHE, ignore_errors=True)
        env = dict(os.environ)
        env.update(c.get("env", {}))
        cmd = [PY, os.path.join(HERE, "b5_rollout.py"), *c["args"], "--tag", c.get("tag", "")]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1500, env=env)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            r = json.loads(line[0][7:]) if line else {"error": p.stderr[-800:], "cmd": " ".join(c["args"])}
        except subprocess.TimeoutExpired:
            r = {"error": "timeout", "cmd": " ".join(c["args"])}
        r["env_override"] = c.get("env", {})
        out.append(r)
        print(json.dumps(r), flush=True)
    name = "b5_rollout" + ("_extras" if CONFIGS is EXTRAS else "") + ("_" + "_".join(only) if only else "")
    with open(os.path.join(HERE, "results", f"{name}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

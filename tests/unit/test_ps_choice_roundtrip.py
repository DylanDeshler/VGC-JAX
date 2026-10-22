"""M5.1: to_ps_choice / parse_choice round trip over every legal joint action at every decision of a corpus, and PS
accepts every engine-encoded legal action string (acceptance via the legal sets recorded by the oracle)."""
import pathlib
import subprocess

import numpy as np
import pytest

from harness.corpus import iter_battles
from vgcjax.interop import ps_choice

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def battles(tmp_path_factory):
    out = tmp_path_factory.mktemp("rt")
    subprocess.run(["node", str(ROOT / "oracle/bin/oracle.js"), "gen", "--n", "40", "--seed", "rt-choice", "--workers", "8",
                    "--teams", "randdoubles", "--policy", "random-weighted", "--legal", "on", "--out", str(out)],
                   check=True, capture_output=True)
    return [b for f in sorted(out.glob("*.zst")) for b in iter_battles(f)]


def test_roundtrip_all_legal(battles):
    n = 0
    for b in battles:
        for d in (r for r in b if r["type"] == "decision"):
            if d["phase"] == "preview":
                for j in range(360):
                    assert ps_choice.preview_action(ps_choice.to_ps_choice(j, {}, None, "preview")) == j
                continue
            for s in range(2):
                req = d["requests"][s]
                lg = d["legal"][s]
                if not req or req.get("wait") or lg["kind"] == "wait":
                    continue
                order = [x for x in d["projection"]["sides"][s]["pokemon"]]
                ps_order = np.array([p["b"] for p in order])
                for a0, a1 in lg["pairs"]:
                    j = a0 * 29 + a1
                    sstr = ps_choice.to_ps_choice(j, req, ps_order, d["phase"])
                    assert ps_choice.parse_choice(sstr, req, ps_order, d["phase"]) == j, (sstr, j)
                    n += 1
    assert n > 10000

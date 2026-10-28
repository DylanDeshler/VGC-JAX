"""Engine-vs-engine differential test (ADR-107 §4.2): a refactor must not change any outcome, so two engine versions
must reach bit-identical battle states at every decision, for the same teams, seeds and actions.

`record` plays B random battles with uniformly random legal actions and stores, after every decision, a 32-bit hash
of each battle-state field (meta, team, sides, mons, field; the internal queue / scratch / exec / event buffers are
excluded, so the execution layout may change). `compare` reports the first divergence per field.

Random rolls must not depend on how many step calls a turn takes (the refactor changes that): each battle uses one
fixed key, and before every call of decision d the harness sets meta.step_counter = d, so a roll depends only on its
semantic draw key. Both versions are driven identically.

    # reference: a pinned commit in a worktree, the current harness
    git worktree add /tmp/ref <commit>
    PYTHONPATH=/tmp/ref uv run python scripts/diff/engine_diff.py record --out /tmp/ref.npz
    PYTHONPATH=. uv run python scripts/diff/engine_diff.py record --out /tmp/new.npz
    uv run python scripts/diff/engine_diff.py compare /tmp/ref.npz /tmp/new.npz

Teams: --teams <trace dir> builds a bank from the team sets of oracle corpora (e.g. traces/regress/impl_random_f, so
the battles use implemented content); default data/teams/bank_core.npz.
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys
import time

import numpy as np

GROUPS = ("meta", "team", "sides", "mons", "field")
SKIP = {"meta.step_counter"}            # driven by the harness
LAYOUT_FIELDS = {"mons.vol_state", "mons.vol_dur", "mons.vol_fx"}   # compared through mons.vol_view


def _bank_from_traces(d):
    import json
    from vgcjax.env import teambank as TB
    from harness import corpus
    rows = []
    dirs = []
    for part in str(d).split(","):                     # comma-separated dirs; a trailing /* globs subdirs
        dirs += [pathlib.Path(x) for x in sorted(glob.glob(part))] if "*" in part else [pathlib.Path(part)]
    files = [f for dd in dirs for f in corpus.shard_files(dd)]
    for f in files:
        for recs in corpus.iter_battles(f):
            for s in recs[0]["team_sets"]:
                rows.append({"sets": s})
    tmp = pathlib.Path("/tmp/engine_diff_bank.jsonl")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return TB.from_jsonl(tmp)


def record(a):
    import jax
    import jax.numpy as jnp
    from vgcjax.battle import state as S
    from vgcjax.battle import step as STEP
    from vgcjax.config import EngineConfig
    from vgcjax.core.dex import load_dex
    from vgcjax.env import api, teambank as TB
    from vgcjax.battle import init as INIT

    cfg = EngineConfig(stage=a.stage, max_turns=a.max_turns, events=False)
    if a.micro_steps:
        cfg = cfg.replace(micro_steps=a.micro_steps)
    dex = load_dex(a.stage)
    bank = _bank_from_traces(a.teams) if a.teams else TB.load(a.bank)
    bank = jax.tree_util.tree_map(jnp.asarray, bank)
    B, D = a.battles, a.decisions
    root = jax.random.PRNGKey(a.seed)
    bkeys = jax.random.split(root, B)

    def new_state(k):
        team, _ = TB.sample_team_pair(jax.random.fold_in(k, 1), bank)
        return INIT.init(team)

    st = jax.jit(jax.vmap(new_state))(bkeys)

    def vol_view(m):
        """Canonical volatile state [.., NV, 6] (duration, fields 1..3, 0, presence) from either storage layout, so the
        comparison survives the ADR-107 compact layout (vol_dur + vol_fx) vs the old vol_state[NV, 6]."""
        if hasattr(m, "vol_state"):
            return m.vol_state.at[..., 4].set(0)
        fidx = jnp.asarray(S.VOL_FIDX)
        fx = jnp.where((fidx >= 0)[:, None], jnp.take(m.vol_fx, jnp.clip(fidx, 0, None), axis=-2), 0)   # [.., NV, 4]
        z = jnp.zeros(m.vol_dur.shape, m.vol_dur.dtype)
        return jnp.stack([m.vol_dur, fx[..., 0], fx[..., 1], fx[..., 2], z, fx[..., 3]], axis=-1)

    def leaves(st):
        out = {}
        for g in GROUPS:
            sub = getattr(st, g)
            for k, v in jax.tree_util.tree_flatten_with_path(sub)[0]:
                name = g + "." + ".".join(str(getattr(p, "name", getattr(p, "key", p))) for p in k)
                if name not in SKIP and name not in LAYOUT_FIELDS:
                    out[name] = v
        out["mons.vol_view"] = vol_view(st.mons)
        return out

    def digest(x):
        """Per-battle 32-bit polynomial hash of one field (any dtype, batch leading)."""
        flat = x.reshape(x.shape[0], -1)
        if jnp.issubdtype(flat.dtype, jnp.floating):
            u = jax.lax.bitcast_convert_type(flat.astype(jnp.float32), jnp.uint32)
        elif flat.dtype == jnp.uint32:
            u = flat
        else:
            u = jax.lax.bitcast_convert_type(flat.astype(jnp.int32), jnp.uint32)
        n = u.shape[1]
        pw = jnp.cumprod(jnp.full((n,), 0x9E3779B1, jnp.uint32))
        return jnp.sum(u * pw[None, :], axis=1, dtype=jnp.uint32) ^ jnp.uint32(n)

    names = sorted(leaves(st).keys())

    @jax.jit
    def advance(st, d):
        masks = jax.vmap(lambda s: api.legal_masks(s, cfg, dex))(st)
        akeys = jax.vmap(lambda k: jax.random.fold_in(jax.random.fold_in(k, 2), d))(bkeys)
        acts = jax.vmap(api.random_actions)(akeys, masks, st.meta.phase)

        def call(st, acts):
            st = st.replace(meta=st.meta.replace(step_counter=jnp.full_like(st.meta.step_counter, d)))
            st2 = jax.vmap(lambda s, ac, k: STEP.step(s, ac, k, cfg, dex)[0])(st, acts, bkeys)
            done = st.meta.phase == S.DONE
            return jax.tree_util.tree_map(lambda a_, b_: jnp.where(
                done.reshape(done.shape + (1,) * (a_.ndim - 1)), b_, a_), st2, st)

        st = call(st, acts)

        def cont(i, st):
            c = st.meta.phase == S.CONT
            st2 = call(st, acts)
            return jax.tree_util.tree_map(lambda a_, b_: jnp.where(
                c.reshape(c.shape + (1,) * (a_.ndim - 1)), a_, b_), st2, st)
        st = jax.lax.fori_loop(0, a.max_cont, cont, st)
        lv = leaves(st)
        return st, jnp.stack([digest(lv[n]) for n in names]), st.meta.phase == S.CONT

    out = np.zeros((len(names), B, D), np.uint32)
    stuck = 0
    t0 = time.time()
    for d in range(D):
        st, h, c = advance(st, jnp.int32(d))
        out[:, :, d] = np.asarray(h)
        stuck += int(np.asarray(c).sum())
        if d == 0:
            print(f"compiled + first decision in {time.time() - t0:.0f}s", file=sys.stderr)
    done = int(np.asarray(st.meta.phase == S.DONE).sum())
    np.savez_compressed(a.out, names=np.asarray(names), digests=out, seed=a.seed, stage=a.stage,
                        battles=B, decisions=D, teams=str(a.teams or a.bank))
    print(f"recorded {B} battles x {D} decisions ({done} finished, {stuck} decision slots still in CONT after "
          f"{a.max_cont} continuation calls) in {time.time() - t0:.0f}s -> {a.out}")


def compare(a):
    x, y = np.load(a.a), np.load(a.b)
    norm = lambda k, v: pathlib.Path(str(v)).name if k == "teams" else str(v)  # noqa: E731
    for k in ("seed", "stage", "battles", "decisions", "teams"):
        if norm(k, x[k]) != norm(k, y[k]):
            print(f"run parameters differ: {k} {x[k]} vs {y[k]}")
            return 2
    nx, ny = list(x["names"]), list(y["names"])
    common = sorted(set(nx) & set(ny))
    only = sorted(set(nx) ^ set(ny))
    if only:
        print(f"fields present in one version only (not compared): {only}")
    dx = x["digests"][[nx.index(n) for n in common]]
    dy = y["digests"][[ny.index(n) for n in common]]
    diff = dx != dy                                    # [F, B, D]
    if not diff.any():
        print(f"IDENTICAL: {len(common)} fields x {dx.shape[1]} battles x {dx.shape[2]} decisions")
        return 0
    bad_b = np.where(diff.any(axis=(0, 2)))[0]
    print(f"DIVERGED in {len(bad_b)} / {dx.shape[1]} battles")
    for b in bad_b[:10]:
        d0 = int(np.where(diff[:, b].any(axis=0))[0][0])
        fields = [common[i] for i in np.where(diff[:, b, d0])[0]]
        print(f"  battle {b}: first at decision {d0}: {fields[:8]}{' ...' if len(fields) > 8 else ''}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--battles", type=int, default=256)
    r.add_argument("--decisions", type=int, default=120)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--stage", default="impl")
    r.add_argument("--bank", default="data/teams/bank_core.npz")
    r.add_argument("--teams", default=None, help="trace dir(s), comma-separated, globs allowed (e.g. "
                   "'traces/regress/*'): build the bank from their corpora's team sets")
    r.add_argument("--max-turns", type=int, default=60)
    r.add_argument("--max-cont", type=int, default=8)
    r.add_argument("--micro-steps", type=int, default=None, help="engine micro-steps per call (must not matter)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "record":
        record(a)
    else:
        sys.exit(compare(a))


if __name__ == "__main__":
    main()

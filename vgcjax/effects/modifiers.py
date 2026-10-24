"""MUL-hook modifier events (contract §4 SPEED_TIE_HANDLER_MUL; 04 §6 MUL fold), vectorised over target slots.

PS runEvent (sim/battle.ts:765-944) on a MUL-class hook (BasePower, ModifyAtk/Def/SpA/SpD/Spe, ModifyDamage,
ModifyAccuracy, ModifyCritRatio, ModifySTAB): the handlers of the event's target / source / allies / foes / sides /
field (findEventHandlers :1049-1109) are sorted by comparePriority (order, priority, holder speed, subOrder); an
exact tie group of size >= 2 among them is a SPEED_TIE_HANDLER_MUL record (sub = the call's mulhook_ord, ord =
group index, members in holder rank then effect-id order) and is folded in the drawn order. Each handler either
chains a factor into event.modifier (chainModify) or returns a new relay value; the final value is
modify(relay, modifier).

Handlers here are pure (no messages, no state writes) and registered per (hook, effect, prefix) with `mod`. The
candidate list is static: every registered handler times the holders its prefix allows. Mods are only evaluated
where the caller's runEvent happens (the caller keeps counting mulhook_ord with common.mul_tick).
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp

from vgcjax.battle.util import mget
from vgcjax.core import fixedpoint as fp
from vgcjax.core import rng as R
from vgcjax.data import ids
from vgcjax.effects import conditions as CO

I32 = jnp.int32
G_TIES = 2
KMAX = 8            # handlers present at once per call (more: the extra ones are dropped)


@dataclasses.dataclass(frozen=True)
class Mod:
    hook: str
    effect: int          # EffectId
    prefix: int
    fn: Callable         # fn(ctx, holder [T]) -> (applies bool[T], chain m4096 i32[T] | None, replace | None)
    ps: str


MODS: dict[str, list[Mod]] = {}


def mod(hook: str, effect: int, ps: str, prefix: int = ids.PFX_ON):
    """Register a pure MUL-hook handler. fn(ctx, holder) returns (applies, m4096, replace): m4096 = the chainModify
    factor (4096-scale, or a callable (t, relay, modifier) -> factor for handlers reading this.event.modifier) or
    None; replace = scalar relay -> new relay (a handler returning a number) or None."""
    def deco(fn):
        MODS.setdefault(hook, []).append(Mod(hook, int(effect), prefix, fn, ps))
        return fn
    return deco


@dataclasses.dataclass
class Ctx:
    """One runEvent(hook, target, source, move) per target slot: tgt / src [T] refs (-1 none)."""
    st: object
    dex: object
    tgt: jax.Array
    src: jax.Array
    move: jax.Array
    mtype: jax.Array
    extra: dict


def _kind(effect):
    if ids.ABILITY_BASE <= effect < ids.ABILITY_BASE + len(ids.ABILITIES) - 1:
        return "ability"
    if ids.ITEM_BASE <= effect < ids.ITEM_BASE + len(ids.ITEMS) - 1:
        return "item"
    k = CO.kind_of(ids.CONDITIONS[effect - ids.COND_BASE + 1])
    return {CO.STATUS: "status", CO.VOLATILE: "volatile", CO.SIDE: "side"}.get(k, "field")


def _held(st, ref, effect, kind):
    m = st.mons
    r = jnp.clip(ref, 0, 7)
    if kind == "ability":
        return mget(m.ability, r) == effect - ids.ABILITY_BASE + 1
    if kind == "item":
        return mget(m.item, r) == effect - ids.ITEM_BASE + 1
    c = effect - ids.COND_BASE + 1
    if kind == "status":
        return mget(m.status, r) == c
    return CO.has_volatile(st, r, c)


def _field_has(st, effect):
    c = effect - ids.COND_BASE + 1
    k = CO.kind_of(ids.CONDITIONS[c])
    if k == CO.PSEUDO:
        return CO.has_pseudo(st, c)
    return (st.field.weather if k == CO.WEATHER else st.field.terrain).astype(I32) == c


def _rows(ctx: Ctx, mods):
    """Candidate rows (mod index, holder [T], present [T]) per findEventHandlers for each registered mod."""
    st = ctx.st
    m = st.mons
    T = ctx.tgt.shape[0]
    tgt = ctx.tgt
    t_ok = tgt >= 0
    t_side = jnp.clip(tgt, 0, 7) // 4
    mon_level = t_ok & (mget(m.is_active, jnp.clip(tgt, 0, 7)) | ((ctx.src >= 0) & mget(m.is_active, jnp.clip(
        ctx.src, 0, 7))))
    act = st.sides.active.astype(I32)                                    # [2, 2]
    rows = []
    for i, md in enumerate(mods):
        kind = _kind(md.effect)
        if kind in ("ability", "item", "status", "volatile"):
            if md.prefix == ids.PFX_ON:
                hs = [tgt]
            elif md.prefix == ids.PFX_SOURCE:
                hs = [ctx.src]
            else:
                hs = []
                for slot in (0, 1):
                    for rel in ((0,) if md.prefix == ids.PFX_ALLY else (1,) if md.prefix == ids.PFX_FOE else (0, 1)):
                        s = jnp.where(rel == 0, t_side, 1 - t_side) if md.prefix != ids.PFX_ANY else jnp.full(
                            T, rel, I32)
                        b = act[s, slot]
                        hs.append(jnp.where(b >= 0, s * 4 + b, -1))
            for n, h in enumerate(hs):
                ok = (h >= 0) & mon_level & _held(st, h, md.effect, kind)
                if md.prefix not in (ids.PFX_ON, ids.PFX_SOURCE):
                    ok = ok & (mget(m.hp, jnp.clip(h, 0, 7)) > 0)
                rows.append((i, h, ok, (kind, md.prefix, n) if kind in ("ability", "item", "status") else None))
        elif kind == "side":
            c = md.effect - ids.COND_BASE + 1
            sides = [t_side] if md.prefix == ids.PFX_ON else [1 - t_side] if md.prefix == ids.PFX_FOE else [
                jnp.zeros(T, I32), jnp.ones(T, I32)]
            for s in sides:
                rows.append((i, 8 + s, mon_level & jax.vmap(lambda ss: CO.has_side(st, ss, c))(s), None))
        else:
            rows.append((i, jnp.full(T, 10, I32), jnp.broadcast_to(_field_has(st, md.effect), (T,)) & t_ok,
                         ("weather" if CO.kind_of(ids.CONDITIONS[md.effect - ids.COND_BASE + 1]) == CO.WEATHER else
                          "terrain" if CO.kind_of(ids.CONDITIONS[md.effect - ids.COND_BASE + 1]) == CO.TERRAIN else None)
                         ))
    return rows


def run(hook: str, ctx: Ctx, rng, relay, ords, on, final=True):
    """Fold runEvent(hook) for each target slot: relay i32[T] (non-negative), ords i32[T] = the calls' mulhook_ord
    (tie-record sub), on bool[T]. final=False: the relay is not an integer in PS (ModifySTAB's 1.5) so the final
    modify is skipped. Returns (value i32[T], modifier i32[T], rng)."""
    from vgcjax.battle import common as C
    from vgcjax.effects import hooks as H
    from vgcjax.effects.fieldevent import _STR_RANK, holder_rank
    mods = MODS.get(hook, [])
    relay = jnp.asarray(relay, I32)
    if not mods:
        return relay, jnp.full(relay.shape, 4096, I32), rng
    st, dex = ctx.st, ctx.dex
    hook_id = getattr(ids, f"HOOK_{hook.upper()}")
    rows = _rows(ctx, mods)
    K = len(rows)
    T = ctx.tgt.shape[0]
    holder = jnp.stack([r[1] for r in rows], axis=1)                    # [T, K]
    present = jnp.stack([r[2] for r in rows], axis=1) & jnp.asarray(on)[:, None]
    eff_l = [mods[r[0]].effect for r in rows]
    pfx_l = [mods[r[0]].prefix for r in rows]
    effs, pfxs = jnp.array(eff_l, I32), jnp.array(pfx_l, I32)
    results = [mods[r[0]].fn(ctx, r[1]) for r in rows]
    applies = jnp.stack([a for a, _, _ in results], axis=1) & present   # [T, K]
    mv = jnp.stack([jnp.broadcast_to(jnp.asarray(m if (m is not None and not callable(m)) else 4096, I32), (T,))
                    for _, m, _ in results], axis=1)                      # [T, K]
    is_chain = jnp.array([m is not None and not callable(m) for _, m, _ in results])
    dyn = [k for k, (_, m, _) in enumerate(results) if callable(m)]
    reps = [k for k, (_, _, rp) in enumerate(results) if rp is not None]
    # static: can two rows ever tie? equal (order, priority, subOrder) and not mutually exclusive (one ability /
    # item / status per holder role, one weather, one terrain)
    from vgcjax.core.dex import load_tables_np
    ho = load_tables_np()["handler_order"]
    skey = [tuple(int(x) for x in ho[e, hook_id, p]) for e, p in zip(eff_l, pfx_l)]
    tie_possible = any(skey[a] == skey[b] and not (rows[a][3] is not None and rows[a][3] == rows[b][3])
                       for a in range(K) for b in range(a + 1, K))
    KP = min(K, KMAX)
    o = dex.handler_order[effs, hook_id, pfxs].astype(I32)            # [K, 3]
    spd = jnp.where(holder < 8, mget(st.mons.speed, jnp.clip(holder, 0, 7)).astype(I32), 0)
    canon = holder_rank(st, holder) * 4096 + jnp.asarray(_STR_RANK)[effs][None, :]
    out, mods_t = [], []
    for t in range(T):
        sel = jnp.nonzero(present[t], size=KP, fill_value=0)[0]          # present rows (collection order)
        cv = jnp.arange(KP) < jnp.sum(present[t].astype(I32))
        keys = (o[sel, 0], o[sel, 1], spd[t][sel], o[sel, 2], jnp.zeros(KP, I32))
        perm, group = H.sort_order(keys, cv)
        if tie_possible and KP >= 2:
            perm, rng = _draw_ties(st, rng, perm, group, cv, canon[t][sel], ords[t], on[t])
        val = relay[t]
        modifier = jnp.asarray(4096, I32)
        for i in range(KP):
            j = sel[perm[i]]
            live = cv[perm[i]] & applies[t, j]
            modifier = jnp.where(live & is_chain[j], fp.chain(modifier, mv[t, j]), modifier)
            for r in dyn:                                # depends on the fold so far (Technician)
                modifier = jnp.where(live & (j == r), fp.chain(modifier, results[r][1](t, val, modifier)), modifier)
            for r in reps:
                val = jnp.where(live & (j == r), results[r][2](val), val)
        out.append(jnp.where((val >= 0) & final, fp.modify_raw(val, modifier), val))
        mods_t.append(modifier)
    return jnp.stack(out), jnp.stack(mods_t), rng


def _draw_ties(st, rng, perm, group, valid, canon, sub, on):
    """SPEED_TIE_HANDLER_MUL records for the tie groups (size >= 2) of one call; members in canonical order, the
    drawn permutation replaces the group's positions. Returns (perm, rng)."""
    from vgcjax.battle import common as C
    K = perm.shape[0]
    pos = jnp.arange(K)
    live = valid[perm]
    gsize = jnp.zeros(K, I32).at[group].add(live.astype(I32))
    first_live = jnp.full(K, K, I32).at[group].min(jnp.where(live, pos, K))
    is_start = live & (pos == first_live[group]) & (gsize[group] >= 2)
    starts = jnp.nonzero(is_start, size=G_TIES, fill_value=-1)[0]
    new_perm = perm
    M = max(K, 2)
    for g in range(G_TIES):
        start = starts[g]
        act = start >= 0
        mem = act & (group == group[jnp.maximum(start, 0)]) & live
        ck = jnp.where(mem, canon[perm], 1 << 30)
        canon_pos = jnp.argsort(ck, stable=True)
        nm = jnp.sum(mem.astype(I32))
        key = C.tkey(st, ids.DK_SPEED_TIE_HANDLER_MUL, sub, g)
        torder, rng = R.tie_order(rng, key, jnp.arange(M), jnp.arange(M) < nm, active=on & act)
        exec_pos = canon_pos[jnp.clip(torder, 0, K - 1)]
        slots = jnp.nonzero(mem, size=K, fill_value=0)[0]
        for j in range(K):
            ok = act & (j < nm)
            new_perm = new_perm.at[slots[j]].set(jnp.where(ok, perm[exec_pos[j]], new_perm[slots[j]]))
    return new_perm, rng

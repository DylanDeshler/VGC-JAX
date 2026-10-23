"""Hook framework (04 §6, ADR-010; M3.1).

A hook = one PS event name (HookId). For a runEvent-style dispatch the engine builds a fixed-size candidate list
(effect EffectId, prefix, holder), filtered by `dex.handler_mask`, keyed by PS resolvePriority's
(order, priority, speed, subOrder, effectOrder) from `dex.handler_order`, sorted with PS comparePriority and folded
(MUL / GATE / REPL). Bespoke handlers register per (hook, effect) and run through one `lax.switch` per hook.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp

from vgcjax.data import ids

I32 = jnp.int32
BIG = 1 << 30
K_H = 8


# ------------------------------------------------------------------ keys and ordering (PS: sim/battle.ts:957-1023, 409-416)
MAGIC_BOUNCE = ids.ABILITY_BASE + getattr(ids, "ABILITY_MAGICBOUNCE", -10**6) - 1


def handler_key(dex, effect, hook, prefix, holder_speed, effect_order, uses_effect_order=False, holder_stored_spe=None):
    """PS resolvePriority fields for one handler: (order, priority, speed, subOrder, effectOrder), int32.

    order: `on<cb>Order` or 2^32-like last; speed: the holder's cached pokemon.speed (0 for side/field holders);
    effectOrder only for *SwitchIn / *RedirectTarget callbacks (`uses_effect_order`)."""
    o = dex.handler_order[effect, hook, prefix].astype(I32)
    shape = o.shape[:-1]
    if holder_stored_spe is not None:
        # PS sim/battle.ts:1011-1014: Magic Bounce onAllyTryHitSide uses getStat('spe', true, true)
        mb = (jnp.asarray(effect) == MAGIC_BOUNCE) & (jnp.asarray(hook) == ids.HOOK_TRYHITSIDE) & (
            jnp.asarray(prefix) == ids.PFX_ALLY)
        holder_speed = jnp.where(mb, holder_stored_spe, holder_speed)
    eo = jnp.broadcast_to(jnp.where(uses_effect_order, jnp.asarray(effect_order, I32), 0), shape)
    return o[..., 0], o[..., 1], jnp.broadcast_to(jnp.asarray(holder_speed, I32), shape), o[..., 2], eo


def compare_keys(keys):
    """Pack comparePriority into one sortable int64-free tuple of int32 (lexicographic via repeated argsort)."""
    order, prio, speed, sub, eo = keys
    return (order, -prio, -speed, sub, eo)


def sort_order(keys, valid):
    """Stable PS comparePriority order of candidates (invalid last). Returns (perm i32[K], tie_id i32[K]) where
    tie_id[i] is the index of the first candidate in i's exact-key tie group (per sorted position)."""
    k = compare_keys(keys)
    n = valid.shape[0]
    idx = jnp.arange(n, dtype=I32)
    perm = idx
    # lexicographic stable sort: sort by least-significant key first
    for comp in reversed((jnp.where(valid, 0, 1),) + tuple(k)):
        perm = perm[jnp.argsort(comp[perm], stable=True)]
    sk = [c[perm] for c in k]
    same_prev = jnp.ones(n, bool).at[0].set(False)
    for c in sk:
        same_prev = same_prev & (c == jnp.roll(c, 1))
    same_prev = same_prev & valid[perm] & jnp.roll(valid[perm], 1)
    # group id = position of the run start
    starts = jnp.where(same_prev, 0, idx)
    group = jax.lax.cummax(starts)
    return perm, group


# ------------------------------------------------------------------ registry
@dataclasses.dataclass(frozen=True)
class Handler:
    hook: int
    effect: int              # EffectId
    fn: Callable
    ps: str
    prefix: int = ids.PFX_ON


_REGISTRY: dict[tuple[int, int, int], Handler] = {}


def handler(hook: int, effect: int, ps: str, prefix: int = ids.PFX_ON):
    """Register a bespoke handler: fn(hc, st, dex) -> (hc, st). `ps` = 'file:lines' citation (lint-enforced)."""
    def deco(fn):
        key = (hook, prefix, effect)
        if key in _REGISTRY:
            raise ValueError(f"duplicate handler {key}")
        _REGISTRY[key] = Handler(hook, effect, fn, ps, prefix)
        return fn
    return deco


def registered(hook: int, prefix: int | None = None) -> list[Handler]:
    return [h for (hk, p, _), h in sorted(_REGISTRY.items()) if hk == hook and (prefix is None or p == prefix)]


def branch_table(hook: int, effect_ids, prefix: int | None = None):
    """(local index array over `effect_ids` (0 = no bespoke handler), branch fns) for one hook's lax.switch."""
    hs = registered(hook, prefix)
    pos = {h.effect: i + 1 for i, h in enumerate(hs)}
    local = jnp.asarray([pos.get(int(e), 0) for e in effect_ids], I32)
    return local, [h.fn for h in hs]


def dispatch(hook: int, prefix: int | None = None):
    """(eff -> local branch index fn, [fn]) of every registered handler for `hook` (run_event `branches`)."""
    hs = registered(hook, prefix)
    effs = jnp.asarray([h.effect for h in hs] or [-10**6], I32)

    def local(e):
        hit = jnp.asarray(e, I32) == effs
        return jnp.where(jnp.any(hit), jnp.argmax(hit) + 1, 0)
    return local, [h.fn for h in hs]


# ------------------------------------------------------------------ event context and dispatch
from flax import struct  # noqa: E402

from vgcjax.battle import results as RS  # noqa: E402
from vgcjax.battle import state as S  # noqa: E402
from vgcjax.battle.util import mget  # noqa: E402

UNPREFIXED = {"BeforeTurn", "Update", "Weather", "WeatherChange", "TerrainChange"}  # PS sim/battle.ts:1066
LEFT_TO_RIGHT = {"Invulnerability", "TryHit", "DamagingHit", "EntryHazard"}       # PS sim/battle.ts:796-802
MUL_HOOKS = {"BasePower", "ModifyAtk", "ModifyDef", "ModifySpA", "ModifySpD", "ModifySpe", "ModifyDamage",
             "ModifyAccuracy", "ModifyCritRatio", "ModifySTAB"}                    # contract §4 HANDLER_MUL
# holder codes in candidate lists: 0..7 mon ref, 8/9 side 0/1, 10 field
HOLD_SIDE0, HOLD_FIELD = 8, 10


@struct.dataclass
class HookCtx:
    """runEvent context threaded through handlers (fields are int32 / bool scalars or small vectors)."""
    target: jax.Array        # event target ref (-1: none / battle)
    source: jax.Array        # event source ref (-1: none)
    effect: jax.Array        # sourceEffect EffectId (e.g. the active move), -1 none
    relay_t: jax.Array       # relay (PS relayVar) as (tag, value) — results.py tags
    relay_v: jax.Array
    vec: jax.Array           # i32[7] vector relay (boost tables)
    modifier: jax.Array      # 4096-scale event.modifier (chainModify target)
    stop: jax.Array          # bool: fold stopped (falsy return / fastExit)
    holder: jax.Array        # current handler holder code
    cur_effect: jax.Array    # current handler EffectId
    ret_t: jax.Array         # current handler's return (tag UNDEF = undefined)
    ret_v: jax.Array

    @staticmethod
    def make(target, source, effect, relay=(RS.UNDEF, 0), vec=None):
        z = jnp.zeros((), I32)
        return HookCtx(target=jnp.asarray(target, I32), source=jnp.asarray(source, I32),
                       effect=jnp.asarray(effect, I32), relay_t=jnp.asarray(relay[0], I32),
                       relay_v=jnp.asarray(relay[1], I32),
                       vec=jnp.zeros(7, I32) if vec is None else jnp.asarray(vec, I32), modifier=jnp.asarray(4096, I32),
                       stop=jnp.zeros((), bool), holder=z - 1, cur_effect=z - 1, ret_t=z, ret_v=z)

    def chain(self, num, den=1):
        """PS chainModify(numerator, denominator) on event.modifier."""
        from vgcjax.core import fixedpoint as fp
        return self.replace(modifier=fp.chain_modify(self.modifier, num, den))

    def ret(self, r, cond=True):
        """Return value r = (tag, val) from the current handler (when cond)."""
        return self.replace(ret_t=jnp.where(cond, r[0], self.ret_t), ret_v=jnp.where(cond, r[1], self.ret_v))


def candidates(st, dex, hook_name: str, target, source):
    """Candidate list for runEvent(hook, target, source) per PS findEventHandlers (sim/battle.ts:1049-1109), built at
    trace time only from effects that define (hook, prefix) (effects/static.py). Returns dict of arrays [N]: eff, pfx,
    holder, valid. `target` = mon ref (or -1 = battle)."""
    from vgcjax.battle import common as C
    from vgcjax.effects import static as SX
    hook = getattr(ids, f"HOOK_{hook_name.upper()}")
    prefixed = hook_name not in UNPREFIXED
    m, sd, f = st.mons, st.sides, st.field
    target = jnp.asarray(target, I32)
    source = jnp.asarray(source, I32)
    t_ok = target >= 0
    t_active = t_ok & mget(m.is_active, target)
    s_active = (source >= 0) & mget(m.is_active, source)
    mon_level = t_ok & (t_active | s_active)
    effs, pfxs, holders = [], [], []

    def add(eff, pfx, holder):
        eff = jnp.atleast_1d(jnp.asarray(eff, I32))
        if eff.shape[0] == 0:
            return
        effs.append(eff)
        pfxs.append(jnp.full(eff.shape, pfx, I32))
        holders.append(jnp.broadcast_to(jnp.asarray(holder, I32), eff.shape))

    def mon_rows(ref, pfx, cond):
        """status, volatiles, ability, item, slot conditions of `ref` that define (hook, pfx)."""
        r = jnp.maximum(ref, 0)
        side = r // 4
        h = jnp.where(cond, ref, -1)
        if SX.statuses_define(hook, pfx):
            status = mget(m.status, r).astype(I32)
            add(jnp.where(status > 0, C.eff_cond(status), -1), pfx, h)
        kv = SX.cond_subset(hook, pfx, ids.VOL_BASE, S.NV)
        if kv:
            kv = jnp.asarray(kv, I32)
            words = mget(m.vol_bits, r)
            on = ((words[kv // 32] >> (kv % 32).astype(jnp.uint32)) & 1) == 1
            add(jnp.where(on, C.eff_cond(ids.VOL_BASE + kv), -1), pfx, h)
        if SX.abilities_define(hook, pfx):
            add(C.eff_ability(mget(m.ability, r)), pfx, h)
        if SX.items_define(hook, pfx):
            add(C.eff_item(mget(m.item, r)), pfx, h)
        ks = SX.cond_subset(hook, pfx, ids.SLOT_BASE, S.NSL)
        if ks:
            ks = jnp.asarray(ks, I32)
            pos = jnp.where(sd.active[side, 0] == r % 4, 0, jnp.where(sd.active[side, 1] == r % 4, 1, -1))
            bits = sd.slot_cond_bits[side, jnp.clip(pos, 0, 1)]
            on = (((bits >> ks.astype(jnp.uint32)) & 1) == 1) & (pos >= 0)
            add(jnp.where(on, C.eff_cond(ids.SLOT_BASE + ks), -1), pfx, h)

    tside = jnp.where(t_ok, target // 4, 0)
    for side in range(2):
        for slot in range(2):
            b = sd.active[side, slot].astype(I32)
            ref = jnp.where(b >= 0, side * 4 + b, -1)
            alive = (ref >= 0) & (mget(m.hp, jnp.maximum(ref, 0)) > 0)
            ok = (ref >= 0) & mon_level
            roles = [(ids.PFX_ON, ref == target)]
            if prefixed:
                roles += [(ids.PFX_ALLY, alive & (side == tside)), (ids.PFX_ANY, alive),
                          (ids.PFX_FOE, alive & (side != tside)), (ids.PFX_SOURCE, ref == source)]
            for pfx, cond in roles:
                mon_rows(ref, pfx, ok & cond)
    # side conditions: on (target side), Foe (other side), Any (both)
    for side in range(2):
        for pfx, cond in ((ids.PFX_ON, side == tside), (ids.PFX_FOE, (side != tside) & prefixed),
                          (ids.PFX_ANY, jnp.asarray(prefixed))):
            k = SX.cond_subset(hook, pfx, ids.SIDE_BASE, S.NSC)
            if not k:
                continue
            k = jnp.asarray(k, I32)
            on = ((sd.side_cond_bits[side] >> k.astype(jnp.uint32)) & 1) == 1
            add(jnp.where(on, C.eff_cond(ids.SIDE_BASE + k), -1), pfx, jnp.where(cond & mon_level, HOLD_SIDE0 + side, -1))
    # field: pseudo-weathers, weather, terrain (on)
    k = SX.cond_subset(hook, ids.PFX_ON, ids.PW_BASE, S.NPW)
    if k:
        k = jnp.asarray(k, I32)
        on = ((f.pw_bits >> k.astype(jnp.uint32)) & 1) == 1
        add(jnp.where(on, C.eff_cond(ids.PW_BASE + k), -1), ids.PFX_ON, HOLD_FIELD)
    reg_on = SX.registered_effects(hook, ids.PFX_ON)
    from vgcjax.effects import conditions as CO
    if any(CO.eff(w) in reg_on for w in CO.WEATHERS if w in ids.CONDITIONS):
        add(jnp.where(f.weather > 0, C.eff_cond(f.weather), -1), ids.PFX_ON, HOLD_FIELD)
    if any(CO.eff(w) in reg_on for w in CO.TERRAINS if w in ids.CONDITIONS):
        add(jnp.where(f.terrain > 0, C.eff_cond(f.terrain), -1), ids.PFX_ON, HOLD_FIELD)
    if not effs:
        add(-1, 0, -1)
    eff = jnp.concatenate(effs)
    pfx = jnp.concatenate(pfxs)
    holder = jnp.concatenate(holders)
    has = ((dex.handler_mask[jnp.maximum(eff, 0), hook].astype(I32) >> pfx) & 1) == 1
    valid = has & (holder >= 0) & (eff >= 0)
    return dict(eff=eff, pfx=pfx, holder=holder, valid=valid, hook=hook)


def compact(c, k=K_H):
    """Keep the first k valid candidates (collection order). Returns (dict of [k] arrays, overflow)."""
    idx = jnp.nonzero(c["valid"], size=k, fill_value=0)[0]
    n = jnp.sum(c["valid"].astype(I32))
    out = {key: (v[idx] if key != "hook" else v) for key, v in c.items()}
    out["valid"] = jnp.arange(k) < n
    return out, n > k


def holder_speed(st, holder):
    m = st.mons
    is_mon = (holder >= 0) & (holder < 8)
    return jnp.where(is_mon, mget(m.speed, jnp.clip(holder, 0, 7)).astype(I32), 0)


def run_event(hook_name: str, st, dex, rng, hc: HookCtx, branches=None, fast_exit=False, on=True):
    """PS runEvent / priorityEvent fold (sim/battle.ts:765-944) over the hook's candidates.

    `branches`: (local index fn eff -> branch, [fn]) from `branch_table`; fn(hc, st, dex, rng) -> (hc, st, rng) sets
    hc.ret_*. Returns (hc, st, rng); hc.relay is final (with the final modify applied to non-negative numbers).
    Tie groups on MUL hooks emit SPEED_TIE_HANDLER_MUL records (contract §4).
    """
    from vgcjax.battle import common as C
    from vgcjax.core import fixedpoint as fp
    from vgcjax.core import rng as R
    on = jnp.asarray(on)
    c = candidates(st, dex, hook_name, hc.target, hc.source)
    k = min(K_H, int(c["eff"].shape[0]))                    # loop bound <= the static candidate count
    c, overflow = compact(c, k)
    hook = c["hook"]
    spd = holder_speed(st, c["holder"])
    uses_eo = hook_name.endswith("SwitchIn") or hook_name.endswith("RedirectTarget")
    keys = handler_key(dex, c["eff"], hook, c["pfx"], spd, 0, uses_eo)
    if hook_name in LEFT_TO_RIGHT:
        o, p, _, _, _ = keys
        keys = (o, p, jnp.zeros_like(o), jnp.zeros_like(o), jnp.zeros_like(o))
    perm, group = sort_order(keys, c["valid"])
    mul_tie = jnp.zeros((), bool)
    if hook_name in MUL_HOOKS:
        # TODO(M3.x, first MUL-hook content): emit SPEED_TIE_HANDLER_MUL records (members canonical by holder rank,
        # then PS id string order, as oracle/lib/instrument.js) and fold in the drawn order. Until then an exact
        # tie group of size >= 2 raises ERR_UNSUPPORTED_EFFECT instead of silently using a canonical order.
        valid_sorted = c["valid"][perm]
        size = jnp.bincount(jnp.where(valid_sorted, group, k), length=k + 1)[:k]
        mul_tie = jnp.any(size >= 2)
        sc = st.scratch
        st = st.replace(scratch=sc.replace(mulhook_ord=sc.mulhook_ord + on.astype(sc.mulhook_ord.dtype)))
    fns = []
    if branches is not None:
        local_fn, fns = branches

    def body(i, carry):
        hc, st, rng = carry
        j = perm[i]
        live = on & c["valid"][j] & ~hc.stop
        hc = hc.replace(holder=c["holder"][j], cur_effect=c["eff"][j], ret_t=jnp.asarray(RS.UNDEF, I32),
                        ret_v=jnp.zeros((), I32))
        if fns:
            idx = jnp.where(live, local_fn(c["eff"][j]), 0)
            bound = [lambda h, s, r: (h, s, r)] + [(lambda h, s, r, f=f: f(h, s, dex, r)) for f in fns]
            hc2, st2, rng2 = jax.lax.switch(idx, bound, hc, st, rng)
            hc, st, rng = hc2, st2, rng2
        ret = (hc.ret_t, hc.ret_v)
        has_ret = live & (hc.ret_t != RS.UNDEF)
        hc = hc.replace(relay_t=jnp.where(has_ret, ret[0], hc.relay_t), relay_v=jnp.where(has_ret, ret[1], hc.relay_v))
        stop = has_ret & (~RS.truthy(ret) | fast_exit)
        hc = hc.replace(stop=hc.stop | stop)
        hc, st, rng = jax.tree_util.tree_map(lambda a, b: jax.lax.optimization_barrier(a), (hc, st, rng), (hc, st, rng))
        return hc, st, rng

    hc, st, rng = jax.lax.fori_loop(0, k, body, (hc, st, rng))
    # final modify for non-negative integer relays (PS :936-940)
    numeric = (hc.relay_t == RS.NUM) & (hc.relay_v >= 0)
    hc = hc.replace(relay_v=jnp.where(on & numeric, fp.modify_raw(hc.relay_v, hc.modifier), hc.relay_v))
    from vgcjax.battle import errors as E
    from vgcjax.battle.util import or_flag
    st = or_flag(st, E.ERR_OVERFLOW_HOOK, on & overflow)
    st = or_flag(st, E.ERR_UNSUPPORTED_EFFECT, on & mul_tie)
    return hc, st, rng

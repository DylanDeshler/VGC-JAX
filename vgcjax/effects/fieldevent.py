"""fieldEvent('Residual') (04 §9.4, SYSTEM residual_walk). PS: sim/battle.ts:489-574 fieldEvent, :1111-1270
find*EventHandlers, :957-1023 resolvePriority.

Handlers: field (pseudo-weathers, weather, terrain: onFieldResidual), each side (side conditions: onSideResidual) and
each active mon (status, volatiles, ability, item, slot conditions: onResidual). With getKey = 'duration' a handler
is listed when its callback exists *or* its state has a duration. They are speed-sorted; exact ties form
SPEED_TIE_FIELD_EVENT records (contract §4: sub = fieldevent_ord, ord = group index, members in field < side0 <
side1 < p1a < p1b < p2a < p2b then effect-id order, fainted holders omitted). Then, in order: skip fainted holders,
decrement the duration and end the effect at 0, skip effects removed meanwhile, run the callback, resolveFaints.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vgcjax.battle import faint as F
from vgcjax.battle import state as S
from vgcjax.battle.util import mget
from vgcjax.core import rng as R
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects import hooks as H

I16, I32, U32 = jnp.int16, jnp.int32, jnp.uint32
K_RES = 16          # listed handlers per Residual walk (overflow -> ERR_OVERFLOW_RESIDUAL)
K_SWITCHIN = 8      # listed handlers per SwitchIn walk
G_TIES = 4          # tie groups per walk
ORDER_LAST = 1 << 15


def _effect_names():
    names = [""] * (ids.ITEM_BASE + len(ids.ITEMS) + len(ids.ABILITIES) + len(ids.MOVES) + len(ids.CONDITIONS))
    for base, lst in ((ids.MOVE_BASE, ids.MOVES), (ids.ABILITY_BASE, ids.ABILITIES), (ids.ITEM_BASE, ids.ITEMS),
                      (ids.COND_BASE, ids.CONDITIONS)):
        for i, n in enumerate(lst[1:], 1):
            j = base + i - 1
            if j >= len(names):
                names += [""] * (j + 1 - len(names))
            names[j] = n
    return names


_NAMES = _effect_names()
_STR_RANK = np.argsort(np.argsort(np.array(_NAMES, dtype=object), kind="stable")).astype(np.int32)

# default subOrder by effect type when the effect has no Residual callback entry (PS resolvePriority :964-999)
SUB_POKEMON_COND, SUB_SLOT, SUB_SIDE, SUB_FIELD, SUB_ABILITY, SUB_ITEM = 2, 3, 4, 5, 7, 8


def _candidates(st, dex, event: str, targets=None):
    """Static candidate list for fieldEvent(event, targets) (PS :489-511), restricted at trace time to effects that
    define the callback (or, for Residual, carry a duration; effects/static.py). Returns dict of i32 arrays [N]:
    eff, pfx, holder (0..7 ref, 8/9 side, 10 field), loc (storage code), idx (index within the storage), dur (state
    duration), present, sub (default subOrder), eo (effectOrder, *SwitchIn only), spdh (speed holder ref or -1), cust
    (1 = customHolder row)."""
    from vgcjax.effects import static as SX
    m, sd, f = st.mons, st.sides, st.field
    hook = getattr(ids, f"HOOK_{event.upper()}")
    resid = event == "Residual"
    rows = []

    def add(eff, pfx, holder, loc, idx, dur, present, sub, eo=0, spdh=-1, cust=False):
        n = jnp.shape(eff)[0] if jnp.ndim(eff) else 1
        if n == 0:
            return
        b = lambda x: jnp.broadcast_to(jnp.asarray(x, I32), (n,))  # noqa: E731
        rows.append((b(eff), b(pfx), b(holder), b(loc), b(idx), b(dur), jnp.broadcast_to(present, (n,)), b(sub), b(eo),
                     b(spdh), b(int(cust))))

    weather_ids = tuple(CO.cid(w) for w in CO.WEATHERS if w in ids.CONDITIONS)
    terrain_ids = tuple(CO.cid(w) for w in CO.TERRAINS if w in ids.CONDITIONS)

    def named_define(cids, pfx, with_dur):
        return any(SX.cond_subset(hook, pfx, c, 1, with_dur) for c in cids)

    # customHolder rows (a field / side effect's on<Event> listed for an active mon, PS :507-508): getKey undefined
    # (callback only) and no `end` (no duration handling)
    def field_effects(pfx, holder, spdh, on=True):
        cust = pfx == ids.PFX_ON
        wd = resid and pfx == ids.PFX_FIELD
        k = jnp.asarray(SX.cond_subset(hook, pfx, ids.PW_BASE, S.NPW, wd), I32)
        pw_on = ((f.pw_bits >> k.astype(U32)) & 1) == 1
        add(ids.COND_BASE + ids.PW_BASE + k - 1, pfx, holder, CO.PSEUDO, k, f.pw_state[k, 0], pw_on & on, SUB_FIELD,
            0, spdh, cust)
        if named_define(weather_ids, pfx, wd):
            add(ids.COND_BASE + f.weather.astype(I32) - 1, pfx, holder, CO.WEATHER, 0, f.weather_state[0],
                (f.weather > 0) & on, SUB_FIELD, 0, spdh, cust)
        if named_define(terrain_ids, pfx, wd):
            add(ids.COND_BASE + f.terrain.astype(I32) - 1, pfx, holder, CO.TERRAIN, 0, f.terrain_state[0],
                (f.terrain > 0) & on, SUB_FIELD, 0, spdh, cust)

    def side_effects(s, pfx, holder, spdh, on=True):
        cust = pfx == ids.PFX_ON
        k = jnp.asarray(SX.cond_subset(hook, pfx, ids.SIDE_BASE, S.NSC, resid and pfx == ids.PFX_SIDE), I32)
        son = ((sd.side_cond_bits[s] >> k.astype(U32)) & 1) == 1
        add(ids.COND_BASE + ids.SIDE_BASE + k - 1, pfx, holder, CO.SIDE, k, sd.side_cond_state[s, k, 0], son & on,
            SUB_SIDE, 0, spdh, cust)

    def mon_effects(s, pos, r, pfx, on):
        wd = resid and pfx == ids.PFX_ON
        if SX.statuses_define(hook, pfx):
            status = mget(m.status, r).astype(I32)
            add(ids.COND_BASE + status - 1, pfx, r, CO.STATUS, 0, mget(m.status_state, r)[0], on & (status > 0),
                SUB_POKEMON_COND, 0, r)
        kv = jnp.asarray(SX.cond_subset(hook, pfx, ids.VOL_BASE, S.NV, wd), I32)
        words = mget(m.vol_bits, r)
        von = ((words[kv // 32] >> (kv % 32).astype(U32)) & 1) == 1
        add(ids.COND_BASE + ids.VOL_BASE + kv - 1, pfx, r, CO.VOLATILE, kv, mget(m.vol_state, r)[kv, 0], on & von,
            SUB_POKEMON_COND, 0, r)
        if SX.abilities_define(hook, pfx):
            ab = mget(m.ability, r).astype(I32)
            add(ids.ABILITY_BASE + ab - 1, pfx, r, -1, 0, 0, on & (ab > 0), SUB_ABILITY, mget(m.ability_seq, r), r)
        if SX.items_define(hook, pfx):
            it = mget(m.item, r).astype(I32)
            add(ids.ITEM_BASE + it - 1, pfx, r, -1, 0, 0, on & (it > 0), SUB_ITEM, mget(m.item_seq, r), r)
        ks = jnp.asarray(SX.cond_subset(hook, pfx, ids.SLOT_BASE, S.NSL, wd), I32)
        son = ((sd.slot_cond_bits[s, pos] >> ks.astype(U32)) & 1) == 1
        add(ids.COND_BASE + ids.SLOT_BASE + ks - 1, pfx, r, CO.SLOT, ks + 8 * pos, sd.slot_cond_state[s, pos, ks, 0],
            on & son, SUB_SLOT, 0, r)

    field_effects(ids.PFX_FIELD, 10, -1)
    for s in range(2):
        side_effects(s, ids.PFX_SIDE, 8 + s, -1)
    for s in range(2):
        for pos in range(2):
            b = sd.active[s, pos].astype(I32)
            ok = b >= 0
            r = s * 4 + jnp.maximum(b, 0)
            if event == "SwitchIn":
                mon_effects(s, pos, r, ids.PFX_ANY, ok)
                ok = ok & jnp.any(jnp.asarray(targets, I32) == r)
            mon_effects(s, pos, r, ids.PFX_ON, ok)
            side_effects(s, ids.PFX_ON, r, r, ok)          # customHolder = the active mon
            field_effects(ids.PFX_ON, r, r, ok)
    if not rows:                                         # nothing can ever handle this event
        return None
    cols = [jnp.concatenate([row[j] for row in rows]) for j in range(11)]
    keys = ("eff", "pfx", "holder", "loc", "idx", "dur", "present", "sub", "eo", "spdh", "cust")
    return dict(zip(keys, cols))


def holder_rank(st, holder):
    """field 0 < side0 1 < side1 2 < p1a 10 < p1b 11 < p2a 12 < p2b 13 (oracle/lib/instrument.js holderRank)."""
    s = jnp.clip(holder, 0, 7) // 4
    b = jnp.clip(holder, 0, 7) % 4
    pos = jnp.where(st.sides.active[s, 0] == b, 0, 1)
    return jnp.where(holder == 10, 0, jnp.where(holder >= 8, holder - 7, 10 + 2 * s + pos))


def _still_present(st, loc, idx, holder, eff):
    """The effect is still held (PS compares the state object; re-adding during the walk is not modelled)."""
    m, sd, f = st.mons, st.sides, st.field
    r = jnp.clip(holder, 0, 7)
    c = eff - ids.COND_BASE + 1
    vol = CO.has_volatile(st, r, c)
    status = mget(m.status, r).astype(I32) == c
    side = ((sd.side_cond_bits[jnp.clip(holder - 8, 0, 1)] >> jnp.clip(idx, 0, 31).astype(U32)) & 1) == 1
    pw = ((f.pw_bits >> jnp.clip(idx, 0, 31).astype(U32)) & 1) == 1
    slot = ((sd.slot_cond_bits[r // 4, jnp.clip(idx // 8, 0, 1)] >> (idx % 8).astype(U32)) & 1) == 1
    return jnp.where(loc == CO.VOLATILE, vol, jnp.where(loc == CO.STATUS, status, jnp.where(
        loc == CO.SIDE, side, jnp.where(loc == CO.PSEUDO, pw, jnp.where(loc == CO.WEATHER, f.weather.astype(I32) == c,
        jnp.where(loc == CO.TERRAIN, f.terrain.astype(I32) == c, jnp.where(loc == CO.SLOT, slot, True)))))))


def _set_duration(st, loc, idx, holder, value, on):
    m, sd, f = st.mons, st.sides, st.field
    r = jnp.clip(holder, 0, 7)
    v = jnp.asarray(value, I16)
    on = jnp.asarray(on)
    from vgcjax.battle.util import mset
    vs = mget(m.vol_state, r)
    m = m.replace(vol_state=mset(m.vol_state, r, vs.at[jnp.clip(idx, 0, S.NV - 1), 0].set(v), on & (loc == CO.VOLATILE)),
                  status_state=mset(m.status_state, r, mget(m.status_state, r).at[0].set(v), on & (loc == CO.STATUS)))
    hs = jnp.clip(holder - 8, 0, 1)
    sd = sd.replace(side_cond_state=jnp.where(on & (loc == CO.SIDE),
                                              sd.side_cond_state.at[hs, jnp.clip(idx, 0, S.NSC - 1), 0].set(v),
                                              sd.side_cond_state),
                    slot_cond_state=jnp.where(on & (loc == CO.SLOT),
                                              sd.slot_cond_state.at[r // 4, jnp.clip(idx // 8, 0, 1),
                                                                    jnp.clip(idx % 8, 0, S.NSL - 1), 0].set(v),
                                              sd.slot_cond_state))
    f = f.replace(pw_state=jnp.where(on & (loc == CO.PSEUDO), f.pw_state.at[jnp.clip(idx, 0, S.NPW - 1), 0].set(v),
                                     f.pw_state),
                  weather_state=jnp.where(on & (loc == CO.WEATHER), f.weather_state.at[0].set(v), f.weather_state),
                  terrain_state=jnp.where(on & (loc == CO.TERRAIN), f.terrain_state.at[0].set(v), f.terrain_state))
    return st.replace(mons=m, sides=sd, field=f)


def end_effect(st, dex, rng, loc, idx, holder, eff, on):
    """handler.end: removeVolatile / clearStatus / removeSideCondition / removeSlotCondition / removePseudoWeather /
    clearWeather / clearTerrain (PS: sim/battle.ts find*EventHandlers `end`), with the effect's End callback."""
    from vgcjax.effects import statuses as ST
    from vgcjax.effects import sideconds as SC
    on = jnp.asarray(on)
    r = jnp.clip(holder, 0, 7)
    c = eff - ids.COND_BASE + 1
    st, rng, _ = CO.remove_volatile(st, dex, rng, r, c, on & (loc == CO.VOLATILE))
    st = ST.clear_status(st, r, on & (loc == CO.STATUS))
    st, rng = SC.remove_side_condition(st, dex, rng, jnp.clip(holder - 8, 0, 1), c, on & (loc == CO.SIDE))
    st, rng = SC.remove_pseudo_weather(st, dex, rng, c, on & (loc == CO.PSEUDO))
    st, rng = SC.clear_weather(st, dex, rng, on & (loc == CO.WEATHER))
    st, rng = SC.clear_terrain(st, dex, rng, on & (loc == CO.TERRAIN))
    st, rng = SC.remove_slot_condition(st, dex, rng, r // 4, idx // 8, c, on & (loc == CO.SLOT))
    return st, rng


def residual(st, dex, rng, on):
    """fieldEvent('Residual'). Returns (st, rng)."""
    return field_event(st, dex, rng, "Residual", on)


def switch_in(st, dex, rng, targets, on):
    """fieldEvent('SwitchIn', targets) from runSwitch (PS: sim/battle-actions.ts:228-249). Returns (st, rng)."""
    return field_event(st, dex, rng, "SwitchIn", on, targets)


def _speed_key(st, holder_ref, event):
    """resolvePriority speed (PS :1008-1021) x4: pokemon holders use the cached speed; *SwitchIn handlers subtract
    the holder's index in battle.speedOrder / (activePerHalf * 2) = idx / 4 (x4: idx). Side / field holders: 0."""
    is_mon = (holder_ref >= 0) & (holder_ref < 8)
    r = jnp.clip(holder_ref, 0, 7)
    spd = mget(st.mons.speed, r).astype(I32) * 4
    if event == "SwitchIn":
        s, b = r // 4, r % 4
        pos = jnp.where(st.sides.active[s, 0] == b, 0, 1)
        fp = s + 2 * pos
        so = st.meta.speed_order.astype(I32)
        hit = so == fp[..., None]
        idx = jnp.where(jnp.any(hit, axis=-1), jnp.argmax(hit, axis=-1), -1)
        spd = spd - idx
    return jnp.where(is_mon, spd, 0)


def field_event(st, dex, rng, event: str, on, targets=None):
    """Shared fieldEvent walk (module docstring). Residual: getKey = 'duration'; SwitchIn: effectOrder ties."""
    on = jnp.asarray(on)
    residual_mode = event == "Residual"
    sc = st.scratch
    sub = sc.fieldevent_ord
    st = st.replace(scratch=sc.replace(fieldevent_ord=sc.fieldevent_ord + on.astype(I16)))
    c = _candidates(st, dex, event, targets)
    if c is None:                                        # statically empty: fieldEvent runs no handler
        return st, rng
    hook = getattr(ids, f"HOOK_{event.upper()}")
    has_cb = ((dex.handler_mask[jnp.maximum(c["eff"], 0), hook].astype(I32) >> c["pfx"]) & 1) == 1
    with_dur = (c["loc"] >= 0) & (c["dur"] > 0) & (c["cust"] == 0) if residual_mode else jnp.zeros_like(has_cb)
    listed = on & c["present"] & (c["eff"] >= 0) & (has_cb | with_dur)
    K = min(K_RES if residual_mode else K_SWITCHIN, int(c["eff"].shape[0]))   # <= the static candidate count
    n = jnp.sum(listed.astype(I32))
    sel = jnp.nonzero(listed, size=K, fill_value=0)[0]
    valid = jnp.arange(K) < n
    cc = {k: v[sel] for k, v in c.items()}
    from vgcjax.battle.util import or_flag
    from vgcjax.battle import errors as E
    st = or_flag(st, E.ERR_OVERFLOW_RESIDUAL, n > K)
    eff, pfx, holder = cc["eff"], cc["pfx"], cc["holder"]
    o = dex.handler_order[jnp.maximum(eff, 0), hook, pfx].astype(I32)
    has_entry = has_cb[sel]
    order = jnp.where(has_entry, o[:, 0], ORDER_LAST)
    prio = jnp.where(has_entry, o[:, 1], 0)
    subo = jnp.where(has_entry, o[:, 2], cc["sub"])
    spd = _speed_key(st, cc["spdh"], event)
    eo = cc["eo"] if event == "SwitchIn" else jnp.zeros_like(order)
    perm, group = H.sort_order((order, prio, spd, subo, eo), valid)
    # tie groups among handlers that fieldEvent will not skip (fainted holders, except slot conditions)
    fainted = (holder < 8) & mget(st.mons.fainted, jnp.clip(holder, 0, 7)) & (cc["loc"] != CO.SLOT)
    live_sorted = valid[perm] & ~fainted[perm]
    pos = jnp.arange(K)
    gsize = jnp.zeros(K, I32).at[group].add(live_sorted.astype(I32))
    # a group is marked at its first live member (the run may start with a skipped handler)
    first_live = jnp.full(K, K, I32).at[group].min(jnp.where(live_sorted, pos, K))
    is_start = live_sorted & (pos == first_live[group]) & (gsize[group] >= 2)
    # canonical member order inside a group: holder rank, then effect-id string rank
    canon_key = holder_rank(st, holder) * 4096 + jnp.asarray(_STR_RANK)[jnp.clip(eff, 0, len(_NAMES) - 1)]
    new_perm = perm
    starts = jnp.nonzero(is_start, size=G_TIES, fill_value=-1)[0]
    st = or_flag(st, E.ERR_OVERFLOW_RESIDUAL, jnp.sum(is_start.astype(I32)) > G_TIES)
    for g in range(G_TIES):
        start = starts[g]
        act = start >= 0
        mem_sorted = act & (group == group[jnp.maximum(start, 0)]) & live_sorted   # sorted positions in the group
        cand = perm
        ck = jnp.where(mem_sorted, canon_key[cand], 1 << 30)
        canon_pos = jnp.argsort(ck, stable=True)                                 # positions in canonical order
        nm = jnp.sum(mem_sorted.astype(I32))
        mvalid = jnp.arange(K) < nm
        torder, rng = R.tie_order(rng, H_tkey(st, sub, g), jnp.arange(8), jnp.pad(mvalid, (0, max(0, 8 - K)))[:8],
                                  active=on & act)
        exec_pos = canon_pos[jnp.clip(torder, 0, K - 1)]
        slots = jnp.nonzero(mem_sorted, size=8, fill_value=0)[0]
        for j in range(8):
            ok = act & (j < nm)
            new_perm = new_perm.at[slots[j]].set(jnp.where(ok, cand[exec_pos[j]], new_perm[slots[j]]))
    cbs = has_cb[sel]

    def body(i, carry):
        st, rng = carry
        j = new_perm[i]
        live = on & valid[j] & (st.meta.phase != S.DONE)
        h, e, loc, idx = holder[j], eff[j], cc["loc"][j], cc["idx"][j]
        skip = (h < 8) & mget(st.mons.fainted, jnp.clip(h, 0, 7)) & (loc != CO.SLOT)
        live = live & ~skip
        ending = jnp.zeros((), bool)
        if residual_mode:
            dur = _cur_duration(st, loc, idx, h)
            has_dur = live & (loc >= 0) & (dur > 0) & (cc["cust"][j] == 0) & _still_present(st, loc, idx, h, e)
            nd = dur - 1
            st = _set_duration(st, loc, idx, h, nd, has_dur)
            ending = has_dur & (nd <= 0)
            st, rng = end_effect(st, dex, rng, loc, idx, h, e, ending)
        run = live & ~ending & _still_present(st, loc, idx, h, e) & cbs[j]
        hc = H.HookCtx.make(jnp.where(h < 8, h, -1), -1, -1).replace(holder=h)
        ar = (ids.ABILITY_BASE, ids.ABILITY_BASE + len(ids.ABILITIES) - 1)
        hc, st, rng = CO.single_event(hook, e, st, dex, rng, hc, run, prefix=pfx[j], exclude_range=ar)
        if residual_mode:                       # abilities' onResidual, undispatched (effects/ability_reactions.py)
            from vgcjax.effects import ability_reactions as AR
            st, rng = AR.residual(st, dex, rng, h, run & (e >= ar[0]) & (e < ar[1]))
        if event == "SwitchIn":                 # abilities' SwitchIn handlers, undispatched (effects/abilities.py)
            from vgcjax.effects import abilities as AB
            st, rng = AB.switch_in_vec(st, dex, rng, h, run & (e >= ar[0]) & (e < ar[1]))
        if residual_mode:                       # the weathers' onFieldResidual eachEvent('Weather'), undispatched
            from vgcjax.effects import weather as WE
            st, rng = WE.residual_each(st, dex, rng, e - ids.COND_BASE + 1, run & (loc == CO.WEATHER))
        st = F.resolve_faints(st, dex, live)
        return st, rng

    return jax.lax.fori_loop(0, K, body, (st, rng))


def _cur_duration(st, loc, idx, holder):
    m, sd, f = st.mons, st.sides, st.field
    r = jnp.clip(holder, 0, 7)
    vol = mget(m.vol_state, r)[jnp.clip(idx, 0, S.NV - 1), 0]
    status = mget(m.status_state, r)[0]
    side = sd.side_cond_state[jnp.clip(holder - 8, 0, 1), jnp.clip(idx, 0, S.NSC - 1), 0]
    slot = sd.slot_cond_state[r // 4, jnp.clip(idx // 8, 0, 1), jnp.clip(idx % 8, 0, S.NSL - 1), 0]
    pw = f.pw_state[jnp.clip(idx, 0, S.NPW - 1), 0]
    out = jnp.where(loc == CO.VOLATILE, vol, jnp.where(loc == CO.STATUS, status, jnp.where(loc == CO.SIDE, side,
          jnp.where(loc == CO.SLOT, slot, jnp.where(loc == CO.PSEUDO, pw, jnp.where(loc == CO.WEATHER, f.weather_state[0],
          jnp.where(loc == CO.TERRAIN, f.terrain_state[0], 0)))))))
    return out.astype(I32)


def H_tkey(st, sub, g):
    from vgcjax.battle import common as C
    return C.tkey(st, ids.DK_SPEED_TIE_FIELD_EVENT, sub, g)

"""Move batch B1 (data/moves.ts; Champions overrides in data/mods/champions/moves.ts: Dire Claw 30%, Spiky Shield /
Baneful Bunker / King's Shield pp): Spiky Shield, Baneful Bunker, King's Shield (Protect variants punishing contact),
Aurora Veil, Knock Off, Throat Chop, Dire Claw / Tri Attack (secondary status pick), Alluring Voice, Freeze-Dry, Haze.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget, mset
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32


def mid(name):
    return getattr(ids, f"MOVE_{name.upper()}", -999)


def has(name):
    return name in ids.CONDITIONS


# ------------------------------------------------------------------ Protect variants
SHIELDS = [n for n in ("spikyshield", "banefulbunker", "kingsshield") if has(n) and mid(n) > 0]


def _shield_start(hc, st, dex, rng):
    """'-singleturn' X 'move: Protect' (King's Shield: 'Protect')."""
    return hc, EV.emit(st, True, EV.EV_SINGLETURN, actor=hc.target, move=CO.eff("protect")), rng


def _shield_try_hit(name):
    def fn(hc, st, dex, rng):
        """onTryHit (priority 3): blocks like Protect (King's Shield: not Status moves); then a contact move is
        punished: Spiky Shield damage(source.baseMaxhp / 8, source, target); Baneful Bunker source.trySetStatus('psn',
        target, Baneful Bunker); King's Shield boost({atk: -1}, source, target, King's Shield). NOT_FAIL."""
        from vgcjax.battle import move_pipeline as MP
        from vgcjax.effects import protect as PR
        from vgcjax.effects import statuses as STS
        move = PR.move_of(hc.effect)
        status_move = dex.move_category[move] == ids.CAT_STATUS
        block = ~PR.bypasses_protect(dex, move) & ((name != "kingsshield") | ~status_move)
        hc, st = PR._block_activate(hc, st, dex, "PROTECT", block)
        w, b = ids.MF_CONTACT
        contact = block & (((dex.move_flags[move, w] >> b) & 1) == 1)
        src, tgt = hc.source, hc.target
        if name == "spikyshield":
            st, rng = STS.effect_damage(st, dex, rng, src, mget(st.mons.maxhp, src) // 8, CO.eff(name), contact,
                                        source=tgt)
        elif name == "banefulbunker":
            st, rng, _, msgs = STS.try_set_status_v(st, dex, rng, jnp.asarray(src, I32)[None], ids.COND_PSN,
                                                    contact[None], tgt, ids.MOVE_BASE + mid(name) - 1, 0)
            st = STS.emit_status_msgs(st, msgs, 0, src)
        else:
            st, _ = MP.boost(st, src, jnp.zeros(7, I32).at[0].set(-1), contact, source=tgt,
                             effect=ids.MOVE_BASE + mid(name) - 1)
        return hc.ret(RS.R(RS.NOTFAIL), block), st, rng
    return fn


for _n in SHIELDS:
    from vgcjax.effects import protect as _PR
    _e = int(C.eff_move(mid(_n)))
    handler(ids.HOOK_PREPAREHIT, _e, ps=f"data/moves.ts {_n}.onPrepareHit")(_PR._stall_prepare_hit)
    handler(ids.HOOK_HIT, _e, ps=f"data/moves.ts {_n}.onHit")(_PR._stall_on_hit)
    handler(ids.HOOK_START, CO.eff(_n), ps=f"data/moves.ts {_n}.condition.onStart")(_shield_start)
    handler(ids.HOOK_TRYHIT, CO.eff(_n), ps=f"data/moves.ts {_n}.condition.onTryHit")(_shield_try_hit(_n))


# ------------------------------------------------------------------ Aurora Veil
if has("auroraveil"):
    handler(ids.HOOK_TRY, int(C.eff_move(mid("auroraveil"))), ps="data/moves.ts auroraveil.onTry")(
        lambda hc, st, dex, rng: (hc.ret(RS.R(RS.BOOL, _snow(st).astype(I32))), st, rng))

    @handler(CO.DURATION_CALLBACK, CO.eff("auroraveil"), ps="data/moves.ts auroraveil.condition.durationCallback")
    def auroraveil_duration(hc, st, dex, rng):
        src = jnp.asarray(hc.source, I32)
        clay = (src >= 0) & (mget(st.mons.item, jnp.maximum(src, 0)) == getattr(ids, "ITEM_LIGHTCLAY", -1))
        return hc.ret(RS.R(RS.NUM, jnp.where(clay, 8, 5))), st, rng

    handler(ids.HOOK_START, CO.eff("auroraveil"), ps="data/moves.ts auroraveil.condition.onSideStart",
            prefix=ids.PFX_SIDE)(lambda hc, st, dex, rng: (hc, EV.emit(st, True, EV.EV_SIDESTART, actor=hc.holder - 8,
                                                                         move=CO.eff("auroraveil")), rng))
    handler(ids.HOOK_END, CO.eff("auroraveil"), ps="data/moves.ts auroraveil.condition.onSideEnd",
            prefix=ids.PFX_SIDE)(lambda hc, st, dex, rng: (hc, EV.emit(st, True, EV.EV_SIDEEND, actor=hc.holder - 8,
                                                                         move=CO.eff("auroraveil")), rng))

    @mod("ModifyDamage", CO.eff("auroraveil"), ps="data/moves.ts auroraveil.condition.onAnyModifyDamage",
         prefix=ids.PFX_ANY)
    def auroraveil_damage(ctx, h):
        """target !== source on the veil's side, not covered by that side's Reflect (physical) / Light Screen
        (special), not a crit -> chainModify([2732, 4096])."""
        d, a = ctx.src, ctx.tgt
        side = h - 8
        cat = ctx.dex.move_category[ctx.move]
        st = ctx.st
        refl = CO.has_side(st, side, CO.cid("reflect")) & (cat == ids.CAT_PHYSICAL) if has("reflect") else False
        ls = CO.has_side(st, side, CO.cid("lightscreen")) & (cat == ids.CAT_SPECIAL) if has("lightscreen") else False
        ok = (d >= 0) & (d != a) & ((jnp.clip(d, 0, 7) // 4) == side) & ~refl & ~ls & ~ctx.extra["crit"]
        return ok, jnp.asarray(2732, I32), None


def _snow(st):
    from vgcjax.effects import weather as WE
    return WE.is_weather(st, "snowscape")


# ------------------------------------------------------------------ Knock Off
def can_take(st, ref):
    """runEvent('TakeItem') / singleEvent('TakeItem', item): a Mega Stone matching its holder's species can't be
    removed (data/items.ts <stone>.onTakeItem); Sticky Hold: M4."""
    m = st.mons
    r = jnp.clip(ref, 0, 7)
    it = mget(m.item, r).astype(I32)
    sp = mget(m.base_species, r).astype(I32)
    base = jnp.asarray(_base_species_table(), I32)[sp]
    stone = jnp.asarray(_stone_table(), I32)                            # [n_items, 2] base species ids
    locked = jnp.any(stone[jnp.clip(it, 0, stone.shape[0] - 1)] == base)
    return (it > 0) & ~locked


_TABLES = {}


def _base_species_table():
    if "base" not in _TABLES:
        from vgcjax.core.dex import load_tables_np
        t = load_tables_np()
        _TABLES["base"] = t["species_base_species"]
    return _TABLES["base"]


def _stone_table():
    """item -> the base species ids whose Mega Stone it is (species_mega rows)."""
    if "stone" not in _TABLES:
        import numpy as np
        from vgcjax.core.dex import load_tables_np
        t = load_tables_np()
        tab = np.full((len(ids.ITEMS) + 1, 2), -1, np.int32)
        cnt = np.zeros(len(ids.ITEMS) + 1, np.int32)
        for sp in range(t["species_mega"].shape[0]):
            for k in range(2):
                stone, mega = int(t["species_mega"][sp, k, 0]), int(t["species_mega"][sp, k, 1])
                if stone > 0 and mega > 0 and cnt[stone] < 2:
                    tab[stone, cnt[stone]] = int(t["species_base_species"][sp])
                    cnt[stone] += 1
        _TABLES["stone"] = tab
    return _TABLES["stone"]


if mid("knockoff") > 0:
    @mod("BasePower", int(C.eff_move(mid("knockoff"))), ps="data/moves.ts knockoff.onBasePower")
    def knockoff_bp(ctx, h):
        """The target's item can be taken (singleEvent TakeItem) -> chainModify(1.5)."""
        import jax
        ok = jax.vmap(lambda d: can_take(ctx.st, d))(ctx.src)
        return ok & (ctx.src >= 0), jnp.asarray(6144, I32), None


def knock_off(st, dex, t, u, move, on):
    """Knock Off onAfterHit (after DamagingHit, per damaged target): target.takeItem(source) -> '-enditem' target
    item '[from] move: Knock Off' '[of] source' (item = '', itemState cleared; no lastItem)."""
    m = st.mons
    r = jnp.clip(t, 0, 7)
    it = mget(m.item, r).astype(I32)
    go = jnp.asarray(on) & (move == mid("knockoff")) & mget(m.is_active, r) & can_take(st, r)
    m = m.replace(item=mset(m.item, r, 0, go), item_state=mset(m.item_state, r, jnp.zeros_like(mget(m.item_state, r)),
                                                              go), item_seq=mset(m.item_seq, r, 0, go))
    st = st.replace(mons=m)
    return EV.emit(st, go, EV.EV_ENDITEM, actor=r, move=ids.ITEM_BASE + it - 1)


# ------------------------------------------------------------------ Throat Chop
if has("throatchop"):
    def _sound(dex, mids):
        w, b = ids.MF_SOUND
        return ((dex.move_flags[jnp.clip(mids, 0, len(ids.MOVES) - 1), w] >> b) & 1) == 1

    handler(ids.HOOK_START, CO.eff("throatchop"), ps="data/moves.ts throatchop.condition.onStart")(
        lambda hc, st, dex, rng: (hc, EV.emit(st, True, EV.EV_START, actor=hc.target, move=CO.eff("throatchop"),
                                               v1=-1), rng))
    handler(ids.HOOK_END, CO.eff("throatchop"), ps="data/moves.ts throatchop.condition.onEnd")(
        lambda hc, st, dex, rng: (hc, EV.emit(st, True, EV.EV_END, actor=hc.target, move=CO.eff("throatchop"),
                                               v1=-1), rng))

    @handler(ids.HOOK_BEFOREMOVE, CO.eff("throatchop"), ps="data/moves.ts throatchop.condition.onBeforeMove")
    def throatchop_before_move(hc, st, dex, rng):
        """priority 6: a sound move -> 'cant' X 'move: Throat Chop'; false."""
        move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
        block = _sound(dex, move)
        st = EV.emit(st, block, EV.EV_CANT, actor=hc.target, move=CO.eff("throatchop"))
        return hc.ret(RS.R(RS.BOOL, 0), block), st, rng


def disable_moves(st, dex, r, on):
    """Throat Chop onDisableMove: every sound move."""
    if not has("throatchop"):
        return st
    m = st.mons
    mids = mget(m.move_ids, r).astype(I32)
    tc = jnp.asarray(on) & CO.has_volatile(st, r, CO.cid("throatchop"))
    w, b = ids.MF_SOUND
    snd = ((dex.move_flags[jnp.clip(mids, 0, len(ids.MOVES) - 1), w] >> b) & 1) == 1
    dis = jnp.where(tc & snd & (mids > 0), 1, mget(m.move_disabled, r))
    return st.replace(mons=m.replace(move_disabled=mset(m.move_disabled, r, dis.astype(m.move_disabled.dtype), on)))


# ------------------------------------------------------------------ secondary onHit specials
STATUS_PICK = {"direclaw": ("psn", "par", "slp"), "triattack": ("brn", "par", "frz")}


def secondary_special(st, dex, rng, u, move, t, k, ok, hit_no):
    """Secondary onHit callbacks (in place of data effects): Dire Claw / Tri Attack this.sample([...]) ->
    target.trySetStatus(status, source) (SECONDARY_STATUS_PICK draw); Throat Chop target.addVolatile('throatchop');
    Alluring Voice: target.statsRaisedThisTurn -> addVolatile('confusion', source, move). Returns (st, rng)."""
    from vgcjax.core import rng as R
    from vgcjax.effects import statuses as STS
    tr = jnp.clip(t, 0, 7)
    for name, sts in STATUS_PICK.items():
        if mid(name) <= 0:
            continue
        on = ok & (move == mid(name))
        key = C.dkey(st, ids.DK_SECONDARY_STATUS_PICK, actor=u, target=tr, subject=C.eff_move(move), hit=hit_no,
                     ord=k)
        i, rng = R.sample(rng, key, 3, active=on)
        status = jnp.asarray([getattr(ids, f"COND_{s.upper()}") for s in sts], I32)[jnp.clip(i, 0, 2)]
        # sourceEffect = the secondary object (no effectType): no '[from] move:' on the status line
        st, rng, _, msgs = STS.try_set_status_v(st, dex, rng, tr[None], status, on[None], u, -1, 0)
        st = STS.emit_status_msgs(st, msgs, 0, tr)
    if has("throatchop") and mid("throatchop") > 0:
        st, rng, _ = CO.add_volatile(st, dex, rng, tr, CO.cid("throatchop"), ok & (move == mid("throatchop")),
                                     source=u, source_effect=C.eff_move(move))
    if mid("alluringvoice") > 0:
        av = ok & (move == mid("alluringvoice")) & mget(st.mons.stats_raised_this_turn, tr)
        st, rng, _ = CO.add_volatile(st, dex, rng, tr, CO.cid("confusion"), av, source=u,
                                     source_effect=C.eff_move(move))
    return st, rng


# ------------------------------------------------------------------ Haze (onHitField)
if mid("haze") > 0:
    @handler(ids.HOOK_HITFIELD, int(C.eff_move(mid("haze"))), ps="data/moves.ts haze.onHitField")
    def haze_hit_field(hc, st, dex, rng):
        """'-clearallboost'; every active's boosts cleared."""
        st = EV.emit(st, True, EV.EV_CLEARALLBOOST)
        m = st.mons
        act = st.sides.active.reshape(4).astype(I32)
        refs = (jnp.arange(4) // 2) * 4 + jnp.maximum(act, 0)
        live = (act >= 0) & ~mget(m.fainted, refs)
        from vgcjax.battle.util import mset_many
        return hc, st.replace(mons=m.replace(boosts=mset_many(m.boosts, refs, 0, live))), rng

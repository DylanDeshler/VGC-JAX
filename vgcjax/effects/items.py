"""Items, first batch (data/items.ts; no Champions overrides for these): the 18 type-boosting items, Life Orb, Expert
Belt, Leftovers, Rocky Helmet, Focus Sash; Light Clay / weather rocks / Terrain Extender are read by the duration
callbacks (effects/field.py, weather.py, terrain.py). Mega Stones join with Mega Evolution (M3.6).

Item use (PS sim/pokemon.ts useItem): UseItem event (no Reg M-C handler yet), '-enditem' X item, lastItem = item,
item = '', usedItemThisTurn, AfterUseItem (none yet). Modifier callbacks are effects/modifiers.py mods; the others
run undispatched at their event sites (perf rule).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle.util import mget, mset
from vgcjax.data import ids
from vgcjax.effects.modifiers import mod, mod_family

I32 = jnp.int32
TYPE_ITEMS = {"blackbelt": "FIGHTING", "blackglasses": "DARK", "charcoal": "FIRE", "dragonfang": "DRAGON",
              "fairyfeather": "FAIRY", "hardstone": "ROCK", "magnet": "ELECTRIC", "metalcoat": "STEEL",
              "miracleseed": "GRASS", "mysticwater": "WATER", "nevermeltice": "ICE", "poisonbarb": "POISON",
              "sharpbeak": "FLYING", "silkscarf": "NORMAL", "silverpowder": "BUG", "softsand": "GROUND",
              "spelltag": "GHOST", "twistedspoon": "PSYCHIC"}


def iid(name):
    return getattr(ids, f"ITEM_{name.upper()}", -1)


def ieff(name):
    """EffectId of an item."""
    return ids.ITEM_BASE + iid(name) - 1


def holds(st, refs, name):
    return (mget(st.mons.item, jnp.clip(refs, 0, 7)) == iid(name)) if iid(name) > 0 else jnp.zeros(jnp.shape(refs),
                                                                                                    bool)


def use_item(st, ref, on):
    """PS: sim/pokemon.ts useItem (UseItem: no Reg M-C handler): '-enditem' X item; lastItem = item; item = '';
    usedItemThisTurn = true (itemState cleared); runEvent('AfterUseItem'): Unburden."""
    m = st.mons
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    it = mget(m.item, r).astype(I32)
    go = jnp.asarray(on) & (it > 0)
    st = EV.emit(st, go, EV.EV_ENDITEM, actor=r, move=ids.ITEM_BASE + it - 1)
    m = st.mons
    st = st.replace(mons=m.replace(last_item=mset(m.last_item, r, it, go), item=mset(m.item, r, 0, go),
                                   used_item_this_turn=mset(m.used_item_this_turn, r, True, go),
                                   item_state=mset(m.item_state, r, jnp.zeros_like(mget(m.item_state, r)), go),
                                   item_seq=mset(m.item_seq, r, 0, go)))      # clearEffectState: effectOrder = 0
    from vgcjax.effects import abilities as AB
    return AB.unburden(st, r, go)                                        # runEvent('AfterUseItem')


# ------------------------------------------------------------------ modifier items
_TYPE_OF_ITEM = [0] * (len(ids.ITEMS) + 1)
for _n, _t in TYPE_ITEMS.items():
    if iid(_n) > 0:
        _TYPE_OF_ITEM[iid(_n)] = getattr(ids, f"TYPE_{_t}")


@mod_family("BasePower", [ieff(n) for n in TYPE_ITEMS if iid(n) > 0], ps="data/items.ts <type item>.onBasePower")
def type_item_boost(ctx, h):
    """The 18 type-boosting items, onBasePower (priority 15): move.type === the item's type -> chainModify([4915,
    4096]) (table-driven family, ADR-106)."""
    it = mget(ctx.st.mons.item, jnp.clip(h, 0, 7)).astype(I32)
    ty = jnp.asarray(_TYPE_OF_ITEM, I32)[jnp.clip(it, 0, len(ids.ITEMS))]
    return (ty > 0) & (ctx.mtype == ty), jnp.asarray(4915, I32), None


if iid("lifeorb") > 0:
    mod("ModifyDamage", ieff("lifeorb"), ps="data/items.ts lifeorb.onModifyDamage")(
        lambda ctx, h: (jnp.ones(h.shape, bool), jnp.asarray(5324, I32), None))
if iid("expertbelt") > 0:
    mod("ModifyDamage", ieff("expertbelt"), ps="data/items.ts expertbelt.onModifyDamage")(
        lambda ctx, h: (ctx.extra["type_mod"] > 0, jnp.asarray(4915, I32), None))


# ------------------------------------------------------------------ event-site callbacks
def life_orb_recoil(st, dex, rng, u, target, move, on):
    """onAfterMoveSecondarySelf(source, target, move): source !== target, not a Status move, no forceSwitchFlag ->
    this.damage(baseMaxhp / 10, source, source, Life Orb): '-damage' X hp '[from] item: Life Orb'."""
    from vgcjax.effects import statuses as STS
    m = st.mons
    go = jnp.asarray(on) & holds(st, u, "lifeorb") & (target != u) & (dex.move_category[move] != ids.CAT_STATUS) & \
        ~mget(m.force_switch_flag, u)
    return STS.effect_damage(st, dex, rng, u, mget(m.maxhp, u) // 10, ieff("lifeorb"), go, source=u)


def rocky_helmet(st, dex, rng, targets, damaged, u, move):
    """onDamagingHit (order 2, after Rough Skin): contact -> this.damage(source.baseMaxhp / 6, source, target):
    '-damage' attacker hp '[from] item: Rocky Helmet' '[of] holder'."""
    from vgcjax.effects import statuses as STS
    w, b = ids.MF_CONTACT
    contact = ((dex.move_flags[move, w] >> b) & 1) == 1
    for j in range(targets.shape[0]):
        t = jnp.clip(targets[j], 0, 7)
        go = damaged[j] & contact & holds(st, t, "rockyhelmet")
        st, rng = STS.effect_damage(st, dex, rng, u, mget(st.mons.maxhp, u) // 6, ieff("rockyhelmet"), go, source=t)
    return st, rng


def focus_sash(st, targets, amt, go):
    """runEvent('Damage', target, source, move) (Focus Sash onDamage, priority -40): at full HP a move's damage >=
    hp -> useItem, damage = hp - 1. Returns (amount [T], used [T]); the caller applies useItem in target order."""
    m = st.mons
    t = jnp.clip(targets, 0, 7)
    hp, mx = mget(m.hp, t).astype(I32), mget(m.maxhp, t).astype(I32)
    used = go & holds(st, t, "focussash") & (hp == mx) & (amt >= hp)
    return jnp.where(used, hp - 1, amt), used


def residual(st, dex, rng, h, on):
    """The items' onResidual (fieldEvent walk, undispatched): Leftovers (order 5, sub 4: heal baseMaxhp / 16,
    '-heal' X hp '[from] item: Leftovers')."""
    from vgcjax.effects import statuses as STS
    r = jnp.clip(h, 0, 7)
    go = jnp.asarray(on) & holds(st, r, "leftovers")
    st, rng = STS.effect_heal(st, dex, rng, r, mget(st.mons.maxhp, r) // 16, ieff("leftovers"), go)
    return white_herb(st, dex, rng, r, on)                              # White Herb onResidual (order 29)


if iid("leftovers") > 0:
    from vgcjax.effects.hooks import handler as _handler
    _handler(ids.HOOK_RESIDUAL, ieff("leftovers"), ps="data/items.ts leftovers.onResidual (items.residual)")(
        lambda hc, st, dex, rng: (hc, st, rng))


# ------------------------------------------------------------------ terrain seeds (useItem with boosts)
SEEDS = {"grassyseed": ("grassyterrain", 1), "electricseed": ("electricterrain", 1),
         "psychicseed": ("psychicterrain", 3), "mistyseed": ("mistyterrain", 3)}   # boosts: def (1) / spd (3) +1


def seed(st, dex, rng, r, on):
    """A terrain seed's onStart / onTerrainChange: the field's terrain is the seed's -> pokemon.useItem(): '-enditem',
    boost(item.boosts, pokemon, pokemon, item) (not isSecondary / isSelf: a capped stat still prints '-boost' 0),
    Use, item consumed, AfterUseItem (Unburden). Returns (st, rng)."""
    from vgcjax.battle import move_pipeline as MP
    from vgcjax.effects import conditions as CO
    rr = jnp.clip(jnp.asarray(r, I32), 0, 7)
    m = st.mons
    it = mget(m.item, rr).astype(I32)
    t = st.field.terrain.astype(I32)
    go = jnp.zeros((), bool)
    stat = jnp.int32(1)
    for name, (terr, k) in SEEDS.items():
        if iid(name) > 0 and terr in ids.CONDITIONS:
            hit = (it == iid(name)) & (t == CO.cid(terr))
            go = go | hit
            stat = jnp.where(hit, k, stat)
    go = jnp.asarray(on) & go & (mget(m.hp, rr) > 0) & mget(m.is_active, rr)
    eff = ids.ITEM_BASE + it - 1
    st = use_item(st, rr, go)
    st, _ = MP.boost(st, rr, (jnp.arange(7) == stat).astype(I32), go, primary=True, source=rr, effect=eff,
                     simple=True)                                       # self-sourced
    return st, rng


# the seeds' onStart rows must be fieldEvent('SwitchIn') candidates (effects/static.py lists registered handlers);
# their work runs undispatched in the walk body (fieldevent.py -> seed)
from vgcjax.effects.hooks import handler as _seed_handler  # noqa: E402
for _n in SEEDS:
    if iid(_n) > 0:
        _seed_handler(ids.HOOK_SWITCHIN, ieff(_n), ps=f"data/items.ts {_n}.onStart (items.seed)")(
            lambda hc, st, dex, rng: (hc, st, rng))


def terrain_change(st, dex, rng, on):
    """eachEvent('TerrainChange') (actives by speed; ties are a don't-care kind): the seeds' onTerrainChange."""
    if not any(iid(n) > 0 for n in SEEDS):
        return st, rng
    act = st.sides.active.reshape(4).astype(I32)
    refs = jnp.clip((jnp.arange(4) // 2) * 4 + act, 0, 7)
    alive = (act >= 0) & ~mget(st.mons.fainted, refs)
    spd = jnp.where(alive, mget(st.mons.speed, refs).astype(I32), -(1 << 20))
    order = jnp.argsort(-spd, stable=True)
    for i in range(4):
        st, rng = seed(st, dex, rng, refs[order[i]], jnp.asarray(on) & alive[order[i]])
    return st, rng


# ------------------------------------------------------------------ Eject Button (Champions version)
if iid("ejectbutton") > 0:
    from vgcjax.effects.hooks import handler as _handler

    @_handler(ids.HOOK_AFTERMOVESECONDARY, ieff("ejectbutton"),
              ps="data/mods/champions/items.ts ejectbutton.onAfterMoveSecondary")
    def ejectbutton_after_move_secondary(hc, st, dex, rng):
        """priority 2: hit by another Pokemon's non-Status move while it has hp: canSwitch, not already forced out /
        called back, no active with switchFlag === true -> switchFlag = true; useItem() (Champions: the attacker's
        switch is not cancelled)."""
        from vgcjax.battle import faint as F
        from vgcjax.battle.util import mset
        t, s = hc.target, hc.source
        tt = jnp.clip(t, 0, 7)
        m = st.mons
        move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
        act = st.sides.active.reshape(4).astype(I32)
        refs = jnp.clip((jnp.arange(4) // 2) * 4 + act, 0, 7)
        pending = jnp.any((act >= 0) & (mget(m.switch_flag, refs) == 1))
        go = (s >= 0) & (s != t) & (mget(m.hp, tt) > 0) & (dex.move_category[move] != ids.CAT_STATUS) & \
            (F.can_switch(st, tt // 4) > 0) & ~mget(m.force_switch_flag, tt) & ~mget(m.being_called_back, tt) & \
            ~pending & (mget(m.item, tt) == iid("ejectbutton"))
        st = st.replace(mons=m.replace(switch_flag=mset(m.switch_flag, tt, 1, go)))
        st = use_item(st, tt, go)
        return hc, st, rng


# ------------------------------------------------------------------ White Herb
def white_herb(st, dex, rng, r, on):
    """White Herb's onStart / onAnySwitchIn / onAnyAfterMega / onAnyAfterMove / onResidual: any negative boost ->
    useItem(): '-enditem' X White Herb, Use: setBoost(negative stages to 0), '-clearnegativeboost' X '[silent]',
    item consumed, AfterUseItem (Unburden). Returns (st, rng)."""
    from vgcjax.battle.util import mset
    if iid("whiteherb") <= 0:
        return st, rng
    rr = jnp.clip(jnp.asarray(r, I32), 0, 7)
    m = st.mons
    b = mget(m.boosts, rr)
    go = jnp.asarray(on) & holds(st, rr, "whiteherb") & jnp.any(b < 0) & (mget(m.hp, rr) > 0) & mget(m.is_active, rr)
    st = use_item(st, rr, go)
    m = st.mons
    st = st.replace(mons=m.replace(boosts=mset(m.boosts, rr, jnp.where(b < 0, 0, b).astype(b.dtype), go)))
    return EV.emit(st, go, EV.EV_CLEARNEGATIVEBOOST, actor=rr), rng


if iid("whiteherb") > 0:
    from vgcjax.effects.hooks import handler as _wh_handler
    # onAnySwitchIn rows of the switch-in walk (work in fieldevent.py -> white_herb); its onStart is not a SwitchIn
    # handler (PS getCallback skips onStart when onAnySwitchIn exists) and runs only from setItem (Trick)
    _wh_handler(ids.HOOK_SWITCHIN, ieff("whiteherb"), prefix=ids.PFX_ANY,
                ps="data/items.ts whiteherb.onAnySwitchIn (items.white_herb)")(lambda hc, st, dex, rng: (hc, st, rng))
    _wh_handler(ids.HOOK_RESIDUAL, ieff("whiteherb"), ps="data/items.ts whiteherb.onResidual (items.residual)")(
        lambda hc, st, dex, rng: (hc, st, rng))

    def _wh_any(hc, st, dex, rng):
        """onAnyAfterMove / onAnyAfterMega: the holder's onStart check."""
        st, rng = white_herb(st, dex, rng, hc.holder, True)
        return hc, st, rng
    _wh_handler(ids.HOOK_AFTERMOVE, ieff("whiteherb"), prefix=ids.PFX_ANY,
                ps="data/items.ts whiteherb.onAnyAfterMove")(_wh_any)
    _wh_handler(ids.HOOK_AFTERMEGA, ieff("whiteherb"), prefix=ids.PFX_ANY,
                ps="data/items.ts whiteherb.onAnyAfterMega")(_wh_any)

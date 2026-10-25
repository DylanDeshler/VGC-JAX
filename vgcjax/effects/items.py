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
    """PS: sim/pokemon.ts useItem (UseItem / AfterUseItem: no Reg M-C handler): '-enditem' X item; lastItem = item;
    item = ''; usedItemThisTurn = true (itemState cleared)."""
    m = st.mons
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    it = mget(m.item, r).astype(I32)
    go = jnp.asarray(on) & (it > 0)
    st = EV.emit(st, go, EV.EV_ENDITEM, actor=r, move=ids.ITEM_BASE + it - 1)
    m = st.mons
    return st.replace(mons=m.replace(last_item=mset(m.last_item, r, it, go), item=mset(m.item, r, 0, go),
                                     used_item_this_turn=mset(m.used_item_this_turn, r, True, go),
                                     item_state=mset(m.item_state, r, jnp.zeros_like(mget(m.item_state, r)), go),
                                     item_seq=mset(m.item_seq, r, 0, go)))   # clearEffectState: effectOrder = 0


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
    return STS.effect_heal(st, dex, rng, r, mget(st.mons.maxhp, r) // 16, ieff("leftovers"), go)


if iid("leftovers") > 0:
    from vgcjax.effects.hooks import handler as _handler
    _handler(ids.HOOK_RESIDUAL, ieff("leftovers"), ps="data/items.ts leftovers.onResidual (items.residual)")(
        lambda hc, st, dex, rng: (hc, st, rng))

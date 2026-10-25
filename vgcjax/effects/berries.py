"""Berries (data/items.ts; no Champions overrides): Sitrus Berry (onUpdate) and the 18 type-resist berries
(onSourceModifyDamage). PS: sim/pokemon.ts eatItem, sim/battle.ts eachEvent('Update').

eatItem: UseItem (none), TryEatItem (a foe's Unnerve: onFoeTryEatItem -> !unnerved; Sitrus: TryHeal, no Reg M-C
handler), '-enditem' X item '[eat]', singleEvent('Eat'), EatItem (none), lastItem = item, item = '', itemState cleared
(effectOrder 0), usedItemThisTurn, ateBerry, AfterUseItem (none).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle.util import mget, mset
from vgcjax.data import ids
from vgcjax.effects.items import holds, ieff, iid
from vgcjax.effects.modifiers import mod_family

I32 = jnp.int32
RESIST = {"babiriberry": "STEEL", "chartiberry": "ROCK", "chopleberry": "FIGHTING", "cobaberry": "FLYING",
          "colburberry": "DARK", "habanberry": "DRAGON", "kasibberry": "GHOST", "kebiaberry": "POISON",
          "occaberry": "FIRE", "passhoberry": "WATER", "payapaberry": "PSYCHIC", "rindoberry": "GRASS",
          "roseliberry": "FAIRY", "shucaberry": "GROUND", "tangaberry": "BUG", "wacanberry": "ELECTRIC",
          "yacheberry": "ICE"}
_RESIST_TYPE = [0] * (len(ids.ITEMS) + 1)
for _n, _t in RESIST.items():
    if iid(_n) > 0:
        _RESIST_TYPE[iid(_n)] = getattr(ids, f"TYPE_{_t}")


def unnerved(st, ref):
    """A foe active with Unnerve whose effectState.unnerved is set (onFoeTryEatItem returns false)."""
    from vgcjax.effects.abilities import aid
    if aid("unnerve") <= 0:
        return jnp.zeros(jnp.shape(ref), bool)
    r = jnp.clip(ref, 0, 7)
    fs = 1 - r // 4
    out = jnp.zeros(jnp.shape(ref), bool)
    for slot in range(2):
        b = st.sides.active.astype(I32)[fs, slot]
        f = fs * 4 + jnp.maximum(b, 0)
        s = mget(st.mons.ability_state, f)                       # [..., 4]; unnerved = slot 0, presence bit 0
        on = (b >= 0) & ~mget(st.mons.fainted, f) & (mget(st.mons.hp, f) > 0) & \
            (mget(st.mons.ability, f) == aid("unnerve")) & \
            (s[..., 0] != 0) & ((s[..., 3] & 1) == 1)
        out = out | on
    return out


def can_eat(st, ref):
    """eatItem's gates before '-enditem': holds an item, hp > 0, active, not unnerved."""
    r = jnp.clip(ref, 0, 7)
    m = st.mons
    return (mget(m.item, r) > 0) & (mget(m.hp, r) > 0) & mget(m.is_active, r) & ~unnerved(st, r)


def eat_item(st, ref, on):
    """The state / message half of eatItem once its events passed: '-enditem' X item '[eat]', item consumed
    (ateBerry: Cud Chew / Belch, not tracked yet)."""
    m = st.mons
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    it = mget(m.item, r).astype(I32)
    go = jnp.asarray(on) & (it > 0)
    st = EV.emit(st, go, EV.EV_ENDITEM, actor=r, move=ids.ITEM_BASE + it - 1, flags=EV.F_EAT)
    m = st.mons
    return st.replace(mons=m.replace(last_item=mset(m.last_item, r, it, go), item=mset(m.item, r, 0, go),
                                     used_item_this_turn=mset(m.used_item_this_turn, r, True, go),
                                     item_state=mset(m.item_state, r, jnp.zeros_like(mget(m.item_state, r)), go),
                                     item_seq=mset(m.item_seq, r, 0, go)))


# ------------------------------------------------------------------ resist berries (ModifyDamage, Source prefix)
@mod_family("ModifyDamage", [ieff(n) for n in RESIST if iid(n) > 0], ps="data/items.ts <resist berry>.onSourceModifyDamage",
            prefix=ids.PFX_SOURCE)
def resist_berry(ctx, h):
    """onSourceModifyDamage (the defender's): the berry's type, super effective (Substitute: M3), eatItem ->
    chainModify(0.5). Pure part only; resist_berry_eat performs the eat (same conditions) after the event."""
    it = mget(ctx.st.mons.item, jnp.clip(h, 0, 7)).astype(I32)
    ty = jnp.asarray(_RESIST_TYPE, I32)[jnp.clip(it, 0, len(ids.ITEMS))]
    ok = (ty > 0) & (ctx.mtype == ty) & (ctx.extra["type_mod"] > 0) & can_eat(ctx.st, h)
    return ok, jnp.asarray(2048, I32), None


def resist_berry_applies(st, targets, mtype, type_mod, reach):
    it = mget(st.mons.item, jnp.clip(targets, 0, 7)).astype(I32)
    ty = jnp.asarray(_RESIST_TYPE, I32)[jnp.clip(it, 0, len(ids.ITEMS))]
    return reach & (ty > 0) & (mtype == ty) & (type_mod > 0) & can_eat(st, targets)


def resist_berry_eat(st, ref, on):
    """eatItem ('-enditem' X berry '[eat]') then '-enditem' X berry '[weaken]'."""
    it = mget(st.mons.item, jnp.clip(ref, 0, 7)).astype(I32)
    st = eat_item(st, ref, on)
    return EV.emit(st, on, EV.EV_ENDITEM, actor=ref, move=ids.ITEM_BASE + it - 1, flags=EV.F_WEAKEN)


# ------------------------------------------------------------------ eachEvent('Update')
def update(st, dex, rng, on):
    """eachEvent('Update') (sim/battle.ts:470-480; speed-sorted actives, ties a don't-care kind): Sitrus Berry
    onUpdate (hp <= maxhp / 2 -> eatItem -> Eat: heal(baseMaxhp / 4), '-heal' X hp '[from] item: Sitrus Berry')."""
    from vgcjax.effects import statuses as STS
    if iid("sitrusberry") <= 0:
        return st, rng
    on = jnp.asarray(on)
    act = st.sides.active.reshape(4).astype(I32)
    refs = jnp.clip((jnp.arange(4) // 2) * 4 + act, 0, 7)
    alive = (act >= 0) & ~mget(st.mons.fainted, refs)
    spd = jnp.where(alive, mget(st.mons.speed, refs).astype(I32), -(1 << 20))
    order = jnp.argsort(-spd, stable=True)
    for i in range(4):
        r = refs[order[i]]
        m = st.mons
        hp, mx = mget(m.hp, r).astype(I32), mget(m.maxhp, r).astype(I32)
        go = on & alive[order[i]] & holds(st, r, "sitrusberry") & (hp * 2 <= mx) & can_eat(st, r)
        st = eat_item(st, r, go)
        st, rng = STS.effect_heal(st, dex, rng, r, mx // 4, ieff("sitrusberry"), go)
    return st, rng

"""Mega Evolution (05 §4 step 9, M3.6). PS: sim/battle-actions.ts:1952-1970 runMegaEvo, [C]
data/mods/champions/scripts.ts:56-115 formeChange (permanent: no revert on faint), sim/pokemon.ts setSpecies /
updateMaxHp / setAbility, [C] scripts.ts:182-194 canMegaEvo (item.megaStone[species.name]).

runMegaEvo: formeChange(mega, item, true): setSpecies (types, weight, stats, speed = stored Spe); baseSpecies = mega;
'detailschange' X details; updateMaxHp; '-mega' X <base species> <item>; moveThisTurnResult = true; setAbility(mega
ability, isFromFormeChange): End old ability, new abilityState (effectOrder++), Start new ability; baseAbility.
Then canMegaEvo = false for the whole side; runEvent('AfterMega') (no Reg M-C handler yet).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset
from vgcjax.core import stats as ST

I16, I32 = jnp.int16, jnp.int32


def mega_species(st, dex, ref):
    """The Mega forme for ref's held stone (0 none): species_mega[base species] = [(stone, mega)] x2."""
    m = st.mons
    r = jnp.clip(ref, 0, 7)
    sp = mget(m.base_species, r).astype(I32)
    it = mget(m.item, r).astype(I32)
    mg = dex.species_mega[sp].astype(I32)                                # [2, 2]
    hit = (mg[:, 0] == it) & (it > 0) & (mg[:, 1] > 0)
    return jnp.where(jnp.any(hit), mg[jnp.argmax(hit), 1], 0)


def mega_evolve(st, dex, rng, ref, on):
    """runMegaEvo(pokemon). Returns (st, rng)."""
    from vgcjax.battle import switching as SW
    from vgcjax.effects import abilities as AB
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    mega = mega_species(st, dex, r)
    go = jnp.asarray(on) & (mega > 0) & mget(st.mons.can_mega, r)
    m = st.mons
    old_maxhp, old_hp = mget(m.maxhp, r).astype(I32), mget(m.hp, r).astype(I32)
    # setSpecies + baseSpecies
    st = SW.set_species(st, dex, r, mega, go)
    m = st.mons
    st = st.replace(mons=m.replace(base_species=mset(m.base_species, r, mega, go)))
    st = EV.emit(st, go, EV.EV_DETAILSCHANGE, actor=r, v1=mega)
    # updateMaxHp: a different base HP rescales hp keeping the damage taken ('-heal' X hp '[silent]')
    nat = mget(m.nature, r).astype(I32)
    new_max, _ = ST.calc_stats(dex.species_base_stats[mega], mget(m.sp, r).astype(I32), dex.nature_plus[nat],
                               dex.nature_minus[nat])
    new_max = new_max.astype(I32)
    chg = go & (new_max != old_maxhp)
    new_hp = jnp.where(old_hp <= 0, 0, jnp.maximum(1, new_max - (old_maxhp - old_hp)))
    m = st.mons
    st = st.replace(mons=m.replace(maxhp=mset(m.maxhp, r, new_max, chg), hp=mset(m.hp, r, new_hp, chg)))
    st = EV.emit_hp(st, chg & (new_hp > 0), EV.EV_HEAL, r)        # '-heal' X hp '[silent]'
    st = EV.emit(st, go, EV.EV_MEGA, actor=r, move=ids_item_eff(mget(st.mons.item, r)))
    m = st.mons
    st = st.replace(mons=m.replace(move_this_turn_result=mset(m.move_this_turn_result, r, S.MR_TRUE, go)))
    # setAbility(species.abilities[0], null, null, true)
    new_ab = dex.species_abilities[mega, 0].astype(I32)
    alive = go & (mget(st.mons.hp, r) > 0)
    st = AB.end_vec(st, r[None], alive[None])
    m = st.mons
    meta = st.meta
    st = st.replace(mons=m.replace(ability=mset(m.ability, r, new_ab, alive),
                                   base_ability=mset(m.base_ability, r, new_ab, go),
                                   ability_state=mset(m.ability_state, r, jnp.zeros(4, I16), alive),
                                   ability_seq=mset(m.ability_seq, r, meta.effect_seq, alive)),
                    meta=meta.replace(effect_seq=meta.effect_seq + alive.astype(meta.effect_seq.dtype)))
    st, rng = AB.switch_in_vec(st, dex, rng, r, alive, start=True)   # singleEvent('Start', ability)
    # Limit one Mega Evolution per side
    side = r // 4
    m = st.mons
    cm = m.can_mega.at[side].set(jnp.where(go, False, m.can_mega[side]))
    sides = st.sides
    st = st.replace(mons=m.replace(can_mega=cm),
                    sides=sides.replace(mega_used=sides.mega_used.at[side].set(sides.mega_used[side] | go)))
    return st, rng


def ids_item_eff(item):
    from vgcjax.data import ids
    return ids.ITEM_BASE + jnp.asarray(item, I32) - 1

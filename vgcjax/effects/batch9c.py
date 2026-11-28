"""Batch 9c: Wandering Spirit (Runerigus), Mummy (Cofagrigus), Seed Sower (Arboliva), Heavy Metal, Aroma Veil, Tangled
Feet, Symbiosis, Illuminate, Slush Rush, Analytic, Curious Medicine (abilities); King's Rock, Persim Berry (items).

PS: data/abilities.ts, data/items.ts (no Champions override: data/mods/champions/{abilities,items}.ts mention none of
them), sim/battle.ts skillSwap / checkMoveMakesContact, sim/pokemon.ts setAbility / takeItem / setItem / getWeight.
MUL-hook callbacks are effects/modifiers.py mods; the others are undispatched helpers called from one-line hooks at
their PS event sites (perf rule):
- Wandering Spirit / Mummy / Seed Sower onDamagingHit: battle/move_pipeline.py spread_move_hit, per target after the
  target's other ability handlers and before its item's (Air Balloon) and the source's onSource ones (Poison Touch):
  DamagingHit handlers sort by order / priority / target index / insertion (sim/battle.ts:796-797, :426-431);
  `damaging_hit` (inside the target fori_loop: one traced copy, incl. the skill-swapped ability's Start);
- Heavy Metal onModifyWeight: `weight` (getWeight), read by effects/move_callbacks.py base_power;
- Aroma Veil onAllyTryAddVolatile: effects/conditions.py add_volatile (`aroma_veil`);
- Tangled Feet onModifyAccuracy, Analytic onBasePower: mods below;
- Symbiosis onAllyAfterUseItem: `symbiosis`, from the runEvent('AfterUseItem') sites (items.use_item,
  berries.eat_item, the terrain seeds, Air Balloon);
- Illuminate onTryBoost: effects/ability_blocks.py try_boost (as Keen Eye); onModifyMove ignoreEvasion:
  battle/move_pipeline.py accuracy (`ignores_evasion`);
- Slush Rush onModifySpe: battle/common.py spe_chain (the weather-speed loop);
- Curious Medicine onStart: effects/abilities.py switch_in_vec (`curious_medicine`);
- King's Rock onModifyMove: battle/move_pipeline.py secondaries (`kings_rock`);
- Persim Berry onUpdate / onEat: berries.update (`persim_update`), Bug Bite / Cud Chew eat effects.
"""
from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle.util import mget, mset, mset_many
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.abilities import aeff, aid
from vgcjax.effects.items import holds, iid
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32


def _is(ab, name):
    return ab == aid(name) if aid(name) > 0 else jnp.zeros(jnp.shape(ab), bool)


def _aflag(dex, ability, bit):
    return (dex.ability_flags[jnp.maximum(jnp.asarray(ability, I32), 0)].astype(I32) & bit) != 0


def _contact(dex, move):
    w, b = ids.MF_CONTACT
    return ((dex.move_flags[move, w] >> b) & 1) == 1


def _ally(st, r):
    """The other active slot of active r's side: (ref, present: side.allies() keeps hp > 0, sim/side.ts:390-396)."""
    side, b = r // 4, r % 4
    row = st.sides.active[side].astype(I32)
    a = jnp.where(row[0] == b, row[1], row[0])
    h = side * 4 + jnp.maximum(a, 0)
    return h, ((row[0] == b) | (row[1] == b)) & (a >= 0) & (mget(st.mons.hp, h) > 0)


# ------------------------------------------------------------------ Mummy / Wandering Spirit / Seed Sower
def damaging_hit(st, dex, rng, t, damaged, u, move):
    """The target's own DamagingHit ability handlers of this batch (one ability per mon, so mutually exclusive):
      Seed Sower (data/abilities.ts:4120-4128): field.setTerrain('grassyterrain') (sim/field.ts setTerrain: source =
        event.target = the holder, sourceEffect = the ability: '-fieldstart' 'move: Grassy Terrain' '[from] ability:
        Seed Sower' '[of] holder'; nothing when Grassy Terrain is up already). No contact needed; damage 0 counts.
      Mummy (:2780-2794): the source's current ability (source.getAbility()) not cantsuppress and not Mummy, contact
        (checkMoveMakesContact, sim/battle.ts:1303-1312: no Protective Pads in Reg M-C) -> source.setAbility('mummy',
        target) (sim/pokemon.ts:1929-1967: source hp > 0; no SetAbility handler (Ability Shield) in Reg M-C;
        End(old ability) (effects/abilities.py end_vec); new ability + abilityState (effectOrder++); '-activate'
        holder 'ability: Mummy' source '[ability] Old' (ACTIVATE); Start(Mummy): no handler).
      Wandering Spirit (:5356-5364): contact -> skillSwap(source, target) (sim/battle.ts:1325-1358: either fainted /
        either ability failskillswap -> false (effects/batch_copy.py skill_swap_ok); '-activate' source 'Skill
        Swap' ... '[of] target' (ACTIVATE); End(source ability), End(target ability); abilities swapped, new
        abilityStates (source first); Start(source's old ability on the target), Start(Wandering Spirit on the
        source: no handler): effects/abilities.py switch_in_vec(start=True)).
    Mold Breaker (effects/batch_reactive.py masking): none of the three is breakable, so the holder is never masked,
    and the source is the mover (never masked); a breakable ability the target receives is masked again while the
    suppression is live (mold_remask). Called once per target inside the DamagingHit target fori_loop (one traced
    copy of switch_in_vec). Returns (st, rng)."""
    from vgcjax.effects import abilities as AB
    from vgcjax.effects import batch_copy as BC
    from vgcjax.effects import batch_reactive as BR
    z = jnp.zeros((), bool)
    if aid("seedsower") <= 0 and aid("mummy") <= 0 and aid("wanderingspirit") <= 0:
        return st, rng
    t = jnp.clip(jnp.asarray(t, I32), 0, 7)
    u = jnp.clip(jnp.asarray(u, I32), 0, 7)
    damaged = jnp.asarray(damaged)
    m = st.mons
    ab_t = mget(m.ability, t).astype(I32)
    contact = _contact(dex, move)
    if aid("seedsower") > 0 and "grassyterrain" in ids.CONDITIONS:
        from vgcjax.effects import terrain as TR
        ss = damaged & _is(ab_t, "seedsower")
        st, rng, _ = TR.set_terrain(st, dex, rng, CO.cid("grassyterrain"), ss, source=t,
                                    source_effect=aeff("seedsower"))
    m = st.mons
    sa = mget(m.ability, u).astype(I32)
    mu = z
    if aid("mummy") > 0:
        mu = damaged & contact & _is(ab_t, "mummy") & ~_aflag(dex, sa, ids.AF_CANTSUPPRESS) & (sa != aid("mummy")) & \
            (mget(m.hp, u) > 0)
    ws = z
    if aid("wanderingspirit") > 0:
        ws = damaged & contact & _is(ab_t, "wanderingspirit") & BC.skill_swap_ok(st, dex, u, t)
        st = EV.emit(st, ws, EV.EV_ACTIVATE, actor=u, move=C.eff_move(ids.MOVE_SKILLSWAP))
    st = AB.end_vec(st, jnp.stack([u, t]), jnp.stack([mu | ws, ws]))
    st = BC._new_ability(st, u, jnp.where(mu, aid("mummy"), ab_t), mu | ws)
    st = BC._new_ability(st, t, sa, ws)
    if aid("mummy") > 0:
        st = EV.emit(st, mu, EV.EV_ACTIVATE, actor=t, move=aeff("mummy"))
    st = BR.mold_remask(st, t[None], ws)
    return AB.switch_in_vec(st, dex, rng, t, ws, start=True)


# ------------------------------------------------------------------ Heavy Metal
def weight(st, refs):
    """pokemon.getWeight() (sim/pokemon.ts:706-709): runEvent('ModifyWeight', weighthg): Heavy Metal onModifyWeight
    (data/abilities.ts:1859-1868): weighthg * 2 (breakable: a Mold Breaker mover's target has its ability masked);
    max(1, weighthg). No Light Metal / Float Stone in the impl stage. refs [...] -> i32 [...]."""
    r = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    w = mget(st.mons.weight_hg, r).astype(I32)
    w = jnp.where(_is(mget(st.mons.ability, r), "heavymetal"), w * 2, w)
    return jnp.maximum(w, 1)


# ------------------------------------------------------------------ Aroma Veil
AROMA_VEIL = ("attract", "disable", "encore", "healblock", "taunt", "torment")


def aroma_veil(st, ref, c, source_effect):
    """Aroma Veil onAllyTryAddVolatile (data/abilities.ts:234-248; onAlly covers the holder itself: alliesAndSelf):
    status in [attract, disable, encore, healblock, taunt, torment] -> null; effect.effectType === 'Move' ->
    '-block' target 'ability: Aroma Veil' '[of] holder' (BLOCK). Breakable (Mold Breaker masking). The fastest
    holder's handler answers (holders sort by speed; a tie keeps slot order: a don't-care speed tie). Returns
    (blocked, message, holder)."""
    conds = [CO.cid(n) for n in AROMA_VEIL if n in ids.CONDITIONS]
    z = jnp.zeros((), bool)
    if aid("aromaveil") <= 0 or not conds:
        return z, z, jnp.asarray(-1, I32)
    m = st.mons
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    c = jnp.asarray(c, I32)
    hit = jnp.any(jnp.asarray(conds, I32) == c)
    side = r // 4
    best = jnp.asarray(-(1 << 20), I32)
    of = jnp.asarray(-1, I32)
    for b in range(2):
        a = st.sides.active[side, b].astype(I32)
        h = side * 4 + jnp.maximum(a, 0)
        ok = (a >= 0) & ~mget(m.fainted, h) & (mget(m.hp, h) > 0) & _is(mget(m.ability, h), "aromaveil")
        s = mget(m.speed, h).astype(I32)
        win = ok & (s > best)
        best = jnp.where(win, s, best)
        of = jnp.where(win, h, of)
    blocked = hit & (of >= 0)
    se = jnp.asarray(source_effect, I32)
    is_move = (se >= CO.MOVE_RANGE[0]) & (se <= CO.MOVE_RANGE[1])
    return blocked, blocked & is_move, of


# ------------------------------------------------------------------ Tangled Feet
if aid("tangledfeet") > 0 and "confusion" in ids.CONDITIONS:
    @mod("ModifyAccuracy", aeff("tangledfeet"), ps="data/abilities.ts:4892-4905 tangledfeet.onModifyAccuracy")
    def tangledfeet_acc(ctx, h):
        """onModifyAccuracy (priority -1, the defender's): numeric accuracy and target.volatiles['confusion'] ->
        chainModify(0.5). Breakable (Mold Breaker masking)."""
        conf = jax.vmap(lambda r: CO.has_volatile(ctx.st, r, CO.cid("confusion")))(jnp.clip(h, 0, 7))
        return ctx.extra["numeric"] & conf, jnp.asarray(2048, I32), None


# ------------------------------------------------------------------ Analytic
if aid("analytic") > 0:
    @mod("BasePower", aeff("analytic"), ps="data/abilities.ts:110-130 analytic.onBasePower")
    def analytic_bp(ctx, h):
        """onBasePower (priority 21), the attacker's: no other unfainted active (getAllActive, sim/battle.ts:
        1379-1389) still has a queued move action (queue.willMove: fainted -> null, sim/battle-queue.ts:324-332) ->
        chainModify([5325, 4096])."""
        from vgcjax.effects import disables as DS
        st = ctx.st
        act = st.sides.active.reshape(4).astype(I32)
        refs = (jnp.arange(4, dtype=I32) // 2) * 4 + jnp.maximum(act, 0)
        live = (act >= 0) & ~mget(st.mons.fainted, refs)
        will = live & jax.vmap(lambda r: DS.will_move(st, r))(refs)                     # [4]
        hh = jnp.clip(h, 0, 7)
        other = jnp.any(will[None, :] & (refs[None, :] != hh[:, None]), axis=-1)
        return ~other, jnp.asarray(5325, I32), None


# ------------------------------------------------------------------ Illuminate
def ignores_evasion(st, u):
    """Illuminate onModifyMove (data/abilities.ts:2051-2053): move.ignoreEvasion = true."""
    return _is(mget(st.mons.ability, jnp.clip(jnp.asarray(u, I32), 0, 7)), "illuminate")


# ------------------------------------------------------------------ Curious Medicine
if aid("curiousmedicine") > 0:
    from vgcjax.effects.hooks import handler

    @handler(ids.HOOK_SWITCHIN, aeff("curiousmedicine"), ps="data/abilities.ts:772-783 curiousmedicine.onStart "
             "(switch_in_vec)")
    def _curious_medicine_row(hc, st, dex, rng):
        return hc, st, rng                       # listed for the SwitchIn walk; runs in abilities.switch_in_vec


def curious_medicine(st, r, ab, on):
    """Curious Medicine onStart (data/abilities.ts:772-783): for each adjacentAllies() (sim/pokemon.ts:745-747: hp
    > 0, not fainted, the other slot of the side) -> clearBoosts() (sim/pokemon.ts:1253-1258), '-clearboost' ally
    '[from] ability: Curious Medicine' '[of] pokemon' (CLEARBOOST). Runs for SwitchIn and Start alike."""
    if aid("curiousmedicine") <= 0:
        return st
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    h, ok = _ally(st, r)
    go = jnp.asarray(on) & (ab == aid("curiousmedicine")) & ok & ~mget(st.mons.fainted, r)
    m = st.mons
    st = st.replace(mons=m.replace(boosts=mset(m.boosts, h, jnp.zeros_like(mget(m.boosts, h)), go)))
    return EV.emit(st, go, EV.EV_CLEARBOOST, actor=h)


# ------------------------------------------------------------------ Symbiosis
_SYM_TRACING = [False]


@contextlib.contextmanager
def no_symbiosis():
    """Trace-time scope in which `symbiosis` is a no-op: a caller that runs several possible item uses for one mon
    (berries.update's body) calls it once afterwards (an item gone = used: the same lines, fewer traced copies)."""
    prev = _SYM_TRACING[0]
    _SYM_TRACING[0] = True
    try:
        yield
    finally:
        _SYM_TRACING[0] = prev


def symbiosis_many(st, refs, on):
    """`symbiosis` for refs [N] in order, one traced copy (fori_loop)."""
    if aid("symbiosis") <= 0 or _SYM_TRACING[0]:
        return st
    refs = jnp.asarray(refs, I32)
    on = jnp.asarray(on)
    return jax.lax.fori_loop(0, refs.shape[0], lambda i, s: symbiosis(s, None, refs[i], on[i]), st)


def symbiosis(st, dex, r, on):  # dex unused (callers without one pass None)
    """Symbiosis onAllyAfterUseItem (data/abilities.ts:4839-4858), r = the item's user (`on`: its useItem / eatItem
    succeeded). The holder = r's active ally (alliesAndSelf: hp > 0; the holder's own use is a no-op, its item is
    gone). pokemon.switchFlag -> nothing. source.takeItem() (sim/pokemon.ts:1872-1887): no item -> nothing;
    runEvent('TakeItem', holder): a Mega Stone of the holder's species vetoes (moves_b1.takeable; no Unburden /
    Sticky Hold on a Symbiosis holder: Floette-Eternal holding Floettite keeps it); item = '', itemState cleared.
    Then singleEvent('TakeItem', item, ..., pokemon) (a Mega Stone the receiver could use: false; not modelled, no
    Symbiosis holder carries its ally's Mega Stone) and pokemon.setItem(item) (sim/pokemon.ts:1889-1915: hp and
    isActive; new itemState, effectOrder++; Start(item)) -> on failure the holder gets the item back (itemState stays
    cleared); else '-activate' holder 'ability: Symbiosis' item '[of] pokemon' (ACTIVATE). Start(item): Air Balloon
    ('-item') only. NOT modelled (compile size: this runs at every useItem / eatItem site): Metronome's addVolatile,
    and the immediate use of a received White Herb / terrain seed (their useItem needs the rng). Unburden's own onAfterUseItem
    (abilities.unburden) is order-free with this. Trace-time guard: a nested AfterUseItem inside Start is a no-op
    (the holder has just given its item away)."""
    if aid("symbiosis") <= 0 or _SYM_TRACING[0]:
        return st
    prev = _SYM_TRACING[0]
    _SYM_TRACING[0] = True
    try:
        from vgcjax.effects import batch_forme as BF
        from vgcjax.effects import moves_b1 as MB1
        r = jnp.clip(jnp.asarray(r, I32), 0, 7)
        h, ok = _ally(st, r)
        m = st.mons
        it = mget(m.item, h).astype(I32)
        go = jnp.asarray(on) & ok & (h != r) & _is(mget(m.ability, h), "symbiosis") & (mget(m.switch_flag, r) == 0) & \
            (it > 0)
        taken = go & MB1.takeable(st, it, h)
        give = taken & (mget(m.hp, r) > 0) & mget(m.is_active, r)
        seq = st.meta.effect_seq
        hr, gg, tg = jnp.stack([h, r]), jnp.stack([give, give]), jnp.stack([taken, give])
        m = m.replace(item=mset_many(m.item, hr, jnp.stack([0, it]), gg),
                      item_state=mset_many(m.item_state, hr, jnp.zeros((2,) + m.item_state.shape[2:], m.item_state.dtype),
                                           tg),
                      item_seq=mset_many(m.item_seq, hr, jnp.stack([jnp.zeros_like(seq), seq]), tg))
        st = st.replace(mons=m, meta=st.meta.replace(effect_seq=seq + give.astype(seq.dtype)))
        st = BF.air_balloon_start(st, r, give)                       # setItem: singleEvent('Start', item)
        return EV.emit(st, give, EV.EV_ACTIVATE, actor=h, move=aeff("symbiosis"))
    finally:
        _SYM_TRACING[0] = prev


# ------------------------------------------------------------------ King's Rock
def _kr_check():
    """Static: every move with two secondaries has a flinch one, so King's Rock never needs a third slot."""
    from vgcjax.core.dex import load_tables_np
    t = load_tables_np()
    if "flinch" not in ids.CONDITIONS:
        return
    fl = ids.CONDITIONS.index("flinch")
    two = np.asarray(t["move_nsec"]) >= 2
    assert np.all((np.asarray(t["move_sec_volatile"])[two] == fl).any(axis=-1)), "King's Rock needs a 3rd slot"


if iid("kingsrock") > 0:
    _kr_check()


def kings_rock(st, dex, u, move, sf):
    """King's Rock onModifyMove (data/items.ts:3221-3243; priority -1: after Sheer Force's onModifyMove (priority 0)
    deleted the secondaries): a non-Status move whose secondaries have no flinch volatile gets {chance: 10,
    volatileStatus: 'flinch'} appended (a fresh array after Sheer Force: the flinch then still applies,
    hasSheerForce stays). Returns (on, slot): the appended secondary's index (the SECONDARY draw ord). No Serene Grace /
    Shield Dust / Covert Cloak in the impl stage. The appended secondaries array does not reach the
    `!move.secondaries` message tests (TryBoost '-fail', terrain / status blocks): no non-Status move with primary
    boosts or status lacks secondaries in Reg M-C's pool."""
    z = jnp.zeros((), bool)
    if iid("kingsrock") <= 0 or "flinch" not in ids.CONDITIONS:
        return z, jnp.int32(0)
    n = dex.move_nsec[move].astype(I32)
    fl = jnp.any((dex.move_sec_volatile[move].astype(I32) == ids.CONDITIONS.index("flinch")) &
                 (jnp.arange(dex.move_sec_volatile.shape[-1]) < n))
    on = holds(st, jnp.clip(jnp.asarray(u, I32), 0, 7), "kingsrock") & \
        (dex.move_category[move] != ids.CAT_STATUS) & (sf | ~fl)
    return on, jnp.where(sf, 0, n)


# ------------------------------------------------------------------ Persim Berry
def persim_update(st, dex, rng, r, on):
    """Persim Berry onUpdate (data/items.ts:4549-4553): volatiles['confusion'] -> eatItem() (berries.can_eat:
    TryEatItem's Unnerve) -> onEat (:4554-4556): removeVolatile('confusion') ('-end' X 'confusion'). Returns
    (st, rng)."""
    if iid("persimberry") <= 0 or "confusion" not in ids.CONDITIONS:
        return st, rng
    from vgcjax.effects import berries as BE
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    go = jnp.asarray(on) & holds(st, r, "persimberry") & CO.has_volatile(st, r, CO.cid("confusion")) & \
        BE.can_eat(st, r)
    st = BE.eat_item(st, r, go)
    st, rng, _ = CO.remove_volatile(st, dex, rng, r, CO.cid("confusion"), go)
    return st, rng


def is_persim(berry):
    """The berry is a Persim Berry (its onEat for Bug Bite / Cud Chew: removeVolatile('confusion'))."""
    return (berry == iid("persimberry")) if iid("persimberry") > 0 else jnp.zeros(jnp.shape(berry), bool)


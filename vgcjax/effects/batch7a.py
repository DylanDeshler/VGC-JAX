"""Batch 7a: the Mega abilities of Meganiumite / Golurkite / Heracronite / Eelektrossite / Skarmorite (Mega Sol, Unseen
Fist [C], Skill Link, Eelevate, Stalwart) and the items Leek, Light Ball, Metronome (+ its 'metronome' volatile),
Zoom Lens.

PS (Champions overrides first): data/mods/champions/abilities.ts:22-25 eelevate / :50-53 megasol (isNonstandard
only), :91-99 unseenfist (onModifyMove removed, onHitProtect added: the same handler as Piercing Drill);
data/mods/champions/items.ts:514-517 leek (isNonstandard only); data/abilities.ts:1147-1159 eelevate, :2559-2572
megasol, :4298-4311 skilllink, :4504-4514 stalwart; data/items.ts:3334-3349 leek, :3439-3459 lightball, :4011-4052
metronome, :7849-7863 zoomlens; sim/pokemon.ts:2211-2227 effectiveWeather (Mega Sol), :677-689 getBestStat,
:850-852 getMoveTargets (tracksTarget skips RedirectTarget); sim/battle.ts:2456-2460 getTarget (Stalwart).

MUL-hook callbacks are effects/modifiers.py mods. Everything else is an undispatched helper called from its PS event
site (perf rule; no new dispatch, no Python loop over mons):
- Mega Sol: `megasol` is read by the effectiveWeather consumers that run with battle.effect a Move or a Weather
  (effects/weather.py weather_damage_mod / modify_move / sandstorm+snowscape stat mods, effects/locked.py charge skip
  + Solar Beam / Blade onBasePower); ability callbacks (Leaf Guard, Chlorophyll, Solar Power, ...) run with
  battle.effect = the ability and are unaffected (sim/pokemon.ts:2213-2217).
- Unseen Fist [C]: effects/batch_forme.py hit_protect_bypass / emit_protect_bypass (shared with Piercing Drill).
- Skill Link: battle/move_pipeline.py hit-loop setup (`skill_link`).
- Eelevate onSourceAfterFaint: battle/faint.py resolve_faints, next to Moxie (`eelevate`); its airborneness already
  lives in conditions.is_grounded.
- Stalwart: getTarget already checks it (move_pipeline.get_target); getMoveTargets' RedirectTarget skip is `tracks`.
- Metronome: onStart (fieldEvent('SwitchIn') walk + setItem's Start) `metronome_start`; the volatile's onTryMove
  (priority -2: after Armor Tail / Damp) `metronome_try_move`; onModifyDamage is a mod.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import state as S
from vgcjax.battle.util import mget
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.abilities import aeff, aid
from vgcjax.effects.hooks import handler
from vgcjax.effects.items import ieff, iid
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32


def _ab_is(st, refs, name):
    if aid(name) <= 0:
        return jnp.zeros(jnp.shape(refs), bool)
    return mget(st.mons.ability, jnp.clip(jnp.asarray(refs, I32), 0, 7)) == aid(name)


def _sid(name):
    return getattr(ids, f"SPECIES_{name.upper()}", -1)


def _base_species_is(ctx, h, names):
    """toID(pokemon.baseSpecies.baseSpecies) in names: the dex base species of the mon's baseSpecies."""
    bs = ctx.dex.species_base_species[mget(ctx.st.mons.base_species, jnp.clip(h, 0, 7)).astype(I32)].astype(I32)
    hit = jnp.zeros(jnp.shape(h), bool)
    for n in names:
        if _sid(n) > 0:
            hit = hit | (bs == _sid(n))
    return hit


# ================================================================== Mega Sol (Mega Meganium)
def megasol(st, refs):
    """sim/pokemon.ts:2211-2217 effectiveWeather: battle.activePokemon has Mega Sol and battle.effect is Mega Sol, a
    Move or a Weather -> 'sunnyday' (before the Utility Umbrella check; Cloud Nine does not apply). `refs`: the
    activePokemon at the call site (the move user). The ability's own onWeatherModifyDamage (data/abilities.ts:
    2559-2566, priority 1, fast exit) calls sunnyday's handler with battle.effect = Mega Sol, i.e. the same."""
    return _ab_is(st, refs, "megasol")


def megasol_weather(st, w, refs):
    """effectiveWeather(...) of any mon read by a Move / Weather callback while `refs` is the active move user."""
    if aid("megasol") <= 0 or "sunnyday" not in ids.CONDITIONS:
        return w
    return jnp.where(megasol(st, refs), CO.cid("sunnyday"), w)


def megasol_message(st, u, on):
    """effectiveWeather(undefined, true) (Solar Beam / Solar Blade onTryMove, data/moves.ts:17234, :17270): the Mega
    Sol path with field.effectiveWeather() !== 'sunnyday' -> '-activate' pokemon 'ability: Mega Sol'."""
    from vgcjax.battle import events as EV
    from vgcjax.effects import weather as WE
    if aid("megasol") <= 0:
        return st
    go = jnp.asarray(on) & megasol(st, u) & (WE.field_weather(st) != WE.wid("sunnyday"))
    return EV.emit(st, go, EV.EV_ACTIVATE, actor=u, move=aeff("megasol"))


# ================================================================== Skill Link (Mega Heracross)
def skill_link(st, u):
    """Skill Link onModifyMove (data/abilities.ts:4298-4307): an array multihit -> multihit[1] (no
    DK_MULTIHIT_COUNT sample in the hit loop, ch/scripts.ts:433-449); multiaccuracy deleted (no per-hit accuracy
    re-roll, ch/scripts.ts:481)."""
    return _ab_is(st, u, "skilllink")


# ================================================================== Stalwart (Mega Skarmory)
def tracks(st, dex, u, move):
    """move.tracksTarget at getMoveTargets (sim/pokemon.ts:850): the move's data, or Stalwart / Propeller Tail
    onModifyMove (priority 1, data/abilities.ts:4504-4510 / :3487-3493: move.target !== 'scripted'), which runs
    before getMoveTargets (sim/battle-actions.ts:503 -> :531). True -> no RedirectTarget."""
    ab = _ab_is(st, u, "stalwart") | _ab_is(st, u, "propellertail")
    return (dex.move_tracks_target[move] == 1) | (ab & (dex.move_target[move] != ids.TARGET_SCRIPTED))


# ================================================================== Eelevate (Mega Eelektross)
def eelevate(st, src, length, on):
    """Eelevate onSourceAfterFaint(length, target, source, effect) (data/abilities.ts:1149-1154): the faint's effect
    is a Move -> boost({[source.getBestStat(true, true)]: length}, source). getBestStat(unboosted, unmodified):
    the first strict maximum of storedStats atk / def / spa / spd / spe (sim/pokemon.ts:677-689; Wonder Room only
    swaps the boost read). Like Moxie (batch_reactive.moxie): runEvent skips an inactive source; `src` is
    faintData.source when the effect is a Move."""
    if aid("eelevate") <= 0:
        return st
    from vgcjax.battle import move_pipeline as MP
    src = jnp.asarray(src, I32)
    r = jnp.clip(src, 0, 7)
    go = jnp.asarray(on) & (src >= 0) & _ab_is(st, r, "eelevate")
    best = jnp.argmax(mget(st.mons.stats, r).astype(I32))           # stats5 = atk def spa spd spe
    d = (jnp.arange(7) == best).astype(I32) * jnp.asarray(length, I32)
    st, _ = MP.boost(st, r, d, go, source=src, ab_msg=aeff("eelevate"), simple=True)
    return st


# ================================================================== Leek
if iid("leek") > 0:
    @mod("ModifyCritRatio", ieff("leek"), ps="data/items.ts:3340-3344 leek.onModifyCritRatio")
    def leek_crit(ctx, h):
        """onModifyCritRatio(critRatio, user): toID(user.baseSpecies.baseSpecies) is farfetchd / sirfetchd ->
        critRatio + 2."""
        return _base_species_is(ctx, h, ("farfetchd", "sirfetchd")), None, (lambda v: v + 2)


# ================================================================== Light Ball
if iid("lightball") > 0:
    def _light_ball(ctx, h):
        """onModifyAtk / onModifySpA (priority 1): pokemon.baseSpecies.baseSpecies === 'Pikachu' -> chainModify(2)."""
        return _base_species_is(ctx, h, ("pikachu",)), jnp.asarray(8192, I32), None
    mod("ModifyAtk", ieff("lightball"), ps="data/items.ts:3445-3450 lightball.onModifyAtk")(_light_ball)
    mod("ModifySpA", ieff("lightball"), ps="data/items.ts:3451-3456 lightball.onModifySpA")(_light_ball)


# ================================================================== Zoom Lens
if iid("zoomlens") > 0:
    @mod("ModifyAccuracy", ieff("zoomlens"), prefix=ids.PFX_SOURCE,
         ps="data/items.ts:7855-7861 zoomlens.onSourceModifyAccuracy")
    def zoomlens_acc(ctx, h):
        """priority -2: numeric accuracy and !queue.willMove(target) -> chainModify([4915, 4096])."""
        from vgcjax.effects import disables as DS
        t = jnp.clip(ctx.tgt, 0, 7)                     # willMove: a fainted pokemon -> null
        moved = ~(jax.vmap(lambda r: DS.will_move(ctx.st, r))(t) & ~mget(ctx.st.mons.fainted, t))
        return ctx.extra["numeric"] & moved, jnp.asarray(4915, I32), None


# ================================================================== Metronome (item + 'metronome' volatile)
HAS_METRONOME = iid("metronome") > 0 and "metronome" in ids.CONDITIONS and "metronome" in S.VOL_STATEFUL
MT_LAST, MT_NUM = 1, 2                      # volatile slots: effectState.lastMove, numConsecutive
DMG_MOD = (4096, 4915, 5734, 6553, 7372, 8192)

if HAS_METRONOME:
    CO.fields("metronome", ("lastMove", "move"), ("numConsecutive", "int"))
    # the item's onStart row is a fieldEvent('SwitchIn') candidate (effects/static.py lists registered handlers); its
    # work runs undispatched in the walk body (fieldevent.py -> metronome_start)
    handler(ids.HOOK_SWITCHIN, ieff("metronome"),
            ps="data/items.ts:4017-4019 metronome.onStart (batch7a.metronome_start)")(lambda hc, st, dex, rng: (hc, st, rng))

    @mod("ModifyDamage", CO.eff("metronome"), ps="data/items.ts:4040-4045 metronome.condition.onModifyDamage")
    def metronome_dmg(ctx, h):
        """chainModify([dmgMod[min(numConsecutive, 5)], 4096]), dmgMod = [4096, 4915, 5734, 6553, 7372, 8192]."""
        c = CO.cid("metronome")
        n = jax.vmap(lambda r: CO.vol_state(ctx.st, r, c)[MT_NUM])(jnp.clip(h, 0, 7)).astype(I32)
        return jnp.ones(jnp.shape(h), bool), jnp.asarray(DMG_MOD, I32)[jnp.clip(n, 0, 5)], None


def metronome_start(st, r, on):
    """Metronome onStart (data/items.ts:4017-4019): pokemon.addVolatile('metronome'): hp gate, already present ->
    false (no onRestart); new state (effectOrder++ for an active holder), condition onStart (:4021-4024): lastMove = '',
    numConsecutive = 0. Run by the fieldEvent('SwitchIn') walk and setItem's Start (moves_b2._item_start)."""
    if not HAS_METRONOME:
        return st
    from vgcjax.effects.items import holds
    c = CO.cid("metronome")
    rr = jnp.clip(jnp.asarray(r, I32), 0, 7)
    go = jnp.asarray(on) & holds(st, rr, "metronome") & (mget(st.mons.hp, rr) > 0) & ~CO.has_volatile(st, rr, c)
    st = CO._set_vol_bit(st, rr, c, True, go)
    s = jnp.zeros(6, I16).at[5].set((1 << MT_LAST) | (1 << MT_NUM))
    st = CO.set_vol_state(st, rr, c, s, go)
    return CO.next_effect_order(st, go & mget(st.mons.is_active, rr))


def metronome_try_move(st, dex, u, move, on):
    """metronome.condition.onTryMove (priority -2, data/items.ts:4025-4039), reached when runEvent('TryMove') gets to
    it: holder no longer hasItem('metronome') -> removeVolatile (no End); move.callsMove -> nothing; lastMove ===
    move.id and moveLastTurnResult truthy -> numConsecutive++; else twoturnmove volatile -> 1 for a new move, ++ for
    the same; else 0; lastMove = move.id."""
    if not HAS_METRONOME:
        return st
    from vgcjax.effects.items import holds
    c = CO.cid("metronome")
    u = jnp.clip(jnp.asarray(u, I32), 0, 7)
    go = jnp.asarray(on) & CO.has_volatile(st, u, c)
    lost = go & ~holds(st, u, "metronome")
    st = CO._set_vol_bit(st, u, c, False, lost)
    st = CO.set_vol_state(st, u, c, jnp.zeros(6, I16), lost)
    upd = go & ~lost & (dex.move_calls_move[move] != 1)
    s = CO.vol_state(st, u, c)
    last, n = s[MT_LAST].astype(I32), s[MT_NUM].astype(I32)
    same = last == jnp.asarray(move, I32)
    ttm = CO.has_volatile(st, u, CO.cid("twoturnmove")) if "twoturnmove" in ids.CONDITIONS else jnp.zeros((), bool)
    lres = mget(st.mons.move_last_turn_result, u) == S.MR_TRUE
    n2 = jnp.where(same & lres, n + 1, jnp.where(ttm, jnp.where(same, n + 1, 1), 0))
    s = s.at[MT_LAST].set(jnp.asarray(move, I16)).at[MT_NUM].set(jnp.minimum(n2, 32767).astype(I16))
    return CO.set_vol_state(st, u, c, s, upd)

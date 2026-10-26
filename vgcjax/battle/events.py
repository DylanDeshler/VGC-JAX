"""PublicEvents (03 §4.7): what the PS protocol shows players, one i16 record per protocol line.

Record = [t, kind, actor, target, move, v1, v2, flags]; refs are side*4 + brought index (-1 = none); HP values are
the PS Champions public percentage (sim/pokemon.ts getHealth; env/obs.py hp_public) with its colour code. Kinds and
fields (PS protocol line -> record):

  SWITCH     |switch| / |drag|          actor, move = [from] move, v1/v2 = hp pct/colour, flags F_DRAG
  MOVE       |move|                     actor, target (as printed), move, v1 = [spread] slot bits (side*2+pos),
                                        flags F_MISS / F_NOTARGET / F_SPREAD / F_STILL (attrLastMove)
  DAMAGE     |-damage|                  target = the damaged mon (in `actor`), v1/v2 hp, flags F_RECOIL
  HEAL       |-heal|                    actor, target = [of] (drain), v1/v2 hp, flags F_DRAIN
  CRIT / SUPEREFFECTIVE / RESISTED      actor = target mon; v1 = min(|typeMod|, 2) ([C] scripts.ts:270-277)
  IMMUNE     |-immune|                  actor, flags F_LEVITATE ([from] ability: Levitate) / F_OHKO
  MISS       |-miss|                    actor = user, target
  FAIL       |-fail|                    actor, flags F_HEAL ('-fail', target, 'heal')
  BOOST      |-boost| / |-unboost|      actor, v1 = stat (atk def spa spd spe accuracy evasion), v2 = amount,
                                        flags F_UNBOOST
  FAINT      |faint|                    actor
  HITCOUNT   |-hitcount|                actor = targets[0], v1 = hits
  OHKO       |-ohko|
  TURN       |turn|                     v1 = turn
  ACTIVATE   |-activate| X|effect         actor, move = effect EffectId (Struggle, confusion, ...)
  STATUS     |-status| X|s [from]          actor, move = status EffectId, v1 = [from] EffectId, flags F_FROM_*
  CURESTATUS |-curestatus| X|s [msg]       actor, move = status EffectId, flags F_MSG / F_SILENT
  CANT       |cant| X|reason               actor, move = reason EffectId (status / condition / move)
  START/END  |-start| / |-end| X|effect    actor, move = effect EffectId, v1 = [from] EffectId, flags
  SINGLETURN |-singleturn| X|effect        actor, move = effect EffectId (Protect, Wide Guard, ...)
  SIDESTART/SIDEEND |-sidestart| side|eff  actor = side (0 / 1), move = effect EffectId
  FIELDSTART/FIELDEND |-fieldstart| eff   move = effect EffectId, target = [of] ref (-1 none)
  PREPARE    |-prepare| X|move             actor, move = MoveId (a charge move's first turn)
  MUSTRECHARGE |-mustrecharge| X         actor
  FIELDACTIVATE |-fieldactivate| effect   move = effect EffectId
"""
from __future__ import annotations

import contextlib

import jax.numpy as jnp

from vgcjax.battle import state as S
from vgcjax.battle.util import mget

I16, I32 = jnp.int16, jnp.int32
KINDS = ("NONE", "SWITCH", "MOVE", "DAMAGE", "HEAL", "CRIT", "SUPEREFFECTIVE", "RESISTED", "IMMUNE", "MISS", "FAIL",
         "BOOST", "FAINT", "HITCOUNT", "OHKO", "TURN", "ACTIVATE", "STATUS", "CURESTATUS", "CANT", "START", "END", "SINGLETURN", "SIDESTART", "SIDEEND",
         "FIELDSTART", "FIELDEND", "WEATHER", "ABILITY", "ITEM", "BLOCK", "SETBOOST", "ENDITEM", "MEGA", "DETAILSCHANGE", "CLEARALLBOOST",
         "PREPARE", "MUSTRECHARGE", "FIELDACTIVATE")
(EV_NONE, EV_SWITCH, EV_MOVE, EV_DAMAGE, EV_HEAL, EV_CRIT, EV_SUPER, EV_RESIST, EV_IMMUNE, EV_MISS, EV_FAIL, EV_BOOST,
 EV_FAINT, EV_HITCOUNT, EV_OHKO, EV_TURN, EV_ACTIVATE, EV_STATUS, EV_CURESTATUS, EV_CANT, EV_START,
 EV_END, EV_SINGLETURN, EV_SIDESTART, EV_SIDEEND, EV_FIELDSTART, EV_FIELDEND, EV_WEATHER, EV_ABILITY, EV_ITEM, EV_BLOCK, EV_SETBOOST, EV_ENDITEM, EV_MEGA, EV_DETAILSCHANGE, EV_CLEARALLBOOST,
 EV_PREPARE, EV_MUSTRECHARGE, EV_FIELDACTIVATE) = range(len(KINDS))
# kinds whose `move` field holds an EffectId (move / ability / item / condition) rather than a MoveId
EFFECT_KINDS = ("DAMAGE", "HEAL", "STATUS", "CURESTATUS", "CANT", "FAIL", "ACTIVATE", "START", "END", "SINGLETURN", "SIDESTART", "SIDEEND",
         "FIELDSTART", "FIELDEND", "WEATHER", "ABILITY", "ITEM", "IMMUNE", "BLOCK", "SETBOOST", "ENDITEM", "MEGA",
         "FIELDACTIVATE")
F_DRAG = 1
F_MISS, F_NOTARGET, F_SPREAD, F_STILL = 1, 2, 4, 8
F_RECOIL = 1
F_FROM = 2                       # DAMAGE / HEAL: '[from] <effect>' (effect in `move`)
F_DRAIN = 1
F_LEVITATE, F_OHKO = 1, 2
F_HEAL = 1                       # FAIL: ('-fail', target, 'heal')
F_FAIL_UNBOOST = 4               # FAIL: ('-fail', target, 'unboost', [stat] (v2, -1 none), '[from] ability: X' (v1))
F_FAILSTATUS = 2                 # FAIL: ('-fail', target, <status>) (status in `move`)
F_UNBOOST = 1
F_FROM_MOVE, F_FROM_ABILITY, F_FROM_ITEM = 1, 2, 4   # STATUS: '[from] move: / ability: / item:' (effect in v1)
F_FATIGUE = 8                    # START: '[fatigue]'
F_BROKEN = 16                    # ACTIVATE: '[broken]' (protection broken by a breaksProtect move other than Feint)
F_MSG, F_SILENT, F_FROM_MOVE_CURE = 1, 2, 4   # CURESTATUS: '[msg]' / '[silent]' / '[from] move: X' (move in v1)
F_BOOST = 1                      # ABILITY: '-ability' pokemon X 'boost' (ability in `move`)
F_EAT, F_WEAKEN = 1, 2           # ENDITEM: '[eat]' / '[weaken]' (item in `move`)
F_UPKEEP = 1                     # WEATHER: '[upkeep]' (weather in `move`, -1 = 'none'; '[from] ability:' -> F_FROM_ABILITY)


_ENABLED = [True]


@contextlib.contextmanager
def enabled(flag: bool):
    """Trace-time switch (EngineConfig.events): with False every emit / patch below is the identity."""
    old = _ENABLED[0]
    _ENABLED[0] = bool(flag)
    try:
        yield
    finally:
        _ENABLED[0] = old


def hp_public(hp, maxhp):
    """PS Champions getHealth: pct = floor(100 hp / maxhp) || 1 (0 if fainted); colour at 20 / 50 only:
    1 if above the boundary ('y' at 20, 'g' at 50 suffix), else 2."""
    hp = jnp.asarray(hp, I32)
    maxhp = jnp.maximum(jnp.asarray(maxhp, I32), 1)
    pct = jnp.where(hp > 0, jnp.maximum((100 * hp) // maxhp, 1), 0)
    above = jnp.where(pct == 20, hp * 5 > maxhp, jnp.where(pct == 50, hp * 2 > maxhp, False))
    colour = jnp.where((pct == 20) | (pct == 50), jnp.where(above, 1, 2), 0)
    return pct, colour


def emit(st: S.State, on, kind, actor=-1, target=-1, move=0, v1=0, v2=0, flags=0) -> S.State:
    if not _ENABLED[0]:
        return st
    ev = st.events
    on = jnp.asarray(on)
    rec = jnp.stack([jnp.asarray(x, I32) for x in (st.meta.turn, kind, actor, target, move, v1, v2, flags)]).astype(I16)
    # one-hot select instead of a dynamic-index scatter: elementwise, so it vectorises under vmap (CPU scatter is serial)
    hot = (jnp.arange(S.NE) == ev.count % S.NE) & on
    return st.replace(events=ev.replace(buf=jnp.where(hot[:, None], rec[None, :], ev.buf), count=ev.count + on.astype(I32)))


def emit_hp(st: S.State, on, kind, ref, target=-1, move=0, flags=0) -> S.State:
    """A record carrying `ref`'s current public HP (SWITCH / DAMAGE / HEAL)."""
    pct, colour = hp_public(mget(st.mons.hp, ref), mget(st.mons.maxhp, ref))
    return emit(st, on, kind, actor=ref, target=target, move=move, v1=pct, v2=colour, flags=flags)


def emit_move(st: S.State, on, actor, target, move) -> S.State:
    """PS: sim/battle-actions.ts:519 addMove('move', pokemon, movename, target); remembered for attrLastMove."""
    if not _ENABLED[0]:
        return st
    idx = st.events.count
    st = emit(st, on, EV_MOVE, actor=actor, target=target, move=move)
    ev = st.events
    return st.replace(events=ev.replace(last_move=jnp.where(jnp.asarray(on), idx, ev.last_move)))


def attr_last_move(st: S.State, on, or_flags=0, v1=None) -> S.State:
    """PS: sim/battle.ts:3181-3197 attrLastMove: annotate the last MOVE record; [still] also blanks its target."""
    return patch(st, st.events.last_move, on, or_flags, v1, target=-1 if or_flags & F_STILL else None)


def anim_last_move(st: S.State, on) -> S.State:
    """PS addMove('-anim', ...): an unrecorded line becomes battle.lastMoveLine, so later attrLastMove calls no
    longer reach the MOVE record."""
    if not _ENABLED[0]:
        return st
    ev = st.events
    return st.replace(events=ev.replace(last_move=jnp.where(jnp.asarray(on), -1, ev.last_move)))


def retarget_last_move(st: S.State, on, target) -> S.State:
    """PS: sim/battle.ts:3199-3204 retargetLastMove."""
    return patch(st, st.events.last_move, on, target=target)


def patch(st: S.State, idx, on, or_flags=0, v1=None, target=None) -> S.State:
    """OR flags into (and optionally set v1 / target of) the record at absolute index `idx`, if still in the ring."""
    if not _ENABLED[0]:
        return st
    ev = st.events
    idx = jnp.asarray(idx, I32)
    on = jnp.asarray(on) & (idx >= 0) & (idx > ev.count - 1 - S.NE)
    hot = ((jnp.arange(S.NE) == idx % S.NE) & on)[:, None]
    col = jnp.arange(8)[None, :]
    buf = ev.buf
    buf = jnp.where(hot & (col == 7), buf | jnp.asarray(or_flags, I16), buf)
    if v1 is not None:
        buf = jnp.where(hot & (col == 5), jnp.asarray(v1, I16), buf)
    if target is not None:
        buf = jnp.where(hot & (col == 3), jnp.asarray(target, I16), buf)
    return st.replace(events=ev.replace(buf=buf))


def new_this_step(st: S.State):
    """(records [NE, 8] oldest first, valid [NE]) written since the current step began (overflow keeps the last NE)."""
    ev = st.events
    n = jnp.minimum(ev.count - ev.step_start, S.NE)
    j = jnp.arange(S.NE)
    absolute = ev.count - n + j
    return ev.buf[absolute % S.NE], j < n


def last_k(st: S.State, k: int):
    """The last k records, oldest first, zero-padded at the front when fewer exist."""
    ev = st.events
    j = jnp.arange(k)
    absolute = ev.count - k + j
    ok = absolute >= 0
    return jnp.where(ok[:, None], ev.buf[jnp.maximum(absolute, 0) % S.NE], 0)

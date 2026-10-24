"""Terrain (05 §4 step 4): Electric, Grassy, Psychic, Misty, set by moves (Surges, seeds: M4).

PS: data/moves.ts electricterrain / grassyterrain / psychicterrain / mistyterrain `condition` (no Champions
overrides), sim/field.ts setTerrain / clearTerrain. Semi-invulnerability (Fly / Dig / ...) is not in the engine yet,
so `!isSemiInvulnerable()` always holds. The terrains' callbacks inside formulas are explicit and called from the
formula sites: onSetStatus (blocks_status), onTryAddVolatile (blocks_volatile); onBasePower is a modifiers.py mod.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget
from vgcjax.core import fixedpoint as fp
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects import modifiers as MOD
from vgcjax.effects.hooks import HookCtx, handler

I16, I32 = jnp.int16, jnp.int32
TERRAINS = ("electricterrain", "grassyterrain", "psychicterrain", "mistyterrain")


def tid(name):
    return CO.cid(name) if name in ids.CONDITIONS else -1


def terrain(st):
    """field.terrain (ConditionId, 0 = none); the terrain's own handlers are listed whenever it is set."""
    return st.field.terrain.astype(I32)


# ------------------------------------------------------------------ formula-site callbacks
def blocks_status(st, dex, refs, status, source_effect, move_status):
    """runEvent('SetStatus', target, source, effect, status): Electric Terrain onSetStatus (slp; '-activate' for Yawn
    or a move without secondaries) and Misty Terrain onSetStatus (any status; '-activate' when the effect has a
    primary `status`), grounded targets only. Returns (blocked, activate message) per ref."""
    t = terrain(st)
    g = CO.is_grounded(st, refs)
    eff = jnp.asarray(source_effect, I32)
    is_move = (eff >= ids.MOVE_BASE) & (eff < ids.MOVE_BASE + len(ids.MOVES) - 1)
    mv = jnp.clip(eff - ids.MOVE_BASE + 1, 0, len(ids.MOVES) - 1)
    yawn = eff == CO.eff("yawn") if "yawn" in ids.CONDITIONS else False
    elec = (t == tid("electricterrain")) & (status == ids.COND_SLP) & g
    misty = (t == tid("mistyterrain")) & (status > 0) & g
    msg = (elec & (yawn | (is_move & (dex.move_nsec[mv] == 0)))) | (misty & ((move_status > 0) | yawn))
    return elec | misty, msg


def blocks_volatile(st, dex, ref, c, source_effect):
    """runEvent('TryAddVolatile', target, source, effect, status): Misty Terrain blocks confusion (null; '-activate'
    for a move without secondaries), Electric Terrain blocks Yawn (null, '-activate'), grounded targets only.
    Returns (blocked, activate message)."""
    t = terrain(st)
    g = CO.is_grounded(st, ref)
    eff = jnp.asarray(source_effect, I32)
    is_move = (eff >= ids.MOVE_BASE) & (eff < ids.MOVE_BASE + len(ids.MOVES) - 1)
    mv = jnp.clip(eff - ids.MOVE_BASE + 1, 0, len(ids.MOVES) - 1)
    misty = (t == tid("mistyterrain")) & (c == tid_or("confusion")) & g
    elec = (t == tid("electricterrain")) & (c == tid_or("yawn")) & g
    return misty | elec, (misty & is_move & (dex.move_nsec[mv] == 0)) | elec


def tid_or(name):
    return CO.cid(name) if name in ids.CONDITIONS else -99


def terrain_cond(name):
    """ConditionId of a terrain, -1 when the format lacks it."""
    return tid(name)


# ------------------------------------------------------------------ setTerrain
def set_terrain(st, dex, rng, c, on, source=-1, source_effect=-1):
    """PS: sim/field.ts setTerrain(status, source, sourceEffect): false when already active; new state (duration 5,
    durationCallback; no target so effectOrder 0); FieldStart (falsy restores); eachEvent('TerrainChange') (seeds /
    abilities: M4). Returns (st, rng, result)."""
    on = jnp.asarray(on)
    c = jnp.asarray(c, I32)
    e = ids.COND_BASE + c - 1
    f = st.field
    go = on & (f.terrain.astype(I32) != c)
    prev_t, prev_s = f.terrain, f.terrain_state
    dur = dex.cond_duration[c].astype(I32)
    hc = HookCtx.make(-1, source, source_effect).replace(holder=10)
    hc, st, rng = CO.single_event(CO.DURATION_CALLBACK, e, st, dex, rng, hc, go)
    dur = jnp.where(hc.ret_t == RS.NUM, hc.ret_v, dur)
    f = st.field
    st = st.replace(field=f.replace(terrain=jnp.where(go, c, f.terrain).astype(f.terrain.dtype),
                                    terrain_state=jnp.where(go, jnp.zeros(3, I16).at[0].set(dur.astype(I16)),
                                                            f.terrain_state)))
    hc = HookCtx.make(-1, source, source_effect).replace(holder=10)
    hc, st, rng = CO.single_event(ids.HOOK_START, e, st, dex, rng, hc, go, prefix=ids.PFX_FIELD)
    started = (hc.ret_t == RS.UNDEF) | RS.truthy((hc.ret_t, hc.ret_v))
    f = st.field
    st = st.replace(field=f.replace(terrain=jnp.where(go & ~started, prev_t, f.terrain),
                                    terrain_state=jnp.where(go & ~started, prev_s, f.terrain_state)))
    return st, rng, RS.where(go & started, RS.R(RS.BOOL, 1), RS.R(RS.BOOL, 0))


# ------------------------------------------------------------------ the terrain conditions
def _extender(hc, st, dex, rng):
    """durationCallback(source): source holding Terrain Extender -> 8, else 5."""
    item = getattr(ids, "ITEM_TERRAINEXTENDER", -1)
    src = jnp.asarray(hc.source, I32)
    n = jnp.where((src >= 0) & (mget(st.mons.item, jnp.maximum(src, 0)) == item), 8, 5)
    return hc.ret(RS.R(RS.NUM, n)), st, rng


def _msg(kind, name):
    def fn(hc, st, dex, rng):
        """'-fieldstart' / '-fieldend' 'move: <Terrain>', '-fieldstart' with '[from] ability: X' '[of] source' for an
        ability source."""
        ab = (hc.effect >= ids.ABILITY_BASE) & (hc.effect < ids.ABILITY_BASE + len(ids.ABILITIES) - 1)
        of = jnp.where(ab & (kind == EV.EV_FIELDSTART), hc.source, -1)
        return hc, EV.emit(st, True, kind, move=CO.eff(name), target=of), rng
    return fn


for _t in TERRAINS:
    if _t not in ids.CONDITIONS:
        continue
    _e = CO.eff(_t)
    handler(CO.DURATION_CALLBACK, _e, ps=f"data/moves.ts {_t}.condition.durationCallback")(_extender)
    handler(ids.HOOK_START, _e, ps=f"data/moves.ts {_t}.condition.onFieldStart", prefix=ids.PFX_FIELD)(
        _msg(EV.EV_FIELDSTART, _t))
    handler(ids.HOOK_END, _e, ps=f"data/moves.ts {_t}.condition.onFieldEnd", prefix=ids.PFX_FIELD)(
        _msg(EV.EV_FIELDEND, _t))


if "grassyterrain" in ids.CONDITIONS:
    @handler(ids.HOOK_RESIDUAL, CO.eff("grassyterrain"), ps="data/moves.ts grassyterrain.condition.onResidual")
    def grassy_residual(hc, st, dex, rng):
        """heal(baseMaxhp / 16, pokemon, pokemon) for a grounded holder: '-heal' X hp '[from] Grassy Terrain'."""
        from vgcjax.effects import statuses as STS
        r = hc.target
        st, rng = STS.effect_heal(st, dex, rng, r, mget(st.mons.maxhp, r) // 16, CO.eff("grassyterrain"),
                                  CO.is_grounded(st, r))
        return hc, st, rng


if "psychicterrain" in ids.CONDITIONS:
    @handler(ids.HOOK_TRYHIT, CO.eff("psychicterrain"), ps="data/moves.ts psychicterrain.condition.onTryHit")
    def psychic_try_hit(hc, st, dex, rng):
        """A move with priority > 0 (not target 'self') against a grounded non-ally: '-activate' target
        'move: Psychic Terrain', null."""
        from vgcjax.battle import queue as Q
        move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
        prio = Q.move_priority(st, dex, hc.source, move)
        tgt, src = hc.target, hc.source
        ally = (jnp.clip(tgt, 0, 7) // 4) == (jnp.clip(src, 0, 7) // 4)
        block = (prio > 0) & (dex.move_target[move] != ids.TARGET_SELF) & ~ally & CO.is_grounded(st, tgt)
        st = EV.emit(st, block, EV.EV_ACTIVATE, actor=tgt, move=CO.eff("psychicterrain"))
        return hc.ret(RS.R(RS.NULL), block), st, rng


# ------------------------------------------------------------------ onBasePower (priority 6; attacker = event target)
def _bp(name):
    def fn(ctx, h):
        """Electric / Psychic: grounded attacker's Electric / Psychic move x5325/4096; Grassy: Earthquake / Bulldoze
        (/ Magnitude) vs a grounded target x0.5, else a grounded attacker's Grass move x5325/4096; Misty: Dragon moves
        vs a grounded target x0.5."""
        a, d, t = ctx.tgt, ctx.src, ctx.mtype
        ag, dg = CO.is_grounded(ctx.st, a), CO.is_grounded(ctx.st, d)
        if name in ("electricterrain", "psychicterrain"):
            ty = ids.TYPE_ELECTRIC if name == "electricterrain" else ids.TYPE_PSYCHIC
            return ag & (t == ty), jnp.asarray(5325, I32), None
        if name == "mistyterrain":
            return dg & (t == ids.TYPE_DRAGON), jnp.asarray(2048, I32), None
        quake = (ctx.move == getattr(ids, "MOVE_EARTHQUAKE", -1)) | (ctx.move == getattr(ids, "MOVE_BULLDOZE", -1)) | (
            ctx.move == getattr(ids, "MOVE_MAGNITUDE", -1))
        half = quake & dg
        return half | (ag & (t == ids.TYPE_GRASS)), jnp.where(half, 2048, 5325).astype(I32), None
    return fn


for _t in TERRAINS:
    if _t in ids.CONDITIONS:
        MOD.mod("BasePower", CO.eff(_t), ps=f"data/moves.ts {_t}.condition.onBasePower")(_bp(_t))

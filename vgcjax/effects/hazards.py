"""Entry hazards: Stealth Rock, Spikes, Toxic Spikes, Sticky Web (side conditions with a customHolder onSwitchIn),
and Toxic Debris (onDamagingHit -> addSideCondition('toxicspikes')).

PS: data/moves.ts stealthrock / spikes / toxicspikes / stickyweb conditions, data/abilities.ts toxicdebris (no
Champions overrides). Heavy-Duty Boots is not Reg M-C legal. The SwitchIn rows run in the fieldEvent('SwitchIn') walk
(effects/fieldevent.py: side_effects(PFX_ON) customHolder rows, hc.target = the switched-in mon).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler

I16, I32 = jnp.int16, jnp.int32
LAYERS = 1                                         # side state slot of effectState.layers (presence bit 1 << 1)
EO = 2                  # raw side state slot: effectState.effectOrder (SwitchIn tie order; dropped by the projection)
_MAX_LAYERS = {"spikes": 3, "toxicspikes": 2}


def _has(name):
    return name in ids.CONDITIONS


def layers(st, side, name):
    return CO.side_state(st, side, CO.cid(name))[LAYERS].astype(I32)


def _set_layers(st, side, name, n, on):
    c = CO.cid(name)
    s = CO.side_state(st, side, c)
    s = s.at[LAYERS].set(jnp.asarray(n, I16)).at[3].set(s[3] | (1 << LAYERS))
    return CO.set_side(st, side, c, True, s, on)


def _foe_active0(st, r):
    """pokemon.side.foe.active[0] (a fainted mon still occupies the slot; an empty slot -> no source)."""
    fs = 1 - r // 4
    b = st.sides.active[fs, 0].astype(I32)
    return jnp.where(b >= 0, fs * 4 + jnp.maximum(b, 0), -1)


def _sidestart(name, shown):
    def fn(hc, st, dex, rng):
        side = hc.holder - 8
        st = EV.emit(st, True, EV.EV_SIDESTART, actor=side, move=CO.eff(shown))
        c = CO.cid(name)                     # initEffectState took battle.effectOrder++ just before SideStart
        s = CO.side_state(st, side, c)
        st = CO.set_side(st, side, c, True, s.at[EO].set((st.meta.effect_seq - 1).astype(s.dtype)), True)
        if name in _MAX_LAYERS:
            st = _set_layers(st, side, name, 1, True)
        return hc, st, rng
    fn.__doc__ = f"onSideStart: '-sidestart' side '{shown}'" + (" ; layers = 1" if name in _MAX_LAYERS else "")
    return fn


def _siderestart(name):
    def fn(hc, st, dex, rng):
        side = hc.holder - 8
        n = layers(st, side, name)
        full = n >= _MAX_LAYERS[name]
        st = EV.emit(st, ~full, EV.EV_SIDESTART, actor=side, move=CO.eff(name))
        st = _set_layers(st, side, name, n + 1, ~full)
        return hc.ret(RS.R(RS.BOOL, 0), full), st, rng
    fn.__doc__ = f"onSideRestart: layers >= {_MAX_LAYERS[name]} -> false; else '-sidestart', layers++"
    return fn


for _n in ("stealthrock", "spikes", "toxicspikes", "stickyweb"):
    if _has(_n):
        CO.fields(_n, ("layers", "int")) if _n in _MAX_LAYERS else None
        handler(ids.HOOK_START, CO.eff(_n), ps=f"data/moves.ts {_n}.condition.onSideStart",
                prefix=ids.PFX_SIDE)(_sidestart(_n, _n))
        if _n in _MAX_LAYERS:
            handler(ids.HOOK_RESTART, CO.eff(_n), ps=f"data/moves.ts {_n}.condition.onSideRestart",
                    prefix=ids.PFX_SIDE)(_siderestart(_n))


if _has("stealthrock"):
    @handler(ids.HOOK_SWITCHIN, CO.eff("stealthrock"), ps="data/moves.ts stealthrock.condition.onSwitchIn")
    def stealthrock_switch_in(hc, st, dex, rng):
        """typeMod = clamp(runEffectiveness(Rock), -6, 6); damage(maxhp * 2 ** typeMod / 8) ('[from] Stealth Rock')."""
        from vgcjax.core import typechart as TC
        from vgcjax.effects import statuses as STS
        r = hc.target
        tm = jnp.clip(TC.effectiveness(dex, ids.TYPE_ROCK, mget(st.mons.types, r)), -6, 6)
        mx = mget(st.mons.maxhp, r).astype(I32)
        amt = jnp.where(tm >= 0, (mx << jnp.maximum(tm, 0)) // 8, mx // (8 << jnp.maximum(-tm, 0)))
        st, rng = STS.effect_damage(st, dex, rng, r, amt, CO.eff("stealthrock"), True)
        return hc, st, rng


if _has("spikes"):
    @handler(ids.HOOK_SWITCHIN, CO.eff("spikes"), ps="data/moves.ts spikes.condition.onSwitchIn")
    def spikes_switch_in(hc, st, dex, rng):
        """Grounded: damage([0, 3, 4, 6][layers] * maxhp / 24) ('[from] Spikes')."""
        from vgcjax.effects import statuses as STS
        r = hc.target
        n = jnp.clip(layers(st, r // 4, "spikes"), 0, 3)
        amt = jnp.asarray([0, 3, 4, 6], I32)[n] * mget(st.mons.maxhp, r).astype(I32) // 24
        go = CO.is_grounded(st, jnp.asarray(r, I32)[None])[0]
        st, rng = STS.effect_damage(st, dex, rng, r, amt, CO.eff("spikes"), go)
        return hc, st, rng


if _has("toxicspikes"):
    @handler(ids.HOOK_SWITCHIN, CO.eff("toxicspikes"), ps="data/moves.ts toxicspikes.condition.onSwitchIn")
    def toxicspikes_switch_in(hc, st, dex, rng):
        """Grounded: Poison type -> '-sideend' side 'move: Toxic Spikes' '[of] pokemon', removeSideCondition; Steel ->
        nothing; else trySetStatus(layers >= 2 ? 'tox' : 'psn', side.foe.active[0]) (sourceEffect: the condition)."""
        from vgcjax.effects import sideconds as SC
        from vgcjax.effects import statuses as STS
        r = hc.target
        side = r // 4
        types = mget(st.mons.types, r)
        go = CO.is_grounded(st, jnp.asarray(r, I32)[None])[0]
        poison = go & jnp.any(types == ids.TYPE_POISON)
        steel = jnp.any(types == ids.TYPE_STEEL)
        st = EV.emit(st, poison, EV.EV_SIDEEND, actor=side, move=CO.eff("toxicspikes"))
        st, rng = SC.remove_side_condition(st, dex, rng, side, CO.cid("toxicspikes"), poison)
        status = jnp.where(layers(st, side, "toxicspikes") >= 2, ids.COND_TOX, ids.COND_PSN)
        on = go & ~poison & ~steel
        st, rng, _, msgs = STS.try_set_status_v(st, dex, rng, jnp.asarray(r, I32)[None], status, on,
                                                _foe_active0(st, r), CO.eff("toxicspikes"), 0)
        from vgcjax.effects import berries as BE
        st = STS.emit_status_msgs(st, msgs, 0, r)
        st, rng = BE.after_set_status(st, dex, rng, msgs, 0, r)
        return hc, st, rng


if _has("stickyweb"):
    @handler(ids.HOOK_SWITCHIN, CO.eff("stickyweb"), ps="data/moves.ts stickyweb.condition.onSwitchIn")
    def stickyweb_switch_in(hc, st, dex, rng):
        """Grounded: '-activate' pokemon 'move: Sticky Web'; boost({spe: -1}, pokemon, side.foe.active[0], Sticky
        Web)."""
        from vgcjax.battle import common as C
        from vgcjax.battle import move_pipeline as MP
        r = hc.target
        go = CO.is_grounded(st, jnp.asarray(r, I32)[None])[0]
        st = EV.emit(st, go, EV.EV_ACTIVATE, actor=r, move=CO.eff("stickyweb"))
        st, _ = MP.boost(st, r, jnp.zeros(7, I32).at[4].set(-1), go, primary=True, source=_foe_active0(st, r),
                         effect=C.eff_move(ids.MOVES.index("stickyweb")))
        return hc, st, rng


def toxic_debris(st, dex, rng, t, damaged, move):
    """Toxic Debris onDamagingHit: physical and the foe side's (relative to the holder) Toxic Spikes < 2 layers ->
    '-activate' target 'ability: Toxic Debris'; side.addSideCondition('toxicspikes', target). Returns (st, rng)."""
    if "toxicdebris" not in ids.ABILITIES or not _has("toxicspikes"):
        return st, rng
    from vgcjax.effects import sideconds as SC
    tr = jnp.clip(t, 0, 7)
    side = 1 - tr // 4
    c = CO.cid("toxicspikes")
    few = ~CO.has_side(st, side, c) | (layers(st, side, "toxicspikes") < 2)
    on = (jnp.asarray(damaged) & (mget(st.mons.ability, tr) == ids.ABILITIES.index("toxicdebris"))
          & (dex.move_category[move] == ids.CAT_PHYSICAL) & few)
    st = EV.emit(st, on, EV.EV_ACTIVATE, actor=tr, move=ids.ABILITY_BASE + ids.ABILITIES.index("toxicdebris") - 1)
    st, rng, _ = SC.add_side_condition(st, dex, rng, side, c, on, source=tr,
                                       source_effect=ids.ABILITY_BASE + ids.ABILITIES.index("toxicdebris") - 1)
    return st, rng

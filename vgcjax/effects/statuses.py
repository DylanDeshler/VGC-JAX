"""Major statuses, Champions values (05 §4 step 2): brn, par, psn, tox, slp, frz.

PS: data/conditions.ts (brn/par/psn/tox/slp/frz) overridden by data/mods/champions/conditions.ts (par 1/8 full
paralysis, slp sample([2,3,3]), frz thaw cap 3 turns). setStatus / trySetStatus / cureStatus: sim/pokemon.ts.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import faint as F
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget, mset
from vgcjax.core import rng as R
from vgcjax.core import typechart as TC
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import HookCtx, handler

I16, I32 = jnp.int16, jnp.int32
STATUSES = ("brn", "par", "psn", "tox", "slp", "frz")
CO.fields("slp", ("startTime", "int"), ("time", "int"))
CO.fields("frz", ("startTime", "int"), ("time", "int"))
CO.fields("tox", ("stage", "int"))


def _status_slot(name, key):
    return CO.slot_of(name, key)


def status_field(st, ref, key_slot):
    return mget(st.mons.status_state, ref)[key_slot]


def set_status_field(st, ref, key_slot, value, on):
    m = st.mons
    s = mget(m.status_state, ref)
    s = s.at[key_slot].set(jnp.asarray(value, I16)).at[3].set(s[3] | (1 << key_slot))
    return st.replace(mons=m.replace(status_state=mset(m.status_state, ref, s, on)))


def run_status_immunity(st, dex, ref, status):
    """PS: sim/pokemon.ts runStatusImmunity: fainted -> false; dex.getImmunity(type) on the typechart pseudo-keys;
    runEvent('Immunity') joins with the abilities (M4). True = not immune."""
    key = jnp.where(status == ids.COND_TOX, ids.COND_PSN, status)
    # IMMUNITY_KEYS order: brn par psn tox frz ... -> map ConditionId to key index
    kidx = jnp.where(key == ids.COND_BRN, 0, jnp.where(key == ids.COND_PAR, 1, jnp.where(key == ids.COND_PSN, 2,
                     jnp.where(key == ids.COND_FRZ, 4, -1))))
    immune = (kidx >= 0) & TC.status_immune(dex, jnp.maximum(kidx, 0), mget(st.mons.types, ref))
    return ~mget(st.mons.fainted, ref) & ~immune


def set_status(st, dex, rng, ref, status, on, source=-1, source_effect=-1, move_status=0):
    """PS: sim/pokemon.ts:1705-1774 setStatus(status, source, sourceEffect). `move_status` = sourceEffect.status when
    sourceEffect is a move (its fail / immune messages), else 0. Returns (st, rng, result)."""
    on = jnp.asarray(on)
    ref = jnp.asarray(ref, I32)
    status = jnp.asarray(status, I32)
    move_status = jnp.asarray(move_status, I32)
    m = st.mons
    src = jnp.where(jnp.asarray(source, I32) >= 0, source, ref)
    go = on & (mget(m.hp, ref) > 0) & ~(~mget(m.is_active, ref) & (status > 0))
    cur = mget(m.status, ref).astype(I32)
    same = go & (cur == status)
    # PS :1720-1727
    st = EV.emit(st, same & (move_status == cur) & (move_status > 0), EV.EV_FAIL, actor=ref,
                 move=C.eff_cond(cur), flags=EV.F_FAILSTATUS)
    fail_src = same & (move_status != cur) & (move_status > 0)
    st = EV.emit(st, fail_src, EV.EV_FAIL, actor=src)
    st = EV.attr_last_move(st, fail_src, EV.F_STILL)
    go = go & ~same
    # immunity, skipped for psn / tox from a Corrosion user (PS :1729-1731)
    corrosion = (jnp.asarray(source, I32) >= 0) & (mget(m.ability, src) == ids.ABILITY_CORROSION) & (
        (status == ids.COND_PSN) | (status == ids.COND_TOX))
    imm = go & (status > 0) & ~corrosion & ~run_status_immunity(st, dex, ref, status)
    st = EV.emit(st, imm & (move_status > 0), EV.EV_IMMUNE, actor=ref)
    go = go & ~imm
    # SetStatus event: Safeguard / Misty Terrain / abilities (M3.3+)
    m = st.mons
    # status_state[3]: presence bits 0..7 + the owning status << 8 (checkFainted sets status 'fnt' but keeps the
    # state, whose keys the projection still shows)
    st = st.replace(mons=m.replace(status=mset(m.status, ref, status, go),
                                   status_state=mset(m.status_state, ref, jnp.zeros(4, I16).at[3].set(
                                       (status << 8).astype(I16)), go)))
    st = CO.next_effect_order(st, go & (status > 0))           # initEffectState (target is active here)
    hc = HookCtx.make(ref, src, source_effect)
    hc, st, rng = CO.single_event(ids.HOOK_START, C.eff_cond(status), st, dex, rng, hc, go & (status > 0))
    started = (hc.ret_t == RS.UNDEF) | RS.truthy((hc.ret_t, hc.ret_v))
    cancel = go & (status > 0) & ~started
    m = st.mons
    st = st.replace(mons=m.replace(status=mset(m.status, ref, cur, cancel)))
    # AfterSetStatus: Synchronize / Lum-like reactions join with abilities and items (M4)
    res = RS.where(go & ~cancel, RS.R(RS.BOOL, 1), RS.where(cancel, (hc.ret_t, hc.ret_v), RS.R(RS.BOOL, 0)))
    return st, rng, res


def try_set_status(st, dex, rng, ref, status, on, source=-1, source_effect=-1, move_status=0):
    """PS: sim/pokemon.ts:1690-1692 trySetStatus: setStatus(this.status || status, ...)."""
    cur = mget(st.mons.status, ref).astype(I32)
    return set_status(st, dex, rng, ref, jnp.where(cur > 0, cur, status), on, source, source_effect, move_status)


def cure_status(st, ref, on, silent=False):
    """PS: sim/pokemon.ts cureStatus: '-curestatus' X status [msg|silent], then setStatus('')."""
    m = st.mons
    cur = mget(m.status, ref).astype(I32)
    go = jnp.asarray(on) & (mget(m.hp, ref) > 0) & (cur > 0)
    st = EV.emit(st, go, EV.EV_CURESTATUS, actor=ref, move=C.eff_cond(cur), flags=EV.F_SILENT if silent else EV.F_MSG)
    return clear_status(st, ref, go)


def clear_status(st, ref, on):
    m = st.mons
    go = jnp.asarray(on) & (mget(m.hp, ref) > 0) & (mget(m.status, ref) > 0)
    return st.replace(mons=m.replace(status=mset(m.status, ref, 0, go),
                                     status_state=mset(m.status_state, ref, jnp.zeros(4, I16), go)))


def effect_damage(st, dex, rng, target, amount, effect, on, source=-1):
    """PS battle.damage(d, target, source, effect) for a non-move effect -> spreadDamage (sim/battle.ts:2108-2210):
    clamp to >= 1, Damage event (Magic Guard etc., M4), target.damage, '-damage' X hp '[from] effect'."""
    on = jnp.asarray(on) & (mget(st.mons.hp, target) > 0) & mget(st.mons.is_active, target)
    amount = jnp.maximum(jnp.asarray(amount, I32), 1)
    st, dealt = F.damage(st, target, amount, on)
    m = st.mons
    st = st.replace(mons=m.replace(hurt_this_turn=mset(m.hurt_this_turn, target, mget(m.hp, target), on & (dealt != 0))))
    shown = jnp.where(jnp.asarray(effect) == CO.eff("tox"), CO.eff("psn"), effect)   # PS :2158 'tox' -> 'psn'
    return EV.emit_hp(st, on, EV.EV_DAMAGE, target, move=shown, flags=EV.F_FROM), rng


def _start_msg(hc, st, name):
    """'-status' target status [from] ...: a move source gives '[from] move:' only for slp (Champions onStart)."""
    eff = hc.effect
    is_move = (eff >= ids.MOVE_BASE) & (eff < ids.MOVE_BASE + len(ids.MOVES) - 1)
    is_ability = (eff >= ids.ABILITY_BASE) & (eff < ids.ABILITY_BASE + len(ids.ABILITIES) - 1)
    flags = jnp.where(is_ability, EV.F_FROM_ABILITY, jnp.where(is_move & (name == "slp"), EV.F_FROM_MOVE, 0))
    return EV.emit(st, True, EV.EV_STATUS, actor=hc.target, move=CO.eff(name), v1=jnp.where(flags > 0, eff, -1),
                   flags=flags)


# ------------------------------------------------------------------ Start
@handler(ids.HOOK_START, CO.eff("brn"), ps="data/conditions.ts:2-14")
def brn_start(hc, st, dex, rng):
    return hc, _start_msg(hc, st, "brn"), rng


@handler(ids.HOOK_START, CO.eff("par"), ps="data/conditions.ts:23-30")
def par_start(hc, st, dex, rng):
    return hc, _start_msg(hc, st, "par"), rng


@handler(ids.HOOK_START, CO.eff("psn"), ps="data/conditions.ts:84-92")
def psn_start(hc, st, dex, rng):
    return hc, _start_msg(hc, st, "psn"), rng


@handler(ids.HOOK_START, CO.eff("tox"), ps="data/conditions.ts:101-111")
def tox_start(hc, st, dex, rng):
    st = set_status_field(st, hc.target, _status_slot("tox", "stage"), 0, True)
    return hc, _start_msg(hc, st, "tox"), rng


@handler(ids.HOOK_START, CO.eff("slp"), ps="data/mods/champions/conditions.ts:12-30")
def slp_start(hc, st, dex, rng):
    st = _start_msg(hc, st, "slp")
    key = C.dkey(st, ids.DK_SLEEP_DURATION, actor=hc.target, subject=CO.eff("slp"))
    i, rng = R.sample(rng, key, 3)
    t = jnp.array([2, 3, 3], I32)[i]
    st = set_status_field(st, hc.target, _status_slot("slp", "startTime"), t, True)
    st = set_status_field(st, hc.target, _status_slot("slp", "time"), t, True)
    return hc, st, rng                   # Nightmare: not in Reg M-C


@handler(ids.HOOK_START, CO.eff("frz"), ps="data/mods/champions/conditions.ts:32-44")
def frz_start(hc, st, dex, rng):
    st = _start_msg(hc, st, "frz")       # Shaymin-Sky: not in Reg M-C
    st = set_status_field(st, hc.target, _status_slot("frz", "startTime"), 3, True)
    st = set_status_field(st, hc.target, _status_slot("frz", "time"), 3, True)
    return hc, st, rng


# ------------------------------------------------------------------ BeforeMove
def _cant(st, ref, name, on):
    return EV.emit(st, on, EV.EV_CANT, actor=ref, move=CO.eff(name))


@handler(ids.HOOK_BEFOREMOVE, CO.eff("par"), ps="data/mods/champions/conditions.ts:3-9")
def par_before_move(hc, st, dex, rng):
    ref = hc.target
    key = C.dkey(st, ids.DK_PARALYSIS, actor=ref, subject=CO.eff("par"))
    full, rng = R.chance(rng, key, 1, 8)
    st = _cant(st, ref, "par", full)
    return hc.ret(RS.R(RS.BOOL, 0), full), st, rng


@handler(ids.HOOK_BEFOREMOVE, CO.eff("slp"), ps="data/conditions.ts:146-162")
def slp_before_move(hc, st, dex, rng):
    ref = hc.target
    ts = _status_slot("slp", "time")
    early = mget(st.mons.ability, ref) == ids.ABILITY_EARLYBIRD
    t = status_field(st, ref, ts).astype(I32) - 1 - early.astype(I32)
    st = set_status_field(st, ref, ts, t, True)
    wake = t <= 0
    st = cure_status(st, ref, wake)
    st = _cant(st, ref, "slp", ~wake)
    usable = dex.move_sleep_usable[jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)] == 1
    return hc.ret(RS.R(RS.BOOL, 0), ~wake & ~usable), st, rng


@handler(ids.HOOK_BEFOREMOVE, CO.eff("frz"), ps="data/mods/champions/conditions.ts:45-55")
def frz_before_move(hc, st, dex, rng):
    ref = hc.target
    move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
    defrost = ((dex.move_flags[move, ids.MF_DEFROST[0]] >> ids.MF_DEFROST[1]) & 1) == 1
    burnup = (move == getattr(ids, "MOVE_BURNUP", -1)) & ~jnp.any(mget(st.mons.types, ref) == ids.TYPE_FIRE)
    skip = defrost & ~burnup
    ts = _status_slot("frz", "time")
    t = status_field(st, ref, ts).astype(I32) - 1
    st = set_status_field(st, ref, ts, t, ~skip)
    key = C.dkey(st, ids.DK_FREEZE_THAW, actor=ref, subject=CO.eff("frz"))
    thaw, rng = R.chance(rng, key, 1, 4, active=~skip & (t > 0))
    cure = ~skip & ((t <= 0) | thaw)
    st = cure_status(st, ref, cure)
    stuck = ~skip & ~cure
    st = _cant(st, ref, "frz", stuck)
    return hc.ret(RS.R(RS.BOOL, 0), stuck), st, rng


# ------------------------------------------------------------------ Residual / SwitchIn
@handler(ids.HOOK_RESIDUAL, CO.eff("brn"), ps="data/conditions.ts:15-18")
def brn_residual(hc, st, dex, rng):
    ref = hc.holder
    st, rng = effect_damage(st, dex, rng, ref, mget(st.mons.maxhp, ref).astype(I32) // 16, CO.eff("brn"), True)
    return hc, st, rng


@handler(ids.HOOK_RESIDUAL, CO.eff("psn"), ps="data/conditions.ts:93-96")
def psn_residual(hc, st, dex, rng):
    ref = hc.holder
    st, rng = effect_damage(st, dex, rng, ref, mget(st.mons.maxhp, ref).astype(I32) // 8, CO.eff("psn"), True)
    return hc, st, rng


@handler(ids.HOOK_RESIDUAL, CO.eff("tox"), ps="data/conditions.ts:115-121")
def tox_residual(hc, st, dex, rng):
    ref = hc.holder
    ss = _status_slot("tox", "stage")
    stage = jnp.minimum(status_field(st, ref, ss).astype(I32) + 1, 15)
    st = set_status_field(st, ref, ss, stage, True)
    mx = mget(st.mons.maxhp, ref).astype(I32)
    # clampIntRange(baseMaxhp / 16, 1) * stage: clampIntRange floors (PS: sim/battle.ts clampIntRange)
    amount = jnp.maximum(mx // 16, 1) * stage
    st, rng = effect_damage(st, dex, rng, ref, amount, CO.eff("tox"), True)
    return hc, st, rng


@handler(ids.HOOK_SWITCHIN, CO.eff("tox"), ps="data/conditions.ts:112-114")
def tox_switch_in(hc, st, dex, rng):
    return hc, set_status_field(st, hc.holder, _status_slot("tox", "stage"), 0, True), rng


# ------------------------------------------------------------------ freeze thaw paths (base frz, inherited by [C])
@handler(ids.HOOK_MODIFYMOVE, CO.eff("frz"), ps="data/conditions.ts:262-267")
def frz_modify_move(hc, st, dex, rng):
    ref = hc.target
    move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
    defrost = ((dex.move_flags[move, ids.MF_DEFROST[0]] >> ids.MF_DEFROST[1]) & 1) == 1
    st = EV.emit(st, defrost, EV.EV_CURESTATUS, actor=ref, move=CO.eff("frz"), v1=hc.effect, flags=EV.F_FROM_MOVE_CURE)
    return hc, clear_status(st, ref, defrost), rng


@handler(ids.HOOK_AFTERMOVESECONDARY, CO.eff("frz"), ps="data/conditions.ts:268-272")
def frz_after_move_secondary(hc, st, dex, rng):
    move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
    return hc, cure_status(st, hc.target, dex.move_thaws_target[move] == 1), rng


@handler(ids.HOOK_DAMAGINGHIT, CO.eff("frz"), ps="data/conditions.ts:273-277")
def frz_damaging_hit(hc, st, dex, rng):
    move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
    fire = (dex.move_type[move] == ids.TYPE_FIRE) & (dex.move_category[move] != ids.CAT_STATUS) & (
        move != getattr(ids, "MOVE_POLARFLARE", -1))
    return hc, cure_status(st, hc.target, fire), rng

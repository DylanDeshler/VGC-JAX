"""Copy / ability-swap batch: Baton Pass, Skill Swap, Worry Seed, Psych Up, Transform (moves); Imposter, Protean,
Libero (abilities).

PS: data/moves.ts, data/abilities.ts (no Champions overrides for these: data/mods/champions/{moves,abilities}.ts),
sim/pokemon.ts copyVolatileFrom / transformInto / setAbility, sim/battle.ts skillSwap, [C] data/mods/champions/
scripts.ts clearVolatile (moveSlots = baseMoveSlots, transformed = false: battle/switching.py clear_volatile_many).

Layout: the dispatched Hit / TryHit handlers below only compute their result; the state work of Transform, Skill Swap
and Worry Seed (incl. the copied / swapped abilities' Start, i.e. effects/abilities.py switch_in_vec) runs undispatched
in `post_hit`, called once by the hit loop right after the Hit events of the first target (battle/move_pipeline.py):
nothing is emitted between a status move's onHit and that point, and one masked pass keeps switch_in_vec out of the
Hit dispatch (effects/abilities.py: no heavy code inside dispatched handlers). Imposter runs at the head of
switch_in_vec, which then plays the copied ability's Start.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset
from vgcjax.data import ids
from vgcjax.effects import abilities as AB
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler

I16, I32, U32 = jnp.int16, jnp.int32, jnp.uint32


def mid(name):
    return getattr(ids, f"MOVE_{name.upper()}", -999)


def _ab(name):
    return getattr(ids, f"ABILITY_{name.upper()}", -999)


def _flag(dex, ability, bit):
    return (dex.ability_flags[jnp.maximum(jnp.asarray(ability, I32), 0)].astype(I32) & bit) != 0


# ------------------------------------------------------------------ Protean / Libero
def protean(st, dex, u, move, mtype, on):
    """Protean / Libero onPrepareHit (data/abilities.ts protean / libero; runEvent('PrepareHit') after the move's own
    Try / PrepareHit, sim/battle-actions.ts:640-650): unless effectState.protean (libero), move.hasBounced (no Magic
    Bounce / Coat in Reg M-C), a futuremove flag, sourceEffect snatch (none) or move.callsMove; type = move.type
    (after ModifyType; Struggle '???' excluded), getTypes().join() !== type -> setType(type) (no Arceus / Silvally /
    Terastallization in Reg M-C: always true; addedType cleared), effectState.protean = true, '-start' source
    'typechange' type '[from] ability: Protean' (START move = v1 = the ability, v2 = 1: harness/events.py prints
    'typechange'). abilityState is new at every switch-in / setAbility, so this is once per switch-in."""
    u = jnp.clip(jnp.asarray(u, I32), 0, 7)
    m = st.mons
    ab = mget(m.ability, u).astype(I32)
    w, b = ids.MF_FUTUREMOVE
    excl = (((dex.move_flags[move, w] >> b) & 1) == 1) | (dex.move_calls_move[move] == 1)
    t = mget(m.types, u).astype(I32)
    same = (t[0] == mtype) & (t[1] <= 0) & (mget(m.added_type, u) == 0)
    base = jnp.asarray(on) & ~excl & (mtype > 0) & (mtype != ids.TYPE_TYPELESS) & ~same & (mget(m.hp, u) > 0)
    for name in ("protean", "libero"):
        a = _ab(name)
        if a <= 0:
            continue
        slot = [s for k, s, _ in AB.FIELDS[name] if k == name][0]
        done = (((mget(m.ability_state, u)[3] >> slot) & 1) == 1) & (AB.state_field(st, u, name, name) != 0)
        go = base & (ab == a) & ~done
        m = st.mons
        new = jnp.stack([mtype, jnp.int32(-1)]).astype(m.types.dtype)
        st = st.replace(mons=m.replace(types=mset(m.types, u, new, go), added_type=mset(m.added_type, u, 0, go)))
        st = AB.set_state_field(st, u, name, name, 1, go)
        e = AB.aeff(name)
        st = EV.emit(st, go, EV.EV_START, actor=u, move=e, v1=e, v2=1, flags=EV.F_FROM_ABILITY)
    return st


# ------------------------------------------------------------------ Psych Up
if mid("psychup") > 0:
    @handler(ids.HOOK_HIT, int(C.eff_move(mid("psychup"))), ps="data/moves.ts psychup.onHit")
    def psychup_hit(hc, st, dex, rng):
        """source.boosts[i] = target.boosts[i] for every stat; the crit volatiles ['dragoncheer', 'focusenergy',
        'gmaxchistrike', 'laserfocus'] are all removed from the source, then each the target has is added to the
        source in that order (their onStart prints '-start' ... '[silent]' for Psych Up: the same START record), with
        dragoncheer's hasDragonType copied from the target (data/moves.ts:14217-14235; no G-Max Chi Strike / Laser
        Focus in Reg M-C); '-copyboost' source target '[from] move: Psych Up' (COPYBOOST). Returns undefined."""
        t, u = jnp.clip(hc.target, 0, 7), jnp.clip(hc.source, 0, 7)
        m = st.mons
        st = st.replace(mons=m.replace(boosts=mset(m.boosts, u, mget(m.boosts, t), True)))
        crit = [v for v in ("dragoncheer", "focusenergy") if v in ids.CONDITIONS]
        had = {v: CO.has_volatile(st, t, CO.cid(v)) for v in crit}
        for v in crit:
            st, rng, _ = CO.remove_volatile(st, dex, rng, u, CO.cid(v), CO.has_volatile(st, u, CO.cid(v)))
        for v in crit:
            c = CO.cid(v)
            st, rng, _ = CO.add_volatile(st, dex, rng, u, c, had[v], source=u, source_effect=C.eff_move(mid("psychup")))
            if v == "dragoncheer":
                sl = CO.slot_of("dragoncheer", "hasDragonType")
                st = CO.set_vol_field(st, u, c, sl, CO.vol_state(st, t, c)[sl].astype(I32),
                                      had[v] & CO.has_volatile(st, u, c))
        st = EV.emit(st, True, EV.EV_COPYBOOST, actor=u, target=t, move=C.eff_move(mid("psychup")))
        return hc, st, rng


# ------------------------------------------------------------------ Baton Pass
if mid("batonpass") > 0:
    @handler(ids.HOOK_HIT, int(C.eff_move(mid("batonpass"))), ps="data/moves.ts batonpass.onHit")
    def batonpass_hit(hc, st, dex, rng):
        """!canSwitch(target.side) (no Commander in Reg M-C) -> attrLastMove('[still]'), '-fail' target, NOT_FAIL.
        self.onHit only sets skipPursuit (no Pursuit in Reg M-C); moveData.self suppresses the did-nothing '-fail'
        (battle/move_pipeline.py has_self). selfSwitch 'copyvolatile': copy_volatile_from at the switch-in."""
        from vgcjax.battle import faint as F
        t = jnp.clip(hc.target, 0, 7)
        fail = F.can_switch(st, t // 4) == 0
        st = EV.attr_last_move(st, fail, EV.F_STILL)
        st = EV.emit(st, fail, EV.EV_FAIL, actor=t)
        return hc.ret(RS.R(RS.NOTFAIL, 0), fail), st, rng


_NOCOPY = None


def _copy_masks():
    """(copyable volatile words u32[WV], copyable vol_dur bool[NV], copyable vol_fx rows bool[NVF]) from the
    conditions' noCopy flag (tables cond_flags bit 1)."""
    global _NOCOPY
    if _NOCOPY is None:
        from vgcjax.core.dex import load_tables_np
        fl = load_tables_np()["cond_flags"]
        ok = np.array([(int(fl[ids.VOL_BASE + k]) & 2) == 0 for k in range(S.NV)], bool)
        words = np.zeros(S.WV, np.uint32)
        for k in range(S.NV):
            if ok[k]:
                words[k // 32] |= np.uint32(1 << (k % 32))
        fx = np.array([ok[ids.CONDITIONS.index(c) - ids.VOL_BASE] for c in S.VOL_STATEFUL] or [False], bool)
        _NOCOPY = (words, ok, fx)
    return _NOCOPY


def copies_volatiles(dex, source_effect):
    """switchIn's switchCopyFlag: the sourceEffect move's selfSwitch is 'copyvolatile' (codegen SELF_SWITCH 2).
    `source_effect` is the instaswitch's MoveId (0 none)."""
    se = jnp.asarray(source_effect, I32)
    return (se > 0) & (dex.move_self_switch[jnp.maximum(se, 0)] == 2)


def copy_volatile_from(st, r_in, r_old, on):
    """PS: sim/pokemon.ts:1267-1289 copyVolatileFrom(oldActive, 'copyvolatile'), before oldActive.clearVolatile()
    (battle/switching.py switch_in): boosts, and every volatile without noCopy as a shallow clone of its state
    (duration and fields; initEffectState gives effectOrder 0 since the incoming mon is not active yet, and the
    engine keeps no volatile effectOrder). The incoming mon's own clearVolatile is a no-op here (a benched mon is
    already cleared). No linked volatiles (linkedPokemon) and no onCopy handler (Gastro Acid, Power Trick) in the impl
    stage."""
    words, ok, fx = _copy_masks()
    r_in = jnp.clip(jnp.asarray(r_in, I32), 0, 7)
    r_old = jnp.clip(jnp.asarray(r_old, I32), 0, 7)
    m = st.mons
    vb = mget(m.vol_bits, r_old) & jnp.asarray(words, U32)
    vd = jnp.where(jnp.asarray(ok), mget(m.vol_dur, r_old), 0)
    vf = jnp.where(jnp.asarray(fx)[:, None], mget(m.vol_fx, r_old), 0)
    m = m.replace(boosts=mset(m.boosts, r_in, mget(m.boosts, r_old), on),
                  vol_bits=mset(m.vol_bits, r_in, vb, on), vol_dur=mset(m.vol_dur, r_in, vd, on),
                  vol_fx=mset(m.vol_fx, r_in, vf, on))
    return st.replace(mons=m)


# ------------------------------------------------------------------ setAbility pieces
def _new_ability(st, ref, new, on):
    """this.ability = new; this.abilityState = initEffectState({id, target}) (active: effectOrder++)."""
    m = st.mons
    new = jnp.asarray(new, I32)
    m = m.replace(ability=mset(m.ability, ref, new.astype(m.ability.dtype), on),
                  ability_state=mset(m.ability_state, ref, jnp.zeros(4, I16), on),
                  ability_state_of=mset(m.ability_state_of, ref, new.astype(m.ability_state_of.dtype), on),
                  ability_seq=mset(m.ability_seq, ref, st.meta.effect_seq, on))
    return CO.next_effect_order(st.replace(mons=m), on)


# ------------------------------------------------------------------ Transform / Imposter
def transform_ok(st, u, t):
    """transformInto's guard (sim/pokemon.ts:1293-1300): target fainted, a substitute on the target, either side
    already transformed, either side's active Illusion (this.illusion || pokemon.illusion: Events.alias,
    effects/batch_forme.py) -> false (no Eternamax / Ogerpon / Terapagos / Terastallization in Reg M-C)."""
    m = st.mons
    t = jnp.asarray(t, I32)
    tc = jnp.clip(t, 0, 7)
    uc = jnp.clip(jnp.asarray(u, I32), 0, 7)
    sub = CO.has_volatile(st, tc, CO.cid("substitute")) if "substitute" in ids.CONDITIONS else False
    ill = (st.events.alias.astype(I32)[uc] >= 0) | (st.events.alias.astype(I32)[tc] >= 0)
    return (t >= 0) & ~mget(m.fainted, tc) & ~sub & ~mget(m.transformed, tc) & ~mget(m.transformed, u) & ~ill


def transform_into(st, dex, u, t, on, effect=-1):
    """PS: sim/pokemon.ts:1291-1404 transformInto(target, effect) once the guard passed: setSpecies(target species,
    isTransform) (types, weight, storedStats from the species with the user's own set; speed = that Spe:
    battle/switching.py set_species), transformed = true, weighthg, setType(target.getTypes(true, true)) (a roosting
    target: its species' types, the engine's stand-in for roost.typeWas), addedType, storedStats := the target's
    (speed keeps the setSpecies value), moveSlots := the target's moves with pp = maxpp = min(5, move.pp) (the
    user's baseMoveSlots pp / disabled are kept in base_slots), timesAttacked, boosts; the crit volatiles (none in
    the impl stage); '-transform' X target ['[from] ability: Imposter'] (TRANSFORM); setAbility(target.ability,
    this, null, true, true): no cantsuppress / SetAbility checks, singleEvent('End', old) (effects/abilities.py
    end_vec), new ability + abilityState, no message, Start(new) only if it differs (returned; the caller runs it).
    Returns (st, start: bool)."""
    from vgcjax.battle import switching as SW
    u = jnp.clip(jnp.asarray(u, I32), 0, 7)
    t = jnp.clip(jnp.asarray(t, I32), 0, 7)
    on = jnp.asarray(on)
    m = st.mons
    tsp = mget(m.species, t).astype(I32)
    ttypes = mget(m.types, t)
    if "roost" in ids.CONDITIONS:
        ttypes = jnp.where(CO.has_volatile(st, t, CO.cid("roost")), dex.species_types[tsp].astype(ttypes.dtype),
                           ttypes)
    t_added, t_weight, t_stats = mget(m.added_type, t), mget(m.weight_hg, t), mget(m.stats, t)
    t_moves, t_boosts, t_ta = mget(m.move_ids, t).astype(I32), mget(m.boosts, t), mget(m.times_attacked, t)
    old_ab, new_ab = mget(m.ability, u).astype(I32), mget(m.ability, t).astype(I32)
    base = mget(m.pp, u).astype(I32) | (mget(m.move_disabled, u).astype(I32) << 8)
    st = SW.set_species(st, dex, u, tsp, on)
    m = st.mons
    npp = jnp.where(t_moves > 0, jnp.minimum(5, dex.move_pp[t_moves].astype(I32)), 0)
    m = m.replace(transformed=mset(m.transformed, u, True, on), weight_hg=mset(m.weight_hg, u, t_weight, on),
                  types=mset(m.types, u, ttypes, on), added_type=mset(m.added_type, u, t_added, on),
                  stats=mset(m.stats, u, t_stats, on), base_slots=mset(m.base_slots, u, base, on),
                  move_ids=mset(m.move_ids, u, t_moves, on), pp=mset(m.pp, u, npp, on),
                  maxpp=mset(m.maxpp, u, npp, on), move_disabled=mset(m.move_disabled, u, jnp.zeros(4, I32), on),
                  move_used=mset(m.move_used, u, jnp.zeros(4, bool), on),
                  times_attacked=mset(m.times_attacked, u, t_ta, on), boosts=mset(m.boosts, u, t_boosts, on))
    st = st.replace(mons=m)
    st = EV.emit(st, on, EV.EV_TRANSFORM, actor=u, target=t, move=effect)
    go = on & (mget(st.mons.hp, u) > 0)                                    # setAbility: if (!this.hp) return false
    st = AB.end_vec(st, u[None], go[None])
    st = _new_ability(st, u, new_ab, go)
    return st, go & (new_ab != old_ab) & (new_ab > 0)


if _ab("imposter") > 0:
    @handler(ids.HOOK_SWITCHIN, AB.aeff("imposter"), ps="data/abilities.ts imposter.onSwitchIn (switch_in_vec)")
    def _imposter_row(hc, st, dex, rng):
        return hc, st, rng                       # listed for the SwitchIn walk; runs in abilities.switch_in_vec


def imposter(st, dex, r, on):
    """Imposter onSwitchIn (data/abilities.ts imposter): target = pokemon.side.foe.active[1 - pokemon.position] (the
    diagonal foe slot, possibly fainted: the guard then fails); transformInto(target, ability: Imposter). Returns
    (st, start): the copied ability's Start is run by the caller (effects/abilities.py switch_in_vec)."""
    from vgcjax.battle import move_pipeline as MP
    if _ab("imposter") <= 0:
        return st, jnp.zeros((), bool)
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    pos = MP.position(st, r)
    fs = 1 - r // 4
    b = st.sides.active[fs, jnp.clip(1 - pos, 0, 1)].astype(I32)
    t = fs * 4 + jnp.maximum(b, 0)
    ok = jnp.asarray(on) & (mget(st.mons.ability, r) == _ab("imposter")) & (pos >= 0) & (b >= 0) & \
        transform_ok(st, r, jnp.where(b >= 0, t, -1))
    return transform_into(st, dex, r, t, ok, effect=AB.aeff("imposter"))


if mid("transform") > 0:
    @handler(ids.HOOK_HIT, int(C.eff_move(mid("transform"))), ps="data/moves.ts transform.onHit")
    def transform_hit(hc, st, dex, rng):
        """return pokemon.transformInto(target): the guard's result here, the transformation in post_hit."""
        ok = transform_ok(st, jnp.clip(hc.source, 0, 7), hc.target)
        return hc.ret(RS.R(RS.BOOL, ok.astype(I32))), st, rng


# ------------------------------------------------------------------ Skill Swap
def skill_swap_ok(st, dex, u, t):
    """sim/battle.ts:1325-1335 skillSwap guard: either fainted, either ability failskillswap -> false (no Dynamax;
    gen > 5; runEvent('SetAbility'): only Ability Shield, not in Reg M-C)."""
    m = st.mons
    u, t = jnp.clip(jnp.asarray(u, I32), 0, 7), jnp.clip(jnp.asarray(t, I32), 0, 7)
    sa, ta = mget(m.ability, u), mget(m.ability, t)
    return ~mget(m.fainted, u) & ~mget(m.fainted, t) & ~_flag(dex, sa, ids.AF_FAILSKILLSWAP) & \
        ~_flag(dex, ta, ids.AF_FAILSKILLSWAP)


if mid("skillswap") > 0:
    @handler(ids.HOOK_HIT, int(C.eff_move(mid("skillswap"))), ps="data/moves.ts skillswap.onHit")
    def skillswap_hit(hc, st, dex, rng):
        """return this.skillSwap(source, target): false on the guard, else undefined (the swap runs in post_hit)."""
        ok = skill_swap_ok(st, dex, hc.source, hc.target)
        return hc.ret(RS.R(RS.BOOL, 0), ~ok), st, rng


# ------------------------------------------------------------------ Worry Seed
def worry_seed_immune(st, move, targets):
    """Worry Seed onTryImmunity: target.ability === 'truant' || 'insomnia' -> false ('-immune', hitStepTryImmunity).
    targets: i32[3] -> bool[3]."""
    if mid("worryseed") <= 0:
        return jnp.zeros(jnp.shape(targets), bool)
    ab = mget(st.mons.ability, jnp.clip(targets, 0, 7))
    return (move == mid("worryseed")) & ((ab == _ab("insomnia")) | (ab == _ab("truant")))


if mid("worryseed") > 0:
    @handler(ids.HOOK_TRYHIT, int(C.eff_move(mid("worryseed"))), ps="data/moves.ts worryseed.onTryHit")
    def worryseed_try_hit(hc, st, dex, rng):
        """target.getAbility().flags['cantsuppress'] -> false."""
        cant = _flag(dex, mget(st.mons.ability, jnp.clip(hc.target, 0, 7)), ids.AF_CANTSUPPRESS)
        return hc.ret(RS.R(RS.BOOL, 0), cant), st, rng


# ------------------------------------------------------------------ the onHit work site
def post_hit(st, dex, rng, u, t, move, on):
    """The onHit work of the first target `t` of user `u` (battle/move_pipeline.py, after the Hit events):
      Transform: transformInto(target) (transform_into) when the guard passes;
      Skill Swap (sim/battle.ts:1325-1366): '-activate' source 'Skill Swap' ... '[of] target' (ACTIVATE; the
        abilities shown or blank between allies: not in the reduced record), End(source ability), End(target
        ability), the abilities swapped with new abilityStates (source first: effectOrder), Start(source's old
        ability on the target), then Start(target's old ability on the source);
      Worry Seed onHit: target.setAbility('insomnia') (hp > 0; cantsuppress already failed at TryHit; no SetAbility
        handler): End(old), new ability, '-ability' target 'Insomnia' old '[from] move: Worry Seed' (ABILITY),
        Start(Insomnia: none); target asleep -> cureStatus() ('-curestatus' X slp '[msg]');
      Simple Beam / Entrainment onHit (effects/batch7c.py): target.setAbility('simple') / (source.ability, source):
        End(old), new ability, '-ability' target New Old '[from] move: X' ['[of] source'] (ABILITY), Start(new).
    Start = effects/abilities.py switch_in_vec(start=True), one compiled pass over two slots. Returns (st, rng)."""
    from vgcjax.effects import statuses as STS
    on = jnp.asarray(on)
    u = jnp.clip(jnp.asarray(u, I32), 0, 7)
    t = jnp.clip(jnp.asarray(t, I32), 0, 7)
    from vgcjax.effects import batch_reactive as BR
    sa_moves = [mid(x) for x in ("skillswap", "worryseed", "transform", "simplebeam", "entrainment") if mid(x) > 0]
    sa_on = on & jnp.any(jnp.asarray(sa_moves or [-1], I32) == move)
    st = BR.mold_unmask(st, jnp.stack([u, t]), sa_on)      # Mold Breaker's mask is not pokemon.ability in PS
    zero = jnp.zeros((), bool)
    tf, ss, ws = zero, zero, zero
    start0 = zero
    if mid("transform") > 0:
        tf = on & (move == mid("transform")) & transform_ok(st, u, t)
        st, start0 = transform_into(st, dex, u, t, tf)
    if mid("skillswap") > 0:
        ss = on & (move == mid("skillswap")) & skill_swap_ok(st, dex, u, t)
    if mid("worryseed") > 0 and _ab("insomnia") > 0:
        ws = on & (move == mid("worryseed")) & (mget(st.mons.hp, t) > 0)
    from vgcjax.effects import batch7c as B7
    sb, en, sb_en_new = B7.set_ability_moves(st, dex, u, t, move, on)      # Simple Beam / Entrainment
    sbe = sb | en
    m = st.mons
    sa, ta = mget(m.ability, u).astype(I32), mget(m.ability, t).astype(I32)
    st = EV.emit(st, ss, EV.EV_ACTIVATE, actor=u, move=C.eff_move(mid("skillswap")))
    st = AB.end_vec(st, jnp.stack([u, t]), jnp.stack([ss, ss | ws | sbe]))
    st = _new_ability(st, u, ta, ss)
    st = _new_ability(st, t, jnp.where(ss, sa, jnp.where(sbe, sb_en_new, _ab("insomnia"))), ss | ws | sbe)
    st = EV.emit(st, ws, EV.EV_ABILITY, actor=t, move=AB.aeff("insomnia") if _ab("insomnia") > 0 else -1)
    st = EV.emit(st, sbe, EV.EV_ABILITY, actor=t, move=ids.ABILITY_BASE + sb_en_new - 1)
    st = STS.cure_status(st, t, ws & (mget(st.mons.status, t) == ids.COND_SLP))
    refs = jnp.stack([jnp.where(ss | sbe, t, u), u])
    ons = jnp.stack([ss | start0 | sbe, ss])

    def body(i, c):
        st_, rng_ = c
        return AB.switch_in_vec(st_, dex, rng_, refs[i], ons[i], start=True)
    st, rng = jax.lax.fori_loop(0, 2, body, (st, rng))
    st = BR.mold_remask(st, jnp.stack([u, t]), sa_on)
    return st, rng

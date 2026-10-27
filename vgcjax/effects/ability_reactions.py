"""Abilities that react to being hit or to boosts (data/abilities.ts; Champions overrides: Regenerator, Natural Cure
in data/mods/champions/abilities.ts). Undispatched masked passes called from the event sites (perf rule):
TryHit (move_pipeline.move_begin after the dispatched TryHit), DamagingHit (spread_move_hit), Hit (Anger Point),
AfterEachBoost / AfterBoost (boost_v), Flinch (volatiles.flinch), Residual (fieldevent walk), SwitchOut
(switching.switch_in).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget
from vgcjax.core import rng as R
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.abilities import aeff, aid

I32 = jnp.int32


def _is(ab, name):
    return ab == aid(name) if aid(name) > 0 else jnp.zeros(jnp.shape(ab), bool)


def _flag(dex, move, flag):
    w, b = flag
    return ((dex.move_flags[move, w] >> b) & 1) == 1


# ------------------------------------------------------------------ TryHit (left to right; priority 1 then 0)
ABSORB = {"voltabsorb": "TYPE_ELECTRIC", "waterabsorb": "TYPE_WATER", "eartheater": "TYPE_GROUND"}


def try_hit(st, dex, rng, targets, tv, th, u, move, mtype):
    """runEvent('TryHit', targets, pokemon, move) for the abilities (after the dispatched protections, which have
    higher priority): Sap Sipper (priority 1), then Volt Absorb / Water Absorb / Earth Eater (heal 1/4 or
    '-immune'), Motor Drive (+1 Spe or '-immune'), Flash Fire (addVolatile or '-immune'), Soundproof, Bulletproof,
    Good as Gold ('-immune'), Telepathy (an ally's damaging move, '-activate'); each returns null. Targets whose
    relay is already falsy are skipped. Returns ((tags, vals), st, rng, acc_true) where acc_true: Flash Fire set
    move.accuracy = true (every later accuracy check passes)."""
    from vgcjax.battle import move_pipeline as MP
    from vgcjax.effects import statuses as STS
    tt, tvv = th
    m = st.mons
    cat = dex.move_category[move]
    acc_true = jnp.zeros((), bool)
    # the two boost reactions (Sap Sipper +1 Atk, Motor Drive +1 Spe) for every target in one boost (one ability
    # per mon, distinct targets); their messages are emitted in each target's turn below
    T = targets.shape[0]
    tcl = jnp.clip(targets, 0, 7)
    ab_all = mget(st.mons.ability, tcl)
    live0 = tv & RS.truthy((tt, tvv)) & (targets != u)
    sap_all = live0 & _is(ab_all, "sapsipper") & (mtype == ids.TYPE_GRASS)
    md_all = live0 & _is(ab_all, "motordrive") & (mtype == ids.TYPE_ELECTRIC)
    z = jnp.zeros((T, 7), I32)
    dlt = jnp.where(sap_all[:, None], z.at[:, 0].set(1), z.at[:, 4].set(1))
    st, br_all, bmsgs = MP.boost_v(st, targets, dlt, sap_all | md_all, source=u,
                                   ab_msg=jnp.where(sap_all, aeff("sapsipper"), aeff("motordrive")), simple=True)  # +1
    for prio in (1, 0):
        for j in range(targets.shape[0]):
            t = jnp.clip(targets[j], 0, 7)
            live = tv[j] & RS.truthy((tt[j], tvv[j]))
            ab = mget(st.mons.ability, t)
            other = targets[j] != u
            if prio == 1:
                sap = sap_all[j]
                st = MP.emit_boost(st, bmsgs, t, j, mask=sap)
                st = EV.emit(st, sap & ~RS.truthy((br_all[0][j], br_all[1][j])), EV.EV_IMMUNE, actor=t,
                             v1=aeff("sapsipper"))
                hit = sap
            else:
                hit = jnp.zeros((), bool)
                for name, ty in ABSORB.items():
                    a = live & other & _is(ab, name) & (mtype == getattr(ids, ty))
                    hp, mx = mget(st.mons.hp, t), mget(st.mons.maxhp, t)
                    healed = a & (hp < mx) & (hp > 0)
                    st, rng = STS.effect_heal(st, dex, rng, t, mx // 4, aeff(name), healed)
                    st = EV.emit(st, a & ~healed, EV.EV_IMMUNE, actor=t, v1=aeff(name))
                    hit = hit | a
                md = md_all[j]
                st = MP.emit_boost(st, bmsgs, t, j, mask=md)
                st = EV.emit(st, md & ~RS.truthy((br_all[0][j], br_all[1][j])), EV.EV_IMMUNE, actor=t,
                             v1=aeff("motordrive"))
                hit = hit | md
                if aid("flashfire") > 0 and "flashfire" in ids.CONDITIONS:
                    ff = live & other & _is(ab, "flashfire") & (mtype == ids.TYPE_FIRE)
                    st, rng, fr = CO.add_volatile(st, dex, rng, t, CO.cid("flashfire"), ff, source=u,
                                                  source_effect=aeff("flashfire"))
                    st = EV.emit(st, ff & ~RS.truthy(fr), EV.EV_IMMUNE, actor=t, v1=aeff("flashfire"))
                    hit = hit | ff
                    acc_true = acc_true | ff
                tnt = (move == getattr(ids, "MOVE_TAUNT", -1)) | (move == getattr(ids, "MOVE_ATTRACT", -1))
                for name, cond in (("oblivious", tnt), ("soundproof", other & _flag(dex, move, ids.MF_SOUND)),
                                   ("bulletproof", _flag(dex, move, ids.MF_BULLET)),
                                   ("goodasgold", other & (cat == ids.CAT_STATUS))):
                    a = live & _is(ab, name) & cond
                    st = EV.emit(st, a, EV.EV_IMMUNE, actor=t, v1=aeff(name))
                    hit = hit | a
                ally = (t // 4) == (jnp.clip(u, 0, 7) // 4)
                tp = live & other & ally & _is(ab, "telepathy") & (cat != ids.CAT_STATUS)
                st = EV.emit(st, tp, EV.EV_ACTIVATE, actor=t, move=aeff("telepathy"))
                hit = hit | tp
            tt = tt.at[j].set(jnp.where(hit, RS.NULL, tt[j]))
            tvv = tvv.at[j].set(jnp.where(hit, 0, tvv[j]))
    return (tt, tvv), st, rng, acc_true


# ------------------------------------------------------------------ DamagingHit
def damaging_hit_first(st, dex, rng, targets, damaged, u, move, hit_no):
    """Rough Skin (onDamagingHitOrder 1, before the other handlers of every target): contact ->
    this.damage(source.baseMaxhp / 8, source, target)."""
    from vgcjax.effects import statuses as STS
    contact = _flag(dex, move, ids.MF_CONTACT)
    for j in range(targets.shape[0]):
        t = jnp.clip(targets[j], 0, 7)
        rs = damaged[j] & contact & _is(mget(st.mons.ability, t), "roughskin")
        st, rng = STS.effect_damage(st, dex, rng, u, mget(st.mons.maxhp, u) // 8, aeff("roughskin"), rs, source=t)
    return st, rng


def damaging_hit_boosts(st, dex, targets, damaged, u, move, mtype):
    """The holders' self-boost DamagingHit abilities for every target in one boost (one ability per mon, distinct
    targets): Stamina (+1 Def, source = attacker), Weak Armor (physical: -1 Def +2 Spe, self), Justified (Dark: +1
    Atk), Rattled (Dark / Bug / Ghost: +1 Spe). Returns (st, msgs, on[T]); damaging_hit emits target j's messages."""
    from vgcjax.battle import move_pipeline as MP
    tcl = jnp.clip(targets, 0, 7)
    ab = mget(st.mons.ability, tcl)
    cat = dex.move_category[move]
    sta = damaged & _is(ab, "stamina")
    wa = damaged & _is(ab, "weakarmor") & (cat == ids.CAT_PHYSICAL)
    jus = damaged & _is(ab, "justified") & (mtype == ids.TYPE_DARK)
    rat = damaged & _is(ab, "rattled") & ((mtype == ids.TYPE_DARK) | (mtype == ids.TYPE_BUG) | (mtype == ids.TYPE_GHOST))
    z = jnp.zeros((targets.shape[0], 7), I32)
    d = jnp.where(sta[:, None], z.at[:, 1].set(1), jnp.where(wa[:, None], z.at[:, 1].set(-1).at[:, 4].set(2),
                  jnp.where(jus[:, None], z.at[:, 0].set(1), z.at[:, 4].set(1))))
    on = sta | wa | jus | rat
    src = jnp.where(wa, tcl, u)
    st, _, msgs = MP.boost_v(st, targets, d, on, source=src, ab_msg=ids.ABILITY_BASE + ab - 1)
    return st, msgs, on


def damaging_hit(st, dex, rng, j, t, damaged, u, move, mtype, hit_no, bmsgs, b_on):
    """The remaining DamagingHit ability handlers of target t (after its status's): Static / Flame Body / Poison
    Point (contact, 3/10 -> source.trySetStatus), Effect Spore (contact, powder-immunity, random(100) < 11 slp / 21
    par / 30 psn), Spicy Spray (brn), Stamina (+1 Def), Weak Armor (physical: -1 Def +2 Spe, self), Gooey
    ('-ability', -1 Spe on the source, isSecondary), Justified (Dark +1 Atk), Rattled (Dark / Bug / Ghost +1 Spe);
    then the attacker's Poison Touch (onSourceDamagingHit: contact, 3/10, not vs Shield Dust). ABILITY_ONHIT_PROC
    draws: actor holder, target the other mon, subject the ability, hit. Returns (st, rng)."""
    from vgcjax.battle import move_pipeline as MP
    from vgcjax.core import typechart as TC
    from vgcjax.effects import statuses as STS
    tr = jnp.clip(t, 0, 7)
    ab = mget(st.mons.ability, tr)
    contact = _flag(dex, move, ids.MF_CONTACT)
    cat = dex.move_category[move]
    # one trySetStatus on the source for the holder's status ability (one ability per mon); draws stay per ability
    any_on = jnp.zeros((), bool)
    s_val = jnp.zeros((), I32)
    for name, status in (("static", ids.COND_PAR), ("flamebody", ids.COND_BRN), ("poisonpoint", ids.COND_PSN)):
        on = damaged & contact & _is(ab, name)
        key = C.dkey(st, ids.DK_ABILITY_ONHIT_PROC, actor=tr, target=u, subject=aeff(name), hit=hit_no)
        ok, rng = R.chance(rng, key, 3, 10, active=on)
        any_on = any_on | (on & ok)
        s_val = jnp.where(on & ok, status, s_val)
    if aid("effectspore") > 0:
        powder_ok = ~TC.status_immune(dex, ids.IMMUNITY_KEYS.index("powder"), mget(st.mons.types, u))
        on = damaged & contact & _is(ab, "effectspore") & powder_ok & ~mget(st.mons.fainted, u)
        key = C.dkey(st, ids.DK_ABILITY_ONHIT_PROC, actor=tr, target=u, subject=aeff("effectspore"), hit=hit_no)
        r, rng = R.randint(rng, key, 0, 100, active=on)
        status = jnp.where(r < 11, ids.COND_SLP, jnp.where(r < 21, ids.COND_PAR, ids.COND_PSN))
        any_on = any_on | (on & (r < 30))
        s_val = jnp.where(on & (r < 30), status, s_val)
    sp = damaged & _is(ab, "spicyspray")
    any_on = any_on | sp
    s_val = jnp.where(sp, ids.COND_BRN, s_val)
    st, rng = _set_status(st, dex, rng, u, s_val, any_on, tr, ids.ABILITY_BASE + ab - 1)
    z = jnp.zeros(7, I32)
    st = MP.emit_boost(st, bmsgs, tr, j, mask=b_on[j])          # Stamina / Weak Armor / Justified / Rattled
    goo = damaged & contact & _is(ab, "gooey")
    st = EV.emit(st, goo, EV.EV_ABILITY, actor=tr, move=aeff("gooey"))
    st, _ = MP.boost(st, u, z.at[4].set(-1), goo, primary=True, source=tr, effect=aeff("gooey"))
    # the attacker's Poison Touch (Source prefix: after the target's own handlers)
    pt = damaged & contact & _is(mget(st.mons.ability, u), "poisontouch") & ~_is(ab, "shielddust")
    key = C.dkey(st, ids.DK_ABILITY_ONHIT_PROC, actor=u, target=tr, subject=aeff("poisontouch"), hit=hit_no)
    ok, rng = R.chance(rng, key, 3, 10, active=pt)
    st, rng = _set_status(st, dex, rng, tr, ids.COND_PSN, pt & ok, u, aeff("poisontouch"))
    return st, rng


def _set_status(st, dex, rng, ref, status, on, source, eff):
    """target.trySetStatus(status, source) with the ability as sourceEffect, messages emitted."""
    from vgcjax.effects import statuses as STS
    st, rng, _, msgs = STS.try_set_status_v(st, dex, rng, jnp.asarray(ref, I32)[None], status, jnp.asarray(on)[None],
                                            source, eff, 0)
    return STS.emit_status_msgs(st, msgs, 0, ref), rng


# ------------------------------------------------------------------ Flash Fire volatile (onModifyAtk / SpA mods)
def _flash_fire_mod(ctx, h):
    """flashfire condition: a Fire move from a Flash Fire holder -> chainModify(1.5)."""
    ok = (ctx.mtype == ids.TYPE_FIRE) & _is(mget(ctx.st.mons.ability, jnp.clip(h, 0, 7)), "flashfire")
    return ok, jnp.asarray(6144, I32), None


if aid("flashfire") > 0 and "flashfire" in ids.CONDITIONS:
    from vgcjax.effects import modifiers as MOD
    from vgcjax.effects.hooks import handler
    for _hk in ("ModifyAtk", "ModifySpA"):
        MOD.mod(_hk, CO.eff("flashfire"), ps=f"data/abilities.ts flashfire.condition.on{_hk}")(_flash_fire_mod)

    @handler(ids.HOOK_START, CO.eff("flashfire"), ps="data/abilities.ts flashfire.condition.onStart")
    def flashfire_start(hc, st, dex, rng):
        """'-start' target 'ability: Flash Fire'."""
        return hc, EV.emit(st, True, EV.EV_START, actor=hc.target, move=CO.eff("flashfire"), v1=-1), rng


# ------------------------------------------------------------------ Hit / Flinch / Residual / SwitchOut
def anger_point(st, t, crit, on):
    """onHit(target, source, move): a crit on a live holder -> boost({atk: 12}, target, target): '-setboost'
    target atk 6 '[from] ability: Anger Point'."""
    r = jnp.clip(t, 0, 7)
    go = jnp.asarray(on) & crit & _is(mget(st.mons.ability, r), "angerpoint") & (mget(st.mons.hp, r) > 0)
    m = st.mons
    b = mget(m.boosts, r)
    from vgcjax.battle.util import mset
    changed = go & (b[0] < 6)
    st = st.replace(mons=m.replace(boosts=mset(m.boosts, r, b.at[0].set(6), go),
                                   stats_raised_this_turn=mset(m.stats_raised_this_turn, r, True, changed)))
    return EV.emit(st, changed, EV.EV_SETBOOST, actor=r, v1=0, v2=6, move=aeff("angerpoint"))


def steadfast(st, r, on):
    """onFlinch: boost({spe: 1}) ('-ability' X 'Steadfast' 'boost')."""
    from vgcjax.battle import move_pipeline as MP
    go = jnp.asarray(on) & _is(mget(st.mons.ability, jnp.clip(r, 0, 7)), "steadfast")
    st, _ = MP.boost(st, r, jnp.zeros(7, I32).at[4].set(1), go, source=-2, ab_msg=aeff("steadfast"), simple=True)
    return st


if aid("speedboost") > 0:
    from vgcjax.effects.hooks import handler as _handler
    _handler(ids.HOOK_RESIDUAL, aeff("speedboost"), ps="data/abilities.ts speedboost.onResidual (residual)")(
        lambda hc, st, dex, rng: (hc, st, rng))


def residual(st, dex, rng, h, on):
    """The abilities' onResidual (fieldEvent walk, undispatched): Speed Boost (order 28: activeTurns -> +1 Spe)."""
    from vgcjax.battle import move_pipeline as MP
    r = jnp.clip(h, 0, 7)
    go = jnp.asarray(on) & _is(mget(st.mons.ability, r), "speedboost") & (mget(st.mons.active_turns, r) > 0)
    st, _ = MP.boost(st, r, jnp.zeros(7, I32).at[4].set(1), go, source=-2, ab_msg=aeff("speedboost"), simple=True)
    return st, rng


def switch_out(st, dex, rng, r, on):
    """runEvent('SwitchOut', oldActive) for a live switch-out ([C] Regenerator: heal 1/3 with '-heal' X hp
    '[from] ability: Regenerator' '[silent]' when it healed; [C] Natural Cure: a status (not fnt) ->
    '-curestatus' X status '[from] ability: Natural Cure' '[silent]', clearStatus)."""
    from vgcjax.battle import faint as F
    from vgcjax.effects import statuses as STS
    r = jnp.clip(r, 0, 7)
    ab = mget(st.mons.ability, r)
    go = jnp.asarray(on)
    regen = go & _is(ab, "regenerator")
    hp, mx = mget(st.mons.hp, r), mget(st.mons.maxhp, r)
    st, healed = F.heal(st, r, mx // 3, regen)
    st = EV.emit_hp(st, regen & (healed > 0), EV.EV_HEAL, r, move=aeff("regenerator"), flags=EV.F_FROM)
    nc = go & _is(ab, "naturalcure") & (mget(st.mons.status, r) > 0)
    st = EV.emit(st, nc, EV.EV_CURESTATUS, actor=r, move=C.eff_cond(mget(st.mons.status, r)), v1=aeff("naturalcure"),
                 flags=EV.F_FROM_MOVE_CURE)
    st = STS.clear_status(st, r, nc)
    return st, rng

"""Passive abilities / items batch: Sand Veil, Overcoat, Damp, Rain Dish, Dry Skin, Super Luck, Healer; Scope Lens,
Muscle Band, Wise Glasses, Bright Powder, Normal Gem (+ the 'gem' condition).

PS: data/abilities.ts, data/items.ts, data/conditions.ts (Champions override: Healer, data/mods/champions/
abilities.ts:39-49). MUL-hook callbacks are effects/modifiers.py mods; the others are undispatched helpers called
from their event sites (perf rule):
- Sand Veil / Overcoat onImmunity('sandstorm'): effects/ability_blocks.py status_immunity;
- Overcoat onImmunity('powder') (runStatusImmunity('powder'): Effect Spore, Rage Powder): `overcoat`;
- Overcoat onTryHit (priority 1) and Dry Skin onTryHit (priority 0): effects/ability_reactions.py try_hit;
- Rain Dish / Dry Skin onWeather: effects/weather.py each_weather (`weather_abilities`);
- Damp onAnyTryMove: battle/move_pipeline.py runEvent('TryMove') (`damp`);
- Super Luck / Scope Lens onModifyCritRatio: run in move_pipeline.get_damage (runEvent('ModifyCritRatio'));
- Healer onResidual: effects/ability_reactions.py residual (`healer`);
- Normal Gem onSourceTryPrimaryHit: move_pipeline.spread_move_hit step 0 (`normal_gem`).
Not modelled: Damp onAnyDamage (Aftermath, not implemented); 'breakable' (no Mold Breaker family yet).
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
from vgcjax.effects.hooks import handler
from vgcjax.effects.items import holds, ieff, iid
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32


def _is(ab, name):
    return ab == aid(name) if aid(name) > 0 else jnp.zeros(jnp.shape(ab), bool)


def _flag(dex, move, flag):
    w, b = flag
    return ((dex.move_flags[move, w] >> b) & 1) == 1


def _const(m):
    return jnp.asarray(m, I32)


# ------------------------------------------------------------------ MUL-hook modifiers
if aid("sandveil") > 0:
    @mod("ModifyAccuracy", aeff("sandveil"), ps="data/abilities.ts:4011-4018 sandveil.onModifyAccuracy")
    def sandveil_acc(ctx, h):
        """onModifyAccuracy (priority -1, the defender's): numeric accuracy in Sandstorm -> chainModify([3277,
        4096]) (field.isWeather: suppressed by Cloud Nine)."""
        from vgcjax.effects import weather as WE
        return ctx.extra["numeric"] & WE.is_weather(ctx.st, "sandstorm"), _const(3277), None

if aid("dryskin") > 0:
    @mod("BasePower", aeff("dryskin"), prefix=ids.PFX_SOURCE,
         ps="data/abilities.ts:1107-1112 dryskin.onSourceBasePower")
    def dryskin_bp(ctx, h):
        """onSourceBasePower (priority 17, the defender's): a Fire move -> chainModify(1.25)."""
        return jnp.broadcast_to(ctx.mtype == ids.TYPE_FIRE, h.shape), _const(5120), None

if aid("superluck") > 0:
    @mod("ModifyCritRatio", aeff("superluck"), ps="data/abilities.ts:4704-4707 superluck.onModifyCritRatio")
    def superluck_crit(ctx, h):
        """onModifyCritRatio(critRatio): return critRatio + 1."""
        return jnp.ones(h.shape, bool), None, (lambda v: v + 1)

if iid("scopelens") > 0:
    @mod("ModifyCritRatio", ieff("scopelens"), ps="data/items.ts:5584-5586 scopelens.onModifyCritRatio")
    def scopelens_crit(ctx, h):
        """onModifyCritRatio(critRatio): return critRatio + 1."""
        return jnp.ones(h.shape, bool), None, (lambda v: v + 1)

for _n, _cat, _ps in (("muscleband", ids.CAT_PHYSICAL, "data/items.ts:4263-4268"),
                      ("wiseglasses", ids.CAT_SPECIAL, "data/items.ts:7784-7789")):
    if iid(_n) > 0:
        def _band(ctx, h, _cat=_cat):
            """onBasePower (priority 16): move.category matches -> chainModify([4505, 4096])."""
            return jnp.broadcast_to(ctx.dex.move_category[ctx.move] == _cat, h.shape), _const(4505), None
        mod("BasePower", ieff(_n), ps=f"{_ps} {_n}.onBasePower")(_band)

if iid("brightpowder") > 0:
    @mod("ModifyAccuracy", ieff("brightpowder"), ps="data/items.ts:677-682 brightpowder.onModifyAccuracy")
    def brightpowder_acc(ctx, h):
        """onModifyAccuracy (priority -2, the defender's): numeric accuracy -> chainModify([3686, 4096])."""
        return ctx.extra["numeric"], _const(3686), None


# ------------------------------------------------------------------ Overcoat powder immunity
def overcoat(st, refs):
    """Overcoat onImmunity(type): 'powder' -> false (runEvent('Immunity') inside runStatusImmunity('powder'),
    data/abilities.ts:3110-3112). True where the holder is powder-immune by ability."""
    return _is(mget(st.mons.ability, jnp.clip(refs, 0, 7)), "overcoat")


def overcoat_try_hit(st, dex, t, u, move, live):
    """Overcoat onTryHit (priority 1, data/abilities.ts:3113-3119): move.flags.powder, target !== source and
    dex.getImmunity('powder', target) (not Grass) -> '-immune' target '[from] ability: Overcoat'; null.
    Returns (st, hit)."""
    from vgcjax.core import typechart as TC
    if aid("overcoat") <= 0:
        return st, jnp.zeros((), bool)
    t = jnp.clip(t, 0, 7)
    not_immune = ~TC.status_immune(dex, ids.IMMUNITY_KEYS.index("powder"), mget(st.mons.types, t))
    hit = live & _is(mget(st.mons.ability, t), "overcoat") & (t != u) & _flag(dex, move, ids.MF_POWDER) & not_immune
    return EV.emit(st, hit, EV.EV_IMMUNE, actor=t, v1=aeff("overcoat")), hit


# ------------------------------------------------------------------ Rain Dish / Dry Skin onWeather
def weather_abilities(st, dex, rng, r, ok, w):
    """runEvent('Weather', r, null, weather) for Rain Dish (data/abilities.ts:3761-3766: the mon's effectiveWeather
    is Rain -> heal(baseMaxhp / 16)) and Dry Skin (:1113-1120: Rain -> heal(baseMaxhp / 8); Sun ->
    damage(baseMaxhp / 8, target, target)). `ok`: the walk reached r with weather unsuppressed; w = the weather."""
    from vgcjax.effects import statuses as STS
    from vgcjax.effects import weather as WE
    if aid("raindish") <= 0 and aid("dryskin") <= 0:
        return st, rng
    ab = mget(st.mons.ability, r)
    mx = mget(st.mons.maxhp, r)
    own = ok & (WE.mon_weather(st, r) == w)
    rain = own & (w == WE.wid("raindance"))
    sun = own & (w == WE.wid("sunnyday"))
    if aid("raindish") > 0:
        st, rng = STS.effect_heal(st, dex, rng, r, mx // 16, aeff("raindish"), rain & _is(ab, "raindish"))
    if aid("dryskin") > 0:
        ds = _is(ab, "dryskin")
        st, rng = STS.effect_heal(st, dex, rng, r, mx // 8, aeff("dryskin"), rain & ds)
        st, rng = STS.effect_damage(st, dex, rng, r, mx // 8, aeff("dryskin"), sun & ds, source=r)
    return st, rng


# ------------------------------------------------------------------ Damp (runEvent('TryMove'))
DAMP_MOVES = ("explosion", "mindblown", "mistyexplosion", "selfdestruct")


def damp(st, dex, u, move, on):
    """Damp onAnyTryMove (data/abilities.ts:812-818): Explosion / Mind Blown / Misty Explosion / Self-Destruct ->
    attrLastMove('[still]'), 'cant' holder 'ability: Damp' move '[of] user', false. Any active unfainted mon holds
    it (the user included); the first holder in handler order (fastest, lowest ref on a speed tie) shows. Returns
    (st, failed)."""
    if aid("damp") <= 0:
        return st, jnp.zeros((), bool)
    hit_move = jnp.zeros((), bool)
    for n in DAMP_MOVES:
        if getattr(ids, f"MOVE_{n.upper()}", -1) > 0:
            hit_move = hit_move | (move == getattr(ids, f"MOVE_{n.upper()}"))
    m = st.mons
    act = st.sides.active.reshape(4).astype(I32)
    refs = (jnp.arange(4) // 2) * 4 + jnp.maximum(act, 0)
    hold = (act >= 0) & ~mget(m.fainted, refs) & (mget(m.hp, refs) > 0) & _is(mget(m.ability, refs), "damp")
    fail = jnp.asarray(on) & hit_move & jnp.any(hold)
    spd = jnp.where(hold, mget(m.speed, refs).astype(I32) * 8 - refs, -(1 << 30))
    holder = refs[jnp.argmax(spd)]
    st = EV.attr_last_move(st, fail, EV.F_STILL)
    st = EV.emit(st, fail, EV.EV_CANT, actor=holder, move=aeff("damp"))
    return st, fail


# ------------------------------------------------------------------ Healer ([C] onResidual)
if aid("healer") > 0:
    handler(ids.HOOK_RESIDUAL, aeff("healer"),
            ps="data/mods/champions/abilities.ts:39-49 healer.onResidual (batch_passive.healer)")(
        lambda hc, st, dex, rng: (hc, st, rng))


def healer(st, dex, rng, r, on):
    """[C] Healer onResidual (order 5, subOrder 3 inherited from data/abilities.ts:1819-1820): for the adjacent
    ally (active, hp > 0) with a status, randomChance(1, 2) -> '-activate' holder 'ability: Healer',
    ally.cureStatus() ('-curestatus' ally status '[msg]'). ABILITY_RESIDUAL_PROC: actor holder, subject Healer."""
    from vgcjax.effects import statuses as STS
    if aid("healer") <= 0:
        return st, rng
    m = st.mons
    go = jnp.asarray(on) & _is(mget(m.ability, r), "healer")
    s = r // 4
    act = st.sides.active[s].astype(I32)
    b = r % 4
    other = jnp.where(act[0] == b, act[1], act[0])
    ally = s * 4 + jnp.maximum(other, 0)
    ok = go & (other >= 0) & (other != b) & (mget(m.hp, ally) > 0) & ~mget(m.fainted, ally) & \
        (mget(m.status, ally) > 0)
    key = C.dkey(st, ids.DK_ABILITY_RESIDUAL_PROC, actor=r, target=-1, subject=aeff("healer"), hit=0)
    win, rng = R.chance(rng, key, 1, 2, active=ok)
    cure = ok & win
    st = EV.emit(st, cure, EV.EV_ACTIVATE, actor=r, move=aeff("healer"))
    return STS.cure_status(st, ally, cure), rng


# ------------------------------------------------------------------ Normal Gem + the 'gem' condition
if "gem" in ids.CONDITIONS:
    @mod("BasePower", CO.eff("gem"), ps="data/conditions.ts:467-476 gem.onBasePower")
    def gem_bp(ctx, h):
        """onBasePower (priority 14): chainModify([5325, 4096])."""
        return jnp.ones(h.shape, bool), _const(5325), None


def normal_gem(st, dex, u, move, mtype, targets, tv, on):
    """Normal Gem onSourceTryPrimaryHit (data/items.ts:4346-4351), run by tryPrimaryHitEvent for each live target
    before the target's substitute (priority -1): target !== source, not Status, not pledgecombo, the ActiveMove's
    type is Normal and source.useItem() ('-enditem' source 'Normal Gem' '[from] gem') -> addVolatile('gem')
    (duration 1, data/conditions.ts:467-476; no Start). Only the first live target can consume it. Returns st."""
    from vgcjax.effects import items as IT
    if iid("normalgem") <= 0 or "gem" not in ids.CONDITIONS:
        return st
    live = jnp.asarray(tv) & (jnp.asarray(targets) >= 0) & (jnp.asarray(targets) != u)
    pledge = _flag(dex, move, ids.MF_PLEDGECOMBO)
    go = jnp.asarray(on) & jnp.any(live) & holds(st, u, "normalgem") & mget(st.mons.is_active, u) & \
        (dex.move_category[move] != ids.CAT_STATUS) & ~pledge & (mtype == ids.TYPE_NORMAL) & \
        (move != ids.MOVE_STRUGGLE)                                     # Struggle's type is '???'
    st = IT.use_item(st, u, go)
    c = CO.cid("gem")
    add = go & ~CO.has_volatile(st, u, c)
    st = CO._set_vol_bit(st, u, c, True, add)
    st = CO.set_vol_state(st, u, c, jnp.zeros(6, I16).at[0].set(dex.cond_duration[c].astype(I16)), add)
    return CO.next_effect_order(st, add)


# ------------------------------------------------------------------ Cute Charm + the 'attract' volatile
def _gender(st, r):
    """pokemon.gender from the team sheet (0 M, 1 F, 2 N; oracle teams always fix it)."""
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    return st.team.gender[r // 4, mget(st.mons.sheet_idx, r).astype(I32)].astype(I32)


AT_SRC = 1                                   # attract raw slot: effectState.source (dropped by the projection)

if "attract" in ids.CONDITIONS:
    _AT = CO.cid("attract")

    @handler(ids.HOOK_START, CO.eff("attract"), ps="data/moves.ts attract.condition.onStart")
    def attract_start(hc, st, dex, rng):
        """Opposite genders (M / F) else false; runEvent('Attract') (no Reg M-C handler); '-start' pokemon 'Attract'
        (Cute Charm: '[from] ability: Cute Charm' '[of] source'). The source is kept for onUpdate / BeforeMove."""
        r = hc.target
        g, gs = _gender(st, r), _gender(st, hc.source)
        ok = ((g == 0) & (gs == 1)) | ((g == 1) & (gs == 0))
        s = CO.vol_state(st, r, _AT)
        st = CO.set_vol_state(st, r, _AT, s.at[AT_SRC].set(jnp.asarray(hc.source, s.dtype)), ok)
        ab = (hc.effect >= ids.ABILITY_BASE) & (hc.effect < ids.ABILITY_BASE + len(ids.ABILITIES) - 1)
        st = EV.emit(st, ok, EV.EV_START, actor=r, move=CO.eff("attract"), v1=jnp.where(ab, hc.effect, -1),
                     flags=jnp.where(ab, EV.F_FROM_ABILITY, 0))
        return hc.ret(RS.R(RS.BOOL, 0), ~ok), st, rng

    @handler(ids.HOOK_BEFOREMOVE, CO.eff("attract"), ps="data/moves.ts attract.condition.onBeforeMove")
    def attract_before_move(hc, st, dex, rng):
        """priority 2: '-activate' pokemon 'move: Attract' '[of] source'; randomChance(1, 2) -> 'cant' pokemon
        'Attract', false. INFATUATION draw: actor the holder, subject attract."""
        r = hc.target
        st = EV.emit(st, True, EV.EV_ACTIVATE, actor=r, move=CO.eff("attract"))
        key = C.dkey(st, ids.DK_INFATUATION, actor=r, target=-1, subject=CO.eff("attract"))
        cant, rng = R.chance(rng, key, 1, 2)
        st = EV.emit(st, cant, EV.EV_CANT, actor=r, move=CO.eff("attract"))
        return hc.ret(RS.R(RS.BOOL, 0), cant), st, rng

    @handler(ids.HOOK_END, CO.eff("attract"), ps="data/moves.ts attract.condition.onEnd")
    def attract_end(hc, st, dex, rng):
        """'-end' pokemon 'Attract' '[silent]'."""
        return hc, EV.emit(st, True, EV.EV_END, actor=hc.target, move=CO.eff("attract"), v1=-1), rng


def attract_update(st, dex, rng, r, on):
    """attract.condition.onUpdate (data/moves.ts): effectState.source no longer active -> removeVolatile('attract')
    (End: '-end' X 'Attract' '[silent]'). Run in eachEvent('Update') before the mon's ability handlers (volatile
    subOrder 2 < ability 7). Returns (st, rng)."""
    if "attract" not in ids.CONDITIONS or aid("cutecharm") <= 0:
        return st, rng
    c = CO.cid("attract")
    src = CO.vol_state(st, r, c)[AT_SRC].astype(I32)
    gone = jnp.asarray(on) & CO.has_volatile(st, r, c) & ~mget(st.mons.is_active, jnp.clip(src, 0, 7))
    st, rng, _ = CO.remove_volatile(st, dex, rng, r, c, gone)
    return st, rng


def cute_charm(st, dex, rng, tr, u, damaged, move, hit_no):
    """Cute Charm onDamagingHit (data/abilities.ts:799-805): contact -> randomChance(3, 10) ->
    source.addVolatile('attract', holder) (sourceEffect Cute Charm). addVolatile: no Restart (already attracted ->
    false), runStatusImmunity('attract'): Oblivious (silent: not a move source). ABILITY_ONHIT_PROC draw: actor
    holder, target the attacker, subject Cute Charm, hit. Returns (st, rng)."""
    if aid("cutecharm") <= 0 or "attract" not in ids.CONDITIONS:
        return st, rng
    on = damaged & _flag(dex, move, ids.MF_CONTACT) & _is(mget(st.mons.ability, tr), "cutecharm")
    key = C.dkey(st, ids.DK_ABILITY_ONHIT_PROC, actor=tr, target=u, subject=aeff("cutecharm"), hit=hit_no)
    ok, rng = R.chance(rng, key, 3, 10, active=on)
    go = on & ok & ~_is(mget(st.mons.ability, u), "oblivious")
    st, rng, _ = CO.add_volatile(st, dex, rng, u, CO.cid("attract"), go, source=tr, source_effect=aeff("cutecharm"))
    return st, rng

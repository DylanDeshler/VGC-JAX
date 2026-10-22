"""A toy-but-realistically-sized doubles battle step, used by b2 (control flow) and b5 (rollout).

Not a correct Pokemon engine: it has the *shape* of one (per-mon [2,6] arrays, a 6-entry action
queue resolved by dynamic-speed argmax, a fixed 3-target axis, a ~200-op integer damage calc per
target, a 64-way move-effect switch, residuals via a 16-way item switch, faint replacement,
terminal detection) so that compile time / throughput / memory numbers are representative.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from handlers import make_branches  # reuse PS-like handler templates for item residuals

N_TYPES = 18
N_MOVES = 256
N_SPECIES = 64
N_EFFECTS = 64
N_ITEMS = 16
QUEUE = 6
LEVEL = 50

_rng = np.random.default_rng(1234)
TYPE_CHART = jnp.asarray(_rng.choice([0, 2, 4, 4, 4, 8], size=(N_TYPES, N_TYPES)), jnp.int32)  # x/4
BOOST_NUM = jnp.asarray([2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8], jnp.int32)
BOOST_DEN = jnp.asarray([8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2], jnp.int32)
# move table columns: bp, type, category(0 phys,1 spec,2 status), accuracy, priority, target(0 single,1 spread-foes,2 all-adjacent), effect id, eff chance
_mt = np.stack([
    _rng.integers(40, 130, N_MOVES), _rng.integers(0, N_TYPES, N_MOVES), _rng.choice([0, 1, 1, 0, 2], N_MOVES),
    _rng.choice([100, 100, 90, 85, 70], N_MOVES), _rng.choice([0, 0, 0, 0, 1, -1, 2], N_MOVES),
    _rng.choice([0, 0, 0, 1, 2], N_MOVES), _rng.integers(0, N_EFFECTS, N_MOVES), _rng.choice([0, 10, 30, 100], N_MOVES),
], axis=1)
MOVE_TABLE = jnp.asarray(_mt, jnp.int32)
SPECIES_BASE = jnp.asarray(_rng.integers(50, 130, (N_SPECIES, 6)), jnp.int32)
SPECIES_TYPES = jnp.asarray(_rng.integers(0, N_TYPES, (N_SPECIES, 2)), jnp.int32)


class Battle(NamedTuple):
    # per mon [2, 6]
    species: jax.Array
    hp: jax.Array
    maxhp: jax.Array
    stats: jax.Array      # [2,6,6] int16
    boosts: jax.Array     # [2,6,7] int8
    types: jax.Array      # [2,6,2] int8
    status: jax.Array     # [2,6] int8
    moves: jax.Array      # [2,6,4] int16
    pp: jax.Array         # [2,6,4] int8
    item: jax.Array       # [2,6] int8
    vol_bits: jax.Array   # [2,6] uint32 (stand-in for the 64-bit field)
    vol_ctr: jax.Array    # [2,6,16] int8
    # per side
    active: jax.Array     # [2,2] mon index in each slot
    screens: jax.Array    # [2] int8 turns
    tailwind: jax.Array   # [2] int8
    # field
    weather: jax.Array    # [] int8
    weather_turns: jax.Array
    trick_room: jax.Array
    turn: jax.Array       # [] int16
    done: jax.Array       # [] bool
    winner: jax.Array     # [] int8
    key: jax.Array        # per-battle PRNG key


def modify(x, m):
    return (x * m + 2047) >> 12


def damage_calc(att_stat, att_boost, def_stat, def_boost, bp, mtype, att_types, def_types,
                roll_u, crit, spread, weather, burned, screen):
    """~200 int32 ops: PS-order damage formula (base -> spread -> weather -> crit -> roll -> STAB -> type -> burn -> final)."""
    ab = jnp.where(crit, jnp.maximum(att_boost, 0), att_boost) + 6
    db = jnp.where(crit, jnp.minimum(def_boost, 0), def_boost) + 6
    A = att_stat * BOOST_NUM[ab] // BOOST_DEN[ab]
    D = jnp.maximum(def_stat * BOOST_NUM[db] // BOOST_DEN[db], 1)
    base = (((2 * LEVEL // 5 + 2) * bp * A) // D) // 50 + 2
    base = jnp.where(spread, modify(base, 3072), base)
    wmul = jnp.where((weather == 1) & (mtype == 1), 6144, jnp.where((weather == 1) & (mtype == 2), 2048,
           jnp.where((weather == 2) & (mtype == 2), 6144, jnp.where((weather == 2) & (mtype == 1), 2048, 4096))))
    base = modify(base, wmul)
    base = jnp.where(crit, modify(base, 6144), base)
    roll = 85 + jnp.minimum((roll_u * 16).astype(jnp.int32), 15)
    base = base * roll // 100
    stab = (att_types[0] == mtype) | (att_types[1] == mtype)
    base = jnp.where(stab, modify(base, 6144), base)
    eff = TYPE_CHART[mtype, def_types[0]] * TYPE_CHART[mtype, def_types[1]]  # /16
    base = base * eff // 16
    base = jnp.where(burned, modify(base, 2048), base)
    fm = jnp.where(screen & ~crit, 2732, 4096)
    fm = (fm * jnp.where(eff > 16, 4915, 4096) + 2048) >> 12  # expert-belt-like chain
    base = modify(base, fm)
    return jnp.where(eff == 0, 0, jnp.maximum(base, 1))


# ------------------------------------------------------------- 64-way move-effect switch
def _effect_branch(k):
    """Secondary effects: stat drops/raises, status, recoil, drain, weather setting, screens..."""
    kind = k % 8
    amt = (k // 8) % 3 + 1

    def br(op):
        hp, boosts, status, weather, screens, dealt, user_hp, user_max = op
        if kind == 0:   # lower target stat
            boosts = boosts.at[k % 7].set(jnp.maximum(boosts[k % 7] - amt, -6))
        elif kind == 1:  # burn/para/poison
            status = jnp.where(status == 0, (k % 5) + 1, status)
        elif kind == 2:  # recoil
            user_hp = jnp.maximum(user_hp - dealt * amt // 4, 0)
        elif kind == 3:  # drain
            user_hp = jnp.minimum(user_hp + dealt // 2, user_max)
        elif kind == 4:  # set weather
            weather = (k % 4).astype(weather.dtype) if hasattr(k, "astype") else jnp.asarray(k % 4, weather.dtype)
        elif kind == 5:  # screens
            screens = jnp.maximum(screens, 5).astype(screens.dtype)
        elif kind == 6:  # fixed damage
            hp = jnp.maximum(hp - (user_max // (amt + 1)), 0)
        else:            # raise own stat (applied to target array for simplicity)
            boosts = boosts.at[(k + 1) % 7].set(jnp.minimum(boosts[(k + 1) % 7] + amt, 6))
        return hp, boosts, status, weather, screens, dealt, user_hp, user_max
    return br


EFFECT_BRANCHES = [_effect_branch(k) for k in range(N_EFFECTS)]
ITEM_BRANCHES = make_branches(N_ITEMS, seed=7)


def target_slots(side, slot, choice, ttype):
    """Fixed 3-target axis: (foe0, foe1, ally) as (side, slot) plus a mask."""
    foe = 1 - side
    t_side = jnp.array([foe, foe, side])
    t_slot = jnp.array([0, 1, 1 - slot])
    single = (jnp.arange(3) == choice)
    spread = jnp.array([True, True, False])
    alladj = jnp.array([True, True, True])
    mask = jnp.where(ttype == 0, single, jnp.where(ttype == 1, spread, alladj))
    return t_side, t_slot, mask


# ------------------------------------------------------------- per-action body (one queue entry)
HOOK_BRANCHES = [make_branches(64, seed=100 + h) for h in range(16)]


def run_action(b: Battle, side, slot, move_slot, choice, valid, key, target_mode="vector", index_mode="gather",
               extra_hooks=0, barrier=False):
    mon = b.active[side, slot]
    alive = (b.hp[side, mon] > 0) & valid & ~b.done
    move = b.moves[side, mon, move_slot].astype(jnp.int32)
    row = MOVE_TABLE[move]
    bp, mtype, cat, acc, _, ttype, eff_id, eff_ch = [row[i] for i in range(8)]
    t_side, t_slot, tmask = target_slots(side, slot, choice, ttype)
    t_mon = b.active[t_side, t_slot]
    t_alive = b.hp[t_side, t_mon] > 0
    tmask = tmask & t_alive & alive & (cat < 2)
    n_t = tmask.sum()
    spread = n_t > 1
    u = jax.random.uniform(key, (3, 4))  # acc, crit, roll, secondary per target
    a_idx = jnp.where(cat == 0, 1, 3)
    d_idx = jnp.where(cat == 0, 2, 4)
    att_stat = b.stats[side, mon, a_idx].astype(jnp.int32)
    att_boost = b.boosts[side, mon, a_idx - 1].astype(jnp.int32)
    att_types = b.types[side, mon].astype(jnp.int32)
    burned = (b.status[side, mon] == 1) & (cat == 0)

    def per_target(t, hp):
        ts, tm = t_side[t], t_mon[t]
        hit = tmask[t] & (u[t, 0] * 100 < acc)
        crit = u[t, 1] < (1 / 24)
        dmg = damage_calc(att_stat, att_boost, b.stats[ts, tm, d_idx].astype(jnp.int32),
                          b.boosts[ts, tm, d_idx - 1].astype(jnp.int32), bp, mtype, att_types,
                          b.types[ts, tm].astype(jnp.int32), u[t, 2], crit, spread,
                          b.weather.astype(jnp.int32), burned, b.screens[ts] > 0)
        dmg = jnp.where(hit, dmg, 0)
        return dmg, hit

    hp = b.hp.astype(jnp.int32)
    if target_mode == "vector":
        dmgs, hits = jax.vmap(per_target, in_axes=(0, None))(jnp.arange(3), hp)
    else:
        dl, hl = [], []
        for t in range(3):
            d, h = per_target(t, hp)
            dl.append(d); hl.append(h)
        dmgs, hits = jnp.stack(dl), jnp.stack(hl)
    if index_mode == "gather":
        hp = hp.at[t_side, t_mon].add(-dmgs)
    else:  # one-hot masks over the [2,6] mon grid
        oh = (jnp.arange(2)[None, :, None] == t_side[:, None, None]) & (jnp.arange(6)[None, None, :] == t_mon[:, None, None])
        hp = hp - (oh * dmgs[:, None, None]).sum(0)
    hp = jnp.maximum(hp, 0)

    # optional extra hook dispatch points (each a 64-way batched switch), to project compile time/runtime
    # for a realistic handler count: extra_hooks x 64 handlers per action
    if extra_hooks:
        ctx = (bp, att_stat, att_stat, hp[side, mon], b.maxhp[side, mon].astype(jnp.int32), b.weather.astype(jnp.int32),
               mtype, b.vol_bits[side, mon].astype(jnp.int32) & 255, jnp.int32(4096))
        for h in range(extra_hooks):
            idx = (b.item[side, mon].astype(jnp.int32) * 7 + move * (h + 1)) % 64
            ctx = lax.switch(idx, HOOK_BRANCHES[h], ctx)
            if barrier:
                ctx = lax.optimization_barrier(ctx)
        hp = hp.at[side, mon].set(jnp.where(alive, jnp.clip(ctx[3] + (ctx[8] - 4096) // 4096, 0, ctx[4]), hp[side, mon]))

    # secondary effect on the first hit target (64-way switch, batched index -> all branches + select)
    first = jnp.argmax(hits)
    fs, fm = t_side[first], t_mon[first]
    do_sec = hits.any() & (u[first, 3] * 100 < eff_ch)
    op = (hp[fs, fm], b.boosts[fs, fm].astype(jnp.int32), b.status[fs, fm].astype(jnp.int32),
          b.weather.astype(jnp.int32), b.screens[side].astype(jnp.int32), dmgs[first], hp[side, mon], b.maxhp[side, mon].astype(jnp.int32))
    out = lax.switch(eff_id, EFFECT_BRANCHES, op)
    out = jax.tree.map(lambda n, o: jnp.where(do_sec, n, o), out, op)
    t_hp, t_boosts, t_status, weather, scr, _, u_hp, _ = out
    hp = hp.at[fs, fm].set(t_hp).at[side, mon].set(u_hp)
    boosts = b.boosts.at[fs, fm].set(t_boosts.astype(b.boosts.dtype))
    status = b.status.at[fs, fm].set(t_status.astype(b.status.dtype))
    pp = b.pp.at[side, mon, move_slot].add(jnp.where(alive, -1, 0).astype(b.pp.dtype))
    return b._replace(hp=hp.astype(b.hp.dtype), boosts=boosts, status=status, pp=pp,
                      weather=weather.astype(b.weather.dtype),
                      screens=b.screens.at[side].set(scr.astype(b.screens.dtype)))


def eff_speed(b: Battle, side, slot):
    mon = b.active[side, slot]
    s = b.stats[side, mon, 5].astype(jnp.int32)
    bo = b.boosts[side, mon, 4].astype(jnp.int32) + 6
    s = s * BOOST_NUM[bo] // BOOST_DEN[bo]
    s = jnp.where(b.tailwind[side] > 0, s * 2, s)
    s = jnp.where(b.status[side, mon] == 2, s // 2, s)
    return jnp.where(b.trick_room > 0, 10000 - s, s)


def residuals(b: Battle):
    """End of turn: per active mon, a 16-way item switch (Leftovers-like), weather chip, counters."""
    hp = b.hp.astype(jnp.int32)
    for side in range(2):
        for slot in range(2):
            mon = b.active[side, slot]
            ctx = (jnp.int32(60), jnp.int32(100), jnp.int32(100), hp[side, mon], b.maxhp[side, mon].astype(jnp.int32),
                   b.weather.astype(jnp.int32), b.types[side, mon, 0].astype(jnp.int32), jnp.int32(0), jnp.int32(4096))
            out = lax.switch(b.item[side, mon].astype(jnp.int32), ITEM_BRANCHES, ctx)
            new_hp = jnp.where(hp[side, mon] > 0, out[3], 0)
            chip = jnp.where((b.weather == 3) & (hp[side, mon] > 0), b.maxhp[side, mon].astype(jnp.int32) // 16, 0)
            hp = hp.at[side, mon].set(jnp.maximum(new_hp - chip, 0))
    return b._replace(hp=hp.astype(b.hp.dtype),
                      screens=jnp.maximum(b.screens - 1, 0).astype(b.screens.dtype),
                      tailwind=jnp.maximum(b.tailwind - 1, 0).astype(b.tailwind.dtype),
                      weather_turns=jnp.maximum(b.weather_turns - 1, 0).astype(b.weather_turns.dtype),
                      weather=jnp.where(b.weather_turns <= 1, 0, b.weather).astype(b.weather.dtype),
                      vol_ctr=jnp.maximum(b.vol_ctr - 1, 0).astype(b.vol_ctr.dtype))


def replace_fainted(b: Battle):
    active = b.active
    for side in range(2):
        alive = b.hp[side] > 0
        on_field = (jnp.arange(6)[:, None] == active[side][None, :]).any(-1)
        bench = alive & ~on_field
        nxt = jnp.argmax(bench)
        for slot in range(2):
            need = (b.hp[side, active[side, slot]] <= 0) & bench.any()
            active = active.at[side, slot].set(jnp.where(need, nxt, active[side, slot]).astype(active.dtype))
            bench = bench & ~((jnp.arange(6) == nxt) & need)
            nxt = jnp.argmax(bench)
    return b._replace(active=active.astype(b.active.dtype))


def step(b: Battle, actions, loop_mode="fori", **kw):
    """actions: [2,2] int32 in [0, 12): move_slot = a // 3, target choice = a % 3."""
    key, k_order, k_act = jax.random.split(b.key, 3)
    sides = jnp.array([0, 0, 1, 1, 0, 1])
    slots = jnp.array([0, 1, 0, 1, 0, 1])
    valid = jnp.array([True, True, True, True, False, False])  # 2 dummy queue entries (switch/mega placeholders)
    acts = jnp.concatenate([actions.reshape(4), jnp.zeros(2, actions.dtype)])
    prio = jnp.array([MOVE_TABLE[b.moves[sides[i], b.active[sides[i], slots[i]], acts[i] // 3].astype(jnp.int32), 4]
                      for i in range(QUEUE)])
    tieb = jax.random.uniform(k_order, (QUEUE,))
    act_keys = jax.random.split(k_act, QUEUE)

    def body(i, carry):
        b, done_mask = carry
        spd = jnp.stack([eff_speed(b, sides[j], slots[j]) for j in range(QUEUE)])
        score = jnp.where(done_mask | ~valid, -1e9, prio * 1e5 + spd + tieb)
        j = jnp.argmax(score)
        b = run_action(b, sides[j], slots[j], acts[j] // 3, acts[j] % 3, valid[j] & ~done_mask[j], act_keys[j], **kw)
        return b, done_mask.at[j].set(True)

    carry = (b, jnp.zeros(QUEUE, bool))
    if loop_mode == "fori":
        carry = lax.fori_loop(0, QUEUE, body, carry)
    else:
        for i in range(QUEUE):
            carry = body(i, carry)
    b = carry[0]
    b = residuals(b)
    b = replace_fainted(b)
    side_alive = (b.hp > 0).any(-1)
    done = b.done | ~side_alive.all()
    winner = jnp.where(b.done, b.winner, jnp.where(side_alive[0] & ~side_alive[1], 0,
                                                   jnp.where(side_alive[1] & ~side_alive[0], 1, -1)))
    done = done | (b.turn >= 100)
    return b._replace(done=done, winner=winner.astype(b.winner.dtype), turn=(b.turn + 1).astype(b.turn.dtype), key=key)


def init(key):
    ks = jax.random.split(key, 6)
    species = jax.random.randint(ks[0], (2, 6), 0, N_SPECIES)
    base = SPECIES_BASE[species]
    stats = (base * 2 + 5).at[..., 0].set(base[..., 0] * 2 + 60)
    return Battle(
        species=species.astype(jnp.int16), hp=stats[..., 0].astype(jnp.int16), maxhp=stats[..., 0].astype(jnp.int16),
        stats=stats.astype(jnp.int16), boosts=jnp.zeros((2, 6, 7), jnp.int8),
        types=SPECIES_TYPES[species].astype(jnp.int8), status=jnp.zeros((2, 6), jnp.int8),
        moves=jax.random.randint(ks[1], (2, 6, 4), 0, N_MOVES).astype(jnp.int16), pp=jnp.full((2, 6, 4), 16, jnp.int8),
        item=jax.random.randint(ks[2], (2, 6), 0, N_ITEMS).astype(jnp.int8), vol_bits=jnp.zeros((2, 6), jnp.uint32),
        vol_ctr=jnp.zeros((2, 6, 16), jnp.int8), active=jnp.array([[0, 1], [0, 1]], jnp.int8),
        screens=jnp.zeros(2, jnp.int8), tailwind=jnp.zeros(2, jnp.int8), weather=jnp.int8(0), weather_turns=jnp.int8(0),
        trick_room=jnp.int8(0), turn=jnp.int16(0), done=jnp.bool_(False), winner=jnp.int8(-1), key=ks[5])


def observe(b: Battle, player):
    """~250-float observation from `player`'s perspective (own side first)."""
    me, op = player, 1 - player
    def side_feats(s):
        return jnp.concatenate([
            (b.hp[s] / jnp.maximum(b.maxhp[s], 1)).astype(jnp.float32),
            jax.nn.one_hot(b.status[s], 6).reshape(-1),
            b.boosts[s].reshape(-1).astype(jnp.float32) / 6,
            jax.nn.one_hot(b.active[s], 6).reshape(-1),
            (b.species[s].astype(jnp.float32) / N_SPECIES),
        ])
    field = jnp.stack([b.weather, b.weather_turns, b.trick_room, b.screens[me], b.screens[op],
                       b.tailwind[me], b.tailwind[op]]).astype(jnp.float32)
    return jnp.concatenate([side_feats(me), side_feats(op), field, jnp.array([b.turn / 100.0])])


def legal_mask(b: Battle, player):
    mons = b.active[player]
    pp_ok = b.pp[player, mons] > 0  # [2,4]
    return jnp.repeat(pp_ok, 3, axis=-1).reshape(2, 12)

"""`step` (04 §2): apply both players' choices, then run the turn until the next decision point or the end.

Control structure (ADR-104 option 4): one `fori` of `cfg.micro_steps` micro-steps. A micro-step either selects the
next queued action (light action, residual, or a move: its preludes and first hit) or runs one more hit of the move
in progress (`st.exec`); then the move's tails if its hits are done, the runAction tail, and end of turn once the
queue is drained. A turn that needs more micro-steps continues in the next `step` call with phase CONT (no decision
for either player; actions are ignored).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import errors as E
from vgcjax.battle import events as EV
from vgcjax.battle import faint as F
from vgcjax.battle import init as INIT
from vgcjax.battle import move_pipeline as MP
from vgcjax.battle import queue as Q
from vgcjax.battle import requests as REQ
from vgcjax.battle import state as S
from vgcjax.battle import switching as SW
from vgcjax.effects import fieldevent as FE
from vgcjax.effects import registry  # noqa: F401  (handler registration)
from vgcjax.battle.util import mget, mset, or_flag
from vgcjax.core import rng as R
from vgcjax.data import ids

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32
NPHAZE = 2
LIGHT = (S.Q_BEFORE_TURN, S.Q_START, S.Q_SWITCH, S.Q_INSTASWITCH, S.Q_RUN_SWITCH, S.Q_MEGA, S.Q_BEFORE_TURN_MOVE,
         S.Q_PRIORITY_CHARGE, S.Q_REVIVAL)
ACTION_NAME = {S.Q_BEFORE_TURN: "beforeTurn", S.Q_START: "start", S.Q_SWITCH: "switch", S.Q_INSTASWITCH: "instaswitch",
               S.Q_RUN_SWITCH: "runSwitch", S.Q_MEGA: "megaEvo", S.Q_REVIVAL: "revivalblessing", S.Q_RESIDUAL: "residual"}
STEP_TAG = {S.Q_SWITCH: ids.STEP_A_SWITCH, S.Q_INSTASWITCH: ids.STEP_A_SWITCH, S.Q_MEGA: ids.STEP_A_MEGA,
            S.Q_BEFORE_TURN: ids.STEP_A_OTHER, S.Q_BEFORE_TURN_MOVE: ids.STEP_A_OTHER, S.Q_REVIVAL: ids.STEP_A_OTHER,
            S.Q_PRIORITY_CHARGE: ids.STEP_A_OTHER, S.Q_START: ids.STEP_A_OTHER, S.Q_MOVE: ids.STEP_B_MOVE,
            S.Q_RESIDUAL: ids.STEP_RESIDUAL, S.Q_RUN_SWITCH: ids.STEP_SWITCHIN}


def _lookup(table: dict, kind, default):
    out = jnp.asarray(default, I32)
    for k, v in table.items():
        out = jnp.where(kind == k, v, out)
    return out


def entry_step_ctx(st, kind, ref, move_id, is_move):
    """step_tag / step_actor / step_subject / action_ord of a queue entry (contract §3.1)."""
    tag = _lookup(STEP_TAG, kind, ids.STEP_COMMIT)
    subj = _lookup({k: C.eff_action(v) for k, v in ACTION_NAME.items()}, kind, -1)
    subj = jnp.where(move_id > 0, C.eff_move(move_id), subj)
    actor = jnp.where(kind == S.Q_RESIDUAL, -1, ref)
    aord = jnp.where(is_move, mget(st.scratch.move_actions, jnp.clip(ref, 0, 7)), 0)
    return tag, actor, subj, aord


# ------------------------------------------------------------------ commits
def commit_preview(st, dex, a, on):
    """PS: commitChoices for team preview, then runAction('team') x8 and queue [start]."""
    st2 = INIT.apply_preview(st, dex, a)
    meta = st2.meta
    st2 = st2.replace(meta=meta.replace(mid_turn=jnp.asarray(True), phase=jnp.asarray(S.MOVE, I8)))
    from vgcjax.battle.util import tree_where
    return tree_where(on, st2, st)


def decode_move_slot(st, dex, side, slot, a):
    """Engine slot action -> (is_move, is_switch, move_slot, move_id, target_loc, mega, switch_b). 04 §4.3."""
    a = jnp.asarray(a, I32)
    b = st.sides.active[side, slot].astype(I32)
    u = side * 4 + jnp.maximum(b, 0)
    m = st.mons
    usable = (mget(m.pp, u) > 0) & (mget(m.move_disabled, u) == 0) & (mget(m.move_ids, u) > 0)
    struggle = ~jnp.any(usable)
    mi = a // 6
    t = (a % 6) // 2
    mega = (a % 2) == 1
    move_id = jnp.where(struggle, ids.MOVE_STRUGGLE, mget(m.move_ids, u)[jnp.clip(mi, 0, 3)])
    move_slot = jnp.where(struggle, -1, mi)
    tt = dex.move_target[move_id]
    choosable = REQ.choosable(tt)
    ally_loc = -(2 - slot)
    self_loc = -(slot + 1)
    loc = jnp.where(tt == ids.TARGET_ADJACENTALLYORSELF, jnp.where(t == 0, self_loc, ally_loc),
                    jnp.where(t == 0, 1, jnp.where(t == 1, 2, ally_loc)))
    loc = jnp.where(choosable & ~struggle, loc, 0)
    return (a < 24), (a >= 24) & (a < 28), move_slot, move_id, loc, mega & ~struggle, a - 24


def resolve_loc(st, dex, u, move_id, loc):
    """PS resolveAction (sim/battle-queue.ts:268-272): an action without a target location gets
    getLocOf(getRandomTarget()). Self-type targets -> own location; adjacentAlly -> the ally if alive; everything else
    -> a random living foe (a don't-care draw, contract §6): the engine takes the first living foe."""
    tt = dex.move_target[move_id]
    pos = MP.position(st, u)
    self_loc = -(pos + 1)
    selfish = ((tt == ids.TARGET_SELF) | (tt == ids.TARGET_ALL) | (tt == ids.TARGET_ALLYSIDE) |
               (tt == ids.TARGET_ALLYTEAM) | (tt == ids.TARGET_ADJACENTALLYORSELF))
    ally = MP.living_ally(st, u)
    foes, fok = MP.living_foes(st, u)
    foe_loc = jnp.where(fok[0], 1, jnp.where(fok[1], 2, 1))
    new = jnp.where(selfish, self_loc, jnp.where(tt == ids.TARGET_ADJACENTALLY,
                                                 jnp.where(ally >= 0, -(2 - pos), 0), foe_loc))
    return jnp.where(loc == 0, new, loc)


def commit_move(st, dex, rng, actions, on):
    """PS: commitChoices for a MOVE request (04 §5.2): updateSpeed, resolve each slot action, then (fresh turn)
    beforeTurn + residual. Returns (st, rng)."""
    on = jnp.asarray(on)
    st = C.update_speed(st, dex, on)
    st = C.set_step(st, ids.STEP_COMMIT, -1, -1, 0, on)
    for side in range(2):
        for slot in range(2):
            a = actions[side][slot]
            b = st.sides.active[side, slot].astype(I32)
            u = side * 4 + jnp.maximum(b, 0)
            is_move, is_switch, mslot, mid, loc, mega, sb = decode_move_slot(st, dex, side, slot, a)
            occ = on & (b >= 0)
            spd = C.action_speed(st, dex, u)
            # megaEvo satellite (order 104), then the move (order 200)
            st = Q.add_entry(st, occ & is_move & mega, kind=S.Q_MEGA, side=side, b=b, speed=spd)
            loc = resolve_loc(st, dex, u, mid, loc)
            frac, rng = Q.frac_priority(st, dex, rng, u, mid, occ & is_move)
            prio = 10 * Q.move_priority(st, dex, u, mid) + frac
            orig = MP.at_loc(st, u, loc)
            st = Q.add_entry(st, occ & is_move, kind=S.Q_MOVE, side=side, b=b, priority10=prio, frac10=frac, speed=spd,
                             move_slot=mslot, move_id=mid, target_loc=loc, original_target=orig)
            st = Q.add_entry(st, occ & is_switch, kind=S.Q_SWITCH, side=side, b=b, speed=spd, switch_target_b=sb)
    fresh = on & ~st.meta.mid_turn
    st = Q.add_entry(st, fresh, kind=S.Q_BEFORE_TURN, speed=1)
    st = Q.add_entry(st, fresh, kind=S.Q_RESIDUAL, speed=1)
    meta = st.meta
    return st.replace(meta=meta.replace(mid_turn=meta.mid_turn | on)), rng


def commit_switch(st, dex, actions, on):
    """PS: commitChoices for a SWITCH request (04 §10.3): instaswitch entries (prepend) before the remainder."""
    on = jnp.asarray(on)
    st = C.update_speed(st, dex, on)
    st = C.set_step(st, ids.STEP_COMMIT, -1, -1, 0, on)
    for side in range(2):
        for slot in range(2):
            a = jnp.asarray(actions[side][slot], I32)
            b = st.sides.active[side, slot].astype(I32)
            u = side * 4 + jnp.maximum(b, 0)
            m = st.mons
            flag = mget(m.switch_flag, u)
            go = on & (b >= 0) & (flag != 0) & (a >= 24) & (a < 28)
            src = jnp.where(flag >= 2, flag - 2, 0)
            st = Q.add_entry(st, go, kind=S.Q_INSTASWITCH, side=side, b=b, speed=C.action_speed(st, dex, u),
                             switch_target_b=a - 24, source_effect=src, prepend=True)
            m = st.mons
            st = st.replace(mons=m.replace(switch_flag=mset(m.switch_flag, u, 0, go)))
    return st


# ------------------------------------------------------------------ actions
def run_light(st, dex, rng, cfg, idx, on):
    """Execute a light (non-move) queue entry. PS: sim/battle.ts:2718-2859 runAction per choice."""
    q = st.queue
    kind = q.kind[idx]
    side = q.side[idx].astype(I32)
    b = q.actor_b[idx].astype(I32)
    ref = jnp.where(b >= 0, side * 4 + b, -1)
    on = jnp.asarray(on)
    # start runs once, outside the action loops (run_start)
    # switch / instaswitch (PS :2814-2820): switchIn(target, pokemon.position, sourceEffect)
    is_sw = on & ((kind == S.Q_SWITCH) | (kind == S.Q_INSTASWITCH))
    pos = MP.position(st, ref)
    st, rng = SW.switch_in(st, dex, rng, side, jnp.maximum(pos, 0), q.switch_target_b[idx], is_sw & (pos >= 0),
                           source_effect=q.source_effect[idx])
    # runSwitch (batched)
    st, rng = SW.run_switch(st, dex, rng, [ref], on & (kind == S.Q_RUN_SWITCH))
    # beforeTurn: eachEvent('BeforeTurn') (hooks M3); megaEvo (M3.6); others unsupported in core
    unsupported = on & ((kind == S.Q_MEGA) | (kind == S.Q_BEFORE_TURN_MOVE) | (kind == S.Q_PRIORITY_CHARGE) |
                        (kind == S.Q_REVIVAL))
    st = or_flag(st, E.ERR_UNSUPPORTED_EFFECT, unsupported)
    return st, rng


def run_start(st, dex, rng, on):
    """The `start` action queued by team preview (PS: sim/battle.ts:2723-2756: leads switch in, p1a p1b p2a p2b),
    run directly after the preview commit instead of inside the action loops (it is the queue's only entry then),
    followed by its runAction tail. Returns (st, rng)."""
    on = jnp.asarray(on)
    st = C.set_step(st, STEP_TAG[S.Q_START], -1, C.eff_action("start"), 0, on)
    sides = st.sides
    st = st.replace(sides=sides.replace(pokemon_left=jnp.where(on & (sides.pokemon_left > 0), 4,
                                                               sides.pokemon_left).astype(I8)))

    def lead(i, c):
        st, rng = c
        return SW.switch_in(st, dex, rng, i // 2, i % 2, i % 2, on)
    st, rng = jax.lax.fori_loop(0, 4, lead, (st, rng))
    return post_action(st, dex, rng, S.Q_START, on)


def post_action(st, dex, rng, kind, on):
    """PS runAction tail (sim/battle.ts:2861-2986; 04 §7.9). Returns (st, rng)."""
    on = jnp.asarray(on) & (st.meta.phase != S.DONE)
    # (a) phazing in p1a, p1b, p2a, p2b order. One action flags at most NPHAZE slots (forceSwitch moves are
    #     single-target; Red Card on spread targets is the multi-flag case), so the first NPHAZE flagged slots are
    #     dragged in order; more raise ERR_OVERFLOW_PHAZE.
    def flagged(st):
        act = st.sides.active.reshape(4).astype(I32)
        refs = (jnp.arange(4) // 2) * 4 + jnp.maximum(act, 0)
        return on & (act >= 0) & mget(st.mons.force_switch_flag, refs), refs
    for _ in range(NPHAZE):
        fl, refs = flagged(st)
        i = jnp.argmax(fl)
        go, r = jnp.any(fl), refs[i]
        st, rng = SW.drag_in(st, dex, rng, i // 2, i % 2, go & (mget(st.mons.hp, r) > 0))
        m = st.mons
        st = st.replace(mons=m.replace(force_switch_flag=mset(m.force_switch_flag, r, False, go)))
    st = or_flag(st, E.ERR_OVERFLOW_PHAZE, jnp.any(flagged(st)[0]))
    # (b) faints + win
    st = F.resolve_faints(st, dex, on, check_win_if_no_faints=True)
    on = on & (st.meta.phase != S.DONE)
    # (c) end-of-queue faint flags
    _, nk, anyv = Q.head(st.queue)
    st = F.check_fainted(st, on & ~anyv)
    # (d) Update / EmergencyExit: hooks (M3)
    # (e) switch requests
    m = st.mons
    want = []
    for side in range(2):
        flagged = jnp.zeros((), bool)
        refs = []
        for slot in range(2):
            b = st.sides.active[side, slot].astype(I32)
            r = side * 4 + jnp.maximum(b, 0)
            refs.append((r, b >= 0))
            flagged = flagged | ((b >= 0) & (mget(m.switch_flag, r) != 0))
        cs = F.can_switch(st, side) > 0
        clear = on & flagged & ~cs
        for r, ok in refs:   # (Revival Blessing slots keep their flag: v1)
            m = m.replace(switch_flag=mset(m.switch_flag, r, 0, clear & ok))
        want.append(on & flagged & cs)
    st = st.replace(mons=m)
    pause = want[0] | want[1]
    meta = st.meta
    return st.replace(meta=meta.replace(phase=jnp.where(pause, S.SWITCH, meta.phase).astype(I8))), rng


def residual(st, dex, rng, on):
    """PS residual action (sim/battle.ts:2851-2858): updateSpeed, fieldEvent('Residual') (effects/fieldevent.py)."""
    on = jnp.asarray(on)
    st = C.update_speed(st, dex, on)
    return FE.residual(st, dex, rng, on)


def end_turn(st, dex, cfg, on) -> S.State:
    """PS: sim/battle.ts:1640-1818 endTurn (04 §11.3)."""
    on = jnp.asarray(on)
    meta = st.meta
    turn = meta.turn.astype(I32) + on.astype(I32)
    m = st.mons
    for side in range(2):
        for slot in range(2):
            b = st.sides.active[side, slot].astype(I32)
            r = side * 4 + jnp.maximum(b, 0)
            go = on & (b >= 0)
            reset = go & (turn != 1)
            m = m.replace(
                move_last_turn_result=mset(m.move_last_turn_result, r, mget(m.move_this_turn_result, r), go),
                move_this_turn_result=mset(m.move_this_turn_result, r, S.MR_UNDEF, go),
                move_this_turn=mset(m.move_this_turn, r, 0, go),
                newly_switched=mset(m.newly_switched, r, False, go),
                used_item_this_turn=mset(m.used_item_this_turn, r, False, reset),
                stats_raised_this_turn=mset(m.stats_raised_this_turn, r, False, reset),
                stats_lowered_this_turn=mset(m.stats_lowered_this_turn, r, False, reset),
                hurt_this_turn=mset(m.hurt_this_turn, r, -1, reset),
                move_disabled=mset(m.move_disabled, r, jnp.zeros(4, I8), go),   # DisableMove hooks (M3)
                trapped=mset(m.trapped, r, 0, go),                              # TrapPokemon hooks (M3)
                active_turns=mset(m.active_turns, r, mget(m.active_turns, r) + 1, go & ~mget(m.fainted, r)),
            )
    st = st.replace(mons=m)
    # DisableMove (PS :1702-1714): runEvent('DisableMove', pokemon) (volatile / ability / item handlers: M3+), then
    # singleEvent('DisableMove', move) per move slot ([C] Fake Out)
    from vgcjax.effects import conditions as CO
    from vgcjax.effects import hooks as H
    move_handlers = [h for h in H.registered(ids.HOOK_DISABLEMOVE) if CO.MOVE_RANGE[0] <= h.effect < CO.MOVE_RANGE[1]]
    for side in range(2):
        for slot in range(2):
            b = st.sides.active[side, slot].astype(I32)
            r = side * 4 + jnp.maximum(b, 0)
            go = on & (b >= 0)
            for h in move_handlers:          # each registered move handler once per active mon that knows the move
                knows = jnp.any(C.eff_move(mget(st.mons.move_ids, r)) == h.effect)
                hc = H.HookCtx.make(r, -1, h.effect)
                hc2, st2, _ = h.fn(hc, st, dex, None)
                st = jax.tree_util.tree_map(lambda a, b_: jnp.where(go & knows, a, b_), st2, st)
    m = st.mons
    sides = st.sides
    sides = sides.replace(fainted_last_turn=jnp.where(on, sides.fainted_this_turn, sides.fainted_last_turn),
                          fainted_this_turn=jnp.where(on, False, sides.fainted_this_turn))
    tie = on & (turn > 1000)
    cap = on & (turn > cfg.max_turns) & ~tie
    phase = jnp.where(on, S.MOVE, meta.phase)
    phase = jnp.where(tie | cap, S.DONE, phase)
    winner = jnp.where(tie, 2, meta.winner)
    meta = meta.replace(turn=turn.astype(I16), mid_turn=jnp.where(on, False, meta.mid_turn),
                        last_successful_move=jnp.where(on, 0, meta.last_successful_move).astype(I16),
                        phase=phase.astype(I8), winner=winner.astype(I8),
                        ended_reason=jnp.where(tie, S.END_PS_TIE, jnp.where(cap, S.END_TURN_CAP, meta.ended_reason)).astype(I8))
    sc = st.scratch
    z = jnp.zeros((), I16)
    sc = sc.replace(select_ord=jnp.where(on, z, sc.select_ord), runswitch_ord=jnp.where(on, z, sc.runswitch_ord),
                    fieldevent_ord=jnp.where(on, z, sc.fieldevent_ord), eachevent_ord=jnp.where(on, z, sc.eachevent_ord),
                    mulhook_ord=jnp.where(on, z, sc.mulhook_ord), insert_ord=jnp.where(on, z, sc.insert_ord),
                    move_actions=jnp.where(on, jnp.zeros((2, 4), I8), sc.move_actions))
    q = st.queue
    q = q.replace(valid=jnp.where(on, False, q.valid))
    st = st.replace(mons=m, sides=sides, meta=meta, scratch=sc, queue=q)
    return EV.emit(st, on & ~tie, EV.EV_TURN, v1=turn)                  # PS: sim/battle.ts:1802 add('turn', turn)


# ------------------------------------------------------------------ micro-steps
def active(st):
    """Mid-turn work may remain: the MOVE phase after a commit (mid_turn); not awaiting a decision, not ended."""
    return (st.meta.phase == S.MOVE) & st.meta.mid_turn


def micro(st, dex, rng, cfg):
    """One micro-step (module docstring). Returns (st, rng)."""
    act = active(st)
    q = st.queue
    _, hk, anyv = Q.head(q)
    go = act & ~st.exec.in_move & anyv & (hk != S.Q_RESIDUAL)      # residual runs once per call (run_until_decision)
    st = Q.rekey(st, dex, go & (hk == S.Q_MOVE))
    # tied entries share an order class, so the head's kind gives the step_tag of the SELECT tie record
    st = C.set_step(st, _lookup(STEP_TAG, hk, ids.STEP_COMMIT), -1, -1, 0, go)
    st, rng, idx, passes = Q.select(st, dex, rng, go)
    q = st.queue
    kind = q.kind[idx]
    ref = Q.entry_ref(q)[idx]
    is_move = kind == S.Q_MOVE
    tag, actor, subj, aord = entry_step_ctx(st, kind, ref, q.move_id[idx], is_move)
    st = C.set_step(st, tag, actor, subj, aord, go)
    run = go & passes                     # early-returning actions skip runAction's tail too
    sc = st.scratch
    st = st.replace(scratch=sc.replace(move_actions=mset(sc.move_actions, ref, mget(sc.move_actions, ref) + 1,
                                                         run & is_move)))
    st, rng = run_light(st, dex, rng, cfg, idx, run & ~is_move)
    st, rng = MP.move_begin(st, dex, rng, cfg, q.side[idx], q.actor_b[idx], q.move_slot[idx], q.move_id[idx],
                            q.target_loc[idx], q.original_target[idx], run & is_move)
    st, rng = MP.move_hit(st, dex, rng, cfg)
    move_done = st.exec.in_move & st.exec.stopped
    st, rng = MP.move_end(st, dex, rng, cfg)
    st, rng = post_action(st, dex, rng, kind, (run & ~is_move) | move_done)
    # end of turn when the queue is drained and no move is in progress
    return end_turn(st, dex, cfg, active(st) & ~st.exec.in_move & ~jnp.any(st.queue.valid)), rng


def block_residual(st, dex, rng, cfg):
    """The residual action once everything before it ran (it is the last queue entry of a turn): selected outside the
    micro-steps so the residual walk is traced once per step call, then the runAction tail and end of turn."""
    q = st.queue
    _, hk, anyv = Q.head(q)
    go = active(st) & ~st.exec.in_move & anyv & (hk == S.Q_RESIDUAL)
    st = C.set_step(st, ids.STEP_RESIDUAL, -1, C.eff_action("residual"), 0, go)
    idx = jnp.argmax(q.valid & (q.kind == S.Q_RESIDUAL))
    st = st.replace(queue=q.replace(valid=q.valid.at[idx].set(jnp.where(go, False, q.valid[idx]))))
    st, rng = residual(st, dex, rng, go)
    st, rng = post_action(st, dex, rng, S.Q_RESIDUAL, go)
    return end_turn(st, dex, cfg, active(st) & ~st.exec.in_move & ~jnp.any(st.queue.valid)), rng


def run_until_decision(st, dex, rng, cfg):
    st, rng = jax.lax.fori_loop(0, cfg.micro_steps, lambda i, c: micro(c[0], dex, c[1], cfg), (st, rng))
    st, rng = block_residual(st, dex, rng, cfg)
    # work left for the next call: continuation phase (no decision)
    pending = active(st) & (jnp.any(st.queue.valid) | st.exec.in_move)
    return st.replace(meta=st.meta.replace(phase=jnp.where(pending, S.CONT, st.meta.phase).astype(I8))), rng


# ------------------------------------------------------------------ step
def step(st: S.State, actions, key, cfg, dex, forced: R.ForcedOutcomes | None = None):
    """actions: i32[2] (PREVIEW: 0..359; MOVE/SWITCH: joint index slot0*29 + slot1; ignored in CONT).
    Returns (st', rng)."""
    with EV.enabled(cfg.events):
        return _step(st, actions, key, cfg, dex, forced)


def _step(st: S.State, actions, key, cfg, dex, forced):
    log = R.DrawLog.empty(cfg.draw_log_size) if cfg.draw_log else None
    rng = R.make_rng(key, st.meta.step_counter, forced=forced if cfg.forced else None, log=log)
    cont = st.meta.phase == S.CONT
    # PublicEvents "new this step" span a whole decision, continuation calls included
    st = st.replace(events=st.events.replace(step_start=jnp.where(cont, st.events.step_start, st.events.count)))
    done = st.meta.phase == S.DONE
    phase = st.meta.phase
    st, actions = REQ.sanitize(st, dex, jnp.asarray(actions, I32))
    slots = [(actions[s] // 29, actions[s] % 29) for s in range(2)]
    st = commit_preview(st, dex, actions, (phase == S.PREVIEW) & ~done)
    st, rng = run_start(st, dex, rng, (phase == S.PREVIEW) & ~done)
    st, rng = commit_move(st, dex, rng, slots, (phase == S.MOVE) & ~done)
    st = commit_switch(st, dex, slots, (phase == S.SWITCH) & ~done)
    st = st.replace(meta=st.meta.replace(phase=jnp.where(((phase == S.SWITCH) | cont) & ~done, S.MOVE,
                                                         st.meta.phase).astype(I8)))
    st, rng = run_until_decision(st, dex, rng, cfg)
    meta = st.meta
    st = st.replace(meta=meta.replace(step_counter=meta.step_counter + 1, error_flags=meta.error_flags | rng.err))
    return st, rng

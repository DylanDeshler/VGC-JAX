"""Battle state pytree (03 §4). One battle; vmap adds the batch dimension.

Mon arrays have leading dims [2 sides, 4 brought]; a mon is addressed by ref = side*4 + b. Team-sheet data is [2, 6].
Field semantics follow PS (ADR-007); comments name the PS field each mirrors.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from vgcjax.data import ids

I8, I16, I32, U32 = jnp.int8, jnp.int16, jnp.int32, jnp.uint32
QMAX = 14            # 04 §12
FAINTQ = 8
NE = 32              # PublicEvents ring (03 §4.7)
K_ATT = 8
NV = ids.VOL_END - ids.VOL_BASE
NSC = ids.SIDE_END - ids.SIDE_BASE
NSL = ids.SLOT_END - ids.SLOT_BASE
NPW = ids.PW_END - ids.PW_BASE
WV = (NV + 31) // 32

# phases (03 §4.1)
PREVIEW, MOVE, SWITCH, DONE, CONT = 0, 1, 2, 3, 4   # CONT: mid-turn work left, no decision (ADR-104 option 4)
# queue kinds (03 §4.5)
Q_NONE, Q_BEFORE_TURN, Q_BEFORE_TURN_MOVE, Q_REVIVAL, Q_SWITCH, Q_INSTASWITCH, Q_RUN_SWITCH, Q_MEGA, \
    Q_PRIORITY_CHARGE, Q_MOVE, Q_RESIDUAL, Q_START = range(12)
# PS order classes (04 §5.1)
ORDER = {Q_INSTASWITCH: 3, Q_BEFORE_TURN: 4, Q_BEFORE_TURN_MOVE: 5, Q_REVIVAL: 6, Q_RUN_SWITCH: 101, Q_SWITCH: 103,
         Q_MEGA: 104, Q_PRIORITY_CHARGE: 107, Q_MOVE: 200, Q_RESIDUAL: 300, Q_START: 2}
# ended_reason
END_NONE, END_KO, END_TURN_CAP, END_PS_TIE, END_ERROR = range(5)
# move_this_turn_result codes (PS moveThisTurnResult): undefined / null / false / true
MR_UNDEF, MR_NULL, MR_FALSE, MR_TRUE = 0, 1, 2, 3


@struct.dataclass
class Team:
    """Team sheets (03 §3), sheet order. Static within a battle."""
    species: jax.Array   # i16[2,6]
    item: jax.Array      # i16[2,6]
    ability: jax.Array   # i16[2,6]
    nature: jax.Array    # i8[2,6]
    gender: jax.Array    # i8[2,6]  0 M, 1 F, 2 N
    moves: jax.Array     # i16[2,6,4]
    sp: jax.Array        # i8[2,6,6]


@struct.dataclass
class Meta:
    phase: jax.Array               # i8   PS requestState
    turn: jax.Array                # i16  battle.turn
    step_counter: jax.Array        # i32  RNG coordinate
    mid_turn: jax.Array            # bool battle.midTurn
    winner: jax.Array              # i8   -1 none, 0/1 side, 2 tie
    ended_reason: jax.Array        # i8
    effect_seq: jax.Array          # i32  battle.effectOrder counter
    last_move_battle: jax.Array    # i16  battle.lastMove (MoveId)
    last_successful_move: jax.Array  # i16 battle.lastSuccessfulMoveThisTurn
    error_flags: jax.Array         # u32
    faint_queue: jax.Array         # i8[8] refs, FIFO
    faint_queue_len: jax.Array     # i8
    speed_order: jax.Array         # i8[4] battle.speedOrder: field positions (side + 2*pos) by runSwitch speed sort


@struct.dataclass
class Sides:
    brought_sheet_idx: jax.Array   # i8[2,4]
    ps_order: jax.Array            # i8[2,4] side.pokemon order as brought indices
    active: jax.Array              # i8[2,2] brought index in slot (incl. fainted); -1 before start
    pokemon_left: jax.Array        # i8[2]
    total_fainted: jax.Array       # i8[2]
    fainted_this_turn: jax.Array   # bool[2]
    fainted_last_turn: jax.Array   # bool[2]
    mega_used: jax.Array           # bool[2]
    side_cond_bits: jax.Array      # u32[2]   bit k = SIDE_BASE + k
    side_cond_state: jax.Array     # i16[2,NSC,4] duration, layers, seq, extra
    slot_cond_bits: jax.Array      # u32[2,2]
    slot_cond_state: jax.Array     # i16[2,2,NSL,6]


@struct.dataclass
class Mons:
    sheet_idx: jax.Array           # i8[2,4]
    species: jax.Array             # i16  current forme
    base_species: jax.Array        # i16
    types: jax.Array               # i8[2,4,2] (-1 none)
    added_type: jax.Array          # i8
    stats: jax.Array               # i16[2,4,5] storedStats
    hp: jax.Array                  # i16
    maxhp: jax.Array               # i16
    ability: jax.Array             # i16
    base_ability: jax.Array        # i16
    ability_state: jax.Array       # i16[2,4,4]
    ability_seq: jax.Array         # i32 abilityState.effectOrder
    item: jax.Array                # i16
    item_state: jax.Array          # i16[2,4,2]
    item_seq: jax.Array            # i32
    last_item: jax.Array           # i16
    nature: jax.Array              # i8
    sp: jax.Array                  # i8[2,4,6]
    move_ids: jax.Array            # i16[2,4,4]
    pp: jax.Array                  # i8[2,4,4]
    maxpp: jax.Array               # i8[2,4,4]
    move_disabled: jax.Array       # i8[2,4,4] 0 / 1 true / 2 hidden
    move_used: jax.Array           # bool[2,4,4]
    status: jax.Array              # i8 ConditionId (0 none)
    status_state: jax.Array        # i16[2,4,4] slp startTime/time, tox stage, frz time; [3] = seq
    boosts: jax.Array              # i8[2,4,7]
    vol_bits: jax.Array            # u32[2,4,WV]
    vol_state: jax.Array           # i16[2,4,NV,6]
    fainted: jax.Array             # bool
    faint_queued: jax.Array        # bool
    is_active: jax.Array           # bool
    active_turns: jax.Array        # i16
    active_move_actions: jax.Array  # i16
    newly_switched: jax.Array      # bool
    being_called_back: jax.Array   # bool
    is_started: jax.Array          # bool
    switch_flag: jax.Array         # i16 0 false, 1 true, else source MoveId + 2
    force_switch_flag: jax.Array   # bool
    speed: jax.Array               # i16 cached pokemon.speed (signed)
    last_move: jax.Array           # i16 MoveId
    last_move_used: jax.Array      # i16
    last_move_target_loc: jax.Array  # i8
    move_this_turn: jax.Array      # i16
    move_this_turn_result: jax.Array  # i8 MR_*
    move_last_turn_result: jax.Array  # i8
    hurt_this_turn: jax.Array      # i16 -1 = null
    last_damage: jax.Array         # i16
    times_attacked: jax.Array      # i8
    stats_raised_this_turn: jax.Array  # bool
    stats_lowered_this_turn: jax.Array  # bool
    used_item_this_turn: jax.Array  # bool
    trapped: jax.Array             # i8 0 / 1 true / 2 hidden
    can_mega: jax.Array            # bool canMegaEvo
    weight_hg: jax.Array           # i16
    previously_switched_in: jax.Array  # i8


@struct.dataclass
class Field:
    weather: jax.Array             # i16 ConditionId
    weather_state: jax.Array       # i16[3] duration, seq, source
    terrain: jax.Array             # i16
    terrain_state: jax.Array       # i16[3]
    pw_bits: jax.Array             # u32
    pw_state: jax.Array            # i16[NPW,3]


@struct.dataclass
class Queue:
    valid: jax.Array               # bool[Q]
    prepend: jax.Array             # bool[Q]
    kind: jax.Array                # i8[Q]
    order: jax.Array               # i16[Q]
    priority10: jax.Array          # i16[Q]
    frac10: jax.Array              # i8[Q]
    speed: jax.Array               # i16[Q]
    side: jax.Array                # i8[Q]
    actor_b: jax.Array             # i8[Q]
    move_slot: jax.Array           # i8[Q]
    move_id: jax.Array             # i16[Q]
    target_loc: jax.Array          # i8[Q]
    original_target: jax.Array     # i8[Q] ref or -1
    mega: jax.Array                # bool[Q]
    switch_target_b: jax.Array     # i8[Q]
    source_effect: jax.Array       # i16[Q]
    seq: jax.Array                 # i16[Q] insertion order (stable tiebreak)


@struct.dataclass
class Scratch:
    """Per-turn scratch (03 §4.6); ordinals persist across SWITCH steps and reset in end_turn."""
    select_ord: jax.Array          # i16
    runswitch_ord: jax.Array       # i16
    fieldevent_ord: jax.Array      # i16
    eachevent_ord: jax.Array       # i16
    mulhook_ord: jax.Array         # i16
    insert_ord: jax.Array          # i16 runSwitch insertions this turn (RUNSWITCH_INSERT_TIE sub, contract v2)
    move_actions: jax.Array        # i8[2,4] move actions started this turn (action_ord source)
    queue_seq: jax.Array           # i16 next queue insertion seq
    # step context of the running action (contract §3.1)
    step_tag: jax.Array            # i8
    step_actor: jax.Array          # i8 ref or -1
    step_subject: jax.Array        # i16 EffectId or -1
    action_ord: jax.Array          # i8
    cur_select_ord: jax.Array      # i16
    # runAction tail continuation (ADR-107 §3.2): a forced switch interrupts the tail; it resumes in a later micro-step
    tail_pending: jax.Array        # bool the tail stopped before its forced switches were done
    tail_kind: jax.Array           # i8 queue kind of the action whose tail is pending
    tail_pos: jax.Array            # i8 next active position (p1a p1b p2a p2b = 0..3) to scan for forceSwitchFlag


@struct.dataclass
class Events:
    """PublicEvents ring buffer (03 §4.7): records [t, kind, actor, target, move, v1, v2, flags] (vgcjax/battle/events.py).
    `count` is the total ever written (slot = count % NE); `step_start` = count when the current step began."""
    buf: jax.Array                 # i16[NE, 8]
    count: jax.Array               # i32
    step_start: jax.Array          # i32
    last_move: jax.Array           # i32 absolute index of the last MOVE record (attrLastMove target), -1 = none


@struct.dataclass
class Exec:
    """A move in progress across micro-steps (ADR-104 option 4): what runMove / useMove / trySpreadMoveHit / the
    Champions hit loop keep on their stacks between hits (vgcjax/battle/move_pipeline.py move_begin/hit/end)."""
    in_move: jax.Array             # bool
    u: jax.Array                   # i32 user ref
    move: jax.Array                # i32 MoveId
    targets: jax.Array             # i32[3] refs (-1 none)
    tv: jax.Array                  # bool[3] targets left after immunity / accuracy
    spread: jax.Array              # bool spread modifier
    smart: jax.Array               # bool smartTarget still active
    spread_hit: jax.Array          # bool move.spreadHit
    on_rm: jax.Array               # bool runMove ran
    go_rm: jax.Array               # bool useMove ran (PP ok)
    go_um: jax.Array               # bool useMove got past the fainted / no-target checks
    has_t: jax.Array               # bool getMoveTargets found targets
    side_field: jax.Array          # bool side / field target (tryMoveHit path)
    on_ts: jax.Array               # bool trySpreadMoveHit ran
    on_hl: jax.Array               # bool the hit loop ran
    fail_any: jax.Array            # bool atLeastOneFailure
    pre_ok: jax.Array              # bool the Try / PrepareHit preamble passed
    pre_notfail: jax.Array         # bool the preamble's falsy result was NOT_FAIL
    side: jax.Array                # bool a side / field move (tryMoveHit: one moveHit, no hit-loop tails)
    hits: jax.Array                # i32 targetHits
    multiacc: jax.Array            # bool
    single: jax.Array              # bool one target
    h: jax.Array                   # i32 next hit number (1-based)
    hit_done: jax.Array            # i32 PS `hit` counter (1 + hits done)
    stopped: jax.Array             # bool no more hits
    null_damage: jax.Array         # bool
    dt: jax.Array                  # i32[3] damage[] tags (results.py)
    dv: jax.Array                  # i32[3] damage[] values
    lt: jax.Array                  # i32[3] last moveDamage tags
    lv: jax.Array                  # i32[3] last moveDamage values
    total: jax.Array               # i32 move.totalDamage
    mtype: jax.Array               # i32 ActiveMove type after ModifyType / ModifyMove
    bp: jax.Array                  # i32 ActiveMove basePower after ModifyMove
    acc: jax.Array                 # i32 ActiveMove accuracy after ModifyMove (0 = true)
    tm_null: jax.Array             # bool TryMove returned null (a charge turn: effects/locked.py)


def empty_exec() -> Exec:
    b, i, a3, b3 = _z((), bool), _z((), I32), _z(3, I32), _z(3, bool)
    return Exec(in_move=b, u=i, move=i, targets=_z(3, I32, -1), tv=b3, spread=b, smart=b, spread_hit=b, on_rm=b,
                go_rm=b, go_um=b, has_t=b, side_field=b, on_ts=b, on_hl=b, fail_any=b, pre_ok=b, pre_notfail=b,
                side=b, hits=i, multiacc=b, single=b,
                h=i, hit_done=i, stopped=_z((), bool, True), null_damage=b, dt=a3, dv=a3, lt=a3, lv=a3, total=i,
                mtype=i, bp=i, acc=i, tm_null=b)


@struct.dataclass
class State:
    meta: Meta
    team: Team
    sides: Sides
    mons: Mons
    field: Field
    queue: Queue
    scratch: Scratch
    events: Events
    exec: Exec


def _z(shape, dt, fill=0):
    return jnp.full(shape, fill, dt)


def empty_queue() -> Queue:
    return Queue(valid=_z(QMAX, bool), prepend=_z(QMAX, bool), kind=_z(QMAX, I8), order=_z(QMAX, I16),
                 priority10=_z(QMAX, I16), frac10=_z(QMAX, I8), speed=_z(QMAX, I16), side=_z(QMAX, I8),
                 actor_b=_z(QMAX, I8, -1), move_slot=_z(QMAX, I8, -1), move_id=_z(QMAX, I16),
                 target_loc=_z(QMAX, I8), original_target=_z(QMAX, I8, -1), mega=_z(QMAX, bool),
                 switch_target_b=_z(QMAX, I8, -1), source_effect=_z(QMAX, I16), seq=_z(QMAX, I16))


def empty_mons() -> Mons:
    s = (2, 4)
    return Mons(
        sheet_idx=_z(s, I8, -1), species=_z(s, I16), base_species=_z(s, I16), types=_z(s + (2,), I8, -1),
        added_type=_z(s, I8), stats=_z(s + (5,), I16), hp=_z(s, I16), maxhp=_z(s, I16), ability=_z(s, I16),
        base_ability=_z(s, I16), ability_state=_z(s + (4,), I16), ability_seq=_z(s, I32), item=_z(s, I16),
        item_state=_z(s + (2,), I16), item_seq=_z(s, I32), last_item=_z(s, I16), nature=_z(s, I8),
        sp=_z(s + (6,), I8), move_ids=_z(s + (4,), I16), pp=_z(s + (4,), I8), maxpp=_z(s + (4,), I8),
        move_disabled=_z(s + (4,), I8), move_used=_z(s + (4,), bool), status=_z(s, I8),
        status_state=_z(s + (4,), I16), boosts=_z(s + (7,), I8), vol_bits=_z(s + (WV,), U32),
        vol_state=_z(s + (NV, 6), I16), fainted=_z(s, bool), faint_queued=_z(s, bool), is_active=_z(s, bool),
        active_turns=_z(s, I16), active_move_actions=_z(s, I16), newly_switched=_z(s, bool),
        being_called_back=_z(s, bool), is_started=_z(s, bool), switch_flag=_z(s, I16),
        force_switch_flag=_z(s, bool), speed=_z(s, I16), last_move=_z(s, I16), last_move_used=_z(s, I16),
        last_move_target_loc=_z(s, I8), move_this_turn=_z(s, I16), move_this_turn_result=_z(s, I8),
        move_last_turn_result=_z(s, I8), hurt_this_turn=_z(s, I16, -1), last_damage=_z(s, I16),
        times_attacked=_z(s, I8), stats_raised_this_turn=_z(s, bool), stats_lowered_this_turn=_z(s, bool),
        used_item_this_turn=_z(s, bool), trapped=_z(s, I8), can_mega=_z(s, bool), weight_hg=_z(s, I16),
        previously_switched_in=_z(s, I8))


def empty_scratch() -> Scratch:
    return Scratch(select_ord=_z((), I16), runswitch_ord=_z((), I16), fieldevent_ord=_z((), I16),
                   eachevent_ord=_z((), I16), mulhook_ord=_z((), I16), insert_ord=_z((), I16), move_actions=_z((2, 4), I8),
                   queue_seq=_z((), I16), step_tag=_z((), I8), step_actor=_z((), I8, -1),
                   step_subject=_z((), I16, -1), action_ord=_z((), I8), cur_select_ord=_z((), I16, -1),
                   tail_pending=_z((), bool), tail_kind=_z((), I8), tail_pos=_z((), I8))


def empty_state(team: Team) -> State:
    meta = Meta(phase=_z((), I8, PREVIEW), turn=_z((), I16), step_counter=_z((), I32), mid_turn=_z((), bool),
                winner=_z((), I8, -1), ended_reason=_z((), I8), effect_seq=_z((), I32), last_move_battle=_z((), I16),
                last_successful_move=_z((), I16), error_flags=_z((), U32), faint_queue=_z(FAINTQ, I8, -1),
                faint_queue_len=_z((), I8), speed_order=jnp.arange(4, dtype=I8))
    sides = Sides(brought_sheet_idx=_z((2, 4), I8, -1), ps_order=jnp.tile(jnp.arange(4, dtype=I8), (2, 1)),
                  active=_z((2, 2), I8, -1), pokemon_left=_z(2, I8, 6), total_fainted=_z(2, I8),
                  fainted_this_turn=_z(2, bool), fainted_last_turn=_z(2, bool), mega_used=_z(2, bool),
                  side_cond_bits=_z(2, U32), side_cond_state=_z((2, NSC, 4), I16), slot_cond_bits=_z((2, 2), U32),
                  slot_cond_state=_z((2, 2, NSL, 6), I16))
    field = Field(weather=_z((), I16), weather_state=_z(3, I16), terrain=_z((), I16), terrain_state=_z(3, I16),
                  pw_bits=_z((), U32), pw_state=_z((NPW, 3), I16))
    return State(meta=meta, team=team, sides=sides, mons=empty_mons(), field=field, queue=empty_queue(),
                 scratch=empty_scratch(), events=Events(buf=_z((NE, 8), I16), count=_z((), I32), step_start=_z((), I32),
                                                       last_move=_z((), I32, -1)), exec=empty_exec())

"""Error bitfield (ADR-014): no exceptions inside jit; every abnormal event sets a bit in `state.error_flags`."""

ERR_ILLEGAL_ACTION = 1 << 0        # an illegal action was replaced by the first legal one
ERR_UNSUPPORTED_EFFECT = 1 << 1    # an entity outside the compiled stage executed
ERR_OVERFLOW_BLOCK_A = 1 << 2
ERR_OVERFLOW_BLOCK_B = 1 << 3
ERR_OVERFLOW_QUEUE = 1 << 4
ERR_OVERFLOW_HOOK = 1 << 5
ERR_INVARIANT = 1 << 6
ERR_FORCED_MISSING = 1 << 7        # forced mode: a non-degenerate draw found no forced entry
ERR_DRAWLOG_OVERFLOW = 1 << 8
ERR_OVERFLOW_FAINT_QUEUE = 1 << 9
ERR_OVERFLOW_ATTACKED_BY = 1 << 10
ERR_OVERFLOW_EVENTS = 1 << 11
ERR_OVERFLOW_RESIDUAL = 1 << 12
ERR_TURN_CAP = 1 << 13             # informational: game ended by the turn cap
ERR_OVERFLOW_PHAZE = 1 << 14       # more than step.NPHAZE slots flagged for phazing after one action

NAMES = {v: k for k, v in globals().items() if k.startswith("ERR_")}


def describe(flags: int) -> list[str]:
    return [name for bit, name in sorted(NAMES.items()) if flags & bit]

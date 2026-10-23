"""Trace-time (static) facts used to keep hook candidate lists small (09 §1.1a).

PS collects a handler only when the effect defines the callback (getCallback) — or, for fieldEvent('Residual'),
when its state has a duration. A callback the engine has not implemented cannot run, so candidates are restricted
to effects with a *registered* handler for (hook, prefix) (effects/hooks.py registry; every content module is
imported by effects/registry.py before tracing), plus data-driven duration rows for the residual walk. Entities
outside the implemented set are kept out of battles by the stage (tools/datagen/impl_stage.py).
"""
from __future__ import annotations

from vgcjax.core.dex import load_tables_np
from vgcjax.data import ids


DATA_MODE = [False]     # tests: collect every effect whose PS data defines the hook (handler_mask), not only registered


def registered_effects(hook: int, prefix: int) -> frozenset[int]:
    if DATA_MODE[0]:
        t = load_tables_np()
        col = t["handler_mask"][:, hook].astype("int64")
        return frozenset(int(e) for e in ((col >> prefix) & 1).nonzero()[0])
    from vgcjax.effects import hooks as H
    return frozenset(h.effect for h in H.registered(hook, prefix))


def cond_subset(hook: int, prefix: int, base: int, count: int, with_duration: bool = False) -> tuple[int, ...]:
    """Indices k in [0, count) of conditions `base + k` with a registered (hook, prefix) handler (or a duration /
    durationCallback in the data when with_duration)."""
    t = load_tables_np()
    reg = registered_effects(hook, prefix)
    out = []
    for k in range(count):
        cid = base + k
        e = ids.COND_BASE + cid - 1
        dur = with_duration and (int(t["cond_duration"][cid]) > 0 or (int(t["cond_flags"][cid]) & 1))
        if e in reg or dur:
            out.append(k)
    return tuple(out)


def any_defines(hook: int, prefix: int, base: int, count: int) -> bool:
    """Is a (hook, prefix) handler registered for any effect base .. base+count-1 (EffectIds)?"""
    return any(base <= e < base + count for e in registered_effects(hook, prefix))


def statuses_define(hook: int, prefix: int) -> bool:
    return bool(cond_subset(hook, prefix, 1, 6))


def abilities_define(hook: int, prefix: int) -> bool:
    return any_defines(hook, prefix, ids.ABILITY_BASE, len(ids.ABILITIES) - 1)


def items_define(hook: int, prefix: int) -> bool:
    return any_defines(hook, prefix, ids.ITEM_BASE, len(ids.ITEMS) - 1)

"""Static engine configuration (ADR-006): a frozen, hashable dataclass passed as a static argument to `jit`.

Python `if`s on it run at trace time. Feature flags for known PS-vs-game disputes default to PS (ADR-019).
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Flags:
    """ADR-019 dispute flags + test-only switches. Conformance always runs with defaults."""

    fire_mane_atk_mod: bool = False         # PS issue #12377: PS applies a base-power modifier
    simultaneous_intimidate: bool = False   # PS resolves Intimidate per target sequentially
    encore_target_reroll: bool = False      # PS re-rolls the Encore target at resolve
    canary_crit_2048: bool = False          # ADR-029 canary: crit x1.25 instead of x1.5 (tests only)


@dataclasses.dataclass(frozen=True)
class EngineConfig:
    stage: str = "core"                 # core | v0 | v1 | v2 (03 §2 supported masks)
    max_turns: int = 1000               # ADR-023: conformance 1000 (PS hard tie); training default 40
    forced: bool = False                # forced-outcome mode (ADR-012)
    draw_log: bool = False              # draw-log mode (ADR-012)
    kf: int = 160                       # forced entries per step (04 §12)
    draw_log_size: int = 192            # draw-log capacity per step
    debug_checks: bool = False          # checkify invariants (ADR-014)
    flags: Flags = Flags()
    handlers: frozenset[str] = frozenset()  # bespoke handler ids compiled in (empty = all registered for the stage)
    turn_cap_outcome: str = "tiebreak"  # 08 §5: tiebreak | draw | truncate
    preview_canonical: bool = True      # 08 §3: policy sees the canonical preview subset
    reward_01: bool = False             # 08 §5: 1/0 rewards instead of +1/-1
    events: bool = True                 # 03 §4.7 PublicEvents (obs.events); off compiles every emit out (train
                                        # builds whose policy ignores events; ~15% fewer kernels per step)

    def replace(self, **kw) -> "EngineConfig":
        return dataclasses.replace(self, **kw)

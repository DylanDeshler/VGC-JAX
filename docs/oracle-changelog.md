# Oracle changelog (PS re-pins, ADR-021)

| Date | PS commit | Notes |
|---|---|---|
| 2026-10-21 | d601443456c795f77c6e4f2b5b459167f2b10a3c | initial pin (ADR-002) |

## 2026-10-23 — CONFUSION_SELFHIT_DAMAGE_ROLL subject
The oracle keyed confusion self-hit damage rolls with subject null; contract/drawkey.md §5 specifies `confusion`
(oracle/lib/instrument.js buildDrawKey). Found by the first impl-stage corpus with confusion moves; no earlier
corpus contained the kind.

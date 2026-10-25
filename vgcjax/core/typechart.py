"""Type effectiveness and immunity from the generated PS `damageTaken` table (03 §2, M1.4).

`dex.type_damage_taken[def_type, atk_type]` holds PS codes {0 neutral, 1 super effective, 2 resisted, 3 immune};
`dex.type_immunity[def_type, k]` the pseudo-keys in `ids.IMMUNITY_KEYS` (brn, par, psn, tox, frz, sandstorm, hail,
powder, prankster, trapped). Defender types are a 2-vector with -1 for "no second type".
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.data import ids

I32 = jnp.int32
_CODE_TO_MOD = jnp.array([0, 1, -1, 0], I32)


def _codes(table, def_types, col):
    t = jnp.asarray(def_types, I32)
    valid = t > 0
    codes = table[jnp.where(valid, t, 0), col]
    return jnp.where(valid, codes, 0)


def effectiveness(dex, atk_type, def_types, freeze_dry=False):
    """Sum of per-type log2 multipliers. PS: sim/dex.ts getEffectiveness (immunity not included); freeze_dry: the
    move's onEffectiveness makes Water 1 (data/moves.ts freezedry)."""
    c = _codes(dex.type_damage_taken, def_types, jnp.asarray(atk_type, I32)[..., None])
    mods = _CODE_TO_MOD[c]
    mods = jnp.where(jnp.asarray(freeze_dry) & (jnp.asarray(def_types) == ids.TYPE_WATER), 1, mods)
    return jnp.sum(mods, axis=-1)


def immune(dex, atk_type, def_types):
    """True if any defending type is immune to `atk_type`. PS: sim/dex.ts getImmunity (negated)."""
    c = _codes(dex.type_damage_taken, def_types, jnp.asarray(atk_type, I32)[..., None])
    return jnp.any(c == 3, axis=-1)


def status_immune(dex, key_idx, def_types):
    """Immunity to a pseudo-key (index into ids.IMMUNITY_KEYS). PS: sim/dex.ts getImmunity on damageTaken[key]."""
    c = _codes(dex.type_immunity, def_types, jnp.asarray(key_idx, I32)[..., None])
    return jnp.any(c == 3, axis=-1)

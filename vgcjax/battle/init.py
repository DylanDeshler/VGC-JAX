"""Battle construction and team preview (04 §3)."""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import state as S
from vgcjax.core import stats as ST

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32


def init(team: S.Team) -> S.State:
    """State at the PREVIEW decision. PS: sim/battle.ts:192-317 (construction), :1922-1986 start()."""
    return S.empty_state(team)


def build_mon(dex, team: S.Team, side, sheet):
    """Constructed Pokemon fields for sheet slot `sheet` of `side` (PS: sim/pokemon.ts constructor + setSpecies +
    clearVolatile). Returns a dict of per-mon scalars/vectors."""
    sp = team.species[side, sheet].astype(I32)
    nat = team.nature[side, sheet].astype(I32)
    sp6 = team.sp[side, sheet].astype(I32)
    maxhp, stats = ST.calc_stats(dex.species_base_stats[sp], sp6, dex.nature_plus[nat], dex.nature_minus[nat])
    moves = team.moves[side, sheet].astype(I32)
    maxpp = jnp.where(moves > 0, dex.move_pp[moves], 0)
    item = team.item[side, sheet].astype(I32)
    mega = dex.species_mega[sp]  # [2, (stone, mega)]
    can_mega = jnp.any((mega[:, 0] == item) & (item > 0) & (mega[:, 1] > 0))  # [C] scripts.ts:182-194
    return dict(
        sheet_idx=sheet, species=sp, base_species=sp, types=dex.species_types[sp], stats=stats, hp=maxhp,
        maxhp=maxhp, ability=team.ability[side, sheet], base_ability=team.ability[side, sheet], item=item,
        nature=nat, sp=sp6, move_ids=moves, pp=maxpp, maxpp=maxpp, can_mega=can_mega,
        weight_hg=dex.species_weight_hg[sp], newly_switched=True, speed=stats[4],
    )


def apply_preview(st: S.State, dex, orders) -> S.State:
    """Both sides' preview choices (indices into PREVIEW_ORDERS). PS: sim/side.ts:1031-1095 chooseTeam,
    sim/battle.ts:2803-2810 runAction('team'): side.pokemon := the picked 4 in order (b = position)."""
    m = st.mons
    sides = st.sides
    for side in range(2):
        order = dex.preview_orders[jnp.asarray(orders[side], I32)].astype(I32)
        sides = sides.replace(brought_sheet_idx=sides.brought_sheet_idx.at[side].set(order.astype(I8)))
        for b in range(4):
            f = build_mon(dex, st.team, side, order[b])
            upd = {}
            for k, v in f.items():
                arr = getattr(m, k)
                upd[k] = arr.at[side, b].set(jnp.asarray(v, arr.dtype))
            m = m.replace(**upd)
    sides = sides.replace(ps_order=jnp.tile(jnp.arange(4, dtype=I8), (2, 1)))
    return st.replace(mons=m, sides=sides)

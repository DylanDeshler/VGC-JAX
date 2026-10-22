"""Observations (08 §4, ADR-022): information-state only, canonical frame (index 0 = observer, 1 = opponent).

Hidden from the observer about the opponent: SP spread and exact stats, exact HP (PS Champions percentage rule only),
unrevealed brought mons and bench order, PP, RNG. Under open team sheets the opponent's six sets are public.
PublicEvents (03 §4.7) are not emitted by the engine yet; `events` is all zeros until they are (M5.2 follow-up).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from vgcjax.battle import state as S

I16, I32 = jnp.int16, jnp.int32


@struct.dataclass
class Obs:
    meta: jax.Array        # i16[8]: phase, turn, turns_to_cap, mega_available, must_act slot0/1, mid_turn, opp_mega_used
    field: jax.Array       # i16[4 + 2*NPW]: weather, weather turns left, terrain, terrain turns left, pw bits/durations
    side: jax.Array        # i16[2, 3 + NSC]: mons left, fainted count, mega used, side-condition durations
    sheet: jax.Array       # i16[2, 6, 11]: species, item, ability, nature, gender, moves[4], brought(0 unk/1 yes/2 no), revealed
    own: jax.Array         # i16[6, 18]: (self only, by sheet idx) stats[5], sp[6], hp, maxhp, pp[4], brought_pos
    mon: jax.Array         # i16[2, 4, 22]: per brought mon of each side (opponent: zeros until revealed)
    events: jax.Array      # i16[16, 8]: last public events (TODO: engine PublicEvents)
    masks_slot: jax.Array  # bool[2, 29]
    masks_pair: jax.Array  # bool[29, 29]
    masks_preview: jax.Array  # bool[360]


MON_FIELDS = ("present", "sheet", "slot", "hp", "hp_colour", "maxhp_known", "status", "fainted", "species", "item",
              "ability", "last_move", "active_turns", "boosts0", "boosts1", "boosts2", "boosts3", "boosts4", "boosts5",
              "boosts6", "can_mega", "is_active")


def hp_public(hp, maxhp):
    """PS Champions getHealth (sim/pokemon.ts getHealth): pct = floor(100 hp / maxhp) || 1 (0 if fainted);
    colour suffix only at 20 / 50: 1 if above the boundary ('y' at 20, 'g' at 50), else 2."""
    hp = jnp.asarray(hp, I32)
    maxhp = jnp.maximum(jnp.asarray(maxhp, I32), 1)
    pct = jnp.where(hp > 0, jnp.maximum((100 * hp) // maxhp, 1), 0)
    above = jnp.where(pct == 20, hp * 5 > maxhp, jnp.where(pct == 50, hp * 2 > maxhp, False))
    colour = jnp.where((pct == 20) | (pct == 50), jnp.where(above, 1, 2), 0)
    return pct, colour


def observe(st: S.State, player, cfg, masks) -> Obs:
    """Observation for `player` (0/1). `masks` = requests.legal_masks(st, dex)."""
    p = jnp.asarray(player, I32)
    o = 1 - p
    sides_order = jnp.stack([p, o])
    m = st.mons
    sd = st.sides
    meta = st.meta
    turns_to_cap = jnp.maximum(cfg.max_turns - meta.turn.astype(I32), 0)
    slot_masks = masks["slot"][p]
    must = jnp.stack([jnp.sum(slot_masks[k].astype(I32)) > 1 for k in range(2)])
    mega_av = jnp.any(m.can_mega[p]) & ~sd.mega_used[p]
    meta_v = jnp.stack([meta.phase.astype(I32), meta.turn.astype(I32), turns_to_cap, mega_av.astype(I32),
                        must[0].astype(I32), must[1].astype(I32), meta.mid_turn.astype(I32),
                        sd.mega_used[o].astype(I32)]).astype(I16)
    f = st.field
    pw_on = ((f.pw_bits >> jnp.arange(S.NPW, dtype=jnp.uint32)) & 1).astype(I32)
    field_v = jnp.concatenate([jnp.stack([f.weather.astype(I32), f.weather_state[0].astype(I32), f.terrain.astype(I32),
                                          f.terrain_state[0].astype(I32)]), pw_on, f.pw_state[:, 0].astype(I32)]).astype(I16)
    side_rows = []
    for s in (p, o):
        on = ((sd.side_cond_bits[s] >> jnp.arange(S.NSC, dtype=jnp.uint32)) & 1) == 1
        side_rows.append(jnp.concatenate([jnp.stack([sd.pokemon_left[s].astype(I32), sd.total_fainted[s].astype(I32),
                                                     sd.mega_used[s].astype(I32)]),
                                          jnp.where(on, jnp.maximum(sd.side_cond_state[s, :, 0].astype(I32), 1), 0)]))
    side_v = jnp.stack(side_rows).astype(I16)
    team = st.team
    revealed = m.previously_switched_in > 0          # has been on the field
    sheet_rows = []
    for k, s in enumerate((p, o)):
        brought_sheet = sd.brought_sheet_idx[s].astype(I32)        # [4]
        rows = []
        for i in range(6):
            in_brought = jnp.any(brought_sheet == i)
            b = jnp.argmax(brought_sheet == i)
            rev = in_brought & revealed[s, b]
            # self knows the brought set; for the opponent only revealed mons are known to be brought
            brought_code = jnp.where(k == 0, jnp.where(in_brought, 1, 2),
                                     jnp.where(rev, 1, jnp.where(meta.phase == S.PREVIEW, 0, 0)))
            rows.append(jnp.concatenate([jnp.stack([team.species[s, i], team.item[s, i], team.ability[s, i],
                                                    team.nature[s, i], team.gender[s, i]]).astype(I32),
                                         team.moves[s, i].astype(I32), jnp.stack([brought_code, rev.astype(I32)])]))
        sheet_rows.append(jnp.stack(rows))
    sheet_v = jnp.stack(sheet_rows).astype(I16)
    own_rows = []
    for i in range(6):
        b = jnp.argmax(sd.brought_sheet_idx[p] == i)
        inb = jnp.any(sd.brought_sheet_idx[p] == i)
        pos = jnp.argmax(sd.ps_order[p] == b)
        own_rows.append(jnp.where(inb, jnp.concatenate([m.stats[p, b].astype(I32), m.sp[p, b].astype(I32),
                                                        jnp.stack([m.hp[p, b], m.maxhp[p, b]]).astype(I32),
                                                        m.pp[p, b].astype(I32), pos[None].astype(I32)]), 0))
    own_v = jnp.stack(own_rows).astype(I16)
    mon_rows = []
    for k, s in enumerate((p, o)):
        rows = []
        for b in range(4):
            vis = (k == 0) | revealed[s, b]
            pos = jnp.where(sd.active[s, 0] == b, 0, jnp.where(sd.active[s, 1] == b, 1, -1))
            pct, colour = hp_public(m.hp[s, b], m.maxhp[s, b])
            hp = jnp.where(k == 0, m.hp[s, b].astype(I32), pct)
            row = jnp.concatenate([jnp.stack([jnp.asarray(1, I32), sd.brought_sheet_idx[s, b].astype(I32), pos,
                                              hp, jnp.where(k == 0, 0, colour),
                                              jnp.where(k == 0, m.maxhp[s, b].astype(I32), 100),
                                              m.status[s, b].astype(I32), m.fainted[s, b].astype(I32),
                                              m.species[s, b].astype(I32), m.item[s, b].astype(I32),
                                              m.ability[s, b].astype(I32), m.last_move[s, b].astype(I32),
                                              m.active_turns[s, b].astype(I32)]),
                                   m.boosts[s, b].astype(I32),
                                   jnp.stack([m.can_mega[s, b].astype(I32), m.is_active[s, b].astype(I32)])])
            rows.append(jnp.where(vis, row, 0))
        rows = jnp.stack(rows)
        if k == 1:
            # opponent: canonical order by sheet index, unrevealed last (brought / bench order is hidden)
            key = jnp.where(revealed[s], sd.brought_sheet_idx[s].astype(I32), 99)
            rows = rows[jnp.argsort(key, stable=True)]
        mon_rows.append(rows)
    mon_v = jnp.stack(mon_rows).astype(I16)
    return Obs(meta=meta_v, field=field_v, side=side_v, sheet=sheet_v, own=own_v, mon=mon_v,
               events=jnp.zeros((16, 8), I16), masks_slot=masks["slot"][p], masks_pair=masks["pair_ok"][p],
               masks_preview=masks["preview"][p])

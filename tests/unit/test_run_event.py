"""M3.1: run_event fold mechanics with test-only handlers (candidate gathering, prefix roles, MUL chain, falsy stop,
final modify, fast exit)."""
import jax
import jax.numpy as jnp
import numpy as np

from vgcjax.battle import init as INIT
from vgcjax.battle import results as RS
from vgcjax.battle import state as S
from vgcjax.core import rng as R
from vgcjax.core.dex import load_dex
from vgcjax.data import ids
from vgcjax.effects import hooks as H


def battle():
    from vgcjax.battle.state import Team
    sp = np.full((2, 6), ids.SPECIES_GARCHOMP, np.int16)
    ab = np.full((2, 6), ids.ABILITY_ROUGHSKIN, np.int16)
    ab[0, 0] = ids.ABILITY_TECHNICIAN
    it = np.zeros((2, 6), np.int16)
    it[1, 0] = ids.ITEM_LIFEORB
    team = Team(species=sp, item=it, ability=ab, nature=np.zeros((2, 6), np.int8), gender=np.zeros((2, 6), np.int8),
                moves=np.full((2, 6, 4), ids.MOVE_EARTHQUAKE, np.int16), sp=np.zeros((2, 6, 6), np.int8))
    dex = load_dex("v2")
    st = INIT.apply_preview(INIT.init(team), dex, jnp.zeros(2, jnp.int32))
    m = st.mons
    st = st.replace(mons=m.replace(is_active=m.is_active.at[:, :2].set(True)),
                    sides=st.sides.replace(active=jnp.array([[0, 1], [0, 1]], jnp.int8)))
    return st, dex


def test_candidates_roles():
    st, dex = battle()
    c = H.candidates(st, dex, "BasePower", 0, 4)
    eff = np.asarray(c["eff"])[np.asarray(c["valid"])]
    tech = ids.ABILITY_BASE + ids.ABILITY_TECHNICIAN - 1
    assert tech in eff.tolist()
    c2 = H.candidates(st, dex, "ModifyDamage", 0, 4)  # Life Orb on p2a is onModifyDamage for its holder only
    eff2 = np.asarray(c2["eff"])[np.asarray(c2["valid"])]
    lo = ids.ITEM_BASE + ids.ITEM_LIFEORB - 1
    assert lo not in eff2.tolist()
    c3 = H.candidates(st, dex, "ModifyDamage", 4, 0)
    assert lo in np.asarray(c3["eff"])[np.asarray(c3["valid"])].tolist()


def test_fold_mul_and_final_modify():
    st, dex = battle()
    tech = ids.ABILITY_BASE + ids.ABILITY_TECHNICIAN - 1

    def technician(hc, st, dex, rng):  # test-only: x1.5 when bp <= 60 (data/abilities.ts technician)
        return hc.chain(6144, 4096), st, rng
    local = lambda e: jnp.where(e == tech, 1, 0)  # noqa: E731
    hc = H.HookCtx.make(0, 4, -1, relay=(RS.NUM, 40))
    hc, st2, _ = H.run_event("BasePower", st, dex, R.make_rng(jax.random.PRNGKey(0), 0), hc, branches=(local, [technician]))
    assert int(hc.relay_v) == 60 and int(st2.meta.error_flags) == 0

    def gate(hc, st, dex, rng):
        return hc.ret(RS.R(RS.BOOL, 0)), st, rng
    hc = H.HookCtx.make(0, 4, -1, relay=(RS.NUM, 40))
    hc, _, _ = H.run_event("BasePower", st, dex, R.make_rng(jax.random.PRNGKey(0), 0), hc, branches=(local, [gate]))
    assert int(hc.relay_t) == RS.BOOL and int(hc.relay_v) == 0 and bool(hc.stop)

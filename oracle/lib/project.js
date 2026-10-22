'use strict';
/**
 * Canonical "state projection" of a PS battle: the JSON both PS and the JAX engine emit at every
 * decision point, and that the differential tester diffs field by field.
 *
 * Design rules
 *  - only engine-relevant state (no log, no request bookkeeping, no RNG state)
 *  - references become labels (p1a / p1:speciesid), effects become ids
 *  - effect states are reduced to primitive fields; bookkeeping keys are dropped
 *  - deterministic key order (sorted) so JSON.stringify equality == state equality
 */
const { monLabel } = require('./context');

const DROP_STATE_KEYS = new Set(['id', 'effectOrder', 'target', 'source', 'sourceSlot', 'sourceEffect', 'started', 'isSlotCondition', 'name']);

function effState(st) {
	if (!st) return null;
	const out = {};
	for (const k of Object.keys(st).sort()) {
		if (DROP_STATE_KEYS.has(k)) continue;
		const v = st[k];
		if (v === undefined || typeof v === 'function') continue;
		if (v === null || typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string') out[k] = v;
		else if (Array.isArray(v)) out[k] = v.map(x => (x && typeof x === 'object') ? (monLabel(x) || x.id || '?') : x);
		else if (v.species && v.side) out[k] = monLabel(v);
		else if (v.id) out[k] = v.id;
		else if (v.effectType) out[k] = v.name;
		// nested plain objects (rare) are ignored on purpose
	}
	return out;
}

function effMap(obj) {
	const out = {};
	for (const id of Object.keys(obj).sort()) out[id] = effState(obj[id]);
	return out;
}

function projectPokemon(p) {
	// id = side:sheet_index (07 §5.1); baseSpecies changes on Mega in Champions (RF #32)
	const ident = p.battle.__ident;
	const sheet = ident ? ident.sheet(p) : p.side.team.findIndex(t => t.species === p.set.species);
	return {
		id: `${p.side.id}:${sheet}`,
		b: ident ? ident.b(p) : -1,
		slot: p.isActive ? `${p.side.id}${'ab'[p.position]}` : null,
		species: p.species.id,
		types: p.getTypes(true),
		hp: p.hp, maxhp: p.maxhp, fainted: !!p.fainted,
		status: p.status || '', statusState: p.status ? effState(p.statusState) : null,
		boosts: { ...p.boosts },
		stats: { ...p.storedStats },
		volatiles: effMap(p.volatiles),
		item: p.item || '', itemState: p.item ? effState(p.itemState) : null, lastItem: p.lastItem || '',
		ability: p.ability, abilityState: effState(p.abilityState),
		moves: p.moveSlots.map(m => ({ id: m.id, pp: m.pp, maxpp: m.maxpp, disabled: !!m.disabled })),
		canMegaEvo: !!p.canMegaEvo,
		lastMove: p.lastMove ? p.lastMove.id : null,
		trapped: !!p.trapped,
		weightkg: p.weighthg / 10,
	};
}

function projectBattle(battle) {
	const f = battle.field;
	return {
		turn: battle.turn,
		ended: battle.ended, winner: battle.winner || null,
		field: {
			weather: f.weather || '', weatherState: f.weather ? effState(f.weatherState) : null,
			terrain: f.terrain || '', terrainState: f.terrain ? effState(f.terrainState) : null,
			pseudoWeather: effMap(f.pseudoWeather),
		},
		sides: battle.sides.map(side => ({
			id: side.id,
			pokemonLeft: side.pokemonLeft,
			sideConditions: effMap(side.sideConditions),
			slotConditions: side.slotConditions.map(effMap),
			active: side.active.map(p => (p ? monLabel(p) + '=' + p.baseSpecies.id : null)),
			pokemon: side.pokemon.map(projectPokemon),
		})),
	};
}

/** Field-level diff of two projections (for reporting): returns list of [path, a, b]. */
function diffProjection(a, b, pth = '', out = []) {
	if (a === b) return out;
	if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) { out.push([pth, a, b]); return out; }
	const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
	for (const k of keys) diffProjection(a[k], b[k], pth ? `${pth}.${k}` : k, out);
	return out;
}

module.exports = { projectBattle, projectPokemon, diffProjection, effState };

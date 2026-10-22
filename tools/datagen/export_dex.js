#!/usr/bin/env node
'use strict';
/**
 * PS Dex.mod(<mod>) -> data/export/*.json (raw, lossless, PS field names, merged mod values) — 06 §2, ADR-020.
 *   node tools/datagen/export_dex.js --mod champions --format gen9championsvgc2026regmc --out data/export
 *
 * Outputs: species, moves, abilities, items, conditions, typechart, natures, formats, handler_order, meta (.json).
 * Function-valued keys are dropped by JSON; their (deep) names are kept in `fnKeys` for the census cross-check.
 * Champions PP is computed here and asserted against a constructed PS battle's moveSlots[].maxpp for every move.
 * Handler (order, priority, subOrder) come from PS's own battle.resolvePriority on synthetic handlers.
 * Deterministic: keys sorted, entities sorted by id.
 */
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

function parseArgs(argv) {
	const out = {};
	for (let i = 0; i < argv.length; i++) if (argv[i].startsWith('--')) out[argv[i].slice(2)] = argv[++i];
	return out;
}
const args = parseArgs(process.argv.slice(2));
const MOD = args.mod || 'champions';
const FORMAT = args.format || 'gen9championsvgc2026regmc';
const OUT = args.out || path.join(__dirname, '..', '..', 'data', 'export');
const PS_ROOT = path.resolve(process.env.PS_ROOT || path.join(__dirname, '..', '..', 'oracle', 'ps'));
const { Dex, Battle } = require(path.join(PS_ROOT, 'dist', 'sim'));
const { Side } = require(path.join(PS_ROOT, 'dist', 'sim', 'side'));
const { Field } = require(path.join(PS_ROOT, 'dist', 'sim', 'field'));

const dex = Dex.mod(MOD);

// ------------------------------------------------------------------ generic serialisation
/** JSON-safe deep copy with sorted keys; functions dropped (their paths collected in `fns`). */
function plain(v, fns, p) {
	if (typeof v === 'function') { fns.push(p); return undefined; }
	if (v === null || typeof v !== 'object') return v === undefined ? undefined : v;
	if (Array.isArray(v)) return v.map((x, i) => { const r = plain(x, fns, `${p}[${i}]`); return r === undefined ? null : r; });
	const out = {};
	for (const k of Object.keys(v).sort()) {
		const r = plain(v[k], fns, p ? `${p}.${k}` : k);
		if (r !== undefined) out[k] = r;
	}
	return out;
}
function entity(obj) {
	const fns = [];
	const data = plain(obj, fns, '');
	data.fnKeys = fns.sort();
	return data;
}
function sortedMap(list, f) {
	const out = {};
	for (const x of list.slice().sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) out[x.id] = f(x);
	return out;
}
function write(name, obj) {
	fs.writeFileSync(path.join(OUT, `${name}.json`), JSON.stringify(obj, null, 1) + '\n');
}

// ------------------------------------------------------------------ species / megas
function exportSpecies() {
	const species = sortedMap(dex.species.all().filter(s => s.exists), s => entity(s));
	const megas = {};
	for (const it of dex.items.all()) {
		if (!it.exists || !it.megaStone) continue;
		for (const [base, mega] of Object.entries(it.megaStone).sort()) {
			(megas[dex.toID(base)] ||= []).push({ item: it.id, mega: dex.toID(mega) });
		}
	}
	for (const k of Object.keys(megas)) megas[k].sort((a, b) => (a.item < b.item ? -1 : 1));
	return { species, megas };
}

// ------------------------------------------------------------------ moves + Champions PP
/** CH §1.1-1.2: pp_c = min(pp, 20); maxpp = noPPBoosts ? pp_c : floor(pp_c / 5 + 1) * 4 ... asserted below. */
function champsMaxPP(m) {
	const ppc = Math.min(m.pp, 20);
	return m.noPPBoosts ? ppc : Math.floor(ppc / 5 + 1) * 4;
}
function psMaxPP(moveIds) {
	// one throwaway battle per 24 moves (6 mons x 4 slots on p1); species/legality irrelevant for moveSlots
	const out = {};
	const mkTeam = ids => Array.from({ length: 6 }, (_, i) => ({
		name: `m${i}`, species: 'Smeargle', item: '', ability: 'owntempo', moves: ids.slice(i * 4, i * 4 + 4),
		nature: 'Hardy', evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
		ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50, gender: 'M',
	})).filter(s => s.moves.length);
	for (let i = 0; i < moveIds.length; i += 24) {
		const ids = moveIds.slice(i, i + 24);
		const battle = new Battle({ formatid: FORMAT, seed: 'sodium,' + '0'.repeat(32) });
		battle.setPlayer('p1', { name: 'a', team: mkTeam(ids) });
		battle.setPlayer('p2', { name: 'b', team: mkTeam(['tackle']) });
		for (const p of battle.sides[0].pokemon) for (const ms of p.baseMoveSlots) out[ms.id] = ms.maxpp;
	}
	return out;
}
function exportMoves() {
	const all = dex.moves.all().filter(m => m.exists);
	const ps = psMaxPP(all.filter(m => !m.isZ && !m.isMax && m.id !== 'struggle').map(m => m.id));
	const mismatches = [];
	const moves = sortedMap(all, m => {
		const e = entity(m);
		e.champions_maxpp = champsMaxPP(m);
		// PS iterates boosts dicts in key order (battle.boost: for (k in boost)) and logs in that order; keys are
		// sorted above, so record the original orders explicitly.
		const ko = b => (b ? Object.keys(b) : []);
		e.boost_key_order = { boosts: ko(m.boosts), self: ko(m.self && m.self.boosts),
			selfBoost: ko(m.selfBoost && m.selfBoost.boosts),
			secondaries: (m.secondaries || []).map(x => ({ boosts: ko(x.boosts), self: ko(x.self && x.self.boosts) })) };
		if (ps[m.id] !== undefined) {
			e.ps_maxpp = ps[m.id];
			if (ps[m.id] !== e.champions_maxpp) mismatches.push([m.id, m.pp, m.noPPBoosts, ps[m.id], e.champions_maxpp]);
		}
		return e;
	});
	if (mismatches.length) {
		console.error(`PP formula mismatches (id, pp, noPPBoosts, ps, formula): ${JSON.stringify(mismatches.slice(0, 20))}`);
		throw new Error(`${mismatches.length} Champions PP mismatches`);
	}
	return moves;
}

// ------------------------------------------------------------------ conditions
function exportConditions() {
	const out = {};
	const add = (id, source) => {
		const c = dex.conditions.get(id);
		if (!c.exists) return;
		const e = entity(c);
		e.conditionSource = source;
		out[id] = e;
	};
	for (const id of Object.keys(dex.data.Conditions).sort()) add(id, 'conditions');
	for (const [kind, list] of [['move', dex.moves.all()], ['ability', dex.abilities.all()], ['item', dex.items.all()]]) {
		for (const x of list) if (x.exists && x.condition && !out[x.id]) add(x.id, kind);
	}
	return Object.fromEntries(Object.keys(out).sort().map(k => [k, out[k]]));
}

// ------------------------------------------------------------------ handler order metadata
/**
 * For every effect x callback, PS resolvePriority's (order, priority, subOrder). Conditions get one subOrder per
 * holder kind (pokemon / slot / side / field) because PS derives the default from the holder (sim/battle.ts
 * resolvePriority). order=false is exported as null.
 */
function exportHandlerOrder(conditions) {
	const battle = new Battle({ formatid: FORMAT, seed: 'sodium,' + '0'.repeat(32) });
	const team = [{ name: 'x', species: 'Smeargle', moves: ['tackle'], ability: 'owntempo', level: 50, gender: 'M',
		evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } }];
	battle.setPlayer('p1', { name: 'a', team }); battle.setPlayer('p2', { name: 'b', team });
	const side = battle.sides[0]; const field = battle.field;
	if (!(side instanceof Side) || !(field instanceof Field)) throw new Error('Side/Field class mismatch');
	const holders = {
		pokemon: { target: null }, slot: { target: side, isSlotCondition: true }, side: { target: side }, field: { target: field },
	};
	const resolve = (effect, cb, state) => {
		const h = { effect, callback: () => {}, state: { ...state }, end: null, effectHolder: null };
		battle.resolvePriority(h, cb);
		return { order: h.order === false ? null : h.order, priority: h.priority, subOrder: h.subOrder };
	};
	// callbacks = function-valued on* keys, plus callbacks named only by their Order/Priority/SubOrder fields
	// (duration-only effects such as Tailwind take part in fieldEvent('Residual') without a callback)
	const callbacks = obj => {
		const s = new Set(Object.keys(obj).filter(k => /^on[A-Z]/.test(k) && typeof obj[k] === 'function'));
		for (const k of Object.keys(obj)) {
			const m = /^(on[A-Z]\w*?)(Order|Priority|SubOrder)$/.exec(k);
			if (m && typeof obj[k] === 'number') s.add(m[1]);
		}
		return [...s].sort();
	};
	const out = { move: {}, ability: {}, item: {}, condition: {} };
	for (const [kind, list] of [['move', dex.moves.all()], ['ability', dex.abilities.all()], ['item', dex.items.all()]]) {
		for (const e of list.slice().sort((a, b) => (a.id < b.id ? -1 : 1))) {
			if (!e.exists) continue;
			const cbs = callbacks(e);
			if (!cbs.length) continue;
			out[kind][e.id] = Object.fromEntries(cbs.map(cb => [cb, resolve(e, cb, holders.pokemon)]));
		}
	}
	for (const id of Object.keys(conditions)) {
		const c = dex.conditions.get(id);
		const cbs = callbacks(c);
		if (!cbs.length) continue;
		out.condition[id] = Object.fromEntries(cbs.map(cb => {
			const r = {};
			for (const [hk, st] of Object.entries(holders)) r[hk] = resolve(c, cb, st);
			const base = r.pokemon;
			return [cb, { order: base.order, priority: base.priority,
				subOrder: Object.fromEntries(Object.entries(r).map(([k, v]) => [k, v.subOrder])) }];
		}));
	}
	return out;
}

// ------------------------------------------------------------------ type chart / natures / format
const IMMUNITY_KEYS = ['brn', 'par', 'psn', 'tox', 'frz', 'sandstorm', 'hail', 'powder', 'prankster', 'trapped'];
function exportTypechart() {
	const types = dex.types.all().filter(t => t.exists).slice().sort((a, b) => (a.id < b.id ? -1 : 1));
	return {
		immunityKeys: IMMUNITY_KEYS,
		types: Object.fromEntries(types.map(t => [t.id, { name: t.name, isNonstandard: t.isNonstandard ?? null,
			damageTaken: Object.fromEntries(Object.keys(t.damageTaken).sort().map(k => [k, t.damageTaken[k]])) }])),
	};
}
function exportNatures() {
	return sortedMap(dex.natures.all(), n => ({ name: n.name, plus: n.plus || null, minus: n.minus || null }));
}
function exportFormat() {
	const f = Dex.formats.get(FORMAT);
	if (!f.exists) throw new Error(`unknown format ${FORMAT}`);
	const rt = Dex.formats.getRuleTable(f);
	return {
		id: f.id, name: f.name, mod: f.mod, gameType: f.gameType,
		rules: [...rt.keys()].sort(),
		values: Object.fromEntries([...rt.valueRules.entries()].sort()),
		pickedTeamSize: rt.pickedTeamSize ?? null, minTeamSize: rt.minTeamSize, maxTeamSize: rt.maxTeamSize,
		evLimit: rt.evLimit ?? null, adjustLevel: rt.adjustLevel ?? null, defaultLevel: rt.defaultLevel,
		maxLevel: rt.maxLevel, maxMoveCount: rt.maxMoveCount,
	};
}

function main() {
	fs.mkdirSync(OUT, { recursive: true });
	let psCommit = 'unknown';
	try { psCommit = execFileSync('git', ['-C', PS_ROOT, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(); } catch {}
	const { species, megas } = exportSpecies();
	const moves = exportMoves();
	const abilities = sortedMap(dex.abilities.all().filter(a => a.exists), entity);
	const items = sortedMap(dex.items.all().filter(i => i.exists), entity);
	const conditions = exportConditions();
	write('species', species);
	write('megas', megas);
	write('moves', moves);
	write('abilities', abilities);
	write('items', items);
	write('conditions', conditions);
	write('typechart', exportTypechart());
	write('natures', exportNatures());
	write('formats', exportFormat());
	write('handler_order', exportHandlerOrder(conditions));
	write('meta', { ps_commit: psCommit, mod: MOD, format: FORMAT, exporter: 'tools/datagen/export_dex.js',
		counts: { species: Object.keys(species).length, moves: Object.keys(moves).length,
			abilities: Object.keys(abilities).length, items: Object.keys(items).length,
			conditions: Object.keys(conditions).length } });
	console.error(`export_dex: wrote ${OUT} (ps ${psCommit.slice(0, 7)})`);
}
main();

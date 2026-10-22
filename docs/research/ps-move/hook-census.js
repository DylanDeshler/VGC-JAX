'use strict';
/**
 * Hook census: for a set of effects (moves, abilities, items, plus the conditions they create),
 * count how many define a handler for each PS event (split by prefix: on / onAlly / onFoe / onAny / onSource,
 * and by holder kind). This is the METHOD for sizing a static hook table; plug in any dex subset.
 *
 * Usage: node hook-census.js [movesJson abilitiesJson itemsJson]
 *   default input = the full Reg M-C legal pool computed by the dex agent (research/champions/legal_*.json),
 *   which gives UPPER BOUNDS. Pass usage-truncated lists for realistic numbers.
 */
const path = require('path');
const fs = require('fs');
const H = require('./harness');
const dex = H.Dex.mod('champions');
const toID = H.Dex.toID;

const champDir = path.join(__dirname, '..', 'champions');
function loadNames(file, key) {
	const x = JSON.parse(fs.readFileSync(file, 'utf8'));
	const v = x[key];
	if (Array.isArray(v)) return v.map(e => (typeof e === 'string' ? e : e.name));
	return Object.keys(v);
}
const moves = (process.argv[2] ? JSON.parse(fs.readFileSync(process.argv[2])) : loadNames(path.join(champDir, 'legal_moves.json'), 'movesReachable')).map(toID);
const abilities = (process.argv[3] ? JSON.parse(fs.readFileSync(process.argv[3])) : loadNames(path.join(champDir, 'legal_abilities.json'), 'abilities')).map(toID);
const items = (process.argv[4] ? JSON.parse(fs.readFileSync(process.argv[4])) : loadNames(path.join(champDir, 'legal_items.json'), 'items')).map(toID);

const PREFIXES = ['Ally', 'Foe', 'Any', 'Source', 'Side', 'Field'];
const META = /(Priority|Order|SubOrder)$/;
const census = new Map(); // event -> {prefix -> Set(effect)}
function add(effectKind, id, obj) {
	for (const k of Object.keys(obj)) {
		if (!k.startsWith('on') || META.test(k) || obj[k] === undefined) continue;
		let ev = k.slice(2), prefix = '';
		for (const p of PREFIXES) if (ev.startsWith(p) && ev.length > p.length && /[A-Z]/.test(ev[p.length])) { prefix = p; ev = ev.slice(p.length); break; }
		if (!census.has(ev)) census.set(ev, new Map());
		const m = census.get(ev);
		const key = prefix || '(self)';
		if (!m.has(key)) m.set(key, new Set());
		m.get(key).add(`${effectKind}:${id}`);
		classes.set(`${ev}|${effectKind}:${id}|${key}`, classify(obj[k]));
	}
}
/** Heuristic handler classification from source text (method demo, not exact). */
const classes = new Map();
function classify(fn) {
	if (typeof fn !== 'function') return 'constant';
	const src = fn.toString().replace(/this\.(debug|add|hint|attrLastMove)\([^;]*\);?/g, '');
	const sideEffects = /this\.(damage|heal|boost|directDamage|faint|addVolatile|removeVolatile|setStatus|trySetStatus|cureStatus|useItem|eatItem|setItem|takeItem|setAbility|field\.|queue\.|actions\.|singleEvent|runEvent|add\b)|\.(addVolatile|removeVolatile|setStatus|trySetStatus|cureStatus|useItem|eatItem|setItem|takeItem|setAbility|formeChange|switchFlag|setType|addSideCondition|removeSideCondition|boost|damage|heal)\b/.test(src);
	const mult = /chainModify|this\.modify\(/.test(src);
	if (mult && !sideEffects) return 'multiplier';
	if (!sideEffects && /return (false|null|this\.NOT_FAIL|0|true)\b/.test(src)) return 'gate';
	if (!sideEffects) return 'pure-other';
	return 'side-effecting';
}
const conditionIds = new Set();
for (const id of moves) {
	const mv = dex.moves.get(id);
	if (!mv.exists) continue;
	add('move', id, mv);
	for (const f of ['basePowerCallback', 'damageCallback', 'beforeMoveCallback', 'beforeTurnCallback', 'priorityChargeCallback']) {
		if (mv[f]) add('move', id, { ['on' + f[0].toUpperCase() + f.slice(1)]: true });
	}
	if (mv.condition) conditionIds.add(id);
	for (const k of ['volatileStatus', 'sideCondition', 'slotCondition', 'pseudoWeather', 'weather', 'terrain', 'status']) if (mv[k]) conditionIds.add(mv[k]);
	for (const s of mv.secondaries || []) { if (s.volatileStatus) conditionIds.add(s.volatileStatus); if (s.status) conditionIds.add(s.status); }
	if (mv.self && mv.self.volatileStatus) conditionIds.add(mv.self.volatileStatus);
}
for (const id of abilities) { const a = dex.abilities.get(id); if (a.exists) { add('ability', id, a); if (a.condition) conditionIds.add(id); } }
for (const id of items) { const it = dex.items.get(id); if (it.exists) { add('item', id, it); if (it.condition) conditionIds.add(id); } }
// global conditions always reachable: statuses, weathers, core volatiles
for (const id of Object.keys(dex.data.Conditions)) conditionIds.add(id);
for (const id of conditionIds) { const c = dex.conditions.get(id); if (c.exists !== false) add('cond', id, c); }

const rows = [...census.entries()].map(([ev, m]) => {
	const total = new Set([...m.values()].flatMap(s => [...s]));
	const cls = {};
	for (const [p, set] of m) for (const e of set) { const c = classes.get(`${ev}|${e}|${p}`); cls[c] = (cls[c] || 0) + 1; }
	return { ev, total: total.size, byPrefix: Object.fromEntries([...m.entries()].map(([p, s]) => [p, s.size])), classes: cls, sample: [...total].slice(0, 6) };
}).sort((a, b) => b.total - a.total);
console.log(`input: ${moves.length} moves, ${abilities.length} abilities, ${items.length} items, ${conditionIds.size} conditions`);
console.log('event'.padEnd(26), 'effects', 'by prefix');
for (const r of rows) console.log(r.ev.padEnd(26), String(r.total).padStart(5), '  ', JSON.stringify(r.byPrefix), JSON.stringify(r.classes), '  e.g.', r.sample.slice(0, 4).join(' '));
fs.writeFileSync(path.join(__dirname, 'hook-census.json'), JSON.stringify(rows, null, 1));

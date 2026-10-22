#!/usr/bin/env node
// Handler census over the legal pool: every function-valued key and data field of each
// move / ability / item / condition, as merged by Dex.mod(<mod>).
// Usage: node handler_census.js <ps_root> <mod> <legal_pool.json> <out.json>
'use strict';
const [,, psRoot, mod, legalPath, outPath] = process.argv;
if (!outPath) {
	console.error('usage: node handler_census.js <ps_root> <mod> <legal_pool.json> <out.json>');
	process.exit(1);
}
const fs = require('fs');
const path = require('path');
const {Dex} = require(path.join(psRoot, 'dist/sim'));
const dex = Dex.mod(mod);
const legal = JSON.parse(fs.readFileSync(legalPath, 'utf8'));

// ---- field classification -------------------------------------------------------------
const META = new Set([
	'name', 'id', 'num', 'fullname', 'effectType', 'exists', 'gen', 'isNonstandard', 'kind', 'desc', 'shortDesc',
	'sourceEffect', 'duration', 'noCopy', 'affectsFainted', 'status', 'contestType', 'realMove', 'isZ', 'isMax',
	'zMove', 'maxMove', 'zMoveEffect', 'zMoveBoost', 'zMovePower', 'gmaxPower', 'baseMoveType', 'rating', 'spritenum',
	'tier', 'inherit', 'basePowerModifier', 'flingDamage',
]);
const ZERO_DEFAULTS = (v) => v === undefined || v === null || v === false || v === '' || v === 0 ||
	(Array.isArray(v) && v.length === 0) || (typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 0);

const CORE_CONDITIONS = new Set(['brn', 'par', 'psn', 'tox', 'slp', 'frz', 'confusion', 'flinch']);
// Conditions the engine core needs regardless of dex (driven by battle code, not by data refs)
const ENGINE_CONDITIONS = ['brn', 'par', 'psn', 'tox', 'slp', 'frz', 'confusion', 'flinch', 'trapped', 'partiallytrapped',
	'lockedmove', 'twoturnmove', 'choicelock', 'mustrecharge', 'futuremove', 'healreplacement', 'stall'];

function normSrc(src) {
	return src
		.replace(/\/\*[\s\S]*?\*\//g, '')
		.replace(/\/\/[^\n]*/g, '')
		.replace(/`(?:\\.|[^`\\])*`/g, 'S')
		.replace(/'(?:\\.|[^'\\])*'/g, 'S')
		.replace(/"(?:\\.|[^"\\])*"/g, 'S')
		.replace(/\b\d+(\.\d+)?\b/g, 'N')
		.replace(/\s+/g, ' ')
		.trim();
}
function locOf(src) {
	return src.split('\n').filter(l => l.trim() && !l.trim().startsWith('//')).length;
}

// Collect all known ids for reference extraction
const knownIds = {move: new Set(), ability: new Set(), item: new Set(), condition: new Set()};
for (const m of dex.moves.all()) knownIds.move.add(m.id);
for (const a of dex.abilities.all()) knownIds.ability.add(a.id);
for (const i of dex.items.all()) knownIds.item.add(i.id);
const condTable = dex.data.Conditions;
for (const c of Object.keys(condTable)) knownIds.condition.add(c);

const REF_PATTERNS = [
	// "creates" references (used for condition closure)
	[/addVolatile\(\s*['"]([a-z0-9]+)['"]/g, 'volatile'],
	[/addSideCondition\(\s*['"]([a-z0-9]+)['"]/g, 'sideCondition'],
	[/addSlotCondition\([^,]+,\s*['"]([a-zA-Z0-9]+)['"]/g, 'slotCondition'],
	[/(?:setStatus|trySetStatus)\(\s*['"]([a-z0-9]+)['"]/g, 'status'],
	[/setWeather\(\s*['"]([a-z0-9]+)['"]/g, 'weather'],
	[/setTerrain\(\s*['"]([a-z0-9]+)['"]/g, 'terrain'],
	[/addPseudoWeather\(\s*['"]([a-z0-9]+)['"]/g, 'pseudoWeather'],
	// inspection / removal references (interactions, not creation)
	[/removeVolatile\(\s*['"]([a-z0-9]+)['"]/g, 'checks_volatile'],
	[/(?:volatiles|getVolatile)\[?\(?\s*['"]([a-z0-9]+)['"]/g, 'checks_volatile'],
	[/removeSideCondition\(\s*['"]([a-z0-9]+)['"]/g, 'checks_sideCondition'],
	[/(?:getSideCondition|sideConditions)\[?\(?\s*['"]([a-z0-9]+)['"]/g, 'checks_sideCondition'],
	[/(?:removePseudoWeather|getPseudoWeather|pseudoWeather)\[?\(?\s*['"]([a-z0-9]+)['"]/g, 'checks_pseudoWeather'],
	[/(?:isWeather|weather ===?)\(?\s*\[?['"]([a-z0-9]+)['"]/g, 'checks_weather'],
	[/(?:isTerrain|terrain ===?)\(?\s*\[?['"]([a-z0-9]+)['"]/g, 'checks_terrain'],
	[/hasAbility\(\s*\[?([^)\]]*)\]?\)/g, 'abilityList'],
	[/hasItem\(\s*\[?([^)\]]*)\]?\)/g, 'itemList'],
	[/useMove\(\s*['"]([a-z0-9]+)['"]/g, 'move'],
];

function extractRefs(src) {
	const refs = {};
	const add = (k, v) => { (refs[k] = refs[k] || new Set()).add(v); };
	for (const [re, kind] of REF_PATTERNS) {
		re.lastIndex = 0;
		let m;
		while ((m = re.exec(src))) {
			if (kind === 'abilityList' || kind === 'itemList') {
				const ids = [...m[1].matchAll(/['"]([a-z0-9]+)['"]/g)].map(x => x[1]);
				for (const id of ids) add(kind === 'abilityList' ? 'ability' : 'item', id);
			} else {
				add(kind, kind.startsWith('slot') ? m[1].toLowerCase() : m[1]);
			}
		}
	}
	// any quoted id that matches a known entity (loose interaction signal)
	for (const m of src.matchAll(/['"]([a-z0-9]{3,})['"]/g)) {
		const id = m[1];
		for (const t of ['move', 'ability', 'item', 'condition']) if (knownIds[t].has(id)) add('mentions_' + t, id);
	}
	return Object.fromEntries(Object.entries(refs).map(([k, v]) => [k, [...v].sort()]));
}

// Walk an effect object; collect functions and data fields with dotted paths.
function walk(obj, prefix, fns, data, seenFns, depth = 0) {
	if (!obj || depth > 4) return;
	for (const key of Object.keys(obj)) {
		const v = obj[key];
		const p = prefix ? `${prefix}.${key}` : key;
		if (typeof v === 'function') {
			if (seenFns.has(v)) continue;
			seenFns.add(v);
			const src = v.toString();
			fns.push({path: p, hook: key, loc: locOf(src), norm: normSrc(src), src});
		} else if (key === 'condition' && prefix === '') {
			// owned condition handled separately
			continue;
		} else if (v && typeof v === 'object' && !Array.isArray(v) &&
			['secondary', 'self', 'selfBoost', 'secondaries'].some(k => p.startsWith(k))) {
			walk(v, p, fns, data, seenFns, depth + 1);
			if (!META.has(key)) data[p] = summarize(v);
		} else if (Array.isArray(v) && key === 'secondaries') {
			v.forEach((s, i) => walk(s, `${p}[${i}]`, fns, data, seenFns, depth + 1));
			data[p] = v.map(summarize);
		} else if (!META.has(key) && !ZERO_DEFAULTS(v)) {
			if (prefix === '' || !prefix.startsWith('secondar')) data[p] = summarize(v);
		}
	}
}
function summarize(v) {
	if (v && typeof v === 'object') {
		const o = Array.isArray(v) ? [] : {};
		for (const k of Object.keys(v)) {
			const x = v[k];
			if (typeof x === 'function') o[k] = '<fn>';
			else if (x && typeof x === 'object') o[k] = summarize(x);
			else o[k] = x;
		}
		return o;
	}
	return v;
}

// Default field values of a fresh DataMove/Ability/Item to drop (class defaults).
function defaultsFor(kind) {
	const probe = {move: dex.moves.get('tackle'), ability: dex.abilities.get('noability'), item: dex.items.get('')}[kind];
	return probe;
}

const MOVE_DEFAULT_SKIP = new Set(['basePower', 'accuracy', 'pp', 'category', 'type', 'target', 'priority', 'flags',
	'critRatio', 'secondaries', 'hasSheerForce', 'ignoreImmunity', 'noPPBoosts', 'baseMoveType', 'nonGhostTarget']);

function recordFor(kind, obj, ownedCond) {
	const fns = [], data = {}, seen = new Set();
	walk(obj, '', fns, data, seen);
	const rec = {kind, id: obj.id, name: obj.name, fns: [], data, owned_condition: null};
	rec.fns = fns.map(f => ({path: f.path, hook: f.hook, loc: f.loc, norm: f.norm}));
	let allSrc = fns.map(f => f.src).join('\n');
	if (ownedCond) {
		const cf = [], cd = {};
		walk(ownedCond, '', cf, cd, new Set());
		rec.owned_condition = {fns: cf.map(f => ({path: 'condition.' + f.path, hook: f.hook, loc: f.loc, norm: f.norm})), data: cd};
		allSrc += '\n' + cf.map(f => f.src).join('\n');
	}
	// data-declared references
	const refs = extractRefs(allSrc);
	const dref = {};
	const addd = (k, v) => { if (v && typeof v === 'string') (dref[k] = dref[k] || new Set()).add(v.toLowerCase().replace(/[^a-z0-9]/g, '')); };
	addd('volatile', obj.volatileStatus);
	addd('sideCondition', obj.sideCondition);
	addd('slotCondition', obj.slotCondition);
	addd('pseudoWeather', obj.pseudoWeather);
	addd('weather', obj.weather);
	addd('terrain', obj.terrain);
	addd('status', obj.status);
	for (const s of (obj.secondaries || [])) {
		addd('status', s.status); addd('volatile', s.volatileStatus);
		if (s.self) addd('volatile', s.self.volatileStatus);
	}
	if (obj.self) { addd('volatile', obj.self.volatileStatus); addd('sideCondition', obj.self.sideCondition); addd('pseudoWeather', obj.self.pseudoWeather); }
	rec.data_refs = Object.fromEntries(Object.entries(dref).map(([k, v]) => [k, [...v]]));
	rec.code_refs = refs;
	return rec;
}

function condId(name) { return dex.conditions.get(name).id; }

const records = {move: {}, ability: {}, item: {}, condition: {}};
for (const id of Object.keys(legal.moves)) {
	const m = dex.moves.get(id);
	records.move[id] = recordFor('move', m, m.condition);
	// drop default-valued fields commonly present
	for (const k of Object.keys(records.move[id].data)) {
		if (k === 'flags' || k === 'secondaries') continue;
	}
}
for (const id of Object.keys(legal.abilities)) {
	const a = dex.abilities.get(id);
	records.ability[id] = recordFor('ability', a, a.condition);
}
for (const id of Object.keys(legal.items)) {
	const it = dex.items.get(id);
	records.item[id] = recordFor('item', it, it.condition);
}

// Conditions: base Conditions table entries + owned conditions referenced by legal entities.
const condRefs = {};  // condition id -> set of referencing entities
function noteCond(cid, who) { if (!cid) return; cid = String(cid).toLowerCase().replace(/[^a-z0-9]/g, ''); (condRefs[cid] = condRefs[cid] || new Set()).add(who); }
for (const t of ['move', 'ability', 'item']) {
	for (const [id, r] of Object.entries(records[t])) {
		for (const k of ['volatile', 'sideCondition', 'slotCondition', 'pseudoWeather', 'weather', 'terrain', 'status']) {
			for (const c of (r.data_refs[k] || [])) noteCond(c, `${t}:${id}`);
			for (const c of (r.code_refs[k] || [])) noteCond(c, `${t}:${id}`);
		}
		if (r.owned_condition && r.owned_condition.fns.length) noteCond(id, `${t}:${id}#owner`);
	}
}
for (const c of ENGINE_CONDITIONS) noteCond(c, 'engine');
for (const cid of Object.keys(condRefs)) {
	let c;
	try { c = dex.conditions.get(cid); } catch (e) { continue; }
	if (!c || !c.exists) { records.condition[cid] = {kind: 'condition', id: cid, missing: true, fns: [], data: {}, referenced_by: [...condRefs[cid]]}; continue; }
	const fns = [], data = {};
	walk(c, '', fns, data, new Set());
	const src = fns.map(f => f.src).join('\n');
	records.condition[cid] = {
		kind: 'condition', id: cid, name: c.name, source: condTable[cid] ? 'conditions.ts' : (c.effectType || 'owned'),
		owner: condTable[cid] ? null : (dex.data.Moves[cid] ? 'move' : dex.data.Abilities[cid] ? 'ability' : dex.data.Items[cid] ? 'item' : null),
		fns: fns.map(f => ({path: f.path, hook: f.hook, loc: f.loc, norm: f.norm})), data,
		code_refs: extractRefs(src), referenced_by: [...condRefs[cid]].sort(),
	};
}

// ---- core-code references (special cases in sim/*.ts and the mod's scripts.ts) ----------
const coreFiles = ['sim/battle.ts', 'sim/battle-actions.ts', 'sim/pokemon.ts', 'sim/side.ts', 'sim/field.ts',
	'sim/battle-queue.ts', 'sim/dex-moves.ts', 'sim/dex-items.ts', 'sim/dex-abilities.ts', 'sim/dex-conditions.ts',
	`data/mods/${mod}/scripts.ts`, 'data/scripts.ts'].filter(f => fs.existsSync(path.join(psRoot, f)));
const coreHits = {};
for (const f of coreFiles) {
	const src = fs.readFileSync(path.join(psRoot, f), 'utf8');
	for (const m of src.matchAll(/['"]([a-z0-9]{3,})['"]/g)) {
		const id = m[1];
		(coreHits[id] = coreHits[id] || {});
		coreHits[id][f] = (coreHits[id][f] || 0) + 1;
	}
}
for (const t of ['move', 'ability', 'item', 'condition']) {
	for (const [id, r] of Object.entries(records[t])) r.core_refs = coreHits[id] || {};
}

// ---- templates: normalized (hook, source) shared across legal entities ----------------
const tmplCount = {};
const tmplKey = (f) => f.hook + '::' + f.norm;
for (const t of ['move', 'ability', 'item', 'condition']) {
	for (const r of Object.values(records[t])) {
		const keys = new Set([...r.fns, ...((r.owned_condition && r.owned_condition.fns) || [])].map(tmplKey));
		for (const k of keys) tmplCount[k] = (tmplCount[k] || 0) + 1;
	}
}
const tmplIds = {};
let tmplNext = 0;
for (const k of Object.keys(tmplCount)) tmplIds[k] = tmplNext++;
for (const t of ['move', 'ability', 'item', 'condition']) {
	for (const r of Object.values(records[t])) {
		for (const f of [...r.fns, ...((r.owned_condition && r.owned_condition.fns) || [])]) {
			const k = tmplKey(f);
			f.template = tmplIds[k];
			f.template_shared_by = tmplCount[k];
			delete f.norm;
		}
	}
}

// ---- classification ----------------------------------------------------------------
const TEMPLATE_MIN = 3;
function classify(r) {
	const selfFns = r.fns;
	const bespoke = selfFns.filter(f => f.template_shared_by < TEMPLATE_MIN);
	const ownedFns = (r.owned_condition && r.owned_condition.fns) || [];
	const refsConds = new Set();
	for (const k of ['volatile', 'sideCondition', 'slotCondition', 'pseudoWeather', 'weather', 'terrain', 'status']) {
		for (const c of (r.data_refs && r.data_refs[k]) || []) refsConds.add(c);
	}
	if (r.kind === 'condition') {
		if (!selfFns.length) return 'DATA_ONLY';
		return bespoke.length ? 'CUSTOM' : 'DATA_PLUS_COMMON';
	}
	if (bespoke.length) return 'CUSTOM';
	const nonCore = [...refsConds].filter(c => !CORE_CONDITIONS.has(c));
	if (!selfFns.length && !ownedFns.length && !nonCore.length) return 'DATA_ONLY';
	return 'DATA_PLUS_COMMON';
}
for (const t of ['move', 'ability', 'item', 'condition']) {
	for (const r of Object.values(records[t])) {
		r.class = classify(r);
		const all = [...r.fns, ...((r.owned_condition && r.owned_condition.fns) || [])];
		r.n_hooks = r.fns.length;
		r.n_condition_hooks = (r.owned_condition && r.owned_condition.fns.length) || 0;
		r.loc = all.reduce((a, f) => a + f.loc, 0);
		r.hooks = [...new Set(r.fns.map(f => f.path))];
		r.condition_hooks = [...new Set(((r.owned_condition && r.owned_condition.fns) || []).map(f => f.hook))];
	}
}

const out = {
	ps_root: psRoot, mod, generated: new Date().toISOString(), template_min: TEMPLATE_MIN,
	core_files_scanned: coreFiles, engine_conditions: ENGINE_CONDITIONS, core_conditions: [...CORE_CONDITIONS],
	n_templates: tmplNext, records,
};
fs.writeFileSync(outPath, JSON.stringify(out));
const summary = {};
for (const t of Object.keys(records)) {
	summary[t] = {n: Object.keys(records[t]).length};
	for (const r of Object.values(records[t])) summary[t][r.class] = (summary[t][r.class] || 0) + 1;
}
console.log(JSON.stringify(summary));

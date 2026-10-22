#!/usr/bin/env node
'use strict';
/**
 * Scan the PS TypeScript sources for PRNG call sites and emit a skeleton
 * (file, line, code, enclosing effect/function, method, args, reachability in
 * Reg M-C). Used (a) to build rng_callsites.json and (b) as a drift check:
 * `node tools/scan_callsites.js --check` fails if the PS source has call sites
 * that are not in rng_callsites.json (or vice versa).
 *
 * Usage:
 *   node tools/scan_callsites.js            > /tmp/skeleton.json
 *   node tools/scan_callsites.js --check    (compare against ../rng_callsites.json)
 */
const fs = require('fs');
const path = require('path');
const { PS_ROOT } = require('../lib/ps');

const FILES = [
	'sim/battle.ts', 'sim/battle-actions.ts', 'sim/battle-queue.ts', 'sim/pokemon.ts', 'sim/side.ts', 'sim/field.ts',
	'data/abilities.ts', 'data/items.ts', 'data/moves.ts', 'data/conditions.ts', 'data/rulesets.ts',
	'data/mods/champions/abilities.ts', 'data/mods/champions/conditions.ts', 'data/mods/champions/items.ts',
	'data/mods/champions/moves.ts', 'data/mods/champions/scripts.ts', 'data/mods/champions/rulesets.ts',
];
// Calls that consume PRNG. `speedSort(` only consumes when it meets a tie (via prng.shuffle).
const PAT = /\b(?:this\.battle\.|this\.|battle\.|this\.prng\.|prng\.)(random|randomChance|sample|shuffle|speedSort)\(/g;
// Definitions / wrappers to skip (they are the PRNG plumbing itself)
const SKIP_LINES = new Set([
	'sim/battle.ts:348', 'sim/battle.ts:353', 'sim/battle.ts:357',
]);

function enclosing(lines, idx, file) {
	const isData = file.startsWith('data/');
	let fn = null; let entry = null; let handler = null; let sub = null;
	if (!isData) {
		for (let i = idx; i >= 0; i--) {
			const m = /^\t(?:async |static |override |private |public )*(\w+)\s*[(<]/.exec(lines[i]);
			if (m && !['if', 'for', 'while', 'switch', 'return'].includes(m[1])) { fn = m[1]; break; }
			const m2 = /^\t\t(\w+)\s*\(.*\)\s*\{\s*$/.exec(lines[i]); // scripts.ts: actions: { foo() {
			if (file.includes('scripts.ts') && m2 && !['if', 'for', 'while', 'switch'].includes(m2[1])) { fn = m2[1]; break; }
		}
		if (file.includes('scripts.ts')) {
			for (let i = idx; i >= 0; i--) {
				const m = /^\t(\w+): \{/.exec(lines[i]);
				if (m) { entry = m[1]; break; }
			}
		}
		return { fn, entry };
	}
	// data file: entry is `\t<id>: {`, handler is the nearest deeper `name(...) {` or `name: {`
	let entryLine = -1;
	for (let i = idx; i >= 0; i--) {
		const m = /^\t(\w+): \{/.exec(lines[i]);
		if (m) { entry = m[1]; entryLine = i; break; }
	}
	for (let i = idx; i > entryLine; i--) {
		const m = /^(\t{2,})(\w+)\s*\(.*\)\s*\{\s*$/.exec(lines[i]);
		if (m && !['if', 'for', 'while', 'switch'].includes(m[2])) {
			handler = m[2];
			// is it inside a nested object (condition / secondary / self)?
			for (let j = i - 1; j > entryLine; j--) {
				const n = /^(\t{2,})(\w+): \{/.exec(lines[j]);
				if (n && n[1].length < m[1].length) { sub = n[2]; break; }
				if (/^\t\t\w/.test(lines[j]) && m[1].length === 2) break;
			}
			if (m[1].length === 2) sub = null;
			break;
		}
	}
	return { entry, handler, sub };
}

function scan() {
	const out = [];
	for (const file of FILES) {
		const abs = path.join(PS_ROOT, file);
		if (!fs.existsSync(abs)) continue;
		const lines = fs.readFileSync(abs, 'utf8').split('\n');
		lines.forEach((line, i) => {
			if (/^\s*(\/\/|\*)/.test(line)) return;
			PAT.lastIndex = 0;
			let m;
			while ((m = PAT.exec(line))) {
				const key = `${file}:${i + 1}`;
				if (SKIP_LINES.has(key)) continue;
				// args: balanced parens from the match end
				let depth = 1; let j = m.index + m[0].length; const start = j;
				while (j < line.length && depth > 0) {
					if (line[j] === '(') depth++;
					else if (line[j] === ')') depth--;
					j++;
				}
				const args = line.slice(start, j - 1);
				const enc = enclosing(lines, i, file);
				out.push({ id: key, file, line: i + 1, method: m[1], args, code: line.trim(), ...enc });
			}
		});
	}
	return out;
}

function reachability(sites) {
	const { Dex, TeamValidator } = require('../lib/ps');
	const dex = Dex.mod('champions');
	const validator = TeamValidator.get('gen9championsvgc2026regmc');
	const legalSpecies = dex.species.all().filter(s => {
		if (s.isNonstandard || s.tier === 'Illegal') return false;
		const set = { species: s.name, ability: s.abilities[0], moves: [], item: s.requiredItem || '', evs: {}, level: 50 };
		const problems = validator.checkSpecies(set, s, s, {});
		return !problems;
	});
	const learnCache = new Map();
	const anyLearns = id => {
		if (learnCache.has(id)) return learnCache.get(id);
		const move = dex.moves.get(id);
		let r = false;
		if (move.exists && !move.isNonstandard) {
			for (const s of legalSpecies) {
				if (!validator.checkCanLearn(move, s, validator.allSources(s), { species: s.name })) { r = true; break; }
			}
		}
		learnCache.set(id, r);
		return r;
	};
	const anyAbility = id => legalSpecies.some(s => Object.values(s.abilities).some(a => dex.toID(a) === id));
	for (const s of sites) {
		if (!s.file.startsWith('data/') || s.file.endsWith('scripts.ts')) { s.reachable = null; continue; }
		const kind = s.file.includes('abilities') ? 'ability' : s.file.includes('items') ? 'item' :
			s.file.includes('moves') ? 'move' : s.file.includes('conditions') ? 'condition' : 'rule';
		s.effectType = kind;
		if (kind === 'move') {
			const mv = dex.moves.get(s.entry);
			s.reachable = mv.exists && !mv.isNonstandard ? anyLearns(s.entry) : false;
			s.availability = mv.exists ? (mv.isNonstandard || 'available') : 'missing';
		} else if (kind === 'ability') {
			const ab = dex.abilities.get(s.entry);
			s.reachable = ab.exists && !ab.isNonstandard && anyAbility(s.entry);
			s.availability = ab.exists ? (ab.isNonstandard || 'available') : 'missing';
		} else if (kind === 'item') {
			const it = dex.items.get(s.entry);
			s.reachable = it.exists && !it.isNonstandard;
			s.availability = it.exists ? (it.isNonstandard || 'available') : 'missing';
		} else {
			s.reachable = null; // conditions/rules: decided manually
		}
	}
	return { sites, legalSpeciesCount: legalSpecies.length };
}

if (require.main === module) {
	const sites = scan();
	if (process.argv.includes('--check')) {
		const inv = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'rng_callsites.json'), 'utf8'));
		const known = new Map(inv.callsites.map(c => [c.id, c]));
		let bad = 0;
		for (const s of sites) {
			const k = known.get(s.id);
			if (!k) { console.log(`NEW/MOVED call site not in inventory: ${s.id}  ${s.code}`); bad++; continue; }
			if (k.code && k.code !== s.code) { console.log(`CHANGED code at ${s.id}:\n  inv: ${k.code}\n  now: ${s.code}`); bad++; }
		}
		const now = new Set(sites.map(s => s.id));
		for (const c of inv.callsites) if (!now.has(c.id) && !c.virtual) { console.log(`STALE inventory entry: ${c.id}`); bad++; }
		console.log(bad ? `${bad} drift problems` : `inventory matches PS source (${sites.length} call sites)`);
		process.exit(bad ? 1 : 0);
	}
	const r = reachability(sites);
	console.log(JSON.stringify(r, null, 1));
}
module.exports = { scan, reachability };

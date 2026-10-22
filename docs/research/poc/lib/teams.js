'use strict';
/**
 * Legal Reg M-C team sources.
 *  - randomDoubles(prng):   PS's own Champions random-doubles generator, fixed up (level 50, Item Clause).
 *  - chaosTeam(rng):        sets sampled from Smogon chaos usage stats (species/ability/item/spread/moves).
 * Both are validated with TeamValidator('gen9championsvgc2026regmc').
 */
const fs = require('fs');
const { Dex, Teams, TeamValidator, PRNG, FORMAT } = require('./ps');

const dex = Dex.forFormat(FORMAT);
let _validator = null;
const validator = () => (_validator ||= TeamValidator.get(FORMAT));

/** Tiny seeded RNG for harness-side randomness (never touches battle.prng). */
class HarnessRNG {
	constructor(seed = 1) { this.s = (seed >>> 0) || 1; }
	next() { // xorshift32
		let x = this.s; x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; this.s = x; return x / 4294967296;
	}
	int(n) { return Math.floor(this.next() * n); }
	pick(arr) { return arr[this.int(arr.length)]; }
	chance(p) { return this.next() < p; }
	weighted(entries) { // [[value, weight], ...]
		let tot = 0; for (const e of entries) tot += e[1];
		let r = this.next() * tot;
		for (const e of entries) { r -= e[1]; if (r <= 0) return e[0]; }
		return entries[entries.length - 1][0];
	}
	shuffle(arr) { for (let i = arr.length - 1; i > 0; i--) { const j = this.int(i + 1); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; }
}

// ---------------------------------------------------------------- random doubles
let _gen = null;
function randomDoubles(seed) {
	if (!_gen) _gen = Teams.getGenerator('gen9championsrandomdoublesbattle', seed ?? PRNG.generateSeed());
	else if (seed) _gen.setSeed(seed);
	const team = _gen.getTeam();
	const used = new Set();
	for (const s of team) {
		s.level = 50; // Adjust Level = 50 anyway; generator emits 50-60 levels
		if (s.item && used.has(s.item)) s.item = '';
		used.add(s.item);
	}
	return team;
}

// ---------------------------------------------------------------- chaos stats
const STATS_PATH = process.env.CHAOS_STATS ||
	['/tmp/stats_dl', '/tmp/stats_dl_rng'].map(d => `${d}/gen9championsvgc2026regmcbo3-1760.json`).find(f => fs.existsSync(f)) ||
	'/tmp/stats_dl/gen9championsvgc2026regmcbo3-1760.json'; // https://www.smogon.com/stats/2026-09/chaos/
const STAT_IDS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
let _chaos = null;
function loadChaos() {
	if (_chaos) return _chaos;
	const raw = JSON.parse(fs.readFileSync(STATS_PATH, 'utf8')).data;
	const entries = [];
	for (const [name, d] of Object.entries(raw)) {
		const sp = dex.species.get(name);
		if (!sp.exists) continue;
		const isMega = !!sp.isMega || /-Mega/.test(sp.name);
		const base = isMega ? dex.species.get(sp.changesFrom || sp.baseSpecies) : sp;
		const top = (obj, n = 99) => Object.entries(obj).filter(([k, v]) => k && k !== 'nothing' && v > 0)
			.sort((a, b) => b[1] - a[1]).slice(0, n);
		// for megas the listed ability is the Mega ability; pull the base forme's ability distribution
		let abilities = top(d.Abilities);
		if (isMega) abilities = raw[base.name] ? top(raw[base.name].Abilities) : [[dex.toID(base.abilities[0]), 1]];
		entries.push({
			name, species: base.name, baseSpecies: base.baseSpecies, usage: d.usage,
			megaItem: isMega ? dex.toID(sp.requiredItem) : null,
			abilities, items: top(d.Items, 20), spreads: top(d.Spreads, 30), moves: top(d.Moves, 20),
		});
	}
	_chaos = entries.filter(e => e.usage > 0 && e.moves.length >= 4);
	return _chaos;
}

function chaosSet(e, rng, usedItems) {
	const ability = e.abilities.length ? rng.weighted(e.abilities) : dex.toID(dex.species.get(e.species).abilities[0]);
	let item = e.megaItem;
	if (!item) {
		const items = e.items.filter(([it]) => !usedItems.has(it));
		item = items.length ? rng.weighted(items) : '';
	}
	const spread = e.spreads.length ? rng.weighted(e.spreads) : 'Serious:0/0/0/0/0/0';
	const [nature, sps] = spread.split(':');
	const evs = {}; sps.split('/').forEach((v, i) => { evs[STAT_IDS[i]] = Number(v); });
	const moves = []; const pool = e.moves.slice();
	while (moves.length < 4 && pool.length) {
		const m = rng.weighted(pool);
		moves.push(m);
		pool.splice(pool.findIndex(p => p[0] === m), 1);
	}
	// Always fix gender: otherwise PS draws it from battle.prng at construction (GENDER draws, sim/pokemon.ts:361)
	const sp = dex.species.get(e.species);
	const gender = sp.gender || (rng.next() < (sp.genderRatio ? sp.genderRatio.M : 0.5) ? 'M' : 'F');
	return {
		name: '', species: e.species, item, ability, moves, nature, evs, gender, level: 50,
		ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
	};
}

/** Returns {team, attempts}. Usage-weighted species, Species/Item clause, validated. */
function chaosTeam(rng, { maxAttempts = 50 } = {}) {
	const pool = loadChaos();
	for (let attempt = 1; attempt <= maxAttempts; attempt++) {
		const usedBase = new Set(); const usedItems = new Set(); const team = [];
		let guard = 0;
		while (team.length < 6 && guard++ < 200) {
			const e = rng.weighted(pool.map(p => [p, p.usage]));
			if (usedBase.has(e.baseSpecies)) continue;
			if (e.megaItem && usedItems.has(e.megaItem)) continue;
			const set = chaosSet(e, rng, usedItems);
			usedBase.add(e.baseSpecies); if (set.item) usedItems.add(set.item);
			team.push(set);
		}
		const problems = validator().validateTeam(team);
		if (!problems) return { team, attempts: attempt };
	}
	throw new Error('chaosTeam: could not build a valid team');
}

function validate(team) { return validator().validateTeam(team); }

module.exports = { HarnessRNG, randomDoubles, chaosTeam, loadChaos, validate, dex };

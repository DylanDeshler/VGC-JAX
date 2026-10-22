'use strict';
/**
 * Pool B: usage-stats team sampler restricted to a content stage (06 §4.1; port of research/dex/scripts/team_sources.js
 * sampleTeam). Species by usage chained through chaos Teammates, abilities/items/moves restricted to the stage and the
 * species' learnset, SP spreads from chaos Spreads, Mega Stone holders get their *base* ability from the frozen
 * real-sheet corpus (dex-scope §2). Gender is always set. Every team passes TeamValidator (retry otherwise).
 */
const fs = require('fs');
const path = require('path');
const { Dex, Teams, TeamValidator, FORMAT } = require('./ps');

const ROOT = path.join(__dirname, '..', '..');
const dex = Dex.mod('champions');
const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const cache = new Map();

function megaBaseAbilities() {
	const out = {};
	const f = path.join(ROOT, 'data', 'teams', 'real_sheets_mc_v1.jsonl');
	for (const line of fs.readFileSync(f, 'utf8').split('\n')) {
		if (!line) continue;
		for (const s of Teams.unpack(JSON.parse(line).packed)) {
			const it = dex.items.get(s.item);
			if (!it.megaStone) continue;
			const forme = toID(it.megaStone[s.species] || Object.values(it.megaStone)[0]);
			const a = toID(s.ability);
			(out[forme] ||= {})[a] = ((out[forme] || {})[a] || 0) + 1;
		}
	}
	return out;
}

function info(stage, chaosPath) {
	const key = `${stage}|${chaosPath}`;
	if (cache.has(key)) return cache.get(key);
	const st = require(path.join(ROOT, 'data', 'export', 'stages', `${stage}.json`));
	const legal = require(path.join(ROOT, 'data', 'export', 'legal_pool.json'));
	const allowed = { species: new Set(st.species), move: new Set(st.moves), ability: new Set(st.abilities),
		item: new Set(st.items) };
	const chaos = JSON.parse(fs.readFileSync(chaosPath, 'utf8')).data;
	const names = Object.keys(chaos).filter(n => allowed.species.has(toID(n)) &&
		(Object.keys(chaos[n].Abilities).some(a => allowed.ability.has(a)) || legal.battleFormes[toID(n)]));
	const usage = Object.fromEntries(names.map(n => [n, Math.max(1e-6, chaos[n].usage)]));
	const v = { st, legal, allowed, chaos, names, usage, megaBase: megaBaseAbilities(), validator: TeamValidator.get(FORMAT) };
	cache.set(key, v);
	return v;
}

function pick(rng, entries) {
	const tot = entries.reduce((a, e) => a + e[1], 0);
	let r = rng.next() * tot;
	for (const e of entries) { r -= e[1]; if (r <= 0) return e[0]; }
	return entries[entries.length - 1][0];
}

function sampleOnce(I, rng) {
	const { chaos, names, usage, legal, allowed, megaBase, validator } = I;
	const team = []; const nums = new Set(); const items = new Set();
	const logs = Object.fromEntries(names.map(n => [n, Math.log(usage[n])]));
	for (let slot = 0; slot < 6; slot++) {
		const cand = names.filter(n => !nums.has(dex.species.get(n).num));
		if (!cand.length) return null;
		const mx = Math.max(...cand.map(n => logs[n]));
		const nm = pick(rng, cand.map(n => [n, Math.exp(logs[n] - mx)]));
		const e = chaos[nm]; const id = toID(nm);
		const w = Object.values(e.Abilities).reduce((a, b) => a + b, 0) || 1;
		nums.add(dex.species.get(nm).num);
		for (const n of names) {
			const cond = ((e.Teammates[n] || 0) / w) + 1e-4;
			logs[n] += Math.log(Math.min(1e3, Math.max(1e-3, cond / usage[n])));
		}
		const forme = legal.battleFormes[id];
		let species; let item; let ability;
		if (forme && forme.isMega) {
			species = legal.species[forme.from[0]].name;
			item = dex.items.get(forme.requiredItem).name;
			if (!allowed.item.has(toID(item))) return null;
			const legalAbs = legal.species[forme.from[0]].abilities.map(a => a.id).filter(a => allowed.ability.has(a));
			const dist = Object.entries(megaBase[id] || {}).filter(([a]) => legalAbs.includes(a));
			if (!dist.length && !legalAbs.length) return null;
			ability = dist.length ? pick(rng, dist) : legalAbs[0];
		} else {
			species = dex.species.get(nm).name;
			const abs = Object.entries(e.Abilities).filter(([a, val]) => allowed.ability.has(a) && val > 0);
			if (!abs.length) return null;
			ability = pick(rng, abs);
			const its = Object.entries(e.Items).filter(([i, val]) => val > 0 &&
				(i === 'nothing' || (allowed.item.has(i) && !items.has(i))) && !dex.items.get(i).megaStone);
			item = its.length ? pick(rng, its) : 'nothing';
			if (item === 'nothing') item = '';
		}
		if (item) items.add(toID(item));
		const sp = dex.species.get(species);
		const pool = Object.entries(e.Moves).filter(([m, val]) => m && val > 0 && allowed.move.has(m) &&
			!validator.checkCanLearn(dex.moves.get(m), sp, validator.allSources(sp), { name: species, level: 50 }));
		const moves = [];
		while (moves.length < 4 && pool.length) {
			const m = pick(rng, pool);
			moves.push(dex.moves.get(m).name);
			pool.splice(pool.findIndex(x => x[0] === m), 1);
		}
		if (!moves.length) return null;
		const spreads = Object.entries(e.Spreads).sort((a, b) => b[1] - a[1]).slice(0, 200);
		const [nature, sps] = (spreads.length ? pick(rng, spreads) : 'Serious:0/0/0/0/0/0').split(':');
		const vv = sps.split('/').map(Number);
		const gender = sp.gender || (sp.genderRatio && sp.genderRatio.M === 0 ? 'F' : (rng.next() < 0.5 ? 'M' : 'F'));
		team.push({ name: species, species, item, ability: dex.abilities.get(ability).name, moves, nature, gender,
			evs: { hp: vv[0], atk: vv[1], def: vv[2], spa: vv[3], spd: vv[4], spe: vv[5] },
			ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50 });
	}
	return team;
}

/** A validated stage-restricted usage-sampled team. `rng` is a HarnessRNG. */
function poolBTeam(stage, rng, { chaos = process.env.CHAOS_STATS ||
	path.join(ROOT, 'data/raw/smogon-chaos/2026-09/gen9championsvgc2026regmcbo3-1760.json'), maxAttempts = 200 } = {}) {
	const I = info(stage, chaos);
	for (let i = 0; i < maxAttempts; i++) {
		const t = sampleOnce(I, rng);
		if (t && !I.validator.validateTeam(t.map(s => ({ ...s, evs: { ...s.evs }, ivs: { ...s.ivs }, moves: [...s.moves] })))) return t;
	}
	throw new Error(`poolBTeam(${stage}): no valid team after ${maxAttempts} attempts`);
}

module.exports = { poolBTeam, info };

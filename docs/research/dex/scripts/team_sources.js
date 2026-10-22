#!/usr/bin/env node
// Validate candidate team sources against the format and the v1 dex, and prototype a
// usage-stats team sampler that stays inside v1 and passes TeamValidator.
// Usage: node team_sources.js <ps_root> <dex_dir> <chaos_bo3_1760.json> <mc_replay_teams.jsonl> <vgcbench_mb_logs.json> <n_sample>
'use strict';
const [,, psRoot, dexDir, chaosPath, replayPath, vgcbPath, nSampleArg] = process.argv;
const fs = require('fs');
const path = require('path');
const {Dex, Teams, TeamValidator} = require(path.join(psRoot, 'dist/sim'));
const dex = Dex.mod('champions');
const FORMAT = 'gen9championsvgc2026regmc';
const validator = TeamValidator.get(FORMAT);
const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const load = f => JSON.parse(fs.readFileSync(path.join(dexDir, f), 'utf8'));
const v1 = {
	species: new Set(load('v1_species.json').species.map(s => s.id)),
	move: new Set(load('v1_moves.json').entries.map(e => e.id)),
	ability: new Set(load('v1_abilities.json').entries.map(e => e.id)),
	item: new Set(load('v1_items.json').entries.map(e => e.id)),
};
const legal = load('legal_pool.json');
const megaBase = load('out/dex_options.json').mega_base_abilities;

function inV1(team) {
	const miss = [];
	for (const s of team) {
		const sp = dex.species.get(s.species);
		let spid = sp.id;
		if (!legal.species[spid] && sp.baseSpecies && legal.species[toID(sp.baseSpecies)]) spid = toID(sp.baseSpecies);
		if (!v1.species.has(spid)) miss.push('species:' + spid);
		const it = dex.items.get(s.item);
		if (it.id && !v1.item.has(it.id)) miss.push('item:' + it.id);
		if (!v1.ability.has(toID(s.ability))) miss.push('ability:' + toID(s.ability));
		if (it.megaStone) {
			const forme = toID(typeof it.megaStone === 'object' ? Object.values(it.megaStone)[0] : it.megaStone);
			const f = legal.battleFormes[forme];
			if (f && !v1.ability.has(f.abilities[0])) miss.push('ability:' + f.abilities[0]);
		}
		for (const m of s.moves) if (!v1.move.has(toID(m))) miss.push('move:' + toID(m));
	}
	return miss;
}

function checkTeams(label, packedList, max = 4000) {
	const res = {label, n: 0, valid: 0, inV1: 0, validAndInV1: 0, problems: {}, v1miss: {}};
	for (const packed of packedList.slice(0, max)) {
		let team;
		try { team = Teams.unpack(packed); } catch (e) { continue; }
		if (!team || team.length !== 6) continue;
		res.n++;
		for (const s of team) {
			s.level = 50;
			// OTS showteam hides Stat Points; use a placeholder legal spread so only sheet legality is tested
			const tot = Object.values(s.evs || {}).reduce((a, b) => a + b, 0);
			if (!tot) s.evs = {hp: 32, atk: 0, def: 0, spa: 0, spd: 0, spe: 32};
			s.ivs = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};  // Champions has no IVs; showteam leaves them blank
		}
		const miss = inV1(team);
		const probs = validator.validateTeam(team.map(s => ({...s, evs: {...s.evs}, ivs: {...s.ivs}})));
		if (!probs) res.valid++;
		else for (const p of probs.slice(0, 3)) { const k = p.slice(0, 100); res.problems[k] = (res.problems[k] || 0) + 1; }
		if (!miss.length) res.inV1++;
		else for (const m of new Set(miss)) res.v1miss[m] = (res.v1miss[m] || 0) + 1;
		if (!probs && !miss.length) res.validAndInV1++;
	}
	const top = o => Object.entries(o).sort((a, b) => b[1] - a[1]).slice(0, 12);
	res.problems = top(res.problems); res.v1miss = top(res.v1miss);
	return res;
}

// ---- source 1: real Reg M-C Bo3 OTS teams from public replays
const mc = [...new Set(fs.readFileSync(replayPath, 'utf8').trim().split('\n').map(l => JSON.parse(l).packed))];
// ---- source 2: VGC-Bench HF logs, Reg M-B Bo3 (previous regulation)
let mb = [];
if (fs.existsSync(vgcbPath)) {
	const logs = JSON.parse(fs.readFileSync(vgcbPath, 'utf8'));
	const s = new Set();
	for (const [, [, log]] of Object.entries(logs)) {
		for (const line of log.split('\n')) if (line.startsWith('|showteam|')) s.add(line.split('|').slice(3).join('|'));
	}
	mb = [...s];
}
// ---- source 3: PS Champions random doubles generator
const rnd = [];
for (let i = 0; i < 1000; i++) {
	const t = Teams.generate('gen9championsrandomdoublesbattle', {seed: [i, 7, 13, 21].map(x => x + 1)});
	rnd.push(Teams.pack(t));
}

const results = [checkTeams('replays_regmc_bo3', mc), checkTeams('vgcbench_regmb_bo3', mb), checkTeams('ps_random_doubles_champions', rnd)];

// ---- source 4: usage-based sampler restricted to v1
const chaos = JSON.parse(fs.readFileSync(chaosPath, 'utf8')).data;
function rngFactory(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function pick(rng, entries) { const tot = entries.reduce((a, e) => a + e[1], 0); let r = rng() * tot; for (const e of entries) { r -= e[1]; if (r <= 0) return e[0]; } return entries[entries.length - 1][0]; }
const names = Object.keys(chaos).filter(n => {
	const id = toID(n);
	if (!v1.species.has(id)) return false;
	return Object.keys(chaos[n].Abilities).some(a => v1.ability.has(a));
});
const usage = Object.fromEntries(names.map(n => [n, Math.max(1e-6, chaos[n].usage)]));
function wOf(n) { return Object.values(chaos[n].Abilities).reduce((a, b) => a + b, 0); }
function sampleTeam(rng) {
	const team = [], nums = new Set(), items = new Set();
	const logs = Object.fromEntries(names.map(n => [n, Math.log(usage[n])]));
	for (let slot = 0; slot < 6; slot++) {
		const cand = names.filter(n => !nums.has(dex.species.get(n).num));
		const mx = Math.max(...cand.map(n => logs[n]));
		const nm = pick(rng, cand.map(n => [n, Math.exp(logs[n] - mx)]));
		const e = chaos[nm], w = wOf(nm), id = toID(nm);
		nums.add(dex.species.get(nm).num);
		for (const n of names) {
			const cond = ((e.Teammates[n] || 0) / w) + 1e-4;
			logs[n] += Math.log(Math.min(1e3, Math.max(1e-3, cond / usage[n])));
		}
		const forme = legal.battleFormes[id];
		let species, item, ability;
		if (forme && forme.isMega) {
			species = legal.species[forme.from[0]].name;
			item = dex.items.get(forme.requiredItem).name;
			const dist = Object.entries(megaBase[id] || {}).filter(([a]) => v1.ability.has(a));
			const legalAbs = legal.species[forme.from[0]].abilities.map(a => a.id).filter(a => v1.ability.has(a));
			ability = dist.length ? pick(rng, dist) : legalAbs[0];
		} else {
			species = dex.species.get(nm).name;
			ability = pick(rng, Object.entries(e.Abilities).filter(([a, v]) => v1.ability.has(a) && v > 0));
			const its = Object.entries(e.Items).filter(([i, v]) => v > 0 && (i === 'nothing' || (v1.item.has(i) && !items.has(i))) && !dex.items.get(i).megaStone);
			item = its.length ? pick(rng, its) : 'nothing';
			if (item === 'nothing') item = '';
		}
		if (item) items.add(toID(item));
		const sp = dex.species.get(species);
		const pool = Object.entries(e.Moves).filter(([m, v]) => m && v > 0 && v1.move.has(m) &&
			!validator.checkCanLearn(dex.moves.get(m), sp, validator.allSources(sp), {name: species, level: 50}));
		const moves = [];
		while (moves.length < 4 && pool.length) {
			const m = pick(rng, pool);
			moves.push(dex.moves.get(m).name);
			pool.splice(pool.findIndex(x => x[0] === m), 1);
		}
		const spreads = Object.entries(e.Spreads).sort((a, b) => b[1] - a[1]).slice(0, 200);
		const [nature, sps] = pick(rng, spreads).split(':');
		const v = sps.split('/').map(Number);
		team.push({name: species, species, item, ability: dex.abilities.get(ability).name, moves, nature, gender: '',
			evs: {hp: v[0], atk: v[1], def: v[2], spa: v[3], spd: v[4], spe: v[5]},
			ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31}, level: 50});
	}
	return team;
}
const nSample = Number(nSampleArg || 2000);
const rng = rngFactory(12345);
const samp = {label: 'usage_sampler_bo3_1760_v1', n: 0, valid: 0, inV1: 0, problems: {}, distinctSpecies: new Set(), distinctTeams: new Set()};
const sampleOut = [];
for (let i = 0; i < nSample; i++) {
	const t = sampleTeam(rng);
	samp.n++;
	const probs = validator.validateTeam(t.map(s => ({...s, evs: {...s.evs}, ivs: {...s.ivs}, moves: [...s.moves]})));
	if (!probs) samp.valid++;
	else for (const p of probs.slice(0, 3)) { const k = p.slice(0, 90); samp.problems[k] = (samp.problems[k] || 0) + 1; }
	if (!inV1(t).length) samp.inV1++;
	t.forEach(s => samp.distinctSpecies.add(s.species));
	samp.distinctTeams.add(t.map(s => s.species).sort().join(','));
	if (i < 200 && !probs) sampleOut.push(Teams.pack(t));
}
samp.problems = Object.entries(samp.problems).sort((a, b) => b[1] - a[1]).slice(0, 12);
samp.distinctSpecies = samp.distinctSpecies.size;
samp.distinctTeams = samp.distinctTeams.size;
results.push(samp);
fs.writeFileSync(path.join(dexDir, 'out', 'team_sources.json'), JSON.stringify(results, null, 1));
fs.writeFileSync(path.join(dexDir, 'out', 'sampled_teams_v1_example.txt'), sampleOut.join('\n') + '\n');
for (const r of results) console.log(JSON.stringify(r));

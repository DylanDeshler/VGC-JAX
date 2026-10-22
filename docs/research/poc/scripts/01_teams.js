#!/usr/bin/env node
'use strict';
/**
 * Team sources for gen9championsvgc2026regmc, validated with PS's TeamValidator.
 *   node scripts/01_teams.js [N=300]   -> results/01_teams.json
 */
const fs = require('fs');
const path = require('path');
const { Teams, TeamValidator, FORMAT } = require('../lib/ps');
const T = require('../lib/teams');

const N = Number(process.argv[2] || 300);
const v = TeamValidator.get(FORMAT);
const now = () => Number(process.hrtime.bigint()) / 1e6;
const reasons = (problems, acc) => { for (const p of problems || []) { const k = p.replace(/^[^:]*?(Pok[eé]mon|limited|level|EV|Item)/, '$1').slice(0, 70); acc[k] = (acc[k] || 0) + 1; } };

const out = { N, format: FORMAT };
// 1) PS champions random-doubles generator, raw
{
	const gen = Teams.getGenerator('gen9championsrandomdoublesbattle', 'gen5,0000000000000001');
	let ok = 0; const why = {}; const t0 = now();
	for (let i = 0; i < N; i++) { const p = v.validateTeam(gen.getTeam()); if (!p) ok++; else reasons(p, why); }
	out.randomDoublesRaw = { validRate: ok / N, msPerTeam: +((now() - t0) / N).toFixed(2), topProblems: why };
}
// 2) same generator + fix-ups (level 50, Item Clause)
{
	let ok = 0; const species = new Set(); const t0 = now();
	for (let i = 0; i < N; i++) { const team = T.randomDoubles(i === 0 ? 'gen5,0000000000000002' : undefined); if (!v.validateTeam(team)) ok++; team.forEach(s => species.add(s.species)); }
	out.randomDoublesFixed = { validRate: ok / N, msPerTeamInclValidation: +((now() - t0) / N).toFixed(2), distinctSpecies: species.size };
}
// 3) chaos-stats sampler
{
	const rng = new T.HarnessRNG(12345); let attempts = 0; const species = new Set(); const megas = new Set(); const t0 = now();
	for (let i = 0; i < N; i++) { const r = T.chaosTeam(rng); attempts += r.attempts; r.team.forEach(s => { species.add(s.species); if (/ite[xyz]?$/.test(s.item) && s.item !== 'eviolite') megas.add(s.item); }); }
	out.chaos = { firstTryValidRate: N / attempts, meanAttempts: attempts / N, msPerTeamInclValidation: +((now() - t0) / N).toFixed(2), distinctSpecies: species.size, distinctMegaStones: megas.size, poolSize: T.loadChaos().length };
}
fs.mkdirSync(path.join(__dirname, '..', 'results'), { recursive: true });
fs.writeFileSync(path.join(__dirname, '..', 'results', '01_teams.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));

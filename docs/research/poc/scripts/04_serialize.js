#!/usr/bin/env node
'use strict';
/**
 * State.serializeBattle / deserializeBattle round trip mid-battle:
 *  - play battle i to decision k (scripted choices from a recording), serialize -> JSON -> deserialize,
 *    continue the deserialized battle with the remaining choices, compare the full log with the recording
 *  - measure serialize / stringify / parse+deserialize cost and size
 *   node scripts/04_serialize.js [N=100]   -> results/04_serialize.json
 */
const fs = require('fs');
const path = require('path');
const { Battle, State, FORMAT } = require('../lib/ps');
const E = require('../lib/experiment');
const R = require('../lib/runner');
const T = require('../lib/teams');
const { InstrumentedPRNG } = require('../lib/rng');
const { projectBattle } = require('../lib/project');

const N = Number(process.argv[2] || 100);
const now = () => Number(process.hrtime.bigint()) / 1e6;
const res = { N, roundTrips: 0, exactContinuation: 0, exactWithInstrumentedPrng: 0, projSameAfterRoundTrip: 0,
	failures: [], cost: { serializeMs: [], stringifyMs: [], deserializeMs: [], projectMs: [], bytes: [] } };

function battleAt(sp, choices, k, prng) {
	const battle = new Battle({ formatid: FORMAT, seed: sp.seed, prng, strictChoices: true });
	if (prng) prng.attach(battle);
	battle.setPlayer('p1', { name: 'p1', team: sp.p1team });
	battle.setPlayer('p2', { name: 'p2', team: sp.p2team });
	R.drive(battle, { script: choices, stopAfterDecisions: k });
	return battle;
}

for (let i = 0; i < N; i++) {
	const sp = E.spec(i);
	const rec = E.run(sp);
	const choices = rec.res.choices;
	const rng = new T.HarnessRNG(i + 1);
	for (let rep = 0; rep < 3; rep++) {
		const k = 1 + rng.int(Math.max(1, choices.length - 1));
		const battle = battleAt(sp, choices, k);
		let t = now(); const state = State.serializeBattle(battle); res.cost.serializeMs.push(now() - t);
		t = now(); const json = JSON.stringify(state); res.cost.stringifyMs.push(now() - t); res.cost.bytes.push(json.length);
		t = now(); const b2 = State.deserializeBattle(json); res.cost.deserializeMs.push(now() - t);
		t = now(); const p1 = JSON.stringify(projectBattle(battle)); res.cost.projectMs.push(now() - t);
		const p2 = JSON.stringify(projectBattle(b2));
		res.roundTrips++;
		if (p1 === p2) res.projSameAfterRoundTrip++;
		// continue the deserialized battle with the remaining choices
		const r2 = R.drive(b2, { script: choices.slice(k) });
		const same = R.normLog(b2) === rec.log;
		if (same) res.exactContinuation++;
		else if (res.failures.length < 5) {
			res.failures.push({ battle: i, k, diverged: r2.diverged, firstDiff: E.firstDiff(rec.log, R.normLog(b2)) });
		}
		// same, but the deserialized battle gets a fresh instrumented PRNG seeded from the serialized seed
		const b3 = State.deserializeBattle(json);
		const prng = new InstrumentedPRNG({ seed: b3.prng.getSeed(), classifier: 'markers' }).attach(b3);
		b3.prng = prng;
		R.drive(b3, { script: choices.slice(k) });
		if (R.normLog(b3) === rec.log) res.exactWithInstrumentedPrng++;
	}
}
const stat = a => { const s = a.slice().sort((x, y) => x - y); return { mean: +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(3), p50: +s[Math.floor(s.length / 2)].toFixed(3), p95: +s[Math.floor(s.length * 0.95)].toFixed(3) }; };
for (const k of Object.keys(res.cost)) res.cost[k] = stat(res.cost[k]);
res.exactRate = res.exactContinuation / res.roundTrips;
fs.mkdirSync(path.join(__dirname, '..', 'results'), { recursive: true });
fs.writeFileSync(path.join(__dirname, '..', 'results', '04_serialize.json'), JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));

#!/usr/bin/env node
'use strict';
/**
 * Single-threaded throughput of the harness configurations (teams pre-generated; random legal play).
 *   node scripts/06_throughput.js [M=200]   -> results/06_throughput.json
 */
const fs = require('fs');
const path = require('path');
const E = require('../lib/experiment');
const R = require('../lib/runner');
const T = require('../lib/teams');
const { randomChoice } = require('../lib/player');
const { InstrumentedPRNG } = require('../lib/rng');
const { projectBattle } = require('../lib/project');
const { captureSite, captureStringStack } = require('../lib/callsite');

const M = Number(process.argv[2] || 200);
const now = () => Number(process.hrtime.bigint()) / 1e6;

let t = now();
const specs = []; for (let i = 0; i < M; i++) specs.push(E.spec(i));
const teamGenMsPerTeam = (now() - t) / (2 * M);
t = now();
for (let i = 0; i < 200; i++) T.randomDoubles();
const randomDoublesMsPerTeam = (now() - t) / 200;

function bench(name, fn, warm = 20) {
	for (let i = 0; i < warm; i++) fn(specs[i % M]);
	let decisions = 0; let draws = 0; let turns = 0;
	const t0 = now();
	for (let i = 0; i < M; i++) { const r = fn(specs[i]); decisions += r.decisions; draws += r.draws || 0; turns += r.turns || 0; }
	const sec = (now() - t0) / 1000;
	const out = { battlesPerSec: +(M / sec).toFixed(1), decisionsPerSec: +(decisions / sec).toFixed(0), drawsPerSec: draws ? +(draws / sec).toFixed(0) : null, msPerBattle: +(1000 * sec / M).toFixed(2), turnsPerBattle: +(turns / M).toFixed(2) };
	console.log(name.padEnd(44), JSON.stringify(out));
	return out;
}

function direct(sp, prngOpts, { sodium = false, project = false } = {}) {
	const seed = sodium ? `sodium,${(sp.i + 1).toString(16).padStart(32, '0')}` : sp.seed;
	const prng = prngOpts ? new InstrumentedPRNG({ seed, ...prngOpts }) : undefined;
	const rng = new T.HarnessRNG(sp.playerSeed);
	const res = R.runDirect({
		p1team: sp.p1team, p2team: sp.p2team, seed, prng, choose: req => randomChoice(req, rng),
		onBattle: b => { if (prng) prng.attach(b); },
		onDecision: project ? b => JSON.stringify(projectBattle(b)) : null,
	});
	return { decisions: res.decisions, draws: prng ? prng.draws.length : 0, turns: res.turns };
}

/** Round-robin all configs battle-by-battle (rotating order) so machine-load drift hits all configs equally. */
function interleaved(configs) {
	const names = Object.keys(configs); const acc = Object.fromEntries(names.map(n => [n, { ms: 0, decisions: 0, draws: 0 }]));
	for (let i = 0; i < 20; i++) for (const n of names) configs[n](specs[i % M]);
	for (let i = 0; i < M; i++) {
		for (let j = 0; j < names.length; j++) {
			const n = names[(i + j) % names.length];
			const t0 = now(); const r = configs[n](specs[i]); acc[n].ms += now() - t0; acc[n].decisions += r.decisions; acc[n].draws += r.draws;
		}
	}
	const base = acc[names[0]].ms;
	const res = {};
	for (const n of names) {
		const a = acc[n];
		res[n] = { battlesPerSec: +(M * 1000 / a.ms).toFixed(1), decisionsPerSec: +(a.decisions * 1000 / a.ms).toFixed(0), relativeCost: +(a.ms / base).toFixed(3), usPerDrawOverhead: a.draws ? +((a.ms - base) * 1000 / a.draws).toFixed(2) : null };
		console.log(('[interleaved] ' + n).padEnd(52), JSON.stringify(res[n]));
	}
	return res;
}

const out = { M, teamGenMsPerTeam: +teamGenMsPerTeam.toFixed(3), randomDoublesMsPerTeam: +randomDoublesMsPerTeam.toFixed(3), configs: {} };
const C = out.configs;
C.direct_uninstrumented_gen5 = bench('direct, uninstrumented (gen5 seed)', sp => direct(sp, null));
C.direct_uninstrumented_sodium = bench('direct, uninstrumented (sodium seed)', sp => direct(sp, null, { sodium: true }));
// stream is async: measure separately
out.configs.stream_uninstrumented_gen5 = null;
C.record_noctx = bench('record decisions, no context', sp => direct(sp, { ctx: false }));
C.record_ctx_markers = bench('record + context + marker classifier', sp => direct(sp, { ctx: true, classifier: 'markers', stack: false }));
C.record_ctx_stack = bench('record + context + stack capture/classifier', sp => direct(sp, { ctx: true, classifier: 'stack', stack: true }));
C.record_ctx_markers_projection = bench('record + markers + projection per decision', sp => direct(sp, { ctx: true, classifier: 'markers', stack: false }, { project: true }));
C.record_ctx_stack_projection = bench('record + stack + projection per decision', sp => direct(sp, { ctx: true, classifier: 'stack', stack: true }, { project: true }));

out.interleaved = interleaved({
	uninstrumented: sp => direct(sp, null),
	record_noctx: sp => direct(sp, { ctx: false }),
	record_ctx_markers: sp => direct(sp, { ctx: true, classifier: 'markers', stack: false }),
	record_ctx_stack: sp => direct(sp, { ctx: true, classifier: 'stack', stack: true }),
	record_ctx_both_projection: sp => direct(sp, { ctx: true, classifier: 'both', stack: true }, { project: true }),
});
out.loadavg = require('os').loadavg();

// micro: cost of one stack capture (called from a shallow frame; real draws are ~10-20 frames deep)
function micro(name, f, n = 200000) {
	for (let i = 0; i < 1000; i++) f();
	const t0 = now(); for (let i = 0; i < n; i++) f(); const us = (now() - t0) * 1000 / n;
	console.log(name.padEnd(44), us.toFixed(3), 'us/call'); return +us.toFixed(3);
}
out.micro = {
	captureSite_structured_mapped_us: micro('captureSite (structured + sourcemap cache)', () => captureSite(3)),
	errorStackString_us: micro('new Error().stack (formatted string)', () => captureStringStack()),
};

(async () => {
	for (let i = 0; i < 20; i++) { const sp = specs[i]; const rng = new T.HarnessRNG(sp.playerSeed); await R.runStream({ p1team: sp.p1team, p2team: sp.p2team, seed: sp.seed, choose: req => randomChoice(req, rng) }); }
	let decisions = 0; const t0 = now();
	for (let i = 0; i < M; i++) { const sp = specs[i]; const rng = new T.HarnessRNG(sp.playerSeed); const r = await R.runStream({ p1team: sp.p1team, p2team: sp.p2team, seed: sp.seed, choose: req => randomChoice(req, rng) }); decisions += r.decisions; }
	const sec = (now() - t0) / 1000;
	C.stream_uninstrumented_gen5 = { battlesPerSec: +(M / sec).toFixed(1), decisionsPerSec: +(decisions / sec).toFixed(0), msPerBattle: +(1000 * sec / M).toFixed(2) };
	console.log('stream, uninstrumented (gen5 seed)'.padEnd(44), JSON.stringify(C.stream_uninstrumented_gen5));
	fs.mkdirSync(path.join(__dirname, '..', 'results'), { recursive: true });
	fs.writeFileSync(path.join(__dirname, '..', 'results', '06_throughput.json'), JSON.stringify(out, null, 1));
})();

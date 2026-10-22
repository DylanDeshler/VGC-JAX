#!/usr/bin/env node
'use strict';
/**
 * worker_threads scaling: W workers each play random battles for D seconds (teams pre-generated
 * inside the worker, not timed). Config: 'plain' (uninstrumented) or 'oracle' (record + context +
 * marker classifier + projection per decision).
 *   node scripts/07_workers.js [D=8] [W list=1,4,8,12] [config=plain]  -> results/07_workers_<config>.json
 */
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const os = require('os');

if (isMainThread) {
	const fs = require('fs');
	const path = require('path');
	const D = Number(process.argv[2] || 8);
	const Ws = (process.argv[3] || '1,4,8,12').split(',').map(Number);
	const config = process.argv[4] || 'plain';
	(async () => {
		const results = { D, config, cpus: os.cpus().length, model: os.cpus()[0].model, runs: [] };
		for (const W of Ws) {
			const load0 = os.loadavg()[0];
			const t0 = Date.now();
			const parts = await Promise.all([...Array(W).keys()].map(w => new Promise((resolve, reject) => {
				const wk = new Worker(__filename, { workerData: { w, D, config } });
				wk.on('message', resolve); wk.on('error', reject);
			})));
			const battles = parts.reduce((a, p) => a + p.battles, 0);
			const decisions = parts.reduce((a, p) => a + p.decisions, 0);
			const sec = Math.max(...parts.map(p => p.sec));
			const r = { W, battlesPerSec: +(battles / sec).toFixed(1), decisionsPerSec: +(decisions / sec).toFixed(0), perWorker: +(battles / sec / W).toFixed(1), loadavgBefore: +load0.toFixed(1), wallSec: (Date.now() - t0) / 1000 };
			results.runs.push(r);
			console.log(JSON.stringify(r));
		}
		fs.mkdirSync(path.join(__dirname, '..', 'results'), { recursive: true });
		fs.writeFileSync(path.join(__dirname, '..', 'results', `07_workers_${config}.json`), JSON.stringify(results, null, 1));
	})();
} else {
	const E = require('../lib/experiment');
	const R = require('../lib/runner');
	const T = require('../lib/teams');
	const { randomChoice } = require('../lib/player');
	const { InstrumentedPRNG } = require('../lib/rng');
	const { projectBattle } = require('../lib/project');
	const { w, D, config } = workerData;
	const specs = []; for (let i = 0; i < 60; i++) specs.push(E.spec(w * 1000 + i));
	const play = sp => {
		const prng = config === 'oracle' ? new InstrumentedPRNG({ seed: sp.seed, classifier: 'markers', stack: false }) : undefined;
		const rng = new T.HarnessRNG(sp.playerSeed);
		return R.runDirect({
			p1team: sp.p1team, p2team: sp.p2team, seed: sp.seed, prng, choose: req => randomChoice(req, rng),
			onBattle: b => { if (prng) prng.attach(b); },
			onDecision: config === 'oracle' ? b => JSON.stringify(projectBattle(b)) : null,
		});
	};
	for (let i = 0; i < 10; i++) play(specs[i]); // warm-up
	let battles = 0; let decisions = 0; const t0 = Date.now(); let i = 0;
	while (Date.now() - t0 < D * 1000) { const r = play(specs[i++ % specs.length]); battles++; decisions += r.decisions; }
	parentPort.postMessage({ battles, decisions, sec: (Date.now() - t0) / 1000 });
}

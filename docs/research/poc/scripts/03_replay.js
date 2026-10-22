#!/usr/bin/env node
'use strict';
/**
 * Forced replay experiments.
 *   Mode 1  stream replay: recorded decisions answered in stream order (different underlying seed)
 *   Mode 2  context-keyed forcing with key variants naive | step | step+ord
 *   Mode 2x observability: step+ord forcing but kind K answered by an unrelated PRNG (one K at a time)
 *   Mode 3  override policies (max roll / no crit / always hit, ...)
 *   node scripts/03_replay.js [N=300]   -> results/03_replay.json
 */
const fs = require('fs');
const path = require('path');
const E = require('../lib/experiment');
const { InstrumentedPRNG, buildKeyedDict, keyFor } = require('../lib/rng');
const { projectBattle } = require('../lib/project');

const N = Number(process.argv[2] || 300);
const OTHER_SEED = '1,2,3,4'; // underlying PRNG for forced runs: anything that leaks through will diverge
const FALLBACK = '5,6,7,8';

function record(sp) {
	const proj = [];
	const r = E.run(sp, { prngOpts: { classifier: 'markers' }, onDecision: b => proj.push(JSON.stringify(projectBattle(b))) });
	proj.push(JSON.stringify(projectBattle(r.res.battle))); // final state too
	return { draws: r.prng.draws, choices: r.res.choices, log: r.log, proj };
}

function replay(sp, rec, prngOpts) {
	const proj = [];
	const prng = new InstrumentedPRNG({ seed: OTHER_SEED, fallbackSeed: FALLBACK, ...prngOpts });
	const r = E.run(sp, { prng, script: rec.choices, onDecision: b => proj.push(JSON.stringify(projectBattle(b))) });
	proj.push(JSON.stringify(projectBattle(r.res.battle)));
	return { prng, log: r.log, proj, diverged: r.res.diverged, battle: r.res.battle };
}

/** First draw where the replay differs from the recording (kind/key/outcome). */
function firstDrawDiff(a, b) {
	for (let i = 0; i < Math.max(a.length, b.length); i++) {
		const x = a[i]; const y = b[i];
		if (!x || !y) return { i, rec: x && { kind: x.kind, key: x.key }, rep: y && { kind: y.kind, key: y.key }, why: 'length' };
		const ox = JSON.stringify(x.method === 'shuffle' ? x.outLabels : x.out);
		const oy = JSON.stringify(y.method === 'shuffle' ? y.outLabels : y.out);
		if (x.key !== y.key || ox !== oy) return { i, rec: { kind: x.kind, key: x.key, out: ox }, rep: { kind: y.kind, key: y.key, out: oy } };
	}
	return null;
}

const res = {
	N, mode1: { logSame: 0, projSame: 0, desync: 0 },
	mode2: {}, observability: {}, mode3: {},
};
const KEYMODES = ['naive', 'step', 'step+ord', 'engine'];
// 'engine' = what the JAX side can actually implement: step key, ordinal only where it is semantic
// (secondary index), and the kinds measured as state-irrelevant answered by an unrelated PRNG.
const DONT_CARE = new Set(['RANDOM_TARGET_NOMINAL', 'SPEED_TIE_EACH_EVENT', 'SPEED_TIE_HANDLER_ORDER', 'SPEED_TIE_FIELD_EVENT', 'QUEUE_INSERT_TIE', 'SELF_EFFECT_ROLL']);
for (const km of KEYMODES) res.mode2[km] = { logSame: 0, projSame: 0, conflictsBattles: 0, missBattles: 0, failKinds: {}, examples: [] };
const OBS_KINDS = ['RANDOM_TARGET_NOMINAL', 'RANDOM_TARGET', 'SPEED_TIE_EACH_EVENT', 'SPEED_TIE_HANDLER_ORDER', 'SPEED_TIE_FIELD_EVENT',
	'SPEED_TIE_ACTION_ORDER', 'SPEED_TIE_SWITCHIN_ORDER', 'QUEUE_INSERT_TIE', 'SELF_EFFECT_ROLL', 'SECONDARY', 'CRIT', 'DAMAGE_ROLL', 'ACCURACY'];
for (const k of OBS_KINDS) res.observability[k] = { battlesWithKind: 0, stateChanged: 0, logChanged: 0 };

const POLICIES = {
	'maxroll_nocrit_alwayshit': d => (d.kind === 'DAMAGE_ROLL' ? 0 : d.kind === 'CRIT' ? false : d.kind === 'ACCURACY' ? true : undefined),
	'minroll_alwayscrit_noparafullpara': d => (d.kind === 'DAMAGE_ROLL' ? 15 : d.kind === 'CRIT' ? true : d.kind === 'PARALYSIS' ? false : undefined),
	'no_speed_tie_randomness': d => (d.method === 'shuffle' ? [...Array(d.args[0]).keys()] : undefined),
};
for (const p of Object.keys(POLICIES)) res.mode3[p] = { battles: 0, forced: 0, critLines: 0, missLines: 0, fullParaLines: 0, completed: 0 };

const t0 = Date.now();
for (let i = 0; i < N; i++) {
	const sp = E.spec(i);
	const rec = record(sp);
	// ---- Mode 1
	{
		const r = replay(sp, rec, { mode: 'replay', script: rec.draws, ctx: false });
		if (r.log === rec.log) res.mode1.logSame++;
		if (JSON.stringify(r.proj) === JSON.stringify(rec.proj)) res.mode1.projSame++;
		if (r.prng.stats.desync) res.mode1.desync++;
	}
	// ---- Mode 2
	for (const km of KEYMODES) {
		const { dict, conflicts } = buildKeyedDict(rec.draws, { keyMode: km });
		const r = replay(sp, rec, { mode: 'keyed', dict, keyMode: km, classifier: 'markers', ignoreKinds: km === 'engine' ? DONT_CARE : null });
		const st = res.mode2[km];
		if (conflicts) st.conflictsBattles++;
		if (r.prng.stats.misses) st.missBattles++;
		const logSame = r.log === rec.log; const projSame = JSON.stringify(r.proj) === JSON.stringify(rec.proj);
		if (logSame) st.logSame++;
		if (projSame) st.projSame++;
		if (!logSame) {
			const fd = firstDrawDiff(rec.draws, r.prng.draws);
			const k = fd ? (fd.rec ? fd.rec.kind : 'extra-draw') : 'no-draw-diff';
			st.failKinds[k] = (st.failKinds[k] || 0) + 1;
			if (st.examples.length < 4) st.examples.push({ battle: i, firstDrawDiff: fd, dictConflicts: conflicts, misses: r.prng.stats.misses });
		}
	}
	// ---- Mode 2x: observability of each kind (randomize only that kind)
	const kindsHere = new Set(rec.draws.map(d => d.kind));
	const { dict } = buildKeyedDict(rec.draws, { keyMode: 'step+ord' });
	for (const k of OBS_KINDS) {
		if (!kindsHere.has(k)) continue;
		const r = replay(sp, rec, { mode: 'keyed', dict, keyMode: 'step+ord', classifier: 'markers', ignoreKinds: new Set([k]) });
		const o = res.observability[k]; o.battlesWithKind++;
		if (JSON.stringify(r.proj) !== JSON.stringify(rec.proj)) o.stateChanged++;
		if (r.log !== rec.log) o.logChanged++;
	}
	// ---- Mode 3 (fresh random play under the policy, own choices)
	for (const [name, policy] of Object.entries(POLICIES)) {
		const prng = new InstrumentedPRNG({ seed: sp.seed, mode: 'override', policy, classifier: 'markers' });
		const r = E.run(sp, { prng });
		const st = res.mode3[name]; st.battles++; st.forced += prng.stats.forced;
		if (r.res.battle.ended) st.completed++;
		const L = r.res.battle.log;
		st.critLines += L.filter(l => l.startsWith('|-crit|')).length;
		st.missLines += L.filter(l => l.startsWith('|-miss|')).length;
		st.fullParaLines += L.filter(l => l.startsWith('|cant|') && l.endsWith('|par')).length;
	}
}
res.seconds = (Date.now() - t0) / 1000;
for (const km of KEYMODES) {
	const s = res.mode2[km]; s.logSameRate = s.logSame / N; s.projSameRate = s.projSame / N;
}
res.mode1.logSameRate = res.mode1.logSame / N;
for (const k of OBS_KINDS) { const o = res.observability[k]; o.stateChangedRate = o.battlesWithKind ? +(o.stateChanged / o.battlesWithKind).toFixed(3) : null; }
fs.mkdirSync(path.join(__dirname, '..', 'results'), { recursive: true });
fs.writeFileSync(path.join(__dirname, '..', 'results', '03_replay.json'), JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));

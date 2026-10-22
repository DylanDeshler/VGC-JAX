'use strict';
/** Shared helpers for the scripts: deterministic team pairs, seeds, and one-call battle runs. */
const T = require('./teams');
const { randomChoice } = require('./player');
const R = require('./runner');
const { InstrumentedPRNG } = require('./rng');

/** Deterministic battle spec #i: teams from chaos stats (or PS random doubles), battle seed, player seed. */
function spec(i, { source = 'chaos' } = {}) {
	const rng = new T.HarnessRNG(1000003 * (i + 1));
	const p1team = source === 'chaos' ? T.chaosTeam(rng).team : T.randomDoubles(`gen5,${(i + 1).toString(16).padStart(16, '0')}`);
	const p2team = source === 'chaos' ? T.chaosTeam(rng).team : T.randomDoubles();
	// Gen5 RNG seeds are ~4x cheaper per draw than the default sodium/ChaCha20 seeds
	return { i, p1team, p2team, seed: `${i + 11},${i * 7 + 3},${i * 13 + 5},${i * 17 + 7}`, playerSeed: 99991 * (i + 1) };
}

/** Run a battle; prngOpts=null => uninstrumented. Returns {res, prng, log}. */
function run(sp, { prngOpts = null, script = null, onDecision = null, prng = null } = {}) {
	const prngObj = prng || (prngOpts ? new InstrumentedPRNG({ seed: sp.seed, ...prngOpts }) : undefined);
	const prng_ = new T.HarnessRNG(sp.playerSeed);
	const res = R.runDirect({
		p1team: sp.p1team, p2team: sp.p2team, seed: sp.seed, prng: prngObj, script, onDecision,
		choose: req => randomChoice(req, prng_),
		onBattle: b => { if (prngObj) prngObj.attach(b); },
	});
	return { res, prng: prngObj, log: R.normLog(res.battle) };
}

function firstDiff(a, b) {
	const A = a.split('\n'); const B = b.split('\n');
	for (let i = 0; i < Math.max(A.length, B.length); i++) if (A[i] !== B[i]) return { line: i, a: A[i], b: B[i] };
	return null;
}

module.exports = { spec, run, firstDiff };

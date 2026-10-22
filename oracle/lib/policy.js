'use strict';
/**
 * Choice policies (07 §2). All use their own HarnessRNG, never battle.prng.
 *   random          uniform over legal joint choices (per lib/player.js legalOptions + jointOk)
 *   random-weighted biased toward non-pass, Mega and switch actions (the POC chooser)
 */
const { legalOptions, jointOk, randomChoice } = require('./player');

function jointChoices(lo) {
	let combos = [[]];
	for (const opts of lo.slots) {
		const next = [];
		for (const c of combos) for (const o of opts) next.push([...c, o]);
		combos = next;
	}
	return combos.filter(c => jointOk(c, lo));
}

function uniformChoice(req, rng) {
	const lo = legalOptions(req);
	if (lo.kind === 'wait') return null;
	if (lo.kind === 'team') {
		const order = rng.shuffle([...Array(lo.n).keys()].map(i => i + 1));
		return 'team ' + order.slice(0, lo.size).join('');
	}
	const all = jointChoices(lo);
	if (!all.length) return 'default';
	return all[rng.int(all.length)].join(', ');
}

const POLICIES = {
	'random': uniformChoice,
	'random-weighted': (req, rng) => randomChoice(req, rng, { megaProb: 0.6, switchProb: 0.15 }),
};

function getPolicy(name) {
	const p = POLICIES[name];
	if (!p) throw new Error(`unknown policy ${name} (have: ${Object.keys(POLICIES).join(', ')})`);
	return p;
}

module.exports = { getPolicy, jointChoices, uniformChoice, POLICIES };

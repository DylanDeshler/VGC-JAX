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

/**
 * heuristic (07 §2, M3.8b), per slot in order: (1) Mega whenever legal; (2) on the mon's first active turn, Fake Out if
 * legal; (3) otherwise Protect-family with p = 0.25 if legal and not used last turn; (4) otherwise a spread move when
 * both foes are active; (5) otherwise uniform over damaging moves (single targets aimed at a living foe).
 * Overall switch probability 0.05. Deterministic given the policy RNG. Falls back to uniform if nothing applies.
 */
function heuristicChoice(req, rng, battle, sideId) {
	const lo = legalOptions(req);
	if (lo.kind !== 'move' || !battle) return uniformChoice(req, rng);
	const side = battle.getSide(sideId);
	const dex = battle.dex;
	const foesAlive = side.foe.active.filter(p => p && !p.fainted);
	let megaUsed = false;
	const picks = lo.slots.map((opts, i) => {
		if (opts.length === 1) return opts[0];
		const mon = side.active[i];
		const switches = opts.filter(o => o.startsWith('switch'));
		if (switches.length && rng.chance(0.05)) return rng.pick(switches);
		const act = req.active[i];
		const moveOpts = opts.filter(o => o.startsWith('move'));
		const plain = moveOpts.filter(o => !/ mega$/.test(o));
		const withMega = o => {
			const m = o + ' mega';
			if (!megaUsed && act.canMegaEvo && moveOpts.includes(m)) { megaUsed = true; return m; }
			return o;
		};
		const byMove = id => plain.filter(o => act.moves[Number(o.split(' ')[1]) - 1].id === id);
		const info = o => dex.moves.get(act.moves[Number(o.split(' ')[1]) - 1].id);
		if (mon && mon.activeMoveActions === 0) {
			const fo = byMove('fakeout').filter(o => / [12]$/.test(o) || / [12] /.test(o));
			if (fo.length) return withMega(rng.pick(fo));
		}
		const protects = plain.filter(o => info(o).stallingMove);
		const usedLast = mon && mon.lastMove && dex.moves.get(mon.lastMove.id).stallingMove;
		if (protects.length && !usedLast && rng.chance(0.25)) return withMega(rng.pick(protects));
		const damaging = plain.filter(o => info(o).category !== 'Status');
		const foeTarget = o => { const t = o.split(' ')[2]; return t === undefined || t === '1' || t === '2'; };
		const aimed = damaging.filter(o => {
			const t = o.split(' ')[2];
			return t === undefined || (t === '1' && side.foe.active[0] && !side.foe.active[0].fainted) ||
				(t === '2' && side.foe.active[1] && !side.foe.active[1].fainted);
		});
		if (foesAlive.length === 2) {
			const spread = damaging.filter(o => ['allAdjacentFoes', 'allAdjacent'].includes(info(o).target));
			if (spread.length) return withMega(rng.pick(spread));
		}
		const pool = aimed.length ? aimed : damaging.filter(foeTarget);
		if (pool.length) return withMega(rng.pick(pool));
		return withMega(rng.pick(plain.length ? plain : opts));
	});
	if (jointOk(picks, lo)) return picks.join(', ');
	return uniformChoice(req, rng);
}

const POLICIES = {
	'random': uniformChoice,
	'random-weighted': (req, rng) => randomChoice(req, rng, { megaProb: 0.6, switchProb: 0.15 }),
	'heuristic': heuristicChoice,
};

function getPolicy(name) {
	const p = POLICIES[name];
	if (!p) throw new Error(`unknown policy ${name} (have: ${Object.keys(POLICIES).join(', ')})`);
	return p;
}

module.exports = { getPolicy, jointChoices, uniformChoice, POLICIES };

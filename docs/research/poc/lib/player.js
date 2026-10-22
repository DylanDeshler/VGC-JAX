'use strict';
/**
 * Random legal-choice player driven purely by PS request JSON.
 * The same parser produces the "legal action set" we will diff against the JAX mask.
 *
 * Doubles target locations (PS convention): foes are +1/+2 (slot a/b), allies are -1/-2.
 */

// target types that need an explicit location in doubles
const NEEDS_TARGET = new Set(['normal', 'any', 'adjacentAlly', 'adjacentFoe', 'adjacentAllyOrSelf']);

function targetLocs(targetType, slot /* 0|1 */) {
	const self = -(slot + 1); const ally = -(2 - slot);
	switch (targetType) {
	case 'normal': return [1, 2, ally];
	case 'any': return [1, 2, ally];
	case 'adjacentFoe': return [1, 2];
	case 'adjacentAlly': return [ally];
	case 'adjacentAllyOrSelf': return [self, ally];
	default: return [null];
	}
}

/**
 * Enumerate legal per-slot options for a request. Returns
 *  { kind: 'team'|'move'|'switch'|'wait', slots: [[option strings]...] , ...}
 * Per-slot option lists are independent except: switch targets must be distinct and mega at most once
 * per side; `combine` handles those joint constraints.
 */
function legalOptions(req) {
	if (req.wait) return { kind: 'wait' };
	const mons = req.side.pokemon;
	if (req.teamPreview) return { kind: 'team', size: req.maxChosenTeamSize || mons.length, n: mons.length };
	const benchOk = mons.map((p, i) => !p.active && !p.condition.endsWith(' fnt') ? i + 1 : 0).filter(Boolean);
	if (req.forceSwitch) {
		// PS rule: a forced slot may pass only if there are fewer healthy bench mons than forced slots
		// (side.choice.forcedPassesLeft); i.e. exactly min(forced, bench) slots must switch.
		const forced = req.forceSwitch.filter(Boolean).length;
		const passOk = benchOk.length < forced;
		const slots = req.forceSwitch.map((must, i) => {
			if (!must) return ['pass'];
			if (mons[i] && mons[i].reviving) {
				const fainted = mons.map((p, j) => p.condition.endsWith(' fnt') ? j + 1 : 0).filter(Boolean);
				return fainted.map(j => `switch ${j}`);
			}
			return benchOk.map(j => `switch ${j}`).concat(passOk || !benchOk.length ? ['pass'] : []);
		});
		return { kind: 'switch', slots, benchCount: benchOk.length, mustSwitch: Math.min(forced, benchOk.length) };
	}
	// move request
	const slots = req.active.map((act, i) => {
		const mon = mons[i];
		if (!mon || mon.condition.endsWith(' fnt') || mon.commanding) return ['pass'];
		const opts = [];
		act.moves.forEach((m, mi) => {
			if (m.disabled) return;
			const locs = NEEDS_TARGET.has(m.target) ? targetLocs(m.target, i) : [null];
			for (const loc of locs) {
				const base = `move ${mi + 1}${loc === null ? '' : ' ' + loc}`;
				opts.push(base);
				if (act.canMegaEvo) opts.push(base + ' mega');
				if (act.canMegaEvoX) opts.push(base + ' megax');
				if (act.canMegaEvoY) opts.push(base + ' megay');
			}
		});
		if (!act.trapped) for (const j of benchOk) opts.push(`switch ${j}`);
		return opts;
	});
	return { kind: 'move', slots };
}

/** Joint (cross-slot) constraints: distinct switch-ins, one Mega per side, forced-switch count. */
function jointOk(choices, lo) {
	const sw = choices.filter(c => c.startsWith('switch'));
	if (new Set(sw).size !== sw.length) return false;
	if (choices.filter(c => / mega[xy]?$/.test(c)).length > 1) return false;
	if (lo && lo.kind === 'switch' && sw.length !== lo.mustSwitch) return false;
	return true;
}

/** Pick a random legal choice string. `rng` is a HarnessRNG. */
function randomChoice(req, rng, { megaProb = 0.5, switchProb = 0.1 } = {}) {
	const lo = legalOptions(req);
	if (lo.kind === 'wait') return null;
	if (lo.kind === 'team') {
		const order = rng.shuffle([...Array(lo.n).keys()].map(i => i + 1));
		return 'team ' + order.slice(0, lo.size).join('');
	}
	for (let tries = 0; tries < 50; tries++) {
		const picks = lo.slots.map(opts => {
			if (lo.kind === 'switch') {
				const nonPass = opts.filter(o => o !== 'pass');
				return nonPass.length ? rng.pick(nonPass) : 'pass';
			}
			// bias: mostly moves, sometimes switch; mega with megaProb when offered
			const moves = opts.filter(o => o.startsWith('move'));
			const switches = opts.filter(o => o.startsWith('switch'));
			if (opts.length === 1) return opts[0];
			if (switches.length && (!moves.length || rng.chance(switchProb))) return rng.pick(switches);
			const megas = moves.filter(o => / mega[xy]?$/.test(o));
			if (megas.length && rng.chance(megaProb)) return rng.pick(megas);
			const plain = moves.filter(o => !/ mega[xy]?$/.test(o));
			return rng.pick(plain.length ? plain : moves);
		});
		// forced switches: can't pass if a bench mon is available for an unfilled slot
		if (lo.kind === 'switch') {
			const used = new Set();
			for (let i = 0; i < picks.length; i++) {
				if (picks[i] === 'pass') continue;
				if (used.has(picks[i])) {
					const alt = lo.slots[i].filter(o => o !== 'pass' && !used.has(o));
					picks[i] = alt.length ? rng.pick(alt) : 'pass';
				}
				used.add(picks[i]);
			}
		}
		if (jointOk(picks, lo)) return picks.join(', ');
	}
	return 'default';
}

module.exports = { legalOptions, jointOk, randomChoice, targetLocs, NEEDS_TARGET };

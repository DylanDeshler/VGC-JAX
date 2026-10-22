#!/usr/bin/env node
'use strict';
/**
 * Instrumented PRNG: transparency, classification coverage, stack-vs-marker agreement, context sanity.
 *   node scripts/02_instrument.js [N=300]
 * Writes results/02_instrument.json
 */
const fs = require('fs');
const path = require('path');
const E = require('../lib/experiment');

const N = Number(process.argv[2] || 300);
const SOURCE = process.argv[3] || 'chaos'; // 'chaos' (usage-stats sets) | 'random' (PS champions random doubles sets)
const agg = {
	battles: 0, transparent: 0, draws: 0, decisions: 0, turns: 0, unmapped: 0, disagree: 0,
	kinds: {}, kindBattles: {}, randomTargetCallers: {}, unmappedSites: {},
	keyCollisions: { naive: {}, step: {} }, collisionExamples: { naive: {}, step: {} },
	missCheck: { n: 0, ok: 0, bad: [] }, critCheck: { n: 0, ok: 0, bad: [] }, degenerate: 0, sitesHit: {},
};
const prefix = (line, tag) => line.startsWith(tag);
for (let i = 0; i < N; i++) {
	const sp = E.spec(i, { source: SOURCE });
	const a = E.run(sp);
	const b = E.run(sp, { prngOpts: { classifier: 'both' } });
	agg.battles++; agg.decisions += b.res.decisions; agg.turns += b.res.turns;
	if (a.log === b.log) agg.transparent++;
	const draws = b.prng.draws; agg.draws += draws.length; agg.disagree += b.prng.stats.disagree;
	const log = b.res.battle.log;
	const seenKinds = new Set(); const keyCount = { naive: new Map(), step: new Map() }; const kindOf = new Map();
	for (const d of draws) {
		agg.kinds[d.kind] = (agg.kinds[d.kind] || 0) + 1; seenKinds.add(d.kind);
		agg.sitesHit[d.site] = (agg.sitesHit[d.site] || 0) + 1;
		if (d.kind.startsWith('UNMAPPED')) { agg.unmapped++; const k = d.frames.join(' < '); agg.unmappedSites[k] = (agg.unmappedSites[k] || 0) + 1; }
		if (d.kind === 'RANDOM_TARGET') { const k = `${d.fns[2]} (${d.frames[2]})`; agg.randomTargetCallers[k] = (agg.randomTargetCallers[k] || 0) + 1; }
		if ((d.method === 'sample' && d.args[0] === 1) || (d.method === 'randomChance' && (d.args[0] >= d.args[1] || d.args[0] <= 0))) agg.degenerate++;
		keyCount.step.set(d.key, (keyCount.step.get(d.key) || 0) + 1); kindOf.set(d.key, d.kind);
		keyCount.naive.set(d.keyNaive, (keyCount.naive.get(d.keyNaive) || 0) + 1); kindOf.set(d.keyNaive, d.kind);
		// context sanity: a failed ACCURACY draw must be followed by |-miss|<actor>|<target>
		// (per-hit accuracy of Triple Axel / Population Bomb hits 2+ just ends the loop: no -miss line)
		if (d.kind === 'ACCURACY' && d.out === false && !(d.ctx && d.ctx.hit > 1)) {
			agg.missCheck.n++;
			const line = log.slice(d.logPos).find(l => prefix(l, '|-miss|'));
			const parts = line ? line.split('|') : [];
			if (line && parts[2].startsWith(d.actor) && (parts[3] || '').startsWith(d.target)) agg.missCheck.ok++;
			else if (agg.missCheck.bad.length < 5) agg.missCheck.bad.push({ i, key: d.key, line });
		}
		if (d.kind === 'CRIT' && d.out === true) {
			agg.critCheck.n++;
			const line = log.slice(d.logPos).find(l => prefix(l, '|-crit|'));
			if (line && line.split('|')[2].startsWith(d.target)) agg.critCheck.ok++;
			else if (agg.critCheck.bad.length < 5) agg.critCheck.bad.push({ i, key: d.key, line });
		}
	}
	for (const k of seenKinds) agg.kindBattles[k] = (agg.kindBattles[k] || 0) + 1;
	for (const variant of ['naive', 'step']) {
		for (const [k, c] of keyCount[variant]) {
			if (c < 2) continue;
			const kind = kindOf.get(k);
			agg.keyCollisions[variant][kind] = (agg.keyCollisions[variant][kind] || 0) + c;
			if (!agg.collisionExamples[variant][kind]) agg.collisionExamples[variant][kind] = { battle: i, key: k, count: c };
		}
	}
}
const out = {
	N, transparentRate: agg.transparent / agg.battles,
	drawsPerBattle: agg.draws / agg.battles, decisionsPerBattle: agg.decisions / agg.battles, turnsPerBattle: agg.turns / agg.battles,
	unmapped: agg.unmapped, stackVsMarkerDisagreements: agg.disagree, degenerateDraws: agg.degenerate,
	kinds: Object.fromEntries(Object.entries(agg.kinds).sort((x, y) => y[1] - x[1]).map(([k, v]) => [k, { draws: v, perBattle: +(v / agg.battles).toFixed(3), battlesWithKind: agg.kindBattles[k] }])),
	randomTargetCallers: agg.randomTargetCallers, unmappedSites: agg.unmappedSites,
	keyCollisions_drawsInNonUniqueKeys: agg.keyCollisions, collisionExamples: agg.collisionExamples,
	missCheck: agg.missCheck, critCheck: agg.critCheck,
	sitesHit: agg.sitesHit,
	reachableSitesNotObserved: require('../rng_callsites.json').callsites
		.filter(c => c.reachable !== 'no' && !agg.sitesHit[c.id] && c.id !== 'sim/battle.ts:461').map(c => `${c.id} ${c.kind} (${c.enclosing})`),
};
fs.mkdirSync(path.join(__dirname, '..', 'results'), { recursive: true });
out.teamSource = SOURCE;
fs.writeFileSync(path.join(__dirname, '..', 'results', `02_instrument${SOURCE === 'chaos' ? '' : '_' + SOURCE}.json`), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));

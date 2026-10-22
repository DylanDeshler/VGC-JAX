#!/usr/bin/env node
'use strict';
/**
 * Don't-care re-validation (07 §4.3, contract §6). For each battle: record PS; replay it with a different seed forcing
 * every compared draw and tie record by DrawKey while leaving the don't-care draws (and unrecorded tie shuffles)
 * free, with ADR-101's canonical answering disabled. Then per don't-care kind, replay forcing every other don't-care
 * draw from the recording (DrawKey + occurrence) and leaving only that kind's don't-care draws free. Unrecorded tie
 * shuffles (each-event / non-MUL handler ties) are free in every run.
 * A changed projection (protected fields; ext fields reported separately) means the kind is not state-irrelevant.
 *   node oracle/tools/revalidate_dontcare.js --n 2000 --teams chaos --seed reval-1
 */
const tr = require('../lib/trace');
const support = require('../lib/support');
const { diffProjection } = require('../lib/project');

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const N = Number(args.n || 500); const teams = args.teams || 'chaos'; const S = args.seed || 'reval-1';
const KINDS = ['RANDOM_TARGET_NOMINAL', 'RANDOM_TARGET_NOEFFECT', 'QUEUE_INSERT_TIE', 'SELF_EFFECT_ROLL', 'SECONDARY'];
const protectedDiff = (a, b) => {
	const strip = p => JSON.parse(JSON.stringify(p, (k, v) => (k === 'ext' ? undefined : v)));
	return diffProjection(strip(a), strip(b));
};
const extDiff = (a, b) => diffProjection(a, b).filter(([path]) => path.includes('ext') && !path.endsWith('lastMoveTargetLoc'));

const stat = { battles: 0, all: { changed: 0, ext_changed: 0, examples: [] } };
for (const k of KINDS) stat[k] = { battles_with: 0, changed: 0, examples: [] };
function check(rec, decisions, final, out, label, i) {
	let firstDiff = null; let extFirst = null;
	for (let k = 0; k < decisions.length; k++) {
		const r = out.decisions[k];
		if (!r) { firstDiff = [k, 'missing decision']; break; }
		const d = protectedDiff(decisions[k].projection, r.projection);
		if (d.length) { firstDiff = [k, d.slice(0, 3)]; break; }
		if (!extFirst) { const e = extDiff(decisions[k].projection, r.projection); if (e.length) extFirst = [k, e.slice(0, 3)]; }
	}
	if (!firstDiff && final && out.final) {
		const d = protectedDiff(final.projection, out.final.projection);
		if (d.length) firstDiff = ['final', d.slice(0, 3)];
	}
	return { firstDiff, extFirst };
}
for (let i = 0; i < N; i++) {
	const recs = tr.runBattle(S, i, { teams, policy: 'random-weighted' });
	const h = recs[0]; const decisions = recs.filter(r => r.type === 'decision'); const final = recs.at(-1);
	// record mode used ADR-101 canonical answers; the replays below draw QUEUE_INSERT_TIE naturally
	const choices = decisions.map(d => d.choices);
	const draws = decisions.flatMap(d => d.draws); const ties = decisions.flatMap(d => d.ties);
	const base = { teams: h.teams, seed: 'sodium,' + '7'.repeat(32), choices, draws, ties, canonical: false };
	stat.battles++;
	try {
		const all = support.replay(base);
		const c = check(h, decisions, final, all, 'all', i);
		if (c.firstDiff) { stat.all.changed++; if (stat.all.examples.length < 5) stat.all.examples.push([i, c.firstDiff]); }
		if (c.extFirst) stat.all.ext_changed++;
	} catch (e) { stat.all.examples.push([i, 'error ' + e.message]); }
	const present = new Set(draws.filter(d => d.dontcare || (KINDS.includes(d.kind) && d.chance === null)).map(d => d.kind));
	for (const k of KINDS) {
		if (!present.has(k)) continue;
		stat[k].battles_with++;
		try {
			// force every other don't-care draw from the recording; free only kind k's don't-care draws
			const out = support.replay({ ...base, free_kinds: [k], force_dontcare: true,
				canonical: k === 'QUEUE_INSERT_TIE' ? false : undefined });
			const c = check(h, decisions, final, out, k, i);
			if (c.firstDiff) { stat[k].changed++; if (stat[k].examples.length < 3) stat[k].examples.push([i, c.firstDiff]); }
		} catch (e) { stat[k].examples.push([i, 'error ' + e.message]); }
	}
}
console.log(JSON.stringify({ teams, n: N, seed: S, ...stat }, null, 1));

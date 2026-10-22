#!/usr/bin/env node
'use strict';
/**
 * Merge scanner output + hand annotations -> ../rng_callsites.json, and print a markdown table.
 *   node tools/build_inventory.js            # writes rng_callsites.json, prints table
 *   node tools/build_inventory.js --table    # print table only
 */
const fs = require('fs');
const path = require('path');
const { scan, reachability } = require('./scan_callsites');
const ANN = require('./inventory_annotations');
const { PS_ROOT } = require('../lib/ps');

const { sites, legalSpeciesCount } = reachability(scan());
const missing = sites.filter(s => !ANN[s.id]);
if (missing.length) {
	console.error('UNANNOTATED call sites:'); for (const m of missing) console.error('  ', m.id, m.code);
	process.exit(1);
}
const extra = Object.keys(ANN).filter(k => !sites.some(s => s.id === k));
if (extra.length) { console.error('annotations without a call site:', extra); process.exit(1); }

const callsites = sites.map(s => {
	const a = ANN[s.id];
	const enclosing = s.file.startsWith('data/') ?
		`${s.entry}${s.sub ? '.' + s.sub : ''}.${s.handler || '?'}` : `${s.entry ? s.entry + '.' : ''}${s.fn}`;
	return {
		id: s.id, file: s.file, line: s.line, enclosing, method: s.method, args: s.args, code: s.code,
		kind: a.kind, decides: a.decides, distribution: a.dist, outcome: a.out,
		context: { actor: a.actor, target: a.target, key: ['turn', 'kind', 'actor', 'target', ...a.key] },
		reachable: a.reach, notes: a.why,
		dexAvailability: s.availability || null, dexReachable: s.reachable,
	};
});
const kinds = {};
for (const c of callsites) {
	if (c.reachable === 'no') continue;
	(kinds[c.kind] ||= []).push(c.id);
}
const out = {
	generated: new Date().toISOString(),
	ps: { root: PS_ROOT, commit: 'd601443', format: 'gen9championsvgc2026regmc', mod: 'champions' },
	legalSpeciesCount,
	summary: {
		total: callsites.length,
		reachable: callsites.filter(c => c.reachable === 'yes').length,
		conditional: callsites.filter(c => c.reachable === 'cond').length,
		unreachable: callsites.filter(c => c.reachable === 'no').length,
	},
	kinds,
	contextFields: {
		turn: 'battle.turn (choices for turn N are committed while battle.turn === N; turn 0 = leads/team preview)',
		actor: 'slot label p1a/p1b/p2a/p2b of the acting/affected mon (resolved from the context expression)',
		target: 'slot label of the other mon',
		move: 'battle.activeMove.id',
		hit: 'battle.activeMove.hit (1-based multi-hit index; 0/1 for single hit)',
		effect: 'battle.effect.id (ability/item/condition whose handler is drawing)',
		ord: 'occurrence index of an otherwise-identical key within the battle (e.g. secondary #2 of Fire Fang)',
		tied: 'sorted canonical labels of the tied elements of a speedSort shuffle (perm outcome = labels in result order)',
		event: 'id of the runEvent/fieldEvent whose handler list is being sorted',
		caller: 'speedSort caller frame',
	},
	callsites,
};

// attach observed frequencies from the latest instrumentation runs, if present
for (const [tag, f] of [['chaos', '02_instrument.json'], ['randomDoubles', '02_instrument_random.json']]) {
	const fp = path.join(__dirname, '..', '..', 'docs', 'research', 'poc', 'results', f);
	if (!fs.existsSync(fp)) continue;
	const r = JSON.parse(fs.readFileSync(fp, 'utf8'));
	out.observed = out.observed || {};
	out.observed[tag] = { battles: r.N, drawsPerBattle: r.drawsPerBattle, perBattleByKind: Object.fromEntries(Object.entries(r.kinds).map(([k, v]) => [k, v.perBattle])), sitesHit: r.sitesHit };
}

const table = () => {
	const rows = ['| # | call site | enclosing | method(args) | decides | EVENT_KIND | out | key extras | reach |', '|---|---|---|---|---|---|---|---|---|'];
	callsites.forEach((c, i) => {
		const args = c.args.length > 38 ? c.args.slice(0, 35) + '...' : c.args;
		rows.push(`| ${i + 1} | \`${c.id}\` | ${c.enclosing} | \`${c.method}(${args.replace(/\|/g, '\\|')})\` | ${c.decides} | ${c.kind} | ${c.outcome} | ${c.context.key.slice(4).join(',') || '-'} | ${c.reachable} |`);
	});
	return rows.join('\n');
};
if (!process.argv.includes('--table')) {
	fs.writeFileSync(path.join(__dirname, '..', 'data', 'rng_callsites.json'), JSON.stringify(out, null, 1));
	console.error(`wrote rng_callsites.json: ${JSON.stringify(out.summary)}`);
}
console.log(table());

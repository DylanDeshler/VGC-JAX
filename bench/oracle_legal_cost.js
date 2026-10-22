#!/usr/bin/env node
'use strict';
// Per-decision cost of acceptance enumeration (support.legalForBattle) vs a plain traced battle (M0.8).
//   node bench/oracle_legal_cost.js [N=40]
const tr = require('../oracle/lib/trace');
const N = Number(process.argv[2] || 40);
function time(legal) {
	let decisions = 0; const t0 = process.hrtime.bigint();
	for (let i = 0; i < N; i++) decisions += tr.runBattle('legal-bench', i, { legal }).filter(r => r.type === 'decision').length;
	return { ms: Number(process.hrtime.bigint() - t0) / 1e6, decisions };
}
const off = time('off'); const on = time('on');
const out = {
	battles: N, decisions: on.decisions,
	ms_per_battle_off: +(off.ms / N).toFixed(2), ms_per_battle_on: +(on.ms / N).toFixed(2),
	legal_ms_per_decision: +((on.ms - off.ms) / on.decisions).toFixed(2),
};
console.log(JSON.stringify(out));

#!/usr/bin/env node
'use strict';
// Probe (ADR-102): which core-stage entities could contribute handlers to fieldEvent('Residual'|'SwitchIn')?
// A SPEED_TIE_FIELD_EVENT record needs >= 2 such handlers with equal keys. Prints every candidate callback.
const path = require('path');
const { Dex } = require('../../lib/ps');
const dex = Dex.mod('champions');
const st = require(path.join(__dirname, '..', '..', '..', 'data', 'export', 'stages', 'core.json'));
const rx = /^on(Side|Field|Any)?(Residual|SwitchIn|Start)$/;
const found = [];
for (const [kind, list, get] of [['move', st.moves, id => dex.moves.get(id)], ['ability', st.abilities, id => dex.abilities.get(id)],
	['item', st.items, id => dex.items.get(id)], ['condition', st.conditions, id => dex.conditions.get(id)]]) {
	for (const id of list) {
		const e = get(id);
		const objs = [e, e.condition || {}];
		for (const o of objs) for (const k of Object.keys(o)) if (rx.test(k) || (k === 'duration' && o !== e)) found.push(`${kind}:${id}.${k}`);
	}
}
// Conditions the engine can create in core: none (core moves reference no conditions; statuses are never inflicted).
console.log(JSON.stringify({ core_conditions: st.conditions, candidates: found }, null, 1));

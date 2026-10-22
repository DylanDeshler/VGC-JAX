#!/usr/bin/env node
// List entities whose Champions-mod entry changes more than isNonstandard (behavior/data tweaks).
// Usage: node mod_diff.js <ps_root> <mod> <out.json>
'use strict';
const [,, psRoot, mod, outPath] = process.argv;
const path = require('path');
const fs = require('fs');
require(path.join(psRoot, 'dist/sim'));
const out = {};
for (const [file, key] of [['moves', 'Moves'], ['items', 'Items'], ['abilities', 'Abilities'], ['conditions', 'Conditions']]) {
	const p = path.join(psRoot, 'dist/data/mods', mod, file + '.js');
	if (!fs.existsSync(p)) continue;
	const table = require(p)[key];
	out[file] = {};
	for (const [id, e] of Object.entries(table)) {
		const keys = Object.keys(e).filter(k => !['inherit', 'isNonstandard'].includes(k));
		if (keys.length) out[file][id] = keys;
	}
}
fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
for (const [f, t] of Object.entries(out)) console.log(f, Object.keys(t).length, JSON.stringify(t).slice(0, 1500));

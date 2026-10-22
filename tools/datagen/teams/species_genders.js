#!/usr/bin/env node
// Dump per-species (incl. cosmetic formes) gender data for a mod: {id: {gender: 'M'|'F'|'N'|'', ratioM, ratioF}} (06 §4.0 gender fill).
// Usage: node tools/datagen/teams/species_genders.js --mod champions > genders.json
'use strict';
const {parseArgs, loadSim, sortKeys} = require('../common');

const args = parseArgs(process.argv.slice(2), [], {mod: 'champions'});
const {Dex} = loadSim();
const dex = Dex.mod(args.mod);
const out = {};
for (const sp of dex.species.all()) {
	const g = {gender: sp.gender || '', ratioM: sp.genderRatio.M, ratioF: sp.genderRatio.F};
	out[sp.id] = g;
	// cosmetic formes (Alcremie-Salted-Cream, Vivillon-Polar, ...) are not separate entries in species.all()
	for (const c of sp.cosmeticFormes || []) out[dex.species.get(c).id] ||= g;
}
process.stdout.write(JSON.stringify(sortKeys(out)) + '\n');

'use strict';
// Single place that loads the pinned Pokemon Showdown build (read-only).
const fs = require('fs');
// Pinned checkout: smogon/pokemon-showdown @ d601443 (2026-10-21), built with `node build`.
const path = require('path');
const PS_ROOT = path.resolve(process.env.PS_ROOT || path.join(__dirname, '..', 'ps'));
if (!fs.existsSync(`${PS_ROOT}/dist/sim/index.js`)) throw new Error(`PS build not found at ${PS_ROOT}; run \`make oracle\``);
const sim = require(`${PS_ROOT}/dist/sim`);
const { State } = require(`${PS_ROOT}/dist/sim/state`);
const FORMAT = 'gen9championsvgc2026regmc';

module.exports = {
	PS_ROOT, FORMAT, State,
	Battle: sim.Battle, BattleStream: sim.BattleStream, Dex: sim.Dex, PRNG: sim.PRNG,
	Teams: sim.Teams, TeamValidator: sim.TeamValidator, toID: sim.toID,
};

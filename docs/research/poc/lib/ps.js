'use strict';
// Single place that loads the pinned Pokemon Showdown build (read-only).
const fs = require('fs');
// Pinned checkout: smogon/pokemon-showdown @ d601443 (2026-10-21), built with `node build`.
const PS_ROOT = process.env.PS_ROOT ||
	['/tmp/psrepo/ps', '/tmp/psrepo-rng/ps'].find(p => fs.existsSync(`${p}/dist/sim/index.js`)) || '/tmp/psrepo/ps';
const sim = require(`${PS_ROOT}/dist/sim`);
const { State } = require(`${PS_ROOT}/dist/sim/state`);
const FORMAT = 'gen9championsvgc2026regmc';

module.exports = {
	PS_ROOT, FORMAT, State,
	Battle: sim.Battle, BattleStream: sim.BattleStream, Dex: sim.Dex, PRNG: sim.PRNG,
	Teams: sim.Teams, TeamValidator: sim.TeamValidator, toID: sim.toID,
};

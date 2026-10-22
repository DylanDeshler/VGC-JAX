#!/usr/bin/env node
'use strict';
// Stage-restricted generated teams (validated by PS TeamValidator) as JSON lines of team sets (06 §4.3 pool C/core).
//   node tools/datagen/teams/gen_stage_teams.js --stage core --n 2000 --seed 1 > data/teams/core_generated.jsonl
const path = require('path');
const lib = path.join(__dirname, '..', '..', '..', 'oracle', 'lib');
const { stageTeam } = require(path.join(lib, 'stageteams'));
const T = require(path.join(lib, 'teams'));
const { teamSets } = require(path.join(lib, 'trace'));
const { Teams } = require(path.join(lib, 'ps'));
const seeds = require(path.join(lib, 'seeds'));
const a = Object.fromEntries(process.argv.slice(2).reduce((acc, x, i, arr) => (x.startsWith('--') ? acc.concat([[x.slice(2), arr[i + 1]]]) : acc), []));
const stage = a.stage || 'core'; const n = Number(a.n || 1000); const S = a.seed || '1';
const rng = new T.HarnessRNG(seeds.u32(seeds.sha256('teambank', stage, S)));
const seen = new Set();
let made = 0;
while (made < n) {
	const team = stageTeam(stage, rng);
	const packed = Teams.pack(team);
	if (seen.has(packed)) continue;
	seen.add(packed);
	process.stdout.write(JSON.stringify({ packed, sets: teamSets(packed) }) + '\n');
	made++;
}

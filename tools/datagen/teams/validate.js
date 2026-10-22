#!/usr/bin/env node
// Validate packed teams with PS TeamValidator for a format (06 §4.3).
// Usage: node tools/datagen/teams/validate.js --format <id> < teams.txt
// stdin: one packed team per line. stdout: one JSON object per input line,
//   {"ok": true} or {"ok": false, "problems": [...]}, in input order.
'use strict';
const fs = require('fs');
const {parseArgs, loadSim} = require('../common');

const args = parseArgs(process.argv.slice(2), ['format']);
const {Dex, Teams, TeamValidator} = loadSim();
const format = Dex.formats.get(args.format);
if (!format.exists) throw new Error(`unknown format ${args.format}`);
const validator = TeamValidator.get(format);

const lines = fs.readFileSync(0, 'utf8').split('\n').filter(l => l.length);
const out = [];
for (const packed of lines) {
	const team = Teams.unpack(packed);
	const problems = team ? validator.validateTeam(team) : ['unpack failed'];
	out.push(JSON.stringify(problems ? {ok: false, problems} : {ok: true}));
}
process.stdout.write(out.join('\n') + (out.length ? '\n' : ''));

'use strict';
// RPC round-trip tests (M0.4). Run: node --test oracle/test/
const test = require('node:test');
const assert = require('node:assert');
const { handle } = require('../lib/rpc');
const tr = require('../lib/trace');
const T = require('../lib/teams');
const { Teams } = require('../lib/ps');
const seeds = require('../lib/seeds');

const call = (method, params) => {
	const r = JSON.parse(JSON.stringify(handle({ id: 1, method, params })));
	assert.ok(!r.error, r.error);
	assert.strictEqual(r.ps_commit, seeds.psCommit());
	return r.result;
};

function spec(i) {
	const sd = seeds.battleSeeds('rpc-test', i);
	const { teams } = tr.makeTeams('chaos', sd);
	return { teams: teams.map(t => Teams.pack(t)), seed: sd.psSeed, policy: 'random', policy_seed: sd.policySeed };
}
const strip = d => ({ ...d, k: 0, requests: null }); // requests embed PS-internal rqids

test('ping and validate_team', () => {
	assert.strictEqual(call('ping', {}), 'pong');
	const team = T.chaosTeam(new T.HarnessRNG(3)).team;
	assert.deepStrictEqual(call('validate_team', { team: Teams.pack(team) }).errors, []);
	const bad = team.map(s => ({ ...s, item: team[0].item }));
	assert.ok(call('validate_team', { team: bad }).errors.length > 0);
});

test('run is deterministic and replays from its own choices', () => {
	for (let i = 0; i < 5; i++) {
		const sp = spec(i);
		const a = call('run', sp); const b = call('run', sp);
		assert.deepStrictEqual(a, b);
		const choices = a.decisions.map(d => d.choices);
		const c = call('run', { ...sp, choices });
		assert.deepStrictEqual(c.decisions.map(strip), a.decisions.map(strip));
		assert.deepStrictEqual(c.final, a.final);
	}
});

test('snapshot + from_snapshot reproduce the uninterrupted battle', () => {
	for (let i = 0; i < 8; i++) {
		const sp = spec(100 + i);
		const full = call('run', sp);
		const choices = full.decisions.map(d => d.choices);
		const n = choices.length;
		for (const at of [1, Math.floor(n / 2), n - 1]) {
			const snap = call('snapshot', { ...sp, choices, at });
			const rest = call('from_snapshot', { snapshot: snap.snapshot, choices: choices.slice(at) });
			assert.deepStrictEqual(rest.decisions.map(strip), full.decisions.slice(at).map(strip), `battle ${i} at ${at}`);
			assert.deepStrictEqual(rest.final.projection, full.final.projection);
		}
	}
});

test('serve speaks line-delimited JSON', async () => {
	const { spawn } = require('child_process');
	const path = require('path');
	const p = spawn(process.execPath, [path.join(__dirname, '..', 'bin', 'oracle.js'), 'serve']);
	const lines = [];
	p.stdout.on('data', d => lines.push(...d.toString().split('\n').filter(Boolean)));
	p.stdin.write(JSON.stringify({ id: 7, method: 'ping' }) + '\n');
	p.stdin.write('not json\n');
	p.stdin.end(JSON.stringify({ id: 8, method: 'nope' }) + '\n');
	await new Promise(r => p.on('close', r));
	const rs = lines.map(l => JSON.parse(l));
	assert.deepStrictEqual(rs.map(r => r.id), [7, null, 8]);
	assert.strictEqual(rs[0].result, 'pong');
	assert.ok(rs[1].error && rs[2].error);
});

test('snapshot at every decision (incl. mid-turn SWITCH pauses) reproduces the remainder', () => {
	let midTurn = 0;
	for (let i = 0; i < 25; i++) {
		const sd = seeds.battleSeeds('rpc-test-all', i);
		const { teams } = tr.makeTeams(i % 2 ? 'chaos' : 'randdoubles', sd);
		const sp = { teams: teams.map(t => Teams.pack(t)), seed: sd.psSeed, policy: 'random-weighted', policy_seed: sd.policySeed };
		const full = call('run', sp);
		const choices = full.decisions.map(d => d.choices);
		for (let at = 1; at < choices.length; at++) {
			const d = full.decisions[at];
			if (d.phase === 'switch' && d.turn === full.decisions[at - 1].turn) midTurn++;
			const snap = call('snapshot', { ...sp, choices, at });
			const rest = call('from_snapshot', { snapshot: snap.snapshot, choices: choices.slice(at), stop_after: 2 });
			assert.deepStrictEqual(rest.decisions.map(strip), full.decisions.slice(at, at + 2).map(strip), `battle ${i} at ${at}`);
		}
	}
	assert.ok(midTurn > 0, 'corpus should contain mid-turn switch decisions');
});

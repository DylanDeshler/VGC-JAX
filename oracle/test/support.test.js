'use strict';
// Round-trip tests for the test-support RPCs (M0.8).
const test = require('node:test');
const assert = require('node:assert');
const { handle } = require('../lib/rpc');
const tr = require('../lib/trace');
const T = require('../lib/teams');
const { Teams, Battle } = require('../lib/ps');
const seeds = require('../lib/seeds');

const call = (method, params) => {
	const r = JSON.parse(JSON.stringify(handle({ id: 1, method, params })));
	assert.ok(!r.error, r.error + '\n' + r.stack);
	return r.result;
};

test('fixedpoint: trunc / modify / chain / chainFinal', () => {
	const r = call('fixedpoint', { ops: [['trunc', 7.9], ['trunc', 70000, 16], ['modify', 100, 6144, 4096], ['modify', 57, 3, 2],
		['chain', [[6144, 4096], [5324, 4096]]], ['chainFinal', 100, [[6144, 4096], [5324, 4096]]], ['modify', 1, 2732, 4096]] });
	assert.deepStrictEqual(r.length, 7);
	assert.strictEqual(r[0], 7); assert.strictEqual(r[1], 70000 % 65536);
	// self-consistency with the PS definitions as written in sim/battle.ts (not hand-computed constants)
	assert.strictEqual(r[5], call('fixedpoint', { ops: [['modify', 100, r[4], 4096]] })[0]);
});

test('stats returns PS-constructed numbers for many sets', () => {
	const rng = new T.HarnessRNG(9);
	const sets = [...T.chaosTeam(rng).team, ...T.chaosTeam(rng).team, ...T.chaosTeam(rng).team];
	const r = call('stats', { sets: sets.map(s => Teams.pack([s])) });
	assert.strictEqual(r.length, sets.length);
	for (const x of r) { assert.ok(x.maxhp > 0); assert.strictEqual(Object.keys(x.storedStats).length, 5); assert.strictEqual(x.maxpp.length, 4); }
});

test('sort_actions applies forced permutations to tie groups in PS order', () => {
	const entries = [
		{ id: 'a', order: 200, priority: 0, speed: 100 }, { id: 'b', order: 200, priority: 0, speed: 120 },
		{ id: 'c', order: 200, priority: 0, speed: 100 }, { id: 'd', order: 103, priority: 0, speed: 50 },
		{ id: 'e', order: 200, priority: 1, speed: 10 },
	];
	const id = call('sort_actions', { entries });
	assert.deepStrictEqual(id.order, ['d', 'e', 'b', 'a', 'c']);
	assert.deepStrictEqual(id.groups, [['a', 'c']]);
	const sw = call('sort_actions', { entries, perms: [[1, 0]] });
	assert.deepStrictEqual(sw.order, ['d', 'e', 'b', 'c', 'a']);
});

test('craft builds a normalised snapshot that continues like PS', () => {
	const sd = seeds.battleSeeds('craft-test', 1);
	const { teams } = tr.makeTeams('chaos', sd);
	const packed = teams.map(t => Teams.pack(t));
	const r = call('craft', { teams: packed, preview: ['team 1234', 'team 2143'], turn: 5,
		mons: [{ side: 0, b: 0, hp: 1, boosts: { atk: 2, spe: -1 } }, { side: 1, b: 0, status: 'par' }, { side: 1, b: 1, pp: [1, null, null, null] }] });
	const st = r.snapshot.state;
	assert.strictEqual(st.turn, 5);
	const sp = r.snapshot.ident[0][0].species;
	const p = st.sides[0].pokemon.find(x => x.set.species === sp);
	assert.strictEqual(p.hp, 1); assert.strictEqual(p.boosts.atk, 2); assert.strictEqual(p.boosts.spe, -1);
	const cont = call('from_snapshot', { snapshot: r.snapshot, policy: 'random', policy_seed: 3, stop_after: 2 });
	assert.ok(cont.decisions.length >= 1);
	assert.strictEqual(cont.decisions[0].turn, 5);
});

test('trace_events returns the event dispatch log of one decision', () => {
	const sd = seeds.battleSeeds('trace-ev', 2);
	const { teams } = tr.makeTeams('chaos', sd);
	const spec = { teams: teams.map(t => Teams.pack(t)), seed: sd.psSeed, policy: 'random', policy_seed: 5 };
	const full = call('run', spec);
	const choices = full.decisions.map(d => d.choices);
	const snap = call('snapshot', { ...spec, choices, at: 1 });
	const d1 = full.decisions[1];
	const r = call('trace_events', { snapshot: snap.snapshot, choices: [d1.choices], draws: d1.draws, ties: d1.ties });
	assert.ok(r.events.length > 10);
	assert.ok(r.events.some(e => e.eventid === 'BeforeMove' || e.eventid === 'SwitchIn'));
	assert.deepStrictEqual(r.decision.draws, d1.draws);
	assert.strictEqual(r.forcing.missed.length, 0);
});

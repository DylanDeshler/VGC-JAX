'use strict';
/**
 * JSON-RPC methods (02 §4.2). Request: {id, method, params}. Response: {id, ps_commit, oracle_git, result | error}.
 *
 * Battle spec: {teams: [packed, packed], seed: "sodium,<hex32>", formatid?, policy?, policy_seed?,
 *               choices?: [[p1choice|null, p2choice|null], ...], classifier?}
 */
const { Teams, TeamValidator, FORMAT } = require('./ps');
const T = require('./teams');
const tr = require('./trace');
const seeds = require('./seeds');
const { getPolicy } = require('./policy');

function unpackTeam(t) { return typeof t === 'string' ? Teams.unpack(t) : t; }

function startBattle(spec) {
	const { battle, prng, ident } = tr.newBattle({
		seed: spec.seed, formatid: spec.formatid || FORMAT, classifier: spec.classifier || 'markers',
	});
	battle.setPlayer('p1', { name: 'p1', team: unpackTeam(spec.teams[0]) });
	battle.setPlayer('p2', { name: 'p2', team: unpackTeam(spec.teams[1]) });
	return { battle, prng, ident };
}

function chooser(spec) {
	const choose = getPolicy(spec.policy || 'random');
	const rng = new T.HarnessRNG(spec.policy_seed ?? 1);
	return req => choose(req, rng);
}

const METHODS = {
	ping: () => 'pong',

	/** Full trace records for one battle (script mode when `choices` is given). */
	run(spec) {
		const b = startBattle(spec);
		const out = tr.driveTraced({ ...b, choose: chooser(spec), script: spec.choices || null,
			stopAfter: spec.choices ? spec.choices.length : Infinity });
		return { decisions: out.decisions, final: out.final, rejected: out.rejected };
	},

	/** Serialized PS state at decision index `at` (before that decision's choices). */
	snapshot(spec) {
		const at = spec.at ?? 0;
		const b = startBattle(spec);
		tr.driveTraced({ ...b, choose: chooser(spec), script: spec.choices || null, stopAfter: at });
		return { k: at, snapshot: tr.snapshotOf(b.battle), ended: b.battle.ended };
	},

	/** Continue from a snapshot: trace records for the following decisions. */
	from_snapshot(params) {
		const b = tr.fromSnapshot(params.snapshot, { classifier: params.classifier || 'markers' });
		const out = tr.driveTraced({ ...b, choose: chooser(params), script: params.choices || null,
			stopAfter: params.stop_after ?? (params.choices ? params.choices.length : Infinity) });
		return { decisions: out.decisions, final: out.final, rejected: out.rejected };
	},

	validate_team(params) {
		const team = unpackTeam(params.team);
		const errors = TeamValidator.get(params.formatid || FORMAT).validateTeam(team);
		return { errors: errors || [] };
	},
};

function handle(req) {
	const base = { id: req.id ?? null, ps_commit: seeds.psCommit(), oracle_git: seeds.oracleGit() };
	const fn = METHODS[req.method];
	if (!fn) return { ...base, error: `unknown method ${req.method}` };
	try {
		return { ...base, result: fn(req.params || {}) };
	} catch (e) {
		return { ...base, error: e.message, stack: e.stack };
	}
}

module.exports = { handle, METHODS };

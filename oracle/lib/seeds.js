'use strict';
/**
 * Deterministic per-battle seeds (07 §2.1, ADR-030). Every hash input is the '|'-joined string of its parts.
 *   PS seed     = "sodium," + hex(sha256("battle|S|i"))[:32]
 *   team seed   = sha256("team|S|i")      policy seed = sha256("policy|S|i")
 *   battle_id   = hex(sha256("S|i|ps_commit|oracle_git"))[:16]
 */
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const sha256 = (...parts) => crypto.createHash('sha256').update(parts.map(String).join('|')).digest('hex');
const u32 = hex => parseInt(hex.slice(0, 8), 16) >>> 0 || 1;

let _psCommit = null; let _oracleGit = null;
function psCommit() {
	if (_psCommit) return _psCommit;
	const { PS_ROOT } = require('./ps');
	try {
		_psCommit = execFileSync('git', ['-C', PS_ROOT, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
	} catch {
		_psCommit = 'unknown';
	}
	return _psCommit;
}
/** Hash of oracle/lib/** (ADR-030 `oracle_git`): content-addressed, independent of git state. */
function oracleGit() {
	if (_oracleGit) return _oracleGit;
	const dir = path.join(__dirname);
	const h = crypto.createHash('sha256');
	for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.js')).sort()) {
		h.update(f); h.update('\0'); h.update(fs.readFileSync(path.join(dir, f))); h.update('\0');
	}
	h.update(fs.readFileSync(path.join(__dirname, '..', 'data', 'rng_callsites.json')));
	_oracleGit = h.digest('hex').slice(0, 16);
	return _oracleGit;
}
const CONTRACT_VERSION = 2; // docs/design/contract/drawkey.md `contract_version`

function battleSeeds(S, i) {
	return {
		psSeed: 'sodium,' + sha256('battle', S, i).slice(0, 32),
		teamSeed: u32(sha256('team', S, i)),
		teamHex: sha256('team', S, i),
		policySeed: u32(sha256('policy', S, i)),
		battleId: sha256(S, i, psCommit(), oracleGit()).slice(0, 16),
	};
}

module.exports = { sha256, u32, psCommit, oracleGit, battleSeeds, CONTRACT_VERSION };

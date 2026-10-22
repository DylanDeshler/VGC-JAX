'use strict';
// Shared helpers for tools/datagen/*.js: CLI args, pinned-PS resolution, deterministic JSON output.
const fs = require('fs');
const path = require('path');
const {execFileSync} = require('child_process');

const REPO_ROOT = path.resolve(__dirname, '..', '..');

/** Resolve the PS checkout: env PS_ROOT, else the pinned submodule at oracle/ps. */
function psRoot() {
	const root = path.resolve(process.env.PS_ROOT || path.join(REPO_ROOT, 'oracle', 'ps'));
	if (!fs.existsSync(path.join(root, 'dist', 'sim', 'index.js'))) {
		throw new Error(`PS build not found at ${root}; run \`make oracle\``);
	}
	return root;
}

function loadSim(root = psRoot()) {
	return require(path.join(root, 'dist', 'sim'));
}

/** Short commit hash of the PS checkout ('unknown' if git is unavailable). */
function psCommit(root = psRoot()) {
	try {
		return execFileSync('git', ['-C', root, 'rev-parse', '--short=7', 'HEAD'], {encoding: 'utf8'}).trim();
	} catch (e) {
		return 'unknown';
	}
}

/** Parse `--key value` pairs; `required` lists keys that must be present. */
function parseArgs(argv, required, defaults = {}) {
	const out = {...defaults};
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (!a.startsWith('--')) throw new Error(`unexpected argument ${a}`);
		const key = a.slice(2);
		const val = argv[i + 1];
		if (val === undefined || val.startsWith('--')) throw new Error(`missing value for ${a}`);
		out[key] = val;
		i++;
	}
	const missing = required.filter(k => out[k] === undefined);
	if (missing.length) throw new Error(`missing required args: ${missing.map(k => '--' + k).join(' ')}`);
	return out;
}

/** Recursively sort object keys (arrays keep their order). */
function sortKeys(v) {
	if (Array.isArray(v)) return v.map(sortKeys);
	if (v && typeof v === 'object') {
		const o = {};
		for (const k of Object.keys(v).sort()) o[k] = sortKeys(v[k]);
		return o;
	}
	return v;
}

/** Write JSON with sorted keys and a trailing newline. indent=null -> compact. */
function writeJSON(file, obj, indent = 1) {
	fs.mkdirSync(path.dirname(path.resolve(file)), {recursive: true});
	fs.writeFileSync(file, JSON.stringify(sortKeys(obj), null, indent === null ? undefined : indent) + '\n');
}

module.exports = {REPO_ROOT, psRoot, loadSim, psCommit, parseArgs, sortKeys, writeJSON};

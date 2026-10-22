'use strict';
/**
 * Call-site capture for PRNG draws: structured V8 CallSites (no string formatting), mapped from
 * dist/*.js back to the TypeScript source with the build's source maps, cached per position.
 */
const fs = require('fs');
const path = require('path');
const { PS_ROOT } = require('./ps');
const { SourceMapConsumer } = require('source-map');

const PS_REAL = fs.realpathSync(PS_ROOT);
const DIST = `${PS_REAL}/dist/`;
const consumers = new Map();
const posCache = new Map();

function consumerFor(file) {
	let c = consumers.get(file);
	if (c === undefined) {
		const mapFile = file + '.map';
		c = fs.existsSync(mapFile) ? new SourceMapConsumer(JSON.parse(fs.readFileSync(mapFile, 'utf8'))) : null;
		consumers.set(file, c);
	}
	return c;
}

/** Map a dist position to "sim/battle.ts:2409" (TS path relative to PS root). */
function mapPos(file, line, col) {
	const k = `${file}:${line}:${col}`;
	let r = posCache.get(k);
	if (r !== undefined) return r;
	r = null;
	if (file && file.startsWith(DIST)) {
		const c = consumerFor(file);
		if (c) {
			const p = c.originalPositionFor({ line, column: col - 1 });
			if (p.source) {
				const abs = path.resolve(path.dirname(file), p.source);
				r = `${path.relative(PS_REAL, abs)}:${p.line}`;
			}
		}
		if (!r) r = `dist/${file.slice(DIST.length)}:${line}`;
	}
	posCache.set(k, r);
	return r;
}

const PRNG_PLUMBING = new Set(['random', 'randomChance', 'sample', 'shuffle']);
const prepare = (_, cs) => cs;

/**
 * Returns up to `depth` mapped frames starting at the first frame outside the PRNG plumbing:
 * [site, caller, caller's caller, ...] e.g. ['sim/battle.ts:461', 'sim/battle-queue.ts:426', ...]
 * Also returns the function names.
 */
function captureSite(depth = 3) {
	const origPrep = Error.prepareStackTrace; const origLimit = Error.stackTraceLimit;
	Error.prepareStackTrace = prepare; Error.stackTraceLimit = 14;
	const holder = {};
	Error.captureStackTrace(holder, captureSite);
	const cs = holder.stack;
	Error.prepareStackTrace = origPrep; Error.stackTraceLimit = origLimit;
	const frames = []; const fns = [];
	for (const f of cs) {
		const file = f.getFileName();
		if (!file || !file.startsWith(DIST)) continue; // our harness frames, node internals
		const fn = f.getFunctionName() || f.getMethodName() || '';
		if (file.endsWith('/sim/prng.js')) continue;
		// Battle#random / randomChance / sample wrappers
		if (!frames.length && file.endsWith('/sim/battle.js') && PRNG_PLUMBING.has(fn)) continue;
		frames.push(mapPos(file, f.getLineNumber(), f.getColumnNumber()));
		fns.push(fn);
		if (frames.length >= depth) break;
	}
	return { frames, fns };
}

/** Cheaper variant for benchmarking: capture + format the string stack (what Error().stack costs). */
function captureStringStack() {
	return new Error().stack;
}

module.exports = { captureSite, captureStringStack, mapPos };

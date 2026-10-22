'use strict';
/**
 * InstrumentedPRNG: drop-in replacement for battle.prng (pass as `new Battle({prng})`).
 *
 * Records every draw at the *decision* level (int for random, bool for randomChance, index for
 * sample, permutation for shuffle), with optional context + call-site + classification, and can
 * answer draws instead of the real PRNG:
 *   mode 'record'   - real PRNG (consumption identical to PS's own PRNG), record everything
 *   mode 'replay'   - Mode 1: answer from a recorded draw list in stream order
 *   mode 'keyed'    - Mode 2: answer from a dict keyed by semantic context key (see classify.js)
 *   mode 'override' - Mode 3: policy(draw) returns an outcome or undefined (=> real PRNG)
 */
const { PRNG } = require('./ps');
const { captureSite } = require('./callsite');
const context = require('./context');
const classify = require('./classify');

class InstrumentedPRNG {
	constructor({
		seed = null, mode = 'record', ctx = true, stack = true, markers = true, classifier = 'stack',
		script = null, dict = null, keyMode = 'step+ord', ignoreKinds = null, fallbackSeed = null, policy = null, keep = true,
	} = {}) {
		this.inner = new PRNG(seed);
		this.startingSeed = this.inner.startingSeed;
		this.mode = mode; this.ctx = ctx; this.stack = stack; this.markers = markers; this.classifier = classifier;
		this.script = script; this.dict = dict; this.keyMode = keyMode; this.ignoreKinds = ignoreKinds; this.policy = policy;
		this.fallback = fallbackSeed ? new PRNG(fallbackSeed) : this.inner; // answers for misses / ignored kinds
		this.keep = keep;
		this.draws = []; this.n = 0;
		this.stats = { misses: 0, desync: 0, forced: 0, ignored: 0, disagree: 0 };
		this.ordCount = new Map();
		this.battle = null;
	}

	attach(battle) {
		this.battle = battle;
		if (this.ctx) {
			context.install(battle);
			if (this.markers) classify.installMarkers(battle);
		}
		return this;
	}

	// ---- PRNG interface used by PS
	getSeed() { return this.inner.getSeed(); }
	setSeed(seed) { this.inner.setSeed(seed); }
	clone() { return this.inner.clone(); }
	get rng() { return this.inner.rng; }

	random(from, to) {
		const args = to === undefined ? (from === undefined ? [] : [from]) : [from, to];
		const d = this._begin('random', args, null);
		let out = this._answer(d);
		if (out !== undefined) {
			const lo = to === undefined ? 0 : Math.floor(from); const hi = to === undefined ? from : to;
			if (from !== undefined && !(Number.isInteger(out) && out >= lo && out < hi)) { this.stats.misses++; out = undefined; }
		}
		if (out === undefined) out = this._src(d).random(from, to);
		return this._end(d, out);
	}

	randomChance(numerator, denominator) {
		const d = this._begin('randomChance', [numerator, denominator], null);
		let out = this._answer(d);
		if (typeof out !== 'boolean') out = this._src(d).random(denominator) < numerator;
		return this._end(d, out);
	}

	sample(items) {
		if (items.length === 0) throw new RangeError(`Cannot sample an empty array`);
		const d = this._begin('sample', [items], null);
		let idx = this._answer(d);
		if (!(Number.isInteger(idx) && idx >= 0 && idx < items.length)) {
			if (idx !== undefined) this.stats.misses++;
			idx = this._src(d).random(items.length);
		}
		this._end(d, idx);
		return items[idx];
	}

	shuffle(items, start = 0, end = items.length) {
		const n = end - start;
		if (n < 2) return;
		const tied = items.slice(start, end);
		const d = this._begin('shuffle', [n], tied);
		let perm = this._answer(d);
		if (!(Array.isArray(perm) && perm.length === n)) {
			if (perm !== undefined) this.stats.misses++;
			// Fisher-Yates with identical PRNG consumption to PRNG#shuffle
			perm = [...Array(n).keys()];
			const src = this._src(d);
			for (let s = 0; s < n - 1; s++) {
				const j = src.random(start + s, end) - start;
				if (j !== s) { const t = perm[s]; perm[s] = perm[j]; perm[j] = t; }
			}
		}
		for (let k = 0; k < n; k++) items[start + k] = tied[perm[k]];
		this._end(d, perm);
	}

	// ---- internals
	_begin(method, args, tied) {
		const battle = this.battle;
		const d = { i: this.n++, method, args: method === 'sample' ? [args[0].length] : args, turn: battle ? battle.turn : -1 };
		if (battle) d.logPos = battle.log.length;
		if (method === 'sample') d.items = args[0];
		if (tied) { d.tiedObjs = tied; d.tied = tied.map(context.elemLabel); }
		if (this.ctx && battle && battle.__ctx) {
			d.ctx = context.snapshot(battle);
			if (this.stack) { const s = captureSite(3); d.frames = s.frames; d.fns = s.fns; }
			const callArgs = method === 'sample' ? [args[0]] : args;
			const dd = { method, args: callArgs, frames: d.frames, fns: d.fns, tied: d.tied, tiedObjs: tied };
			let c;
			if (this.classifier === 'stack' || this.classifier === 'both') c = classify.classifyByStack(dd, battle);
			if (this.classifier === 'markers' || this.classifier === 'both') {
				const m = classify.classifyByMarkers(dd, battle);
				if (c) { d.markerKind = m.kind; d.markerKey = m.key; if (m.kind !== c.kind || m.key !== c.key) this.stats.disagree++; } else c = m;
			}
			if (c) {
				d.kind = c.kind; d.key = c.key; d.keyNaive = c.keyNaive; d.actor = c.actor; d.target = c.target; d.site = c.site;
				const k = d.key; const o = this.ordCount.get(k) || 0; this.ordCount.set(k, o + 1); d.ord = o;
			}
		}
		return d;
	}

	_src(d) {
		if (this.ignoreKinds && d.kind && this.ignoreKinds.has(d.kind)) return this.fallback;
		if (this.mode === 'keyed' && d._miss) return this.fallback;
		return this.inner;
	}

	_answer(d) {
		switch (this.mode) {
		case 'record': return undefined;
		case 'replay': {
			const r = this.script[d.i];
			if (!r || r.method !== d.method || (d.method !== 'sample' && d.method !== 'shuffle' && String(r.args) !== String(d.args)) ||
				(d.method === 'sample' && r.args[0] !== d.args[0])) {
				this.stats.desync++; return undefined;
			}
			this.stats.forced++; return r.out;
		}
		case 'keyed': {
			if (this.ignoreKinds && this.ignoreKinds.has(d.kind)) { this.stats.ignored++; return undefined; }
			const v = this.dict.get(keyFor(d, this.keyMode));
			if (v === undefined) { this.stats.misses++; d._miss = true; return undefined; }
			this.stats.forced++;
			if (d.method === 'shuffle') { // stored as labels in result order -> permutation of current tied list
				const pos = new Map(); d.tied.forEach((l, i) => { if (!pos.has(l)) pos.set(l, i); });
				if (pos.size !== d.tied.length) { this.stats.misses++; d._miss = true; return undefined; }
				const perm = v.map(l => pos.get(l));
				if (perm.some(x => x === undefined)) { this.stats.misses++; d._miss = true; return undefined; }
				return perm;
			}
			return v;
		}
		case 'override': {
			const v = this.policy(d, this.battle);
			if (v !== undefined) this.stats.forced++;
			return v;
		}
		}
		return undefined;
	}

	_end(d, out) {
		d.out = out;
		if (d.method === 'shuffle') d.outLabels = out.map(k => d.tied[k]);
		delete d.tiedObjs; delete d.items;
		if (this.keep) this.draws.push(d);
		return out;
	}
}

/**
 * Key variants for Mode 2:
 *   'naive'     turn|kind|actor|target|extras
 *   'step'      turn|step|kind|actor|target|extras      (step = queue action running this turn)
 *   'step+ord'  ... + '#'+occurrence index of that key within the battle
 */
// kinds whose within-step ordinal is engine-observable (index of the secondary in the move's list, Moody's 2nd draw)
const ORD_KINDS = new Set(['SECONDARY', 'ABILITY_MOODY_STAT']);
function keyFor(d, keyMode) {
	if (keyMode === 'naive') return d.keyNaive;
	if (keyMode === 'step') return d.key;
	if (keyMode === 'engine') return ORD_KINDS.has(d.kind) ? `${d.key}#${d.ord}` : d.key;
	return `${d.key}#${d.ord}`;
}

/** Build a Mode-2 dictionary from a recording. Returns {dict, collisions, conflicts}. */
function buildKeyedDict(draws, { keyMode = 'step+ord' } = {}) {
	const dict = new Map(); const seen = new Map(); let collisions = 0; let conflicts = 0; const conflictKinds = {};
	for (const d of draws) {
		if (!d.key) continue;
		const key = keyFor(d, keyMode);
		const val = d.method === 'shuffle' ? d.outLabels : d.out;
		if (dict.has(key)) {
			collisions++;
			if (JSON.stringify(seen.get(key)) !== JSON.stringify(val)) { conflicts++; conflictKinds[d.kind] = (conflictKinds[d.kind] || 0) + 1; }
			continue;
		}
		dict.set(key, val); seen.set(key, val);
	}
	return { dict, collisions, conflicts, conflictKinds };
}

module.exports = { InstrumentedPRNG, buildKeyedDict, keyFor };

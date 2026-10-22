'use strict';
/**
 * Minimal harness around the built Pokemon Showdown sim (read-only use of /tmp/psrepo/ps/dist).
 *  - makeBattle(): builds a [Gen 9 Champions] VGC 2026 Reg M-C battle with fixed teams, finishes team preview.
 *  - instrumentPRNG(): replaces battle.prng with a proxy that logs every draw (with the calling PS function)
 *    and lets a policy force individual outcomes.
 *  - traceEvents(): monkeypatches Battle.prototype event functions to log every event dispatch.
 */
const PS = process.env.PS_DIST || '/tmp/psrepo-move/ps/dist/sim';
const Sim = require(PS);
const { Battle, Dex } = Sim;

const FORMAT = 'gen9championsvgc2026regmc';

function makeBattle(p1team, p2team, opts = {}) {
	const battle = new Battle({ formatid: opts.formatid || FORMAT, seed: opts.seed || 'gen5,0123456789abcdef' });
	battle.setPlayer('p1', { name: 'P1', team: p1team });
	battle.setPlayer('p2', { name: 'P2', team: p2team });
	const pick = opts.pick || 'team 1234';
	if (opts.prePreview) opts.prePreview(battle);
	battle.makeChoices(pick, pick);
	return battle;
}

/** Extract PS function names from the current stack (outermost-last), skipping harness frames. */
function psCallers(depth = 6) {
	const stack = new Error().stack.split('\n').slice(1);
	const names = [];
	for (const line of stack) {
		if (line.includes('harness.js')) continue;
		const m = line.match(/at (?:[A-Za-z_$][\w$]*\.)?([\w$<>.]+) \(([^)]+)\)/) || line.match(/at ([^ ]+)$/);
		if (!m) continue;
		const fn = m[1];
		if (['PRNG.random', 'random', 'randomChance', 'sample', 'shuffle'].includes(fn) && /prng\.js/.test(line)) continue;
		let where = '';
		const loc = (m[2] || '').match(/(dist\/[^:]+):(\d+)/);
		if (loc) where = `${loc[1].replace('dist/', '')}:${loc[2]}`;
		names.push(where ? `${fn}@${where}` : fn);
		if (names.length >= depth) break;
	}
	return names;
}

/**
 * Replace battle.prng with a logging/forcing proxy.
 * policy(draw) may return a number to force the raw result of random(n) / random(m,n)
 * (i.e. the integer that PRNG.random would return); return undefined to use the real PRNG.
 */
function instrumentPRNG(battle, policy = () => undefined) {
	const real = battle.prng;
	const log = [];
	const proxy = {
		get startingSeed() { return real.startingSeed; },
		get rng() { return real.rng; },
		getSeed: () => real.getSeed(),
		setSeed: s => real.setSeed(s),
		clone: () => real.clone(),
		random(from, to) {
			const callers = psCallers();
			const draw = { i: log.length, from, to, callers, turn: battle.turn };
			let forced = policy(draw);
			let result;
			if (forced !== undefined) {
				result = forced;
				real.random(from, to); // keep the underlying stream advancing identically
				draw.forced = true;
			} else {
				result = real.random(from, to);
			}
			draw.result = result;
			log.push(draw);
			return result;
		},
		randomChance(numerator, denominator) {
			return this.random(denominator) < numerator;
		},
		sample(items) {
			return items[this.random(items.length)];
		},
		shuffle(items, start = 0, end = items.length) {
			while (start < end - 1) {
				const nextIndex = this.random(start, end);
				if (start !== nextIndex) [items[start], items[nextIndex]] = [items[nextIndex], items[start]];
				start++;
			}
		},
	};
	battle.prng = proxy;
	return log;
}

/** Monkeypatch event dispatchers. Returns {log, restore}. */
function traceEvents(filter = () => true) {
	const proto = Battle.prototype;
	const orig = {
		runEvent: proto.runEvent, singleEvent: proto.singleEvent, eachEvent: proto.eachEvent,
		fieldEvent: proto.fieldEvent, priorityEvent: proto.priorityEvent,
	};
	const log = [];
	let depth = 0;
	const nm = x => {
		if (x === null || x === undefined) return '-';
		if (Array.isArray(x)) return '[' + x.map(nm).join(',') + ']';
		if (typeof x === 'string') return x;
		if (x.constructor && x.constructor.name === 'Pokemon') return `${x.side.id}${'ab'[x.position]}:${x.name}`;
		if (x.constructor && x.constructor.name === 'Side') return `side:${x.id}`;
		if (x.constructor && x.constructor.name === 'Field') return 'field';
		if (x.constructor && x.constructor.name === 'Battle') return 'battle';
		if (x.effectType) return `${x.effectType}:${x.id || x.name}`;
		return typeof x;
	};
	proto.runEvent = function (eventid, target, source, sourceEffect, relayVar, onEffect, fastExit) {
		const handlers = this.findEventHandlers(target || this, eventid, source && source.constructor.name === 'Pokemon' ? source : null);
		const entry = { depth, kind: fastExit ? 'priorityEvent' : 'runEvent', eventid, target: nm(target), source: nm(source),
			effect: nm(sourceEffect), handlers: handlers.filter(h => h.callback !== undefined).map(h => nm(h.effect)), turn: this.turn };
		if (filter(entry)) log.push(entry);
		depth++;
		try {
			return orig.runEvent.call(this, eventid, target, source, sourceEffect, relayVar, onEffect, fastExit);
		} finally { depth--; }
	};
	proto.singleEvent = function (eventid, effect, state, target, source, sourceEffect, relayVar, customCallback) {
		const has = customCallback !== undefined || (effect && effect[`on${eventid}`] !== undefined);
		const entry = { depth, kind: 'singleEvent', eventid, target: nm(target), source: nm(source), effect: nm(effect),
			handlers: has ? [nm(effect)] : [], turn: this.turn };
		if (filter(entry)) log.push(entry);
		depth++;
		try {
			return orig.singleEvent.call(this, eventid, effect, state, target, source, sourceEffect, relayVar, customCallback);
		} finally { depth--; }
	};
	proto.eachEvent = function (eventid, effect, relayVar) {
		const entry = { depth, kind: 'eachEvent', eventid, target: '-', source: '-', effect: nm(effect), handlers: [], turn: this.turn };
		if (filter(entry)) log.push(entry);
		depth++;
		try { return orig.eachEvent.call(this, eventid, effect, relayVar); } finally { depth--; }
	};
	proto.fieldEvent = function (eventid, targets) {
		const entry = { depth, kind: 'fieldEvent', eventid, target: nm(targets), source: '-', effect: '-', handlers: [], turn: this.turn };
		if (filter(entry)) log.push(entry);
		depth++;
		try { return orig.fieldEvent.call(this, eventid, targets); } finally { depth--; }
	};
	return {
		log,
		restore() { Object.assign(proto, orig); },
	};
}

/** Helper to build a set. Champions: evs are Stat Points (0..32), level 50, ivs ignored. */
function set(species, item, ability, moves, nature = 'Hardy', evs = {}, extra = {}) {
	return Object.assign({
		name: '', species, item, ability, moves, nature, gender: '', level: 50,
		evs: Object.assign({ hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, evs),
		ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
	}, extra);
}

function lastLines(battle, n = 40) {
	return battle.log.slice(-n).join('\n');
}

module.exports = { Sim, Battle, Dex, FORMAT, makeBattle, instrumentPRNG, traceEvents, set, psCallers, lastLines };

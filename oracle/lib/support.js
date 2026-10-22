'use strict';
/**
 * Test-support RPC methods (02 §4.2, M0.8): fixedpoint, stats, sort_actions, legal, replay, craft, damage.
 * All expectations for engine unit/golden tests come from here (never hand-written).
 */
const { Battle, State, Teams, Dex, FORMAT } = require('./ps');
const tr = require('./trace');
const { InstrumentedPRNG } = require('./rng');

const ZERO_SEED = 'sodium,' + '0'.repeat(32);
const unpack = t => (typeof t === 'string' ? Teams.unpack(t) : t);

function dummyBattle(formatid = FORMAT) {
	return new Battle({ formatid, seed: ZERO_SEED });
}

// ------------------------------------------------------------------ fixedpoint
/**
 * ops: [['trunc', num, bits], ['modify', value, num, den], ['chain', [[num, den], ...]] -> int 4096-scale modifier,
 *       ['chainFinal', value, [[num, den], ...]] -> finalModify(value) after chaining]
 * PS: sim/dex.ts:365 trunc; sim/battle.ts chainModify / modify / finalModify.
 */
function fixedpoint({ ops }) {
	const b = dummyBattle();
	return ops.map(([op, ...a]) => {
		switch (op) {
		case 'trunc': return b.trunc(a[0], a[1] || 0);
		case 'modify': return b.modify(a[0], a[1], a[2] ?? 1);
		case 'chain': case 'chainFinal': {
			b.event = { modifier: 1 };
			for (const [n, d] of (op === 'chain' ? a[0] : a[1])) b.chainModify(n, d ?? 1);
			if (op === 'chain') return b.trunc(b.event.modifier * 4096);
			return b.finalModify(a[0]);
		}
		default: throw new Error(`fixedpoint: unknown op ${op}`);
		}
	});
}

// ------------------------------------------------------------------ stats
/**
 * sets -> [{storedStats, maxhp, maxpp:[...], speed, types, weighthg, canMegaEvo, getStat?, actionSpeed?}] from
 * constructed PS Pokemon. Optional per-set `mods[i] = {boosts, status, trickroom}` adds getStat (all 5 stats,
 * sim/pokemon.ts:617-660) and getActionSpeed (sim/pokemon.ts:662-670) under those conditions.
 */
function stats({ sets, mods = [], formatid }) {
	const out = [];
	const per = mods.length ? 2 : 6; // with mods the pair is started so both are active (status handlers need it)
	for (let i = 0; i < sets.length; i += per) {
		const chunk = sets.slice(i, i + per).map(unpack).map(s => (Array.isArray(s) ? s[0] : s));
		const b = new Battle({ formatid: formatid || FORMAT, seed: ZERO_SEED });
		const team = mods.length ? [...chunk, chunk[0], chunk[0], chunk[0]].slice(0, 4) : chunk;
		b.setPlayer('p1', { name: 'a', team });
		b.setPlayer('p2', { name: 'b', team: team.slice(0, 4) });
		if (mods.length) b.makeChoices('team 1234', 'team 1234');
		const mons = mods.length ? b.sides[0].active.slice(0, chunk.length) : b.sides[0].pokemon;
		mons.forEach((p, j) => {
			const r = {
				storedStats: { ...p.storedStats }, maxhp: p.maxhp, maxpp: p.baseMoveSlots.map(m => m.maxpp),
				speed: p.speed, types: p.getTypes(true), weighthg: p.weighthg, canMegaEvo: p.canMegaEvo || null,
			};
			const m = mods[i + j];
			if (m) {
				p.boosts = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0, ...(m.boosts || {}) };
				if (m.status) { p.status = m.status; p.statusState = b.initEffectState({ id: m.status, target: p }); }
				if (m.trickroom) b.field.pseudoWeather.trickroom = b.initEffectState({ id: 'trickroom', duration: 5 });
				r.getStat = Object.fromEntries(['atk', 'def', 'spa', 'spd', 'spe'].map(s => [s, p.getStat(s)]));
				r.actionSpeed = p.getActionSpeed();
				delete b.field.pseudoWeather.trickroom;
				p.status = ''; p.statusState = b.initEffectState({ id: '', target: p });
			}
			out.push(r);
		});
	}
	return out;
}

// ------------------------------------------------------------------ typechart
/** cases: [[attackType | immunityKey, [defTypes...]]] -> [{eff, immune}] via PS Dex.getEffectiveness/getImmunity. */
function typechart({ cases }) {
	const dex = Dex.forFormat(FORMAT);
	return cases.map(([a, types]) => ({
		eff: dex.types.get(a).exists ? dex.getEffectiveness(a, types) : null,
		immune: !dex.getImmunity(a, types),
	}));
}

// ------------------------------------------------------------------ sort_actions
/**
 * entries: [{id, order, priority, speed, subOrder, effectOrder}], perms: list of permutations to apply to tie groups
 * in the order PS's selection sort meets them (missing -> identity). Returns {order: [ids], groups: [[ids before shuffle]]}.
 * PS: sim/battle.ts speedSort / comparePriority.
 */
function sortActions({ entries, perms = [] }) {
	const b = dummyBattle();
	const groups = []; let gi = 0;
	b.prng = {
		shuffle(items, start = 0, end = items.length) {
			const tied = items.slice(start, end);
			groups.push(tied.map(x => x.id));
			const perm = perms[gi++] || tied.map((_, k) => k);
			for (let k = 0; k < tied.length; k++) items[start + k] = tied[perm[k]];
		},
	};
	const list = entries.map(e => ({ ...e }));
	b.speedSort(list);
	return { order: list.map(e => e.id), groups };
}

// ------------------------------------------------------------------ legal (acceptance enumeration, 07 §6)
const CHOOSABLE = new Set(['normal', 'any', 'adjacentFoe', 'adjacentAlly', 'adjacentAllyOrSelf']);
/**
 * PS choice string for engine slot action a (04 §4.3 encoding) given the request, or null if the encoding has no
 * string for it (non-choosable target with t != 0, missing move, mega not offered...).
 */
function slotActionString(req, slot, a, brought2pos) {
	if (a === 28) return 'pass';
	if (a >= 24) {
		const pos = brought2pos[a - 24];
		return pos === undefined || pos < 0 ? null : `switch ${pos + 1}`;
	}
	const act = req.active && req.active[slot];
	if (!act) return null;
	const m = Math.floor(a / 6); const t = Math.floor((a % 6) / 2); const mega = a % 2;
	const mv = act.moves[m];
	if (!mv) return null;
	const locked = act.moves.length === 1 && (mv.target === undefined || mv.id === 'struggle');
	if (locked) return a === 0 ? 'move 1' : null;
	if (mega && !act.canMegaEvo) return null;
	let loc = null;
	if (CHOOSABLE.has(mv.target)) {
		const ally = -(2 - slot); const self = -(slot + 1);
		if (mv.target === 'adjacentAllyOrSelf') loc = t === 0 ? self : t === 2 ? ally : null;
		else loc = t === 0 ? 1 : t === 1 ? 2 : ally;
		if (loc === null) return null;
	} else if (t !== 0) {
		return null;
	}
	return `move ${m + 1}${loc === null ? '' : ' ' + loc}${mega ? ' mega' : ''}`;
}

/**
 * Accepted joint actions for side `sideIdx` of a battle (fresh deserialized clones; side.choose never commits).
 * Returns {pairs: [[a0, a1]...], unavailable: [[a0,a1]...], kind}. Preview returns {kind: 'team', all: true}.
 */
function legalForBattle(battle, sideIdx) {
	const side = battle.sides[sideIdx];
	const req = side.activeRequest;
	if (!req || req.wait) return { kind: 'wait', pairs: [[28, 28]], unavailable: [] };
	if (req.teamPreview) return { kind: 'team', all: true, pairs: [], unavailable: [] };
	const ident = battle.__ident;
	const brought2pos = [0, 1, 2, 3].map(b => side.pokemon.findIndex(p => ident.b(p) === b));
	const snap = JSON.stringify(State.serializeBattle(battle));
	let clone = State.deserializeBattle(JSON.parse(snap));
	const nslots = side.active.length;
	const strs = [];
	for (let s = 0; s < nslots; s++) {
		const list = [];
		for (let a = 0; a < 29; a++) { const str = slotActionString(req, s, a, brought2pos); if (str) list.push([a, str]); }
		strs.push(list);
	}
	const pairs = []; const unavailable = [];
	for (const [a0, s0] of strs[0]) {
		for (const [a1, s1] of strs[1]) {
			const cs = clone.sides[sideIdx];
			let ok; let err = '';
			try { ok = cs.choose(`${s0}, ${s1}`); } catch (e) { ok = false; err = e.message; }
			err = err || cs.choice.error || '';
			if (ok) pairs.push([a0, a1]);
			else if (err.includes('[Unavailable choice]')) unavailable.push([a0, a1]);
			cs.clearChoice();
			if (!ok && err.includes('[Unavailable choice]')) clone = State.deserializeBattle(JSON.parse(snap)); // request refreshed
		}
	}
	return { kind: req.forceSwitch ? 'switch' : 'move', pairs, unavailable };
}
function legal(params) {
	const { battle } = tr.fromSnapshot(params.snapshot);
	const sides = params.side === undefined ? [0, 1] : [params.side === 'p2' || params.side === 1 ? 1 : 0];
	return sides.map(i => legalForBattle(battle, i));
}

// ------------------------------------------------------------------ replay (forced by DrawKey)
const keyStr = (kind, k) => [kind, k.turn, k.step_tag, k.step_actor, k.step_subject, k.actor, k.target, k.subject, k.hit,
	k.ord, k.sub, k.action_ord].join('|');
const tieKeyStr = (kind, k) => [kind, k.turn, k.step_tag, k.sub, k.ord].join('|');

/**
 * Build a forcing policy from draw records ({kind, key, value, pick?}) and tie records ({kind, key, value}).
 * Draws are matched by full DrawKey; sample picks in the engine domain take precedence over raw indices.
 */
function makeForcing(battle, draws = [], ties = []) {
	const dmap = new Map(); for (const d of draws) if (d.key && !d.dontcare) dmap.set(keyStr(d.kind, d.key), d);
	const tmap = new Map(); for (const t of ties) tmap.set(tieKeyStr(t.kind, t.key), t);
	const stats = { forced: 0, missed: [], tiesForced: 0 };
	const policy = (d) => {
		if (d.method === 'shuffle') return battle.__ctx.shuffleForcer(d.tiedObjs || []);
		if (!d.dk || d.dk.dontcare) return undefined;
		const rec = dmap.get(keyStr(d.dk.kind, d.dk.key));
		if (!rec) { if (!d.dk.degenerate) stats.missed.push({ kind: d.dk.kind, key: d.dk.key }); return undefined; }
		stats.forced++;
		if (d.method === 'sample' && rec.pick !== undefined && d._items) {
			const ident = battle.__ident;
			const idx = d._items.findIndex(it => {
				if (it && it.side && it.set) return (d.dk.kind === 'FORCED_SWITCH_PICK' ? ident.b(it) : ident.ref(it)) === rec.pick;
				return it === rec.pick || (it && it.id === rec.pick);
			});
			if (idx >= 0) return idx;
		}
		return rec.value;
	};
	return { policy, tmap, stats };
}

/**
 * Replay a battle (spec + choices) or continue a snapshot with forced draws/ties.
 * params: {teams, seed, choices, draws, ties, until?} | {snapshot, choices, draws, ties, until?}
 */
function replay(params) {
	const prngOpts = { mode: 'override' };
	let b;
	if (params.snapshot) {
		b = tr.fromSnapshot(params.snapshot, { prngOpts });
	} else {
		b = tr.newBattle({ seed: params.seed, formatid: params.formatid || FORMAT, prngOpts });
		b.battle.setPlayer('p1', { name: 'p1', team: unpack(params.teams[0]) });
		b.battle.setPlayer('p2', { name: 'p2', team: unpack(params.teams[1]) });
	}
	const f = makeForcing(b.battle, params.draws, params.ties);
	b.prng.policy = f.policy;
	b.battle.__ctx.tieForcer = (kind, key) => {
		const t = f.tmap.get(tieKeyStr(kind, key));
		if (t) f.stats.tiesForced++;
		return t ? t.value : undefined;
	};
	const n = params.until ?? params.choices.length;
	const out = tr.driveTraced({ ...b, choose: null, script: params.choices, stopAfter: n });
	return { decisions: out.decisions, final: out.final, forcing: f.stats,
		snapshot: params.return_snapshot ? tr.snapshotOf(b.battle) : undefined };
}

// ------------------------------------------------------------------ craft (minimal, 07 §7)
const BOOST_KEYS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
/**
 * scenario: {teams:[packed,packed], seed?, preview:["team 1234","team 1234"], turn?,
 *            mons:[{side:0|1, b:0..3, hp?, status?, statusState?, boosts?, pp?:[...], active_turns?}]}
 * Builds the battle at the turn-1 MOVE decision, JSON-edits the serialized state, deserializes, normalises with one
 * eachEvent('Update') pass and returns {snapshot, normalisation_log}.
 */
function craft(sc) {
	const b = tr.newBattle({ seed: sc.seed || ZERO_SEED, formatid: sc.formatid || FORMAT });
	b.battle.setPlayer('p1', { name: 'p1', team: unpack(sc.teams[0]) });
	b.battle.setPlayer('p2', { name: 'p2', team: unpack(sc.teams[1]) });
	const preview = sc.preview || ['team 1234', 'team 1234'];
	tr.driveTraced({ ...b, choose: null, script: [preview], stopAfter: 1 });
	const snap = tr.snapshotOf(b.battle);
	const st = snap.state;
	const identOf = b.battle.__ident;
	for (const m of sc.mons || []) {
		const sideJ = st.sides[m.side];
		const species = identOf.brought[m.side][m.b].species;
		const pj = sideJ.pokemon.find(p => p.set.species === species);
		if (!pj) throw new Error(`craft: no pokemon side ${m.side} b ${m.b}`);
		if (m.hp !== undefined) { pj.hp = m.hp; if (m.hp === 0) throw new Error('craft: use faint scenarios via APIs (not supported yet)'); }
		if (m.status !== undefined) {
			pj.status = m.status;
			pj.statusState = { id: m.status, effectOrder: 0, target: pj.statusState?.target ?? `[Pokemon:${sideJ.id}${'ab'[pj.position] || 'a'}]`,
				...(m.statusState || {}) };
		}
		if (m.boosts) for (const k of BOOST_KEYS) if (m.boosts[k] !== undefined) pj.boosts[k] = m.boosts[k];
		if (m.pp) m.pp.forEach((v, i) => { if (v !== null && pj.moveSlots[i]) pj.moveSlots[i].pp = v; });
		if (m.active_turns !== undefined) pj.activeTurns = m.active_turns;
	}
	if (sc.turn !== undefined) st.turn = sc.turn;
	for (const s of st.sides) delete s.activeRequest;
	const nb = tr.fromSnapshot(snap);
	const logStart = nb.battle.log.length;
	nb.battle.eachEvent('Update');
	nb.battle.makeRequest('move');
	return { snapshot: tr.snapshotOf(nb.battle), normalisation_log: nb.battle.log.slice(logStart) };
}

// ------------------------------------------------------------------ damage
/**
 * cases: [{attacker: set, defender: set, move: id, atkBoosts?, defBoosts?, atkStatus?, defStatus?, crit: bool,
 *          roll: 0..15, spread?: bool, weather?, terrain?, sideConditions?: {p1:[...], p2:[...]}}]
 * Returns PS actions.getDamage per case (number | false | null | undefined -> null), with crit/roll forced.
 * PS: sim/battle-actions.ts:1639-1776 getDamage, data/mods/champions/scripts.ts:196-312 modifyDamage.
 */
function damage({ cases, formatid }) {
	return cases.map(c => {
		const prng = new InstrumentedPRNG({ seed: ZERO_SEED, mode: 'override', ctx: true, stack: false, markers: true,
			classifier: 'markers' });
		const battle = new Battle({ formatid: formatid || FORMAT, seed: ZERO_SEED, prng });
		prng.attach(battle);
		const filler = unpack(c.attacker); const fill = (Array.isArray(filler) ? filler[0] : filler);
		const A = unpack(c.attacker); const D = unpack(c.defender);
		battle.setPlayer('p1', { name: 'a', team: [Array.isArray(A) ? A[0] : A, fill] });
		battle.setPlayer('p2', { name: 'b', team: [Array.isArray(D) ? D[0] : D, fill] });
		battle.makeChoices('team 12', 'team 12');
		const src = battle.p1.active[0]; const tgt = battle.p2.active[0];
		if (c.atkBoosts) Object.assign(src.boosts, c.atkBoosts);
		if (c.defBoosts) Object.assign(tgt.boosts, c.defBoosts);
		if (c.atkStatus) src.setStatus(c.atkStatus, src);
		if (c.defStatus) tgt.setStatus(c.defStatus, tgt);
		if (c.weather) battle.field.setWeather(c.weather, src);
		if (c.terrain) battle.field.setTerrain(c.terrain, src);
		for (const [sid, list] of Object.entries(c.sideConditions || {})) {
			for (const cond of list) battle[sid].addSideCondition(cond, sid === 'p1' ? src : tgt);
		}
		const move = battle.dex.getActiveMove(c.move);
		if (c.spread) move.spreadHit = true;
		move.hit = 1;
		battle.setActiveMove(move, src, tgt);
		prng.policy = d => {
			if (d.kind === 'CRIT') return !!c.crit;
			if (d.kind === 'DAMAGE_ROLL') return c.roll;
			return undefined;
		};
		const r = battle.actions.getDamage(src, tgt, move, true);
		return typeof r === 'number' ? r : null;
	});
}

// ------------------------------------------------------------------ trace_events (debugging, 11 §3)
/** Instance-level event-dispatch tracer (port of research/ps-move/harness.js traceEvents, without prototype patches). */
function installEventTracer(battle) {
	const log = []; let depth = 0;
	const ident = battle.__ident;
	const nm = x => {
		if (x === null || x === undefined) return '-';
		if (Array.isArray(x)) return '[' + x.map(nm).join(',') + ']';
		if (typeof x === 'string') return x;
		if (x.side && x.species) return `${x.side.id}${x.isActive ? 'ab'[x.position] : ''}:${x.name}#${ident ? ident.ref(x) : '?'}`;
		if (x.pokemonLeft !== undefined) return `side:${x.id}`;
		if (x.pseudoWeather !== undefined) return 'field';
		if (x.sides) return 'battle';
		if (x.effectType) return `${x.effectType}:${x.id || x.name}`;
		return typeof x;
	};
	const wrap = (name, entryOf) => {
		const orig = battle[name];
		battle[name] = function (...args) {
			log.push({ depth, turn: this.turn, ...entryOf.call(this, ...args) });
			depth++;
			try { return orig.apply(this, args); } finally { depth--; }
		};
	};
	wrap('runEvent', function (eventid, target, source, sourceEffect, relayVar, onEffect, fastExit) {
		const hs = this.findEventHandlers(target || this, eventid, source && source.side && source.species ? source : null);
		return { kind: fastExit ? 'priorityEvent' : 'runEvent', eventid, target: nm(target), source: nm(source),
			effect: nm(sourceEffect), handlers: hs.filter(h => h.callback !== undefined).map(h => `${nm(h.effect)}@${nm(h.effectHolder)}`) };
	});
	wrap('singleEvent', function (eventid, effect, state, target, source, sourceEffect, relayVar, customCallback) {
		const has = customCallback !== undefined || (effect && effect[`on${eventid}`] !== undefined);
		return { kind: 'singleEvent', eventid, target: nm(target), source: nm(source), effect: nm(effect), handlers: has ? [nm(effect)] : [] };
	});
	wrap('eachEvent', (eventid, effect) => ({ kind: 'eachEvent', eventid, effect: nm(effect), handlers: [] }));
	wrap('fieldEvent', (eventid, targets) => ({ kind: 'fieldEvent', eventid, target: nm(targets), handlers: [] }));
	return log;
}
/** params: {snapshot, choices: [[p1,p2]] (one decision), draws?, ties?} -> {events, log, decision}. */
function traceEvents(params) {
	const b = tr.fromSnapshot(params.snapshot, { prngOpts: { mode: 'override' } });
	const f = makeForcing(b.battle, params.draws, params.ties);
	b.prng.policy = f.policy;
	b.battle.__ctx.tieForcer = (kind, key) => { const t = f.tmap.get(tieKeyStr(kind, key)); return t ? t.value : undefined; };
	const events = installEventTracer(b.battle);
	const logStart = b.battle.log.length;
	const out = tr.driveTraced({ ...b, choose: null, script: params.choices, stopAfter: 1 });
	return { events, log: b.battle.log.slice(logStart).filter(l => !l.startsWith('|t:|')), decision: out.decisions[0],
		forcing: f.stats };
}

module.exports = {
	traceEvents, installEventTracer, typechart, fixedpoint, stats, sortActions, legal, legalForBattle, slotActionString, replay, makeForcing, craft,
	damage, keyStr, tieKeyStr };

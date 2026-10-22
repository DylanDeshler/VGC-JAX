'use strict';
/**
 * Map a PRNG draw (call-site frames + battle context) onto the EVENT_KIND taxonomy and build its
 * semantic key. Two classifiers:
 *   classifyByStack(): authoritative — uses the TS call site (lib/callsite.js) + rng_callsites.json
 *   classifyByMarkers(): no stack capture — uses core-method wrapper markers (installMarkers) +
 *                        battle.effect id + args signature. ~free; agreement is measured in scripts/.
 */
const path = require('path');
const { monLabel, elemLabel, stepLabel } = require('./context');
const INV = require(path.join(__dirname, '..', 'rng_callsites.json'));
const SITES = new Map(INV.callsites.map(c => [c.id, c]));

// speedSort call sites and other "primitive" sites whose meaning comes from a caller frame
const PRIMITIVE = new Set(['sim/battle.ts:461', 'sim/side.ts:372', 'sim/battle.ts:2523', 'sim/battle.ts:2532', 'sim/battle.ts:2409']);
// Kinds whose context key is the tied set rather than actor/target
// getRandomTarget on a non-single-target move (spread/side/field moves): only picks the nominal target
// shown in the |move| line; never changes who gets hit.
const SINGLE_TARGET = new Set(['normal', 'any', 'adjacentFoe', 'adjacentAlly', 'randomNormal', 'adjacentAllyOrSelf']);
function randomTargetKind(battle) {
	const mv = battle.__ctx.rtMove; if (!mv) return 'RANDOM_TARGET';
	const m = typeof mv === 'string' ? battle.dex.moves.get(mv) : mv;
	return SINGLE_TARGET.has(m.target) ? 'RANDOM_TARGET' : 'RANDOM_TARGET_NOMINAL';
}
const PERM_KINDS = new Set(['SPEED_TIE_ACTION_ORDER', 'SPEED_TIE_HANDLER_ORDER', 'SPEED_TIE_FIELD_EVENT',
	'SPEED_TIE_EACH_EVENT', 'SPEED_TIE_SWITCHIN_ORDER', 'SPEED_TIE_ABILITY', 'SPEED_TIE_DYNAMAX_END']);

function allyLabel(p) { return p && p.side && p.isActive ? `${p.side.id}${'ab'[1 - p.position]}` : null; }

function resolveExpr(expr, battle) {
	if (!expr) return null;
	const ctx = battle.__ctx; const ev = battle.event || {};
	switch (expr) {
	case 'A': return monLabel(battle.activePokemon);
	case 'T': return monLabel(battle.activeTarget);
	case 'ev.target': return monLabel(ev.target);
	case 'ev.source': return monLabel(ev.source);
	case 'rt.pokemon': return monLabel(ctx.rtPokemon);
	case 'sw.side': return ctx.swSide ? ctx.swSide.id : null;
	case 'act.pokemon': return ctx.action ? monLabel(ctx.action.pokemon) : null;
	case 'ally': return allyLabel(ev.target);
	default:
		if (expr.startsWith('last.')) {
			const [, evid, which] = expr.split('.');
			return monLabel(which === 'target' ? ctx.lastT[evid] : ctx.lastS[evid]);
		}
		return null;
	}
}

/** Fill actor/target/key for a kind using the inventory row `row` (may be null). */
function buildKey(kind, row, battle, draw) {
	const ctx = battle.__ctx; const am = battle.activeMove;
	let actor = null; let target = null; const extra = [];
	if (PERM_KINDS.has(kind)) {
		const ev = kind === 'SPEED_TIE_FIELD_EVENT' ? ctx.fieldEvent :
			kind === 'SPEED_TIE_EACH_EVENT' ? ctx.eachEvent :
			kind === 'SPEED_TIE_HANDLER_ORDER' ? ctx.lastEventId : '';
		extra.push(ev || '', draw.tied ? draw.tied.slice().sort().join(',') : '');
	} else if (kind === 'QUEUE_INSERT_TIE') {
		const ins = ctx.inserting; const arr = Array.isArray(ins) ? ins : [ins];
		extra.push(arr.map(c => c ? `${c.choice}:${monLabel(c.pokemon) || ''}` : '').join('+'));
	} else if (kind === 'RANDOM_TARGET' || kind === 'RANDOM_TARGET_NOMINAL') {
		actor = monLabel(ctx.rtPokemon) || (row ? resolveExpr(row.context.actor, battle) : null);
		const mv = ctx.rtMove; extra.push(mv ? (typeof mv === 'string' ? mv : mv.id) : (am ? am.id : ''));
		extra.push(ctx.inGetTarget ? 'getTarget' : ctx.inResolve ? 'resolve' : 'other');
	} else if (kind === 'CONFUSION_SELFHIT_DAMAGE_ROLL') {
		actor = target = monLabel((battle.event || {}).target);
	} else {
		if (row) { actor = resolveExpr(row.context.actor, battle); target = resolveExpr(row.context.target, battle); }
		const keys = row ? row.context.key.slice(4) : [];
		for (const k of keys) {
			if (k === 'move') extra.push(am ? am.id : '');
			else if (k === 'hit') extra.push(am ? (am.hit || 0) : 0);
			else if (k === 'effect') extra.push(battle.effect?.id || '');
			else if (k === 'slot') extra.push(am ? am.id : '');
			// 'ord' is appended by the keyed-forcing layer
		}
	}
	const tail = `${kind}|${actor || '-'}|${target || '-'}|${extra.join('|')}`;
	return { actor, target, keyNaive: `${battle.turn}|${tail}`, key: `${battle.turn}|${stepLabel(battle)}|${tail}` };
}

function classifyByStack(draw, battle) {
	const f = draw.frames || [];
	let site = f[0]; let kind; let row = SITES.get(site);
	if (PRIMITIVE.has(site)) {
		if (site === 'sim/battle.ts:461') { // shuffle inside speedSort -> speedSort call site
			row = SITES.get(f[1]); kind = row ? row.kind : 'UNMAPPED_SPEEDSORT';
			site = f[1];
		} else if (site === 'sim/battle.ts:2409') {
			kind = (draw.fns && draw.fns[1] === 'getConfusionDamage') ? 'CONFUSION_SELFHIT_DAMAGE_ROLL' : 'DAMAGE_ROLL';
		} else {
			kind = randomTargetKind(battle);
		}
	} else {
		kind = row ? row.kind : 'UNMAPPED';
	}
	const k = buildKey(kind, row, battle, draw);
	return { kind, site, ...k };
}

// ----------------------------------------------------------------- marker-based (no stack)
const MARKED = ['hitStepAccuracy', 'getDamage', 'secondaries', 'selfDrops', 'hitStepMoveHitLoop', 'getConfusionDamage', 'modifyDamage', 'dragIn'];
function installMarkers(battle) {
	const ctx = battle.__ctx; ctx.marks = []; // stack of [name, eventDepth]
	const A = battle.actions;
	for (const name of MARKED) {
		const orig = A[name];
		if (typeof orig !== 'function') continue;
		A[name] = function (...args) {
			ctx.marks.push(name, battle.eventDepth);
			try { return orig.apply(this, args); } finally { ctx.marks.length -= 2; }
		};
	}
	const randomizer = battle.randomizer;
	battle.randomizer = function (d) {
		ctx.marks.push('randomizer', battle.eventDepth);
		try { return randomizer.call(this, d); } finally { ctx.marks.length -= 2; }
	};
}

// effect id -> inventory rows (handlers in data/), for marker classification
const BY_EFFECT = new Map();
for (const c of INV.callsites) {
	if (c.reachable === 'no' || !c.file.startsWith('data/') || c.file.endsWith('scripts.ts')) continue;
	const id = c.enclosing.split('.')[0];
	if (!BY_EFFECT.has(id)) BY_EFFECT.set(id, []);
	BY_EFFECT.get(id).push(c);
}

function classifyByMarkers(draw, battle) {
	const ctx = battle.__ctx; const { method, args } = draw;
	let kind = null; let row = null;
	const marks = ctx.marks; const n = marks.length;
	const top = n ? marks[n - 2] : null; const topDepth = n ? marks[n - 1] : -1;
	const inHandler = battle.eventDepth > topDepth; // a handler started after the innermost marker
	if (method === 'shuffle') {
		const t0 = draw.tiedObjs && draw.tiedObjs[0];
		if (t0 && t0.choice) kind = 'SPEED_TIE_ACTION_ORDER';
		else if (t0 && t0.effectHolder !== undefined) kind = (ctx.fieldEvent && ctx.eventsSinceField === 0) ? 'SPEED_TIE_FIELD_EVENT' : 'SPEED_TIE_HANDLER_ORDER';
		else if (battle.effect?.id === 'magician') kind = 'SPEED_TIE_ABILITY';
		else if (ctx.eachEvent) kind = 'SPEED_TIE_EACH_EVENT';
		else kind = 'SPEED_TIE_SWITCHIN_ORDER';
		row = [...SITES.values()].find(r => r.kind === kind) || null;
	} else if (ctx.inserting && method === 'random' && args.length === 2) {
		kind = 'QUEUE_INSERT_TIE';
	} else if (ctx.rtPokemon && method === 'sample') {
		kind = randomTargetKind(battle); row = SITES.get('sim/side.ts:372');
	} else if (top === 'dragIn' && method === 'sample' && !inHandler) {
		kind = 'FORCED_SWITCH_PICK'; row = SITES.get('sim/battle.ts:1586');
	} else if (top && !inHandler) {
		const am = battle.activeMove; const hit = am ? (am.hit || 0) : 0;
		const isChance100 = method === 'randomChance' && args[1] === 100;
		const is100 = method === 'random' && args[0] === 100 && args.length === 1;
		if (top === 'randomizer' && method === 'random' && args[0] === 16) {
			kind = marks.includes('getConfusionDamage') ? 'CONFUSION_SELFHIT_DAMAGE_ROLL' : 'DAMAGE_ROLL';
			row = SITES.get('sim/battle.ts:2409');
		} else if (top === 'hitStepAccuracy' && isChance100) { kind = 'ACCURACY'; row = SITES.get('sim/battle-actions.ts:800'); }
		else if (top === 'getDamage' && method === 'randomChance' && args[0] === 1) { kind = 'CRIT'; row = SITES.get('sim/battle-actions.ts:1695'); }
		else if (top === 'secondaries' && is100) { kind = 'SECONDARY'; row = SITES.get('sim/battle-actions.ts:1399'); }
		else if (top === 'selfDrops' && is100) { kind = 'SELF_EFFECT_ROLL'; row = SITES.get('sim/battle-actions.ts:1381'); }
		else if (top === 'hitStepMoveHitLoop') {
			if (isChance100 && hit > 1) { kind = 'ACCURACY'; row = SITES.get('data/mods/champions/scripts.ts:507'); }
			else if (!hit && (method === 'sample' || method === 'random')) { kind = 'MULTIHIT_COUNT'; row = SITES.get('data/mods/champions/scripts.ts:439'); }
		}
	}
	if (!kind) {
		// handler draw: battle.effect identifies the effect; disambiguate by method
		// (secondary-effect onHit handlers run with battle.effect = the anonymous secondary object -> use the move)
		const byMethod = id => (BY_EFFECT.get(id) || []).filter(c => (c.method === 'speedSort' ? 'shuffle' : c.method) === method);
		let cands = byMethod(battle.effect?.id);
		if (!cands.length && battle.activeMove) cands = byMethod(battle.activeMove.id);
		if (cands.length) { row = cands[0]; kind = row.kind; }
		else if (method === 'random' && args[0] === 5 && args[1] === 7) { kind = 'PARTIAL_TRAP_DURATION'; row = SITES.get('data/conditions.ts:227'); }
		else if (method === 'random' && args[0] === 2 && args[1] === 4) { kind = 'LOCKEDMOVE_DURATION'; row = SITES.get('data/conditions.ts:265'); }
		else if (method === 'random' && args[1] === 6) { kind = 'CONFUSION_DURATION'; row = SITES.get('data/conditions.ts:174'); }
		else if (method === 'sample' && args[0] && args[0].length === 2 && args[0][0] === 'M') { kind = 'GENDER'; row = SITES.get('sim/pokemon.ts:361'); }
		else kind = 'UNMAPPED';
	}
	return { kind, ...buildKey(kind, row, battle, draw) };
}

module.exports = { classifyByStack, classifyByMarkers, installMarkers, PERM_KINDS, SITES, elemLabel };

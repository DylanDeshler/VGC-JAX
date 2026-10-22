'use strict';
/**
 * Per-battle context tracking for PRNG draws. Installs cheap instance-level wrappers (the PS
 * checkout is not modified) that remember:
 *   - the queue action being run (runAction)
 *   - the most recent runEvent target/source per event id (for core draws whose target is a local
 *     variable in PS, e.g. accuracy/crit/damage roll/secondaries)
 *   - the pokemon/side argument of getRandomTarget / getRandomSwitchable
 *   - the event id of an enclosing fieldEvent/eachEvent, and the action being inserted by insertChoice
 */

function monLabel(p) {
	if (!p || typeof p !== 'object') return null;
	if (p.side && p.species) { // Pokemon
		return p.isActive ? `${p.side.id}${'abc'[p.position]}` : `${p.side.id}:${p.species.id}`;
	}
	if (p.pokemonLeft !== undefined && p.id) return p.id; // Side
	if (p.pseudoWeather !== undefined) return 'field';
	if (p.sides) return 'battle';
	return null;
}

/** Canonical label for an element of a speedSort list (Action | EventListener | Pokemon). */
function elemLabel(x) {
	if (!x || typeof x !== 'object') return String(x);
	if (x.choice) { // queue Action
		const who = x.pokemon ? monLabel(x.pokemon) : (x.side ? x.side.id : '');
		return `${x.choice}:${who}${x.move ? ':' + x.move.id : ''}`;
	}
	if (x.effectHolder !== undefined) { // EventListener
		return `${x.effect?.id || '?'}@${monLabel(x.effectHolder) || '?'}`;
	}
	if (x.species) return monLabel(x);
	return '?';
}

class Ctx {
	constructor() {
		this.action = null;
		this.lastT = Object.create(null); // eventid -> target
		this.lastS = Object.create(null); // eventid -> source
		this.lastEventId = '';
		this.rtPokemon = null; this.rtMove = null;
		this.swSide = null;
		this.fieldEvent = null; this.eachEvent = null; this.eventsSinceField = 0;
		this.inserting = null;
		this.turnSeen = -1; this.actionIdx = 0; this.inGetTarget = false; this.inResolve = false;
	}
}

function install(battle) {
	const ctx = new Ctx();
	// non-enumerable so State.serializeBattle ignores it (it throws on class instances)
	Object.defineProperty(battle, '__ctx', { value: ctx, enumerable: false, configurable: true });
	const B = Object.getPrototypeOf(battle);

	const runAction = B.runAction;
	battle.runAction = function (action) {
		if (ctx.turnSeen !== this.turn) { ctx.turnSeen = this.turn; ctx.actionIdx = 0; }
		ctx.action = action; ctx.actionIdx++;
		return runAction.call(this, action);
	};
	const getTarget = B.getTarget;
	battle.getTarget = function (pokemon, move, targetLoc, originalTarget) {
		const prev = ctx.inGetTarget; ctx.inGetTarget = true;
		try { return getTarget.call(this, pokemon, move, targetLoc, originalTarget); } finally { ctx.inGetTarget = prev; }
	};
	const resolveAction = Object.getPrototypeOf(battle.queue).resolveAction;
	battle.queue.resolveAction = function (action, midTurn) {
		const prev = ctx.inResolve; ctx.inResolve = true;
		try { return resolveAction.call(this, action, midTurn); } finally { ctx.inResolve = prev; }
	};

	const runEvent = B.runEvent;
	battle.runEvent = function (eventid, target, source, sourceEffect, relayVar, onEffect, fastExit) {
		ctx.lastT[eventid] = target; ctx.lastS[eventid] = source; ctx.lastEventId = eventid; ctx.eventsSinceField++;
		return runEvent.call(this, eventid, target, source, sourceEffect, relayVar, onEffect, fastExit);
	};

	const fieldEvent = B.fieldEvent;
	battle.fieldEvent = function (eventid, targets) {
		const prev = ctx.fieldEvent; const prevN = ctx.eventsSinceField;
		ctx.fieldEvent = eventid; ctx.eventsSinceField = 0;
		try { return fieldEvent.call(this, eventid, targets); } finally { ctx.fieldEvent = prev; ctx.eventsSinceField = prevN; }
	};
	const eachEvent = B.eachEvent;
	battle.eachEvent = function (eventid, effect, relayVar) {
		const prev = ctx.eachEvent; ctx.eachEvent = eventid;
		try { return eachEvent.call(this, eventid, effect, relayVar); } finally { ctx.eachEvent = prev; }
	};
	const getRandomTarget = B.getRandomTarget;
	battle.getRandomTarget = function (pokemon, move) {
		const pp = ctx.rtPokemon; const pm = ctx.rtMove;
		ctx.rtPokemon = pokemon; ctx.rtMove = move;
		try { return getRandomTarget.call(this, pokemon, move); } finally { ctx.rtPokemon = pp; ctx.rtMove = pm; }
	};
	const getRandomSwitchable = B.getRandomSwitchable;
	battle.getRandomSwitchable = function (side) {
		ctx.swSide = side;
		try { return getRandomSwitchable.call(this, side); } finally { ctx.swSide = null; }
	};
	const q = battle.queue; const QP = Object.getPrototypeOf(q);
	const insertChoice = QP.insertChoice;
	q.insertChoice = function (choices, midTurn) {
		const prev = ctx.inserting; ctx.inserting = choices;
		try { return insertChoice.call(this, choices, midTurn); } finally { ctx.inserting = prev; }
	};
	return ctx;
}

function actionLabel(a) {
	return a ? `${a.choice}${a.pokemon ? ':' + monLabel(a.pokemon) : ''}${a.move ? ':' + a.move.id : ''}` : null;
}
/** Engine-observable sub-step: the queue action currently running this turn, or 'commit' (choice resolution). */
function stepLabel(battle) {
	const ctx = battle.__ctx;
	return ctx && ctx.turnSeen === battle.turn && ctx.action ? actionLabel(ctx.action) : 'commit';
}

/** Snapshot of the context at a draw (labels only, JSON-safe). */
function snapshot(battle) {
	const ctx = battle.__ctx;
	const ev = battle.event || {};
	const am = battle.activeMove;
	const a = ctx && ctx.action;
	return {
		turn: battle.turn,
		action: actionLabel(a), step: stepLabel(battle), actionIdx: ctx && ctx.turnSeen === battle.turn ? ctx.actionIdx : 0,
		effect: battle.effect?.id || '', effectType: battle.effect?.effectType || '',
		event: ev.id || '', evTarget: monLabel(ev.target), evSource: monLabel(ev.source),
		lastEvent: ctx ? ctx.lastEventId : '',
		move: am ? am.id : null, hit: am ? (am.hit || 0) : 0,
		activePokemon: monLabel(battle.activePokemon), activeTarget: monLabel(battle.activeTarget),
		fieldEvent: ctx ? ctx.fieldEvent : null, eachEvent: ctx ? ctx.eachEvent : null,
	};
}

module.exports = { install, snapshot, monLabel, elemLabel, actionLabel, stepLabel };

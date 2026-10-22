'use strict';
/**
 * Coverage counters (07 §9.1), per battle, instance-level only.
 *  - handlers: "<effectType>:<id>|<callbackName>" counted each time the callback actually runs. PS gathers handlers
 *    through findPokemon/Side/Field/BattleEventHandlers; we wrap those four methods so every returned handler's
 *    callback is a counting wrapper (covers runEvent, fieldEvent and priorityEvent dispatch).
 *  - singleEvent callbacks (move onHit, ability onStart via SwitchIn ...) are counted in a singleEvent wrapper when
 *    the effect defines the callback.
 *  - pairs: unordered pairs of effects co-occurring in one runEvent dispatch with >= 2 live (callback) handlers.
 */
function effKey(effect, cb) {
	const t = (effect && effect.effectType) || '?';
	return `${t.toLowerCase()}:${(effect && effect.id) || '?'}|${cb}`;
}

function install(battle) {
	const cov = { handlers: Object.create(null), pairs: Object.create(null) };
	Object.defineProperty(battle, '__cov', { value: cov, enumerable: false, configurable: true });
	const bump = (map, k) => { map[k] = (map[k] || 0) + 1; };
	for (const name of ['findPokemonEventHandlers', 'findSideEventHandlers', 'findFieldEventHandlers', 'findBattleEventHandlers']) {
		const orig = battle[name];
		battle[name] = function (holder, callbackName, ...rest) {
			const hs = orig.call(this, holder, callbackName, ...rest);
			for (const h of hs) {
				if (typeof h.callback !== 'function' || h.callback.__cov) continue;
				const key = effKey(h.effect, callbackName);
				const cb = h.callback;
				const wrapped = function (...args) { bump(cov.handlers, key); return cb.apply(this, args); };
				wrapped.__cov = true;
				h.callback = wrapped;
			}
			return hs;
		};
	}
	const findEventHandlers = battle.findEventHandlers;
	battle.findEventHandlers = function (target, eventName, source) {
		const hs = findEventHandlers.call(this, target, eventName, source);
		const live = [...new Set(hs.filter(h => h.callback).map(h => `${(h.effect && h.effect.id) || '?'}`))].sort();
		if (live.length >= 2) {
			for (let i = 0; i < live.length; i++) {
				for (let j = i + 1; j < live.length; j++) bump(cov.pairs, `${eventName}|${live[i]}|${live[j]}`);
			}
		}
		return hs;
	};
	const singleEvent = battle.singleEvent;
	battle.singleEvent = function (eventid, effect, state, target, source, sourceEffect, relayVar, customCallback) {
		if (!customCallback && effect && typeof effect[`on${eventid}`] === 'function') bump(cov.handlers, effKey(effect, `on${eventid}`));
		return singleEvent.call(this, eventid, effect, state, target, source, sourceEffect, relayVar, customCallback);
	};
	return cov;
}

/** Merge per-battle coverage into an aggregate {handlers: {k: battles_hit}, pairs: {...}}. */
function mergeBattles(agg, cov) {
	for (const kind of ['handlers', 'pairs']) {
		agg[kind] ||= Object.create(null);
		for (const k of Object.keys(cov[kind] || {})) agg[kind][k] = (agg[kind][k] || 0) + 1;
	}
	return agg;
}

module.exports = { install, mergeBattles, effKey };

'use strict';
/**
 * Hand annotations for every PRNG call site found by tools/scan_callsites.js.
 * build_inventory.js merges these with the scanner output into ../rng_callsites.json.
 *
 * Fields
 *   kind     semantic EVENT_KIND (what the JAX engine keys the forced outcome by)
 *   decides  what the draw decides
 *   dist     distribution as written in PS
 *   out      outcome encoding we record/force: bool | int | index | perm
 *   actor    context expression for the acting/affected mon (see lib/classify.js `resolveExpr`)
 *   target   context expression for the other mon (or null)
 *   key      extra key fields beyond (turn, kind, actor, target)
 *   reach    'yes' | 'cond' (reachable only in a rare configuration) | 'no' + why
 *   why      reachability / notes
 *
 * Context expressions:
 *   A                  battle.activePokemon (user of the active move)
 *   T                  battle.activeTarget
 *   ev.target/ev.source  battle.event (the handler currently running)
 *   last.<Event>.target/.source  most recent runEvent('<Event>') entered (tracked by lib/context.js)
 *   rt.pokemon         argument of the enclosing getRandomTarget(pokemon, move) call
 *   sw.side            argument of the enclosing getRandomSwitchable(side) call
 *   act.pokemon        pokemon of the queue action currently being run (runAction)
 *   tied               canonical labels of the tied elements of a speedSort shuffle
 */
const CORE_KEY = ['move', 'hit'];

module.exports = {
	// ------------------------------------------------------------------ sim/battle.ts
	'sim/battle.ts:461': {
		kind: 'SPEED_TIE (primitive)', decides: 'Fisher-Yates order of a run of equal-priority elements inside speedSort',
		dist: 'shuffle(list, start, end): (end-start-1) uniform draws', out: 'perm', actor: null, target: null,
		key: ['caller', 'tied'], reach: 'yes',
		why: 'The only shuffle in a battle. Semantics come from the speedSort caller (rows below); classify by the caller frame.',
	},
	'sim/battle.ts:473': {
		kind: 'SPEED_TIE_EACH_EVENT', decides: 'order in which equal-speed active mons run an eachEvent (e.g. Update after Weather)',
		dist: 'shuffle of tied actives', out: 'perm', actor: null, target: null, key: ['event', 'tied'], reach: 'yes',
		why: 'eachEvent(Weather/Update/...). Usually unobservable; observable when two tied mons both react (e.g. two Booster Energy/seed activations).',
	},
	'sim/battle.ts:512': {
		kind: 'SPEED_TIE_FIELD_EVENT', decides: 'order of tied handlers in fieldEvent (Residual, SwitchIn)',
		dist: 'shuffle of tied handlers', out: 'perm', actor: null, target: null, key: ['event', 'tied'], reach: 'yes',
		why: 'End-of-turn residual order and switch-in ability order among speed-tied holders. Observable (e.g. who gets Intimidated first, Leftovers vs poison order across mons).',
	},
	'sim/battle.ts:801': {
		kind: 'SPEED_TIE_HANDLER_ORDER', decides: 'order of tied handlers inside runEvent',
		dist: 'shuffle of tied handlers', out: 'perm', actor: null, target: null, key: ['event', 'tied'], reach: 'yes',
		why: 'Fires very often (any event with >=2 equal-priority handlers from different holders, e.g. two Pokémon with same Speed both listening to an "Any" event). Mostly unobservable; most frequent draw kind by count.',
	},
	'sim/battle.ts:1586': {
		kind: 'FORCED_SWITCH_PICK', decides: 'which benched mon is dragged in (Roar, Whirlwind, Dragon Tail, Circle Throw, Red Card)',
		dist: 'sample(canSwitchIn)', out: 'index', actor: 'sw.side', target: null, key: ['slot'], reach: 'yes',
		why: 'dragIn(). Outcome = bench index among legal switch-ins (order = side.pokemon order).',
	},
	'sim/battle.ts:1652': {
		kind: 'SPEED_TIE_DYNAMAX_END', decides: 'order of Dynamax-ending messages', dist: 'speedSort', out: 'perm',
		actor: null, target: null, key: [], reach: 'no', why: 'No Dynamax in Champions.',
	},
	'sim/battle.ts:1814': {
		kind: 'QUICK_CLAW_GEN2', decides: 'gen 2 Quick Claw', dist: 'randomChance(60,256)', out: 'bool', actor: null, target: null,
		key: [], reach: 'no', why: 'gen === 2 only.',
	},
	'sim/battle.ts:1815': {
		kind: 'QUICK_CLAW_GEN3', decides: 'gen 3 Quick Claw', dist: 'randomChance(1,5)', out: 'bool', actor: null, target: null,
		key: [], reach: 'no', why: 'gen === 3 only.',
	},
	'sim/battle.ts:2409': {
		kind: 'DAMAGE_ROLL', decides: 'damage random factor (100 - r)%', dist: 'random(16) -> r in 0..15 (0 = max roll)', out: 'int',
		actor: 'last.WeatherModifyDamage.target', target: 'last.WeatherModifyDamage.source', key: CORE_KEY, reach: 'yes',
		why: 'Called from champions modifyDamage (per target, per hit) and from getConfusionDamage (caller frame => CONFUSION_SELFHIT_DAMAGE_ROLL, actor=target=self).',
	},
	'sim/battle.ts:2523': {
		kind: 'RANDOM_TARGET', decides: 'random ally for an adjacentAlly move used without a target', dist: 'sample(adjacentAllies)',
		out: 'index', actor: 'rt.pokemon', target: null, key: ['move'], reach: 'cond',
		why: 'Only when an adjacentAlly move (e.g. Aromatic Mist via Copycat) is called without a chosen target; 1 candidate in doubles => degenerate.',
	},
	'sim/battle.ts:2532': {
		kind: 'RANDOM_TARGET', decides: 'random adjacent foe (triples)', dist: 'sample(adjacentFoes)', out: 'index',
		actor: 'rt.pokemon', target: null, key: ['move'], reach: 'no', why: 'activePerHalf > 2 only (triples / multi).',
	},
	// ------------------------------------------------------------------ sim/battle-actions.ts
	'sim/battle-actions.ts:235': {
		kind: 'SPEED_TIE_SWITCHIN_ORDER', decides: 'speedOrder snapshot for simultaneous switch-ins (used to order SwitchIn handlers)',
		dist: 'speedSort(allActive)', out: 'perm', actor: null, target: null, key: ['tied'], reach: 'yes',
		why: 'Turn 0 leads and simultaneous replacements; observable via switch-in ability order (Intimidate/weather/terrain wars) when Speed ties.',
	},
	'sim/battle-actions.ts:800': {
		kind: 'ACCURACY', decides: 'move hits this target', dist: 'randomChance(acc, 100) with acc after boosts/modifiers', out: 'bool',
		actor: 'last.Accuracy.source', target: 'last.Accuracy.target', key: CORE_KEY, reach: 'yes',
		why: 'hitStepAccuracy, one draw per target (spread moves draw per target). No draw when accuracy===true (alwaysHit, No Guard, Lock-On...).',
	},
	'sim/battle-actions.ts:929': { kind: 'MULTIHIT_COUNT', decides: '2-5 hit count', dist: 'sample 35/35/15/15', out: 'int', actor: 'A', target: null, key: ['move'], reach: 'no', why: 'Overridden by champions scripts.ts hitStepMoveHitLoop (same logic at scripts.ts:439).' },
	'sim/battle-actions.ts:931': { kind: 'MULTIHIT_COUNT', decides: 'Loaded Dice 4-5', dist: '5 - random(2)', out: 'int', actor: 'A', target: null, key: ['move'], reach: 'no', why: 'Overridden by champions scripts.ts:441.' },
	'sim/battle-actions.ts:934': { kind: 'MULTIHIT_COUNT', decides: 'gen<5 2-5 hits', dist: 'sample 3/3/1/1', out: 'int', actor: 'A', target: null, key: ['move'], reach: 'no', why: 'Overridden; gen < 5 only.' },
	'sim/battle-actions.ts:937': { kind: 'MULTIHIT_COUNT', decides: 'generic [m,n] hit count', dist: 'random(m, n+1)', out: 'int', actor: 'A', target: null, key: ['move'], reach: 'no', why: 'Overridden by champions scripts.ts:447.' },
	'sim/battle-actions.ts:940': { kind: 'MULTIHIT_COUNT', decides: 'Loaded Dice Population Bomb', dist: '10 - random(7)', out: 'int', actor: 'A', target: null, key: ['move'], reach: 'no', why: 'Overridden by champions scripts.ts:450.' },
	'sim/battle-actions.ts:997': { kind: 'ACCURACY', decides: 'per-hit accuracy (Triple Axel / Population Bomb)', dist: 'randomChance(acc,100)', out: 'bool', actor: 'A', target: 'last.Accuracy.target', key: CORE_KEY, reach: 'no', why: 'Overridden by champions scripts.ts:507.' },
	'sim/battle-actions.ts:1381': {
		kind: 'SELF_EFFECT_ROLL', decides: 'chance of a move\'s self stat change (move.self.boosts)', dist: 'random(100) < self.chance',
		out: 'int', actor: 'A', target: null, key: CORE_KEY, reach: 'yes',
		why: 'Drawn even when self.chance is undefined (always applies: Close Combat, Draco Meteor...) => usually don\'t-care. Once per move (selfDropped) unless multihit.',
	},
	'sim/battle-actions.ts:1399': {
		kind: 'SECONDARY', decides: 'whether secondary effect i applies to this target', dist: 'random(100) < secondary.chance',
		out: 'int', actor: 'last.ModifySecondaries.source', target: 'last.ModifySecondaries.target', key: CORE_KEY.concat(['ord']), reach: 'yes',
		why: 'One draw per (target, secondary) — drawn even when chance is undefined (don\'t-care). Moves with 2 secondaries (Fire Fang, Ice Fang, Thunder Fang) need the secondary index (ord). Serene Grace doubles chance (no extra draw). King\'s Rock adds a secondary.',
	},
	'sim/battle-actions.ts:1695': {
		kind: 'CRIT', decides: 'critical hit on this target', dist: 'randomChance(1, [0,24,8,2,1][critRatio])', out: 'bool',
		actor: 'last.ModifyCritRatio.target', target: 'last.ModifyCritRatio.source', key: CORE_KEY, reach: 'yes',
		why: 'getDamage, per target per hit. No draw when willCrit is defined (Storm Throw, Flower Trick, Frost Breath) or critRatio clamps to 0. critRatio>=4 still draws randomChance(1,1).',
	},
	// ------------------------------------------------------------------ sim/battle-queue.ts
	'sim/battle-queue.ts:406': {
		kind: 'QUEUE_INSERT_TIE', decides: 'insertion point of a mid-turn action among equal-priority queued actions',
		dist: 'random(firstIndex, lastIndex+1)', out: 'int', actor: null, target: null, key: ['inserted', 'tiedWith'], reach: 'yes',
		why: 'insertChoice for runSwitch/megaEvo/instaswitch etc. Draws only when a speed-tied action is already queued. Outcome is a queue position => key by labels of the inserted action and the actions it ties with.',
	},
	'sim/battle-queue.ts:426': {
		kind: 'SPEED_TIE_ACTION_ORDER', decides: 'order of speed-tied actions in the turn queue (the classic speed tie)',
		dist: 'speedSort(queue)', out: 'perm', actor: null, target: null, key: ['tied'], reach: 'yes',
		why: 'Includes ties between mega-evolution actions, switches, moves of equal priority/speed, and residual pseudo-actions.',
	},
	// ------------------------------------------------------------------ sim/pokemon.ts / side.ts
	'sim/pokemon.ts:361': {
		kind: 'GENDER', decides: 'gender for a set with no gender on a mixed-gender species', dist: "sample(['M','F'])", out: 'index',
		actor: 'self (constructor)', target: null, key: ['side', 'teamIndex'], reach: 'yes',
		why: 'At battle construction (turn 0, before team preview). Avoid by always setting gender in generated sets. Gender affects Attract/Cute Charm/Rivalry.',
	},
	'sim/side.ts:372': {
		kind: 'RANDOM_TARGET', decides: 'random foe when a move has no valid chosen target (randomNormal moves, retarget after the chosen target fainted, Dragon Darts)',
		dist: 'sample(foes)', out: 'index', actor: 'rt.pokemon', target: null, key: ['move'], reach: 'yes',
		why: 'Called from getRandomTarget (doubles fallback). Draws even with 1 foe left (degenerate). Callers: queue.resolveAction (randomNormal: Outrage/Thrash/Petal Dance/Raging Fury/Uproar/Struggle at commit time), useMove retargeting, getTarget for smartTarget, champions Curse.',
	},
	// ------------------------------------------------------------------ data/abilities.ts
	'data/abilities.ts:788': { kind: 'ABILITY_ONHIT_PROC', decides: 'Cursed Body disables the move', dist: 'randomChance(3,10)', out: 'bool', actor: 'ev.source', target: 'ev.target', key: ['effect', 'hit'], reach: 'yes', why: 'onDamagingHit (holder = ev.target). Only draws if source has a lastMove and is not already disabled.' },
	'data/abilities.ts:801': { kind: 'ABILITY_ONHIT_PROC', decides: 'Cute Charm infatuates attacker', dist: 'randomChance(3,10)', out: 'bool', actor: 'ev.source', target: 'ev.target', key: ['effect', 'hit'], reach: 'yes', why: 'Contact only.' },
	'data/abilities.ts:1164': { kind: 'ABILITY_ONHIT_PROC', decides: 'Effect Spore: slp (<11) / par (<21) / psn (<30) / none', dist: 'random(100)', out: 'int', actor: 'ev.source', target: 'ev.target', key: ['effect', 'hit'], reach: 'yes', why: 'Contact and powder-immunity check first.' },
	'data/abilities.ts:1320': { kind: 'ABILITY_ONHIT_PROC', decides: 'Flame Body burns attacker', dist: 'randomChance(3,10)', out: 'bool', actor: 'ev.source', target: 'ev.target', key: ['effect', 'hit'], reach: 'yes', why: 'Contact only.' },
	'data/abilities.ts:1526': { kind: 'ABILITY_FOREWARN_PICK', decides: 'which tied highest-BP foe move Forewarn reveals', dist: 'sample(warnMoves)', out: 'index', actor: 'ev.target', target: null, key: ['effect'], reach: 'yes', why: 'Message-only (no state change) but consumes a draw.' },
	'data/abilities.ts:1805': { kind: 'ABILITY_RESIDUAL_PROC', decides: 'Harvest restores berry (always in sun)', dist: 'randomChance(1,2)', out: 'bool', actor: 'ev.target', target: null, key: ['effect'], reach: 'yes', why: 'Short-circuit: no draw in sun.' },
	'data/abilities.ts:1823': { kind: 'ABILITY_RESIDUAL_PROC', decides: 'Healer (base 30%)', dist: 'randomChance(3,10)', out: 'bool', actor: 'ev.target', target: 'ally', key: ['effect'], reach: 'no', why: 'Overridden by champions abilities.ts:43 (50%).' },
	'data/abilities.ts:2483': { kind: 'SPEED_TIE_ABILITY', decides: 'Magician steal order among speed-tied hit targets', dist: 'speedSort(hitTargets)', out: 'perm', actor: 'ev.source', target: null, key: ['effect', 'tied'], reach: 'yes', why: 'Only matters for spread moves hitting two tied item holders.' },
	'data/abilities.ts:2715': { kind: 'ABILITY_MOODY_STAT', decides: 'Moody stat to raise', dist: 'sample(stats)', out: 'index', actor: 'ev.target', target: null, key: ['effect', 'ord'], reach: 'yes', why: 'Moody is legal by species but usually banned in Smogon formats; Reg M-C has no Moody ban.' },
	'data/abilities.ts:2726': { kind: 'ABILITY_MOODY_STAT', decides: 'Moody stat to lower', dist: 'sample(stats)', out: 'index', actor: 'ev.target', target: null, key: ['effect', 'ord'], reach: 'yes', why: 'Second draw of the same handler => ord=1.' },
	'data/abilities.ts:2952': { kind: 'SPEED_TIE_ABILITY', decides: 'Neutralizing Gas end order', dist: 'speedSort', out: 'perm', actor: null, target: null, key: ['tied'], reach: 'no', why: 'Neutralizing Gas not on any legal species.' },
	'data/abilities.ts:3272': { kind: 'ABILITY_PICKUP_TARGET', decides: 'whose consumed item Pickup takes', dist: 'sample(pickupTargets)', out: 'index', actor: 'ev.target', target: null, key: ['effect'], reach: 'yes', why: 'Residual.' },
	'data/abilities.ts:3348': { kind: 'ABILITY_ONHIT_PROC', decides: 'Poison Point poisons attacker', dist: 'randomChance(3,10)', out: 'bool', actor: 'ev.source', target: 'ev.target', key: ['effect', 'hit'], reach: 'yes', why: 'Contact only.' },
	'data/abilities.ts:3376': { kind: 'ABILITY_ONHIT_PROC', decides: 'Poison Touch poisons target', dist: 'randomChance(3,10)', out: 'bool', actor: 'ev.source', target: 'ev.target', key: ['effect', 'hit'], reach: 'yes', why: 'onSourceDamagingHit: holder is the attacker (ev.source).' },
	'data/abilities.ts:3739': { kind: 'PRIORITY_PROC', decides: 'Quick Draw moves first in bracket', dist: 'randomChance(3,10)', out: 'bool', actor: 'ev.target', target: null, key: ['effect'], reach: 'yes', why: 'Drawn at choice-commit time (queue.resolveAction -> FractionalPriority), i.e. before the turn starts; non-Status moves only.' },
	'data/abilities.ts:4192': { kind: 'ABILITY_RESIDUAL_PROC', decides: 'Shed Skin cures status', dist: 'randomChance(33,100)', out: 'bool', actor: 'ev.target', target: null, key: ['effect'], reach: 'yes', why: 'Only when statused.' },
	'data/abilities.ts:4540': { kind: 'ABILITY_ONHIT_PROC', decides: 'Static paralyzes attacker', dist: 'randomChance(3,10)', out: 'bool', actor: 'ev.source', target: 'ev.target', key: ['effect', 'hit'], reach: 'yes', why: 'Contact only.' },
	'data/abilities.ts:5097': { kind: 'ABILITY_ONHIT_PROC', decides: 'Toxic Chain badly poisons', dist: 'randomChance(3,10)', out: 'bool', actor: 'ev.source', target: 'ev.target', key: ['effect', 'hit'], reach: 'no', why: 'No legal species has Toxic Chain.' },
	'data/abilities.ts:5145': { kind: 'ABILITY_TRACE_TARGET', decides: 'which adjacent foe Trace copies', dist: 'sample(possibleTargets)', out: 'index', actor: 'ev.target', target: null, key: ['effect'], reach: 'yes', why: 'onUpdate; draws even with one candidate.' },
	// ------------------------------------------------------------------ data/items.ts
	'data/items.ts:2273': { kind: 'ITEM_FOCUS_BAND', decides: 'Focus Band survives at 1 HP', dist: 'randomChance(1,10)', out: 'bool', actor: 'ev.target', target: 'ev.source', key: ['hit'], reach: 'yes', why: 'Drawn on EVERY damage instance to the holder (draw happens before the lethal check).' },
	'data/items.ts:5017': { kind: 'QUICK_CLAW', decides: 'Quick Claw moves first in bracket', dist: 'randomChance(1,5)', out: 'bool', actor: 'ev.target', target: null, key: ['effect'], reach: 'yes', why: 'At choice-commit time; only for priority <= 0 moves.' },
	'data/items.ts:6036': { kind: 'ITEM_STARF_STAT', decides: 'Starf Berry stat', dist: 'sample(stats)', out: 'index', actor: 'ev.target', target: null, key: [], reach: 'no', why: 'Starf Berry isNonstandard Past in Champions.' },
	// ------------------------------------------------------------------ data/moves.ts
	'data/moves.ts:155': { kind: 'MOVE_ACUPRESSURE_STAT', decides: 'stat boosted by Acupressure', dist: 'sample(stats not at +6)', out: 'index', actor: 'A', target: 'ev.target', key: ['move'], reach: 'yes', why: '' },
	'data/moves.ts:343': { kind: 'ALLY_SWITCH_STALL', decides: 'consecutive Ally Switch succeeds', dist: 'randomChance(1, counter) counter=3,9,27..', out: 'bool', actor: 'ev.target', target: null, key: [], reach: 'yes', why: 'Gen 9 Ally Switch shares the Protect-style counter (onRestart).' },
	'data/moves.ts:632': { kind: 'MOVE_CALL_PICK', decides: 'Assist move', dist: 'sample', out: 'index', actor: 'A', target: null, key: [], reach: 'no', why: 'Assist Past.' },
	'data/moves.ts:745': { kind: 'INFATUATION', decides: 'infatuated mon is immobilized', dist: 'randomChance(1,2)', out: 'bool', actor: 'ev.target', target: null, key: [], reach: 'yes', why: 'attract condition onBeforeMove.' },
	'data/moves.ts:2839': { kind: 'MOVE_CONVERSION2_TYPE', decides: 'Conversion 2 type', dist: 'sample', out: 'index', actor: 'A', target: null, key: [], reach: 'no', why: 'Past.' },
	'data/moves.ts:3641': { kind: 'SECONDARY_STATUS_PICK', decides: 'Dire Claw status (base)', dist: "sample(['psn','par','slp'])", out: 'index', actor: 'ev.source', target: 'ev.target', key: ['move'], reach: 'no', why: 'Overridden by champions moves.ts:223.' },
	'data/moves.ts:5231': { kind: 'MOVE_FICKLE_BEAM', decides: 'Fickle Beam doubles power', dist: 'randomChance(3,10)', out: 'bool', actor: 'A', target: 'ev.source', key: ['move'], reach: 'yes', why: 'onBasePower: drawn once per target (spread: n/a, single target).' },
	'data/moves.ts:6728': { kind: 'GMAX', decides: 'G-Max Befuddle status', dist: 'sample', out: 'index', actor: null, target: null, key: [], reach: 'no', why: 'Gmax.' },
	'data/moves.ts:7110': { kind: 'GMAX', decides: 'G-Max Replenish', dist: 'randomChance(1,2)', out: 'bool', actor: null, target: null, key: [], reach: 'no', why: 'Gmax.' },
	'data/moves.ts:7202': { kind: 'GMAX', decides: 'G-Max Snooze', dist: 'randomChance(1,2)', out: 'bool', actor: null, target: null, key: [], reach: 'no', why: 'Gmax.' },
	'data/moves.ts:7207': { kind: 'GMAX', decides: 'G-Max Snooze', dist: 'randomChance(1,2)', out: 'bool', actor: null, target: null, key: [], reach: 'no', why: 'Gmax.' },
	'data/moves.ts:7288': { kind: 'GMAX', decides: 'G-Max Stun Shock', dist: 'random(2)', out: 'int', actor: null, target: null, key: [], reach: 'no', why: 'Gmax.' },
	'data/moves.ts:10903': { kind: 'MOVE_MAGNITUDE', decides: 'Magnitude power', dist: 'random(100)', out: 'int', actor: 'A', target: null, key: [], reach: 'no', why: 'Past.' },
	'data/moves.ts:11803': { kind: 'MOVE_CALL_PICK', decides: 'Metronome move', dist: 'sample', out: 'index', actor: 'A', target: null, key: [], reach: 'no', why: 'Past in Champions.' },
	'data/moves.ts:13926': { kind: 'MOVE_PRESENT', decides: 'Present power/heal', dist: 'random(10)', out: 'int', actor: 'A', target: null, key: [], reach: 'no', why: 'Past.' },
	'data/moves.ts:14294': { kind: 'MOVE_PSYWAVE', decides: 'Psywave damage', dist: 'random(50,151)', out: 'int', actor: 'A', target: null, key: [], reach: 'no', why: 'Past.' },
	'data/moves.ts:16232': { kind: 'MOVE_SHELLSIDEARM_TIE', decides: 'Shell Side Arm category when physical == special', dist: 'randomChance(1,2)', out: 'bool', actor: 'A', target: 'ev.source', key: ['move'], reach: 'yes', why: 'Short-circuit: draws only on an exact tie.' },
	'data/moves.ts:16887': { kind: 'MOVE_CALL_PICK', decides: 'move called by Sleep Talk', dist: 'sample(eligible moves)', out: 'index', actor: 'A', target: null, key: ['move'], reach: 'yes', why: 'Outcome index into the filtered move list (moveSlots order minus nosleeptalk/charge).' },
	'data/moves.ts:19853': { kind: 'SECONDARY_STATUS_PICK', decides: 'Tri Attack status (after the 20% SECONDARY passed)', dist: "sample(['brn','par','frz'])", out: 'index', actor: 'ev.source', target: 'ev.target', key: ['move', 'hit'], reach: 'yes', why: 'Second draw after SECONDARY for the same (actor,target).' },
	// ------------------------------------------------------------------ data/conditions.ts
	'data/conditions.ts:41': { kind: 'PARALYSIS', decides: 'full paralysis (base 25%)', dist: 'randomChance(1,4)', out: 'bool', actor: 'ev.target', target: null, key: [], reach: 'no', why: 'Overridden by champions conditions.ts:5 (12.5%).' },
	'data/conditions.ts:59': { kind: 'SLEEP_DURATION', decides: 'sleep turns (base 1-3)', dist: 'random(2,5)', out: 'int', actor: 'ev.target', target: null, key: [], reach: 'no', why: 'Overridden by champions conditions.ts:23.' },
	'data/conditions.ts:99': { kind: 'FREEZE_THAW', decides: 'thaw (base 20%)', dist: 'randomChance(1,5)', out: 'bool', actor: 'ev.target', target: null, key: [], reach: 'no', why: 'Overridden by champions conditions.ts:49.' },
	'data/conditions.ts:174': { kind: 'CONFUSION_DURATION', decides: 'confusion turns', dist: 'random(min,6), min=3 for Axe Kick else 2', out: 'int', actor: 'ev.target', target: null, key: [], reach: 'yes', why: 'onStart of the volatile; also fatigue confusion after lockedmove ends.' },
	'data/conditions.ts:187': { kind: 'CONFUSION_SELFHIT', decides: 'confused mon hits itself', dist: 'randomChance(33,100)', out: 'bool', actor: 'ev.target', target: null, key: [], reach: 'yes', why: 'Followed by a DAMAGE_ROLL from getConfusionDamage when true.' },
	'data/conditions.ts:227': { kind: 'PARTIAL_TRAP_DURATION', decides: 'binding turns (4-5 effective)', dist: 'random(5,7) (Grip Claw: 8, no draw)', out: 'int', actor: 'ev.target', target: 'ev.source', key: [], reach: 'yes', why: 'durationCallback: Bind/Wrap/Fire Spin/Whirlpool/Sand Tomb/Infestation/Snap Trap. battle.event may be the addVolatile context; actor taken from effectState/target.' },
	'data/conditions.ts:265': { kind: 'LOCKEDMOVE_DURATION', decides: 'Outrage/Thrash/Petal Dance/Raging Fury length', dist: 'random(2,4)', out: 'int', actor: 'ev.target', target: null, key: [], reach: 'yes', why: '' },
	'data/conditions.ts:305': { kind: 'RANDOM_TARGET', decides: 'target for a called two-turn move whose target fainted', dist: 'sample(foes)', out: 'index', actor: 'ev.target', target: null, key: ['move'], reach: 'cond', why: 'Only when a charge move is called by another move (Copycat) — very rare.' },
	'data/conditions.ts:456': { kind: 'PROTECT_STALL', decides: 'consecutive Protect/Detect/Endure/... succeeds', dist: 'randomChance(1, counter) counter=3,9,27..729', out: 'bool', actor: 'ev.target', target: null, key: [], reach: 'yes', why: 'No draw on the first use (no stall volatile yet).' },
	// ------------------------------------------------------------------ data/rulesets.ts
	'data/rulesets.ts:2623': { kind: 'RULE_CATEGORY_SWAP', decides: 'Category Swap Mod tie', dist: 'randomChance(1,2)', out: 'bool', actor: 'A', target: null, key: [], reach: 'no', why: 'Rule not in Reg M-C.' },
	// ------------------------------------------------------------------ data/mods/champions
	'data/mods/champions/abilities.ts:43': { kind: 'ABILITY_RESIDUAL_PROC', decides: 'Healer cures adjacent ally (Champions 50%)', dist: 'randomChance(1,2)', out: 'bool', actor: 'ev.target', target: 'ally', key: ['effect'], reach: 'yes', why: 'Per adjacent statused ally.' },
	'data/mods/champions/conditions.ts:5': { kind: 'PARALYSIS', decides: 'full paralysis (Champions 12.5%)', dist: 'randomChance(1,8)', out: 'bool', actor: 'ev.target', target: null, key: [], reach: 'yes', why: 'par onBeforeMove.' },
	'data/mods/champions/conditions.ts:23': { kind: 'SLEEP_DURATION', decides: 'sleep length (Champions: 2 w.p. 1/3, else 3)', dist: 'sample([2,3,3])', out: 'index', actor: 'ev.target', target: null, key: [], reach: 'yes', why: 'Drawn at status start. Record the value (2|3), not the index.' },
	'data/mods/champions/conditions.ts:49': { kind: 'FREEZE_THAW', decides: 'thaw before the 3-turn cap (Champions 25%)', dist: 'randomChance(1,4) (short-circuited when time<=0)', out: 'bool', actor: 'ev.target', target: null, key: [], reach: 'yes', why: 'No draw on the forced-thaw turn.' },
	'data/mods/champions/moves.ts:223': { kind: 'SECONDARY_STATUS_PICK', decides: 'Dire Claw status', dist: "sample(['psn','par','slp'])", out: 'index', actor: 'ev.source', target: 'ev.target', key: ['move', 'hit'], reach: 'yes', why: 'After the 50% SECONDARY passed.' },
	'data/mods/champions/scripts.ts:439': { kind: 'MULTIHIT_COUNT', decides: '2-5 hit count', dist: 'sample: 2,3 w.p. 35% each; 4,5 w.p. 15% each', out: 'int', actor: 'A', target: null, key: ['move'], reach: 'yes', why: 'Bullet Seed, Icicle Spear, Rock Blast, Scale Shot, Pin Missile, Bone Rush, Tail Slap, Water Shuriken. Skill Link sets multihit=5 (no draw).' },
	'data/mods/champions/scripts.ts:441': { kind: 'MULTIHIT_COUNT', decides: 'Loaded Dice reroll to 4-5', dist: '5 - random(2)', out: 'int', actor: 'A', target: null, key: ['move'], reach: 'no', why: 'Loaded Dice not available in Champions.' },
	'data/mods/champions/scripts.ts:444': { kind: 'MULTIHIT_COUNT', decides: 'gen<5 table', dist: 'sample', out: 'int', actor: 'A', target: null, key: ['move'], reach: 'no', why: 'gen < 5.' },
	'data/mods/champions/scripts.ts:447': { kind: 'MULTIHIT_COUNT', decides: 'generic [m,n] range', dist: 'random(m,n+1)', out: 'int', actor: 'A', target: null, key: ['move'], reach: 'no', why: 'No legal move has a multihit range other than [2,5].' },
	'data/mods/champions/scripts.ts:450': { kind: 'MULTIHIT_COUNT', decides: 'Loaded Dice Population Bomb', dist: '10 - random(7)', out: 'int', actor: 'A', target: null, key: ['move'], reach: 'no', why: 'Loaded Dice not available.' },
	'data/mods/champions/scripts.ts:507': { kind: 'ACCURACY', decides: 'per-hit accuracy for hits 2..n (Triple Axel, Population Bomb)', dist: 'randomChance(acc,100)', out: 'bool', actor: 'last.Accuracy.source', target: 'last.Accuracy.target', key: CORE_KEY, reach: 'yes', why: 'hit index disambiguates from the hit-1 ACCURACY draw.' },
};

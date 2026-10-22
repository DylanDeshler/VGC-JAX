'use strict';
/**
 * Trace instrumentation for one Battle instance (installed after InstrumentedPRNG.attach):
 *  - identity: PS Pokemon -> side*4 + brought index (b = team-preview order), and sheet index
 *  - step context for DrawKeys (contract/drawkey.md §3, §3.1): step_tag, step_actor, step_subject, action_ord
 *  - structured DrawKeys for every classified draw (contract §5) incl. don't-care / degenerate flags (§6, §7)
 *  - tie records replacing speedSort shuffles (contract §4), with the per-turn ordinals of 03 §4.6
 *
 * Like context.js this only installs instance-level wrappers; the PS checkout is never modified.
 * Mons are identified by `set.species`, which is unique per side (Species Clause) and survives
 * State serialization (Pokemon/set object identity does not).
 */
const { SINGLE_TARGET } = require('./classify');

const STEP = { COMMIT: 0, A_SWITCH: 1, A_MEGA: 2, A_OTHER: 3, B_MOVE: 4, RESIDUAL: 5, SWITCHIN: 6 };
// PS action.choice -> step_tag (contract §3.1)
const CHOICE_TAG = {
	switch: 1, instaswitch: 1, megaEvo: 2,
	beforeTurn: 3, beforeTurnMove: 3, revivalblessing: 3, priorityChargeMove: 3, start: 3, team: 3,
	move: 4, residual: 5, runSwitch: 6,
};
// actions that take part in SPEED_TIE_SELECT and advance select_ord (contract §4)
const SELECTABLE = new Set(['switch', 'instaswitch', 'megaEvo', 'beforeTurnMove', 'priorityChargeMove', 'revivalblessing', 'move']);
// runAction early-returns for these when the actor is inactive or fainted (PS sim/battle.ts:2756-2800)
const ACTOR_CHECKED = new Set(['move', 'beforeTurnMove', 'priorityChargeMove']);
// MUL-class hooks whose handler-sort ties are forced (contract §4 SPEED_TIE_HANDLER_MUL)
const MUL_EVENTS = new Set(['BasePower', 'ModifyAtk', 'ModifyDef', 'ModifySpA', 'ModifySpD', 'ModifySpe',
	'ModifyDamage', 'ModifyAccuracy', 'ModifyCritRatio', 'ModifySTAB']);

// inventory kind -> contract DrawKind (identity unless listed)
const KIND_MAP = { QUICK_CLAW: 'FRAC_PRIO', PRIORITY_PROC: 'FRAC_PRIO' };
const DONTCARE_KINDS = new Set(['RANDOM_TARGET_NOMINAL', 'RANDOM_TARGET_NOEFFECT', 'QUEUE_INSERT_TIE', 'TEAM_SORT', 'GENDER']);
// contract §5 subjects for condition-driven kinds
const COND_SUBJECT = {
	PARALYSIS: 'par', SLEEP_DURATION: 'slp', FREEZE_THAW: 'frz', CONFUSION_DURATION: 'confusion',
	CONFUSION_SELFHIT: 'confusion', CONFUSION_SELFHIT_DAMAGE_ROLL: 'confusion', PARTIAL_TRAP_DURATION: 'partiallytrapped',
	LOCKEDMOVE_DURATION: 'lockedmove', ALLY_SWITCH_STALL: 'allyswitch', INFATUATION: 'attract',
};
const HANDLER_KINDS = new Set(['ABILITY_ONHIT_PROC', 'ABILITY_RESIDUAL_PROC', 'ABILITY_TRACE_TARGET', 'ABILITY_FOREWARN_PICK',
	'ABILITY_PICKUP_TARGET', 'ABILITY_MOODY_STAT', 'FRAC_PRIO']);

// ------------------------------------------------------------------ identity
class Ident {
	constructor() {
		this.brought = [null, null]; // per side: [{species, sheet}] in brought order
		this.byspecies = [new Map(), new Map()];
	}
	/** At `start`, side.pokemon is the brought order (PS sim/battle.ts:2803-2810). */
	captureFromParty(battle) {
		for (const side of battle.sides) {
			const team = side.team;
			const list = side.pokemon.map(p => {
				let sheet = team.indexOf(p.set);
				if (sheet < 0) sheet = team.findIndex(t => t.species === p.set.species);
				return { species: p.set.species, sheet };
			});
			this.set(side.n, list);
		}
	}
	set(sideN, list) {
		this.brought[sideN] = list;
		this.byspecies[sideN] = new Map(list.map((e, b) => [e.species, b]));
	}
	b(p) {
		if (!p || !p.side || !p.set) return -1;
		const b = this.byspecies[p.side.n].get(p.set.species);
		return b === undefined ? -1 : b;
	}
	/** side*4 + brought index, or -1 */
	ref(p) {
		const b = this.b(p);
		return b < 0 ? -1 : p.side.n * 4 + b;
	}
	sheet(p) {
		const b = this.b(p);
		if (b >= 0) return this.brought[p.side.n][b].sheet;
		if (!p || !p.side) return -1;
		return p.side.team.findIndex(t => t.species === p.set.species);
	}
	toJSON() { return this.brought; }
}

// ------------------------------------------------------------------ helpers
const isPokemon = x => !!(x && x.side && x.species && x.set);
const slotRank = p => p.side.n * 2 + p.position; // p1a, p1b, p2a, p2b
function holderCode(h, ident) {
	if (!h) return '?';
	if (isPokemon(h)) return String(ident.ref(h));
	if (h.pokemonLeft !== undefined) return `s${h.n}`; // Side
	if (h.pseudoWeather !== undefined) return 'field';
	return 'battle';
}
function holderRank(h) {
	if (!h) return 99;
	if (isPokemon(h)) return 10 + slotRank(h);
	if (h.pokemonLeft !== undefined) return 1 + h.n;
	if (h.pseudoWeather !== undefined) return 0;
	return 98;
}
/** Lexicographic rank of permutation `perm` (array of distinct 0..n-1). */
function permRank(perm) {
	const n = perm.length; let rank = 0; let fact = 1;
	for (let i = n - 1; i >= 0; i--) {
		let smaller = 0;
		for (let j = i + 1; j < n; j++) if (perm[j] < perm[i]) smaller++;
		rank += smaller * fact;
		fact *= (n - i);
	}
	return rank;
}
/** Inverse of permRank. */
function permUnrank(rank, n) {
	const fact = [1]; for (let i = 1; i <= n; i++) fact[i] = fact[i - 1] * i;
	const pool = [...Array(n).keys()]; const out = [];
	for (let i = n - 1; i >= 0; i--) {
		const q = Math.floor(rank / fact[i]); rank %= fact[i];
		out.push(pool.splice(q, 1)[0]);
	}
	return out;
}
function isDegenerate(method, args) {
	if (method === 'randomChance') return args[0] >= args[1] || args[0] <= 0;
	if (method === 'sample') return args[0] <= 1;
	if (method === 'random') {
		if (args.length === 1) return args[0] <= 1;
		if (args.length === 2) return args[1] - args[0] <= 1;
	}
	return false;
}

// ------------------------------------------------------------------ install
function install(battle, prng) {
	const ctx = battle.__ctx;
	if (!ctx) throw new Error('instrument.install: context.install must run first (InstrumentedPRNG.attach)');
	const ident = new Ident();
	Object.defineProperty(battle, '__ident', { value: ident, enumerable: false, configurable: true });
	Object.assign(ctx, {
		inCommit: false, commitKind: '', inActionSpeed: 0, stack: [], curAction: null, curActionOrd: 0, curSelectOrd: -1,
		ordTurn: null, ords: null, moveActs: null, prepend: new WeakSet(), lastSecondaries: null, curSecondaryIdx: 0,
		dragOut: null, stepSerial: 0, occ: new Map(), occSerial: -1, ties: [], droppedTies: 0, unrecordedSelectTies: 0,
		addingVolatile: null, wrapped: new WeakSet(),
	});
	/** Per-Pokemon addVolatile wrapper: durationCallbacks (partiallytrapped) run outside any event, with
	 * `this` = the target (PS sim/pokemon.ts addVolatile). Installed once the Pokemon objects exist. */
	function wrapPokemon() {
		for (const side of battle.sides) {
			for (const p of side.pokemon) {
				if (!p || ctx.wrapped.has(p)) continue;
				ctx.wrapped.add(p);
				const addVolatile = p.addVolatile;
				p.addVolatile = function (...args) {
					const prev = ctx.addingVolatile; ctx.addingVolatile = this;
					try { return addVolatile.apply(this, args); } finally { ctx.addingVolatile = prev; }
				};
			}
		}
	}
	ident.wrapPokemon = wrapPokemon;
	/** Per-turn instrumentation state that must survive a mid-turn snapshot (03 §4.6 ordinals persist across
	 * SWITCH steps within a turn). Pokemon are stored as refs, queue entries as list indices. */
	ident.saveTurnState = function () {
		ensureTurn();
		const moveActs = [];
		for (const [p, n] of ctx.moveActs) moveActs.push([ident.ref(p), n]);
		const prepend = battle.queue.list.map((a, i) => (ctx.prepend.has(a) ? i : -1)).filter(i => i >= 0);
		return { ordTurn: ctx.ordTurn, ords: { ...ctx.ords }, moveActs, prepend };
	};
	ident.loadTurnState = function (st) {
		if (!st) return;
		ctx.ordTurn = st.ordTurn; ctx.ords = { ...st.ords }; ctx.moveActs = new Map();
		const byRef = new Map();
		for (const side of battle.sides) for (const p of side.pokemon) byRef.set(ident.ref(p), p);
		for (const [r, n] of st.moveActs) if (byRef.has(r)) ctx.moveActs.set(byRef.get(r), n);
		ctx.prepend = new WeakSet(st.prepend.map(i => battle.queue.list[i]).filter(Boolean));
	};
	function ensureTurn() {
		if (ctx.ordTurn !== battle.turn) {
			ctx.ordTurn = battle.turn;
			ctx.ords = { select: 0, runswitch: 0, fieldevent: 0, eachevent: 0, mulhook: 0 };
			ctx.moveActs = new Map();
		}
	}
	ensureTurn();

	/** Step context of the running queue action (contract §3, §3.1). */
	function stepInfo() {
		const a = ctx.curAction;
		if (ctx.inCommit || !a) return { tag: STEP.COMMIT, actor: -1, subject: null, actionOrd: 0, serial: ctx.stepSerial };
		const tag = ctx.inResolve ? STEP.COMMIT : CHOICE_TAG[a.choice] ?? 7;
		return {
			tag,
			actor: CHOICE_TAG[a.choice] === STEP.RESIDUAL ? -1 : ident.ref(a.pokemon),
			subject: a.move ? a.move.id : a.choice,
			actionOrd: a.choice === 'move' ? ctx.curActionOrd : 0,
			serial: ctx.stepSerial,
		};
	}
	/** occurrence ordinal of `k` within the current step instance */
	function occurrence(k) {
		if (ctx.occSerial !== ctx.stepSerial) { ctx.occ.clear(); ctx.occSerial = ctx.stepSerial; }
		const o = ctx.occ.get(k) || 0; ctx.occ.set(k, o + 1);
		return o;
	}
	function pushTie(kind, sub, ord, tied, value, extra) {
		const st = stepInfo();
		ctx.ties.push({
			kind, key: { turn: battle.turn, step_tag: st.tag, sub, ord }, tied, value,
			step: { actor: st.actor, subject: st.subject }, ...extra,
		});
	}

	// ---- commit / turn loop
	const commitChoices = battle.commitChoices;
	battle.commitChoices = function (...args) {
		ensureTurn();
		ctx.inCommit = true; ctx.commitKind = this.requestState; ctx.stepSerial++;
		try { return commitChoices.apply(this, args); } finally { ctx.inCommit = false; }
	};
	const turnLoop = battle.turnLoop;
	battle.turnLoop = function (...args) {
		ctx.inCommit = false; // commitChoices calls turnLoop at its end
		return turnLoop.apply(this, args);
	};
	const getActionSpeed = battle.getActionSpeed;
	battle.getActionSpeed = function (...args) {
		ctx.inActionSpeed++;
		try { return getActionSpeed.apply(this, args); } finally { ctx.inActionSpeed--; }
	};

	// ---- queue: prepend marks (04 §5.5) and sort ownership
	const q = battle.queue;
	const qsort = q.sort;
	q.sort = function (...args) {
		ctx.stack.push({ tag: 'queue' });
		try { return qsort.apply(this, args); } finally {
			ctx.stack.pop();
			// SWITCH-phase commit sorts only the new choices, which PS then puts in front of the old remainder
			// (sim/battle.ts commitChoices); any other sort (re-sort before a move) re-keys everything.
			ctx.prepend = new WeakSet();
			if (ctx.inCommit && ctx.commitKind === 'switch') for (const a of this.list) ctx.prepend.add(a);
		}
	};

	// ---- runAction: step context, select ties, ordinals
	const runAction = battle.runAction;
	battle.runAction = function (action) {
		ensureTurn();
		if (action.choice === 'start') { ident.captureFromParty(this); wrapPokemon(); }
		const passes = a => !ACTOR_CHECKED.has(a.choice) || (a.pokemon && a.pokemon.isActive && !a.pokemon.fainted);
		const prevSelect = ctx.curSelectOrd;
		ctx.stepSerial++;
		ctx.curAction = action;
		if (SELECTABLE.has(action.choice) && passes(action)) {
			ctx.curSelectOrd = ctx.ords.select++;
			const tied = selectTied(action);
			if (ctx.tieForcer && tied) {
				const want = ctx.tieForcer('SPEED_TIE_SELECT', selectKey(ctx.curSelectOrd));
				const e = want === undefined ? null : tied.find(x => ident.ref(x.pokemon) === want);
				if (e && e !== action) { // forced winner: execute e now, leave `action` in e's queue position
					const q = battle.queue.list; q[q.indexOf(e)] = action;
					action = e; ctx.curAction = action;
				}
			}
			if (tied) pushTie('SPEED_TIE_SELECT', ctx.curSelectOrd, 0,
				tied.map(e => ident.ref(e.pokemon)).sort((a, b) => a - b), ident.ref(action.pokemon));
		}
		if (action.choice === 'move' && passes(action)) {
			const p = action.pokemon;
			ctx.curActionOrd = ctx.moveActs.get(p) || 0;
			ctx.moveActs.set(p, ctx.curActionOrd + 1);
		}
		try { return runAction.call(this, action); } finally { ctx.curAction = null; ctx.curSelectOrd = prevSelect; }
	};
	function selectKey(selectOrd) {
		return { turn: battle.turn, step_tag: stepInfo().tag, sub: selectOrd, ord: 0 };
	}
	/** Entries tied with `action` on the full comparePriority key (same prepend class), restricted to active,
	 * non-fainted actors (contract §4 SPEED_TIE_SELECT); null when fewer than 2. */
	function selectTied(action) {
		const pre = ctx.prepend.has(action);
		const tied = [action];
		let unrecorded = 0;
		for (const e of battle.queue.list) {
			if (!e.pokemon || ctx.prepend.has(e) !== pre || battle.comparePriority(action, e) !== 0) continue;
			if (e.pokemon.isActive && !e.pokemon.fainted) tied.push(e); else unrecorded++;
		}
		const me = action.pokemon;
		if (!(me && me.isActive && !me.fainted)) {
			if (tied.length > 1) ctx.unrecordedSelectTies++;
			return null;
		}
		if (unrecorded) ctx.unrecordedSelectTies++;
		return tied.length >= 2 ? tied : null;
	}

	// ---- sort owners
	const fieldEvent = battle.fieldEvent;
	battle.fieldEvent = function (...args) {
		ensureTurn();
		ctx.stack.push({ tag: 'fieldEvent', id: args[0], ord: ctx.ords.fieldevent++ });
		try { return fieldEvent.apply(this, args); } finally { ctx.stack.pop(); }
	};
	const eachEvent = battle.eachEvent;
	battle.eachEvent = function (...args) {
		ensureTurn();
		const fr = args[0] === 'Weather' ? { tag: 'eachWeather', ord: ctx.ords.eachevent++ } : { tag: 'each', id: args[0] };
		ctx.stack.push(fr);
		try { return eachEvent.apply(this, args); } finally { ctx.stack.pop(); }
	};
	const runEvent = battle.runEvent;
	battle.runEvent = function (...args) {
		ensureTurn();
		const id = args[0];
		ctx.stack.push({ tag: 'runEvent', id, ord: MUL_EVENTS.has(id) ? ctx.ords.mulhook++ : -1 });
		try {
			const r = runEvent.apply(this, args);
			if (id === 'ModifySecondaries') ctx.lastSecondaries = r;
			return r;
		} finally { ctx.stack.pop(); }
	};
	const A = battle.actions;
	const runSwitch = A.runSwitch;
	A.runSwitch = function (...args) {
		ensureTurn();
		ctx.stack.push({ tag: 'runSwitch', ord: ctx.ords.runswitch++ });
		try { return runSwitch.apply(this, args); } finally { ctx.stack.pop(); }
	};
	const dragIn = A.dragIn;
	A.dragIn = function (side, pos) {
		const prev = ctx.dragOut; ctx.dragOut = side.active[pos];
		try { return dragIn.call(this, side, pos); } finally { ctx.dragOut = prev; }
	};

	// ---- speedSort: tie records (contract §4) and tie forcing (replay mode)
	const speedSort = battle.speedSort;
	battle.speedSort = function (list, comparator) {
		const top = ctx.stack[ctx.stack.length - 1] || { tag: '' };
		const spec = list.length >= 2 ? sortSpec(top, list) : null;
		const prevSort = ctx.curSort; ctx.curSort = spec ? { ...spec, gi: 0 } : null;
		try {
			const r = speedSort.call(this, list, comparator);
			if (spec && spec.kind) emitSortTies(spec, list, comparator || this.comparePriority);
			else if (spec) countDropped(list, comparator || this.comparePriority);
			return r;
		} finally { ctx.curSort = prevSort; }
	};
	const handlerMember = h => `${holderCode(h.effectHolder, ident)}:${h.effect?.id || ''}`;
	const handlerOrder = (a, b) => (holderRank(a.effectHolder) - holderRank(b.effectHolder)) ||
		((a.effect?.id || '') < (b.effect?.id || '') ? -1 : (a.effect?.id || '') > (b.effect?.id || '') ? 1 : 0);
	const monMember = p => ident.ref(p);
	const bySlot = (a, b) => slotRank(a) - slotRank(b);
	const byRef = (a, b) => ident.ref(a) - ident.ref(b);
	// omit handlers fieldEvent will skip (fainted holder unless slot condition; sim/battle.ts:517-519)
	const fieldFilter = g => g.filter(h => !(isPokemon(h.effectHolder) && h.effectHolder.fainted && !h.state?.isSlotCondition));
	/** Which tie kind (if any) a speedSort produces, from its caller and element type. kind=null: don't-care. */
	function sortSpec(top, list) {
		const t = top.tag; const pokemonList = isPokemon(list[0]);
		if (t === 'queue') return { kind: null, queue: true }; // SPEED_TIE_SELECT is handled at execution
		if (t === 'runSwitch' && pokemonList) return { kind: 'SPEED_TIE_SWITCHIN', sub: top.ord, member: monMember, order: bySlot };
		if (t === 'eachWeather' && pokemonList) return { kind: 'SPEED_TIE_WEATHER_EACH', sub: top.ord, member: monMember, order: bySlot };
		if (t === 'fieldEvent' && !pokemonList) {
			return { kind: 'SPEED_TIE_FIELD_EVENT', sub: top.ord, member: handlerMember, order: handlerOrder, filter: fieldFilter };
		}
		if (t === 'runEvent' && !pokemonList && top.ord >= 0) {
			return { kind: 'SPEED_TIE_HANDLER_MUL', sub: top.ord, member: handlerMember, order: handlerOrder };
		}
		if (pokemonList && battle.effect?.id === 'magician') {
			return { kind: 'SPEED_TIE_ABILITY', sub: ctx.curSelectOrd, member: monMember, order: byRef };
		}
		return { kind: null };
	}
	function tieGroups(list, cmp) {
		const groups = [];
		for (let i = 0; i < list.length;) {
			let j = i + 1;
			while (j < list.length && cmp(list[i], list[j]) === 0) j++;
			if (j - i >= 2) groups.push(list.slice(i, j));
			i = j;
		}
		return groups;
	}
	function countDropped(list, cmp) {
		const spec = ctx.curSort;
		if (spec && spec.queue) return;
		ctx.droppedTies += tieGroups(list, cmp).length; // don't-care tie kinds (contract §6)
	}
	function emitSortTies(spec, list, cmp) {
		let gi = 0;
		for (let g of tieGroups(list, cmp)) {
			if (spec.filter) { g = spec.filter(g); if (g.length < 2) continue; }
			const canon = g.slice().sort(spec.order);
			const perm = g.map(x => canon.indexOf(x));
			pushTie(spec.kind, spec.sub, gi++, canon.map(spec.member), permRank(perm), { order: g.map(spec.member) });
		}
	}
	/** Replay mode: permutation for a speedSort shuffle of `tied` (PS pre-shuffle order), or undefined. */
	ctx.shuffleForcer = function (tied) {
		const spec = ctx.curSort;
		if (!spec || !spec.kind || !ctx.tieForcer) return undefined;
		const g = spec.filter ? spec.filter(tied) : tied;
		if (g.length < 2) return undefined;
		const key = { turn: battle.turn, step_tag: stepInfo().tag, sub: spec.sub, ord: spec.gi++ };
		const v = ctx.tieForcer(spec.kind, key);
		if (v === undefined) return undefined;
		const canon = g.slice().sort(spec.order);
		const want = permUnrank(v, canon.length).map(i => canon[i]); // desired order of the members
		const slots = tied.map((x, i) => (g.includes(x) ? i : -1)).filter(i => i >= 0);
		const perm = tied.map((_, i) => i);
		slots.forEach((pos, k) => { perm[pos] = tied.indexOf(want[k]); });
		return perm;
	};

	// ---- structured DrawKeys (called by InstrumentedPRNG._begin after classification)
	prng.onClassified = function (d, c, b, callArgs) {
		if (d.method === 'shuffle') return;
		d.dk = buildDrawKey(d, c, callArgs);
	};
	function buildDrawKey(d, c, callArgs) {
		const st = stepInfo();
		const am = battle.activeMove; const ev = battle.event || {};
		const hit = am ? (am.hit || 0) : 0;
		const ref = x => ident.ref(x);
		const holder = battle.effectState?.target;
		let kind = KIND_MAP[c.kind] || c.kind;
		let actor = -1; let target = -1; let subject = null; let h = 0; let ord = 0;
		const extra = {};
		switch (kind) {
		case 'ACCURACY': case 'CRIT': case 'DAMAGE_ROLL':
			actor = ref(c.actorObj); target = ref(c.targetObj); subject = am?.id ?? null; h = hit; break;
		case 'CONFUSION_SELFHIT_DAMAGE_ROLL':
			actor = target = ref(c.actorObj); break;
		case 'SECONDARY': {
			actor = ref(c.actorObj); target = ref(c.targetObj); subject = am?.id ?? null; h = hit;
			const k = occurrence(`SECONDARY|${actor}|${target}|${subject}|${h}`);
			const list = ctx.lastSecondaries; const sec = Array.isArray(list) ? list[k] : null;
			const idx = sec && am?.secondaries ? am.secondaries.indexOf(sec) : -1;
			ord = idx >= 0 ? idx : k;
			ctx.curSecondaryIdx = ord;
			extra.chance = sec ? (sec.chance ?? null) : null;
			if (sec && sec.chance === undefined) extra.dontcare = true;
			break;
		}
		case 'SECONDARY_STATUS_PICK':
			actor = ref(ev.source); target = ref(ev.target); subject = am?.id ?? null; h = hit; ord = ctx.curSecondaryIdx; break;
		case 'SELF_EFFECT_ROLL':
			actor = ref(c.actorObj || battle.activePokemon); subject = am?.id ?? null; h = hit;
			extra.chance = am?.self?.chance ?? null;
			if (!(am && am.self && am.self.chance !== undefined)) extra.dontcare = true;
			break;
		case 'MULTIHIT_COUNT':
			actor = ref(c.actorObj || battle.activePokemon); subject = am?.id ?? null; break;
		case 'RANDOM_TARGET': case 'RANDOM_TARGET_NOMINAL': {
			const mv = ctx.rtMove ? (typeof ctx.rtMove === 'string' ? battle.dex.moves.get(ctx.rtMove) : ctx.rtMove) : am;
			actor = ref(c.actorObj); subject = mv?.id ?? null;
			ord = ctx.inGetTarget ? 1 : ctx.inResolve ? 0 : 2;
			if (kind === 'RANDOM_TARGET') {
				if (mv && !SINGLE_TARGET.has(mv.target)) kind = 'RANDOM_TARGET_NOMINAL';
				else if (ctx.inActionSpeed || (ctx.inResolve && mv?.target === 'randomNormal')) kind = 'RANDOM_TARGET_NOEFFECT';
			}
			break;
		}
		case 'FORCED_SWITCH_PICK':
			actor = ref(ctx.dragOut); subject = ctx.curAction?.move?.id ?? null; break;
		case 'ITEM_FOCUS_BAND': {
			actor = ref(holder || c.actorObj); target = ref(ev.source); subject = ev.effect?.id ?? null;
			h = ev.effect?.effectType === 'Move' ? hit : 0;
			ord = occurrence(`FOCUSBAND|${actor}|${subject}`);
			break;
		}
		case 'MOVE_FICKLE_BEAM':
			actor = ref(c.actorObj); target = ref(c.targetObj); subject = am?.id ?? null; h = hit; break;
		case 'MOVE_SHELLSIDEARM_TIE': case 'MOVE_ACUPRESSURE_STAT':
			actor = ref(c.actorObj); target = ref(c.targetObj); subject = am?.id ?? null; break;
		case 'MOVE_CALL_PICK':
			actor = ref(c.actorObj); subject = am?.id ?? null; break;
		case 'PARTIAL_TRAP_DURATION':
			actor = ref(ctx.addingVolatile || holder || c.actorObj); subject = 'partiallytrapped'; break;
		case 'PROTECT_STALL':
			actor = ref(holder || c.actorObj); subject = am?.id ?? null; break;
		default:
			if (COND_SUBJECT[kind]) {
				actor = ref(holder || c.actorObj); subject = COND_SUBJECT[kind];
			} else if (HANDLER_KINDS.has(kind)) {
				const hold = holder || c.actorObj;
				actor = ref(hold); subject = battle.effect?.id ?? null;
				if (kind === 'ABILITY_ONHIT_PROC') {
					const other = ev.target === hold ? ev.source : ev.target;
					target = ref(other); h = hit;
				}
				if (kind === 'ABILITY_MOODY_STAT') ord = occurrence(`MOODY|${actor}`);
			} else {
				actor = ref(c.actorObj); target = ref(c.targetObj);
			}
		}
		if (DONTCARE_KINDS.has(kind)) extra.dontcare = true;
		if (d.method === 'sample') Object.defineProperty(d, '_items', { value: callArgs[0], enumerable: false });
		return {
			kind,
			key: {
				turn: battle.turn, step_tag: st.tag, step_actor: st.actor, step_subject: st.subject,
				actor, target, subject, hit: h, ord, sub: 0, action_ord: st.actionOrd,
			},
			degenerate: isDegenerate(d.method, d.args),
			dontcare: !!extra.dontcare,
			chance: extra.chance,
		};
	}
	return ident;
}

/** Trace-format draw record from an InstrumentedPRNG draw (07 §3). `pick` = sampled item in the engine domain. */
function drawRecord(d, ident) {
	const dk = d.dk;
	const rec = {
		kind: dk ? dk.kind : (d.kind || 'UNMAPPED'), key: dk ? dk.key : null,
		method: d.method, args: d.args, value: d.out, site: d.site || null,
		dontcare: dk ? dk.dontcare : false, degenerate: dk ? dk.degenerate : false,
	};
	if (dk && dk.chance !== undefined) rec.chance = dk.chance;
	if (d.method === 'sample' && d._items) {
		const it = d._items[d.out];
		if (isPokemon(it)) rec.pick = dk && dk.kind === 'FORCED_SWITCH_PICK' ? ident.b(it) : ident.ref(it);
		else if (it === null || ['number', 'string', 'boolean'].includes(typeof it)) rec.pick = it;
		else if (it && it.id) rec.pick = it.id;
	}
	if (d.markerKind && d.markerKind !== d.kind) rec.classifierDisagree = d.markerKind;
	return rec;
}

module.exports = {
	install, drawRecord, Ident, permRank, permUnrank, isDegenerate, STEP, CHOICE_TAG, SELECTABLE, MUL_EVENTS, KIND_MAP, DONTCARE_KINDS,
};

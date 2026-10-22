'use strict';
/**
 * Verified damage examples for the Champions doubles engine spec.
 * For every example we run the real PS sim (Champions mod, Reg M-C format) once per forced damage
 * roll r in 0..15 (PS draws random(16) and multiplies by 100 - r), record the damage PS dealt,
 * and compare against `ref*` — an explicit-integer reference implementation of the PS formula.
 */
const H = require('./harness');
const { Dex } = H;
const dex = Dex.mod('champions');

// ---------------- reference implementation (explicit integer ops) ----------------
const u32 = x => x >>> 0;                         // Dex.trunc(num) : sim/dex.ts:365-368
const tr16 = x => (x >>> 0) % 65536;              // Dex.trunc(num, 16)
const M = (num, den = 1) => u32(num * 4096 / den); // 4096-based modifier as used by modify/chain
function modify(v, m4096) { return u32((u32(v * m4096) + 2048 - 1) / 4096); } // battle.ts:2348-2359
function chain(m1, m2) { return (m1 * m2 + 2048) >> 12; }                     // battle.ts:2337-2346
const BOOST = [[1, 1], [3, 2], [2, 1], [5, 2], [3, 1], [7, 2], [4, 1]];       // pokemon.ts:604 (as fractions)
function boosted(stat, b) {                        // pokemon.ts:605-611 (floor of float mult/div)
	if (b > 6) b = 6; if (b < -6) b = -6;
	const [n, d] = BOOST[Math.abs(b)];
	return b >= 0 ? Math.floor(stat * n / d) : Math.floor(stat * d / n);
}
function champStat(species, statName, sp, nature) { // data/mods/champions/scripts.ts:10-40 (no Level Clause Mod)
	const base = dex.species.get(species).baseStats[statName];
	if (statName === 'hp') return base + sp + 75;
	let stat = base + sp + 20;
	const nat = dex.natures.get(nature);
	if (nat.plus === statName) stat = u32(tr16(stat * 110) / 100);
	else if (nat.minus === statName) stat = u32(tr16(stat * 90) / 100);
	return stat;
}
/**
 * p = { level, bp, bpMods:[m4096...] (in handler order), atk, def, atkBoost, defBoost, crit, spread,
 *       weatherMods:[...], stab:m4096|null, typeMod:int, burn:bool, finalMods:[...], roll:r }
 */
function refDamage(p) {
	// BasePower event: chained modifier applied once at the end of runEvent (battle.ts:936-940)
	let bpM = 4096; for (const m of p.bpMods || []) bpM = chain(bpM, m);
	let bp = modify(p.bp, bpM); if (bp < 1) bp = 1;
	// crit ignores negative offensive and positive defensive boosts (battle-actions.ts:1740-1755)
	let ab = p.atkBoost || 0, db = p.defBoost || 0;
	if (p.crit && ab < 0) ab = 0;
	if (p.crit && db > 0) db = 0;
	let A = boosted(p.atk, ab), D = boosted(p.def, db);
	let aM = 4096; for (const m of p.atkMods || []) aM = chain(aM, m); A = modify(A, aM);
	let dM = 4096; for (const m of p.defMods || []) dM = chain(dM, m); D = modify(D, dM);
	// base damage (battle-actions.ts:1772)
	let d = u32(u32(u32(u32(2 * p.level / 5 + 2) * bp * A) / D) / 50);
	// modifyDamage (data/mods/champions/scripts.ts:196-312)
	d += 2;
	if (p.spread) d = modify(d, 3072);
	if (p.parentalBondHit2) d = modify(d, 1024);
	let wM = 4096; for (const m of p.weatherMods || []) wM = chain(wM, m); d = modify(d, wM);
	if (p.crit) d = u32(d * 1.5);
	d = u32(u32(d * (100 - p.roll)) / 100);
	if (p.stab) d = modify(d, p.stab);
	const tm = Math.max(-6, Math.min(6, p.typeMod));
	if (tm > 0) for (let i = 0; i < tm; i++) d *= 2;
	if (tm < 0) for (let i = 0; i > tm; i--) d = u32(d / 2);
	if (p.burn) d = modify(d, 2048);
	let fM = 4096; for (const m of p.finalMods || []) fM = chain(fM, m); d = modify(d, fM);
	if (!d) return 1;
	return tr16(d);
}

// ---------------- PS runner ----------------
const filler = [
	H.set('Milotic', 'Leftovers', 'Competitive', ['Scald', 'Protect']),
	H.set('Metagross', 'Leftovers', 'Clear Body', ['Iron Head', 'Protect']),
];

/** Run one turn with a forced roll; return {damage: Map(targetName -> dmg list), log, prng} */
function runPS(ex, roll, extra = {}) {
	const b = H.makeBattle([...ex.p1, ...filler], [...ex.p2, ...filler]);
	if (ex.setup) ex.setup(b);
	const prng = H.instrumentPRNG(b, d => {
		const c = d.callers.join(' ');
		if (/getDamage/.test(c) && d.from === 24) return ex.crit ? 0 : 23;   // crit roll: random(24) < 1
		if (/hitStepAccuracy/.test(c)) return 0;                             // always hit
		if (/randomizer/.test(c)) return roll;                               // damage roll r -> *(100-r)/100
		if (/secondaries|selfDrops/.test(c)) return 99;                      // no secondary procs
		return undefined;
	});
	const dealt = [];
	const origSD = b.spreadDamage;
	b.spreadDamage = function (damage, targets, source, effect, instafaint) {
		const ret = origSD.call(this, damage, targets, source, effect, instafaint);
		if (effect && effect.effectType === 'Move' && targets) {
			for (const [i, t] of targets.entries()) if (t && t !== source) dealt.push({ target: `${t.side.id}${'ab'[t.position]}:${t.name}`, move: effect.id, damage: ret[i], hit: effect.hit });
		}
		return ret;
	};
	b.makeChoices(ex.c1, ex.c2);
	return { dealt, log: b.log, prng, battle: b };
}

function verify(ex) {
	console.log(`\n### ${ex.name}`);
	const rows = [];
	let ok = true;
	let sample;
	for (let roll = 0; roll < 16; roll++) {
		const r = runPS(ex, roll);
		if (roll === 0) sample = r;
		for (const [ti, tgt] of ex.targets.entries()) {
			const got = r.dealt.filter(x => x.target === tgt.name).map(x => x.damage);
			const exp = tgt.ref(roll);
			const expArr = Array.isArray(exp) ? exp : [exp];
			const match = JSON.stringify(got) === JSON.stringify(expArr);
			ok = ok && match;
			rows.push({ roll, target: tgt.name, ps: got.join('+'), ref: expArr.join('+'), match });
		}
	}
	for (const tgt of ex.targets) {
		const ps = rows.filter(x => x.target === tgt.name).map(x => x.ps);
		const ref = rows.filter(x => x.target === tgt.name).map(x => x.ref);
		console.log(`${tgt.name}: PS  rolls r=0..15 -> [${ps.join(', ')}]`);
		console.log(`${' '.repeat(tgt.name.length)}  ref rolls r=0..15 -> [${ref.join(', ')}]`);
	}
	console.log(ok ? 'ALL 16 ROLLS MATCH' : 'MISMATCH', rows.filter(x => !x.match));
	if (ex.trace) ex.trace(sample);
	return ok;
}

// ===================================================================================
// Example 1: spread Earthquake in doubles (ally immune), STAB, SE/resisted, Life Orb.
// ===================================================================================
const chompAtk = champStat('Garchomp', 'atk', 32, 'Jolly');
const ex1 = {
	name: 'Ex1: Garchomp Life Orb Earthquake (spread) vs Incineroar (2x) and Rillaboom (0.5x); ally Corviknight immune',
	p1: [
		H.set('Garchomp', 'Life Orb', 'Rough Skin', ['Earthquake'], 'Jolly', { hp: 2, atk: 32, spe: 32 }),
		H.set('Corviknight', 'Leftovers', 'Pressure', ['Iron Defense']),
	],
	p2: [
		H.set('Incineroar', 'Leftovers', 'Blaze', ['Bulk Up'], 'Careful', { hp: 32, def: 20, spd: 14 }),
		H.set('Rillaboom', 'Leftovers', 'Overgrow', ['Bulk Up'], 'Adamant', { hp: 32, atk: 32, def: 2 }),
	],
	c1: 'move 1, move 1', c2: 'move 1, move 1',
	targets: [
		{ name: 'p2a:Incineroar', ref: roll => refDamage({ level: 50, bp: 100, atk: chompAtk, def: champStat('Incineroar', 'def', 20, 'Careful'),
			spread: true, stab: 6144, typeMod: 1, finalMods: [5324], roll }) },
		{ name: 'p2b:Rillaboom', ref: roll => refDamage({ level: 50, bp: 100, atk: chompAtk, def: champStat('Rillaboom', 'def', 2, 'Adamant'),
			spread: true, stab: 6144, typeMod: -1, finalMods: [5324], roll }) },
	],
	trace(r) {
		console.log('PS log excerpt:\n' + r.log.filter(l => /\|move\||-immune|-supereffective|-resisted|-damage\|p2[ab]: [A-Za-z]+\|\d+\/\d+$|\[spread\]/.test(l) && !/\/100/.test(l)).join('\n'));
		console.log(`stats: Garchomp Atk=${chompAtk} Incineroar Def=${champStat('Incineroar', 'def', 20, 'Careful')} Rillaboom Def=${champStat('Rillaboom', 'def', 2, 'Adamant')}`);
	},
};

// ===================================================================================
// Example 2: crit Close Combat; attacker -1 Atk & burned, defender +2 Def behind Reflect; Life Orb; 4x.
// ===================================================================================
const snAtk = champStat('Sneasler', 'atk', 32, 'Adamant');
const kgDef = champStat('Kingambit', 'def', 32, 'Impish');
const ex2 = {
	name: 'Ex2: crit Sneasler Life Orb Mach Punch (burned, -1 Atk) vs Kingambit (+2 Def, Reflect up), 4x',
	crit: true,
	p1: [
		H.set('Sneasler', 'Life Orb', 'Unburden', ['Mach Punch'], 'Adamant', { atk: 32, spe: 32 }),
		H.set('Corviknight', 'Leftovers', 'Pressure', ['Iron Defense']),
	],
	p2: [
		H.set('Kingambit', 'Leftovers', 'Supreme Overlord', ['Swords Dance'], 'Impish', { hp: 32, def: 32 }),
		H.set('Rillaboom', 'Leftovers', 'Overgrow', ['Bulk Up'], 'Adamant', { hp: 32, atk: 32 }),
	],
	setup(b) {
		const sn = b.p1.active[0], kg = b.p2.active[0];
		sn.setStatus('brn', sn); sn.boosts.atk = -1; kg.boosts.def = 2;
		b.p2.addSideCondition('reflect', kg);
	},
	c1: 'move 1 1, move 1', c2: 'move 1, move 1',
	targets: [
		{ name: 'p2a:Kingambit', ref: roll => refDamage({ level: 50, bp: 40, atk: snAtk, def: kgDef, atkBoost: -1, defBoost: 2, crit: true,
			stab: 6144, typeMod: 2, burn: true, finalMods: [5324 /* Life Orb; Reflect skipped on crit */], roll }) },
	],
	trace(r) {
		console.log('PS log excerpt:\n' + r.log.filter(l => /\|move\||-crit|-supereffective|-damage\|p2a: [A-Za-z]+\|\d+\/\d+/.test(l) && !/\/100/.test(l)).join('\n'));
		console.log(`stats: Sneasler Atk=${snAtk} (boost -1 ignored on crit) Kingambit Def=${kgDef} (+2 ignored on crit)`);
		console.log('RNG draws (turn 1):'); for (const d of r.prng.filter(d => d.turn === 1)) console.log(`  #${d.i} random(${d.from}${d.to ? ',' + d.to : ''}) -> ${d.result}${d.forced ? ' (forced)' : ''}  ${d.callers.slice(0, 3).join(' < ')}`);
	},
};

// ===================================================================================
// Example 3: rain + Helping Hand + Mystic Water + Light Screen (doubles 2732) special, 2x.
// ===================================================================================
const pelSpA = champStat('Pelipper', 'spa', 32, 'Modest');
const zardSpD = champStat('Incineroar', 'spd', 32, 'Careful');
const ex3 = {
	name: 'Ex3: Pelipper Mystic Water Water Pulse in rain + Helping Hand vs Incineroar behind Light Screen, 2x',
	p1: [
		H.set('Pelipper', 'Mystic Water', 'Drizzle', ['Water Pulse'], 'Modest', { hp: 32, spa: 32 }),
		H.set('Whimsicott', 'Focus Sash', 'Prankster', ['Helping Hand'], 'Timid', { spe: 32 }),
	],
	p2: [
		H.set('Incineroar', 'Leftovers', 'Blaze', ['Bulk Up'], 'Careful', { hp: 32, spd: 32 }),
		H.set('Rillaboom', 'Leftovers', 'Overgrow', ['Bulk Up'], 'Adamant', { hp: 32, atk: 32 }),
	],
	setup(b) { b.p2.addSideCondition('lightscreen', b.p2.active[0]); },
	c1: 'move 1 1, move 1 -1', c2: 'move 1, move 1',
	targets: [
		{ name: 'p2a:Incineroar', ref: roll => refDamage({ level: 50, bp: 60,
			bpMods: [M(4915, 4096) /* Mystic Water prio 15 */, M(1.5) /* Helping Hand prio 10 */],
			atk: pelSpA, def: zardSpD, weatherMods: [M(1.5)], stab: 6144, typeMod: 1, finalMods: [2732 /* Light Screen, doubles */], roll }) },
	],
	trace(r) {
		console.log('PS log excerpt:\n' + r.log.filter(l => /\|move\||-singleturn|-supereffective|-damage\|p2a: [A-Za-z]+\|\d+\/\d+/.test(l) && !/\/100/.test(l)).join('\n'));
		const bpM = chain(M(4915, 4096), 6144);
		console.log(`BasePower chain: chain(4915, 6144) = ${bpM}; BP = modify(60, ${bpM}) = ${modify(60, bpM)}; Pelipper SpA=${pelSpA} Incineroar SpD=${zardSpD}`);
	},
};

// ===================================================================================
// Example 4: Mega Kangaskhan (Parental Bond) Double-Edge: two hits, 2nd at 1/4; recoil on total.
// ===================================================================================
const kanAtk = champStat('Kangaskhan-Mega', 'atk', 32, 'Adamant');
const garDef = champStat('Garchomp', 'def', 0, 'Hardy');
const ex4 = {
	name: 'Ex4: Mega Kangaskhan Parental Bond Double-Edge vs Garchomp (two hits, second x0.25)',
	p1: [
		H.set('Kangaskhan', 'Kangaskhanite', 'Scrappy', ['Double-Edge'], 'Adamant', { atk: 32, spe: 32 }),
		H.set('Corviknight', 'Leftovers', 'Pressure', ['Iron Defense']),
	],
	p2: [
		H.set('Garchomp', 'Leftovers', 'Rough Skin', ['Swords Dance'], 'Hardy', { hp: 32 }),
		H.set('Rillaboom', 'Leftovers', 'Overgrow', ['Bulk Up'], 'Adamant', { hp: 32, atk: 32 }),
	],
	c1: 'move 1 1 mega, move 1', c2: 'move 1, move 1',
	targets: [
		{ name: 'p2a:Garchomp', ref: roll => [
			refDamage({ level: 50, bp: 120, atk: kanAtk, def: garDef, stab: 6144, typeMod: 0, roll }),
			refDamage({ level: 50, bp: 120, atk: kanAtk, def: garDef, stab: 6144, typeMod: 0, parentalBondHit2: true, roll }),
		] },
	],
	trace(r) {
		console.log('PS log excerpt:\n' + r.log.filter(l => /\|move\||-mega|-hitcount|-damage\|p[12][ab]: [A-Za-z]+\|\d+\/\d+/.test(l) && !/\/100/.test(l)).join('\n'));
		console.log(`stats: Kangaskhan-Mega Atk=${kanAtk} Garchomp Def=${garDef}`);
		console.log('RNG draws (turn 1):'); for (const d of r.prng.filter(d => d.turn === 1)) console.log(`  #${d.i} random(${d.from}${d.to ? ',' + d.to : ''}) -> ${d.result}${d.forced ? ' (forced)' : ''}  ${d.callers.slice(0, 3).join(' < ')}`);
	},
};

let all = true;
for (const ex of [ex1, ex2, ex3, ex4]) all = verify(ex) && all;
console.log(all ? '\nALL EXAMPLES VERIFIED' : '\nSOME EXAMPLES FAILED');
module.exports = { refDamage, modify, chain, champStat, boosted };

'use strict';
/**
 * Stage-restricted random team generator (M2.9 core teams; reusable for any stage list in data/export/stages).
 * Species Clause by base species, abilities / moves / items restricted to the stage, random legal SP (<=32 each,
 * <=66 total) and nature, gender always set. Every team is checked with TeamValidator; invalid draws are retried.
 * Moves come from the species' learnset (incl. prevo learnsets) and are re-checked by the validator.
 */
const path = require('path');
const { Dex, TeamValidator, FORMAT } = require('./ps');

const dex = Dex.mod('champions');
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const cache = new Map();

function stageInfo(stage) {
	if (cache.has(stage)) return cache.get(stage);
	const st = require(path.join(__dirname, '..', '..', 'data', 'export', 'stages', `${stage}.json`));
	const moves = new Set(st.moves); const abilities = new Set(st.abilities); const items = st.items.slice();
	const pool = [];
	for (const id of st.species) {
		const sp = dex.species.get(id);
		if (sp.battleOnly || sp.isMega) continue;
		const abs = Object.values(sp.abilities).map(a => dex.toID(a)).filter(a => abilities.has(a));
		if (!abs.length) continue;
		const learn = new Set();
		let cur = sp;
		while (cur && cur.exists) {
			const ls = dex.species.getLearnsetData(cur.id);
			for (const m of Object.keys((ls && ls.learnset) || {})) if (moves.has(m)) learn.add(m);
			cur = cur.prevo ? dex.species.get(cur.prevo) : (cur.changesFrom ? dex.species.get(cur.changesFrom) : null);
		}
		if (learn.size < 1) continue;
		pool.push({ id: sp.id, name: sp.name, num: sp.num, abilities: abs, moves: [...learn].sort(), gender: sp.gender,
			genderRatio: sp.genderRatio });
	}
	const info = { pool, items };
	cache.set(stage, info);
	return info;
}

function randomSP(rng) {
	const sp = [0, 0, 0, 0, 0, 0];
	const budget = rng.int(67);
	for (let i = 0; i < budget; i++) { const k = rng.int(6); if (sp[k] < 32) sp[k]++; }
	return Object.fromEntries(STATS.map((s, i) => [s, sp[i]]));
}

function stageTeam(stage, rng, { maxAttempts = 200 } = {}) {
	const { pool, items } = stageInfo(stage);
	const validator = TeamValidator.get(FORMAT);
	const natures = dex.natures.all().map(n => n.name).sort();
	for (let attempt = 0; attempt < maxAttempts; attempt++) {
		const usedNum = new Set(); const usedItems = new Set(); const team = [];
		const order = rng.shuffle(pool.slice());
		for (const e of order) {
			if (team.length >= 6) break;
			if (usedNum.has(e.num)) continue;
			usedNum.add(e.num);
			const moves = rng.shuffle(e.moves.slice()).slice(0, 4);
			let item = '';
			if (items.length) {
				const free = items.filter(i => !usedItems.has(i));
				item = free.length && rng.chance(0.9) ? rng.pick(free) : '';
				if (item) usedItems.add(item);
			}
			const gender = e.gender || (e.genderRatio && e.genderRatio.M === 0 ? 'F' : (rng.chance(0.5) ? 'M' : 'F'));
			team.push({ name: e.name, species: e.name, item, ability: rng.pick(e.abilities), moves, nature: rng.pick(natures),
				evs: randomSP(rng), ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, gender, level: 50 });
		}
		if (team.length < 6) throw new Error(`stage ${stage}: species pool too small for 6 distinct species`);
		if (!validator.validateTeam(team)) return team;
	}
	throw new Error(`stageTeam(${stage}): no valid team after ${maxAttempts} attempts`);
}

module.exports = { stageTeam, stageInfo };

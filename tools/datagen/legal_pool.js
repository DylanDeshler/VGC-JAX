#!/usr/bin/env node
// Enumerate the full legal pool for a PS format using TeamValidator + learnsets.
// Port of docs/research/dex/scripts/legal_pool.js (06 §1).
// Usage: node tools/datagen/legal_pool.js --format <id> --mod <mod> --out data/export/legal_pool.json
// PS is resolved from oracle/ps (env PS_ROOT overrides).
'use strict';
const {parseArgs, psRoot, loadSim, psCommit, writeJSON} = require('./common');

const args = parseArgs(process.argv.slice(2), ['format', 'out'], {mod: 'champions'});
const root = psRoot();
const {Dex, TeamValidator} = loadSim(root);
const formatId = args.format, mod = args.mod, outPath = args.out;
const dex = Dex.mod(mod);
const format = Dex.formats.get(formatId);
if (!format.exists) throw new Error(`unknown format ${formatId}`);
const validator = TeamValidator.get(format);

const ZERO = {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0};
const IV = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};
function mkSet(species, ability, item, moves) {
	return {
		name: species, species, ability, item: item || '', moves,
		nature: 'Hardy', evs: {...ZERO}, ivs: {...IV}, level: 50, gender: '',
	};
}

const allMoves = dex.moves.all();
const legalSpecies = {};      // set-level species (what you put on the team sheet)
const battleFormes = {};      // in-battle formes reachable (megas, etc.)
const rejected = {};

for (const sp of dex.species.all()) {
	// Battle-only formes (megas etc.) are validated via their base forme + required item.
	if (sp.battleOnly) continue;
	// species-level legality prefilter
	const spProb = validator.checkSpecies(mkSet(sp.name, '', '', []), sp, sp, {});
	if (spProb) { rejected[sp.id] = spProb; continue; }
	// learnable moves by validator
	let sources;
	try { sources = validator.allSources(sp); } catch (e) { rejected[sp.id] = 'allSources: ' + e.message; continue; }
	const learnable = [];
	for (const mv of allMoves) {
		if (mv.isZ || mv.isMax) continue;
		let prob;
		try {
			prob = validator.checkCanLearn(mv, sp, validator.allSources(sp), {name: sp.name, level: 50});
		} catch (e) { prob = 'err ' + e.message; }
		if (prob) continue;
		const banProb = validator.checkMove(mkSet(sp.name, '', '', []), mv, {});
		if (banProb) continue;
		learnable.push(mv.id);
	}
	if (!learnable.length) { rejected[sp.id] = 'no learnable moves'; continue; }
	const firstMove = dex.moves.get(learnable[0]).name;
	// ability legality
	const abilities = [];
	let speciesProblem = null;
	for (const slot of Object.keys(sp.abilities)) {
		const ab = sp.abilities[slot];
		const set = mkSet(sp.name, ab, sp.requiredItem || (sp.requiredItems ? sp.requiredItems[0] : ''), [firstMove]);
		const probs = validator.validateSet(set, {});
		if (!probs) abilities.push({slot, id: dex.abilities.get(ab).id, name: ab});
		else speciesProblem = probs.join(' | ');
	}
	if (!abilities.length) { rejected[sp.id] = speciesProblem; continue; }
	legalSpecies[sp.id] = {
		cosmeticFormes: (sp.cosmeticFormes || []).map(f => f.toLowerCase().replace(/[^a-z0-9]/g, '')),
		name: sp.name, baseSpecies: sp.baseSpecies, forme: sp.forme, types: sp.types,
		baseStats: sp.baseStats, abilities, requiredItem: sp.requiredItem || null,
		nLearnable: learnable.length, learnable,
	};
}

// Battle formes: megas / primals / other battleOnly formes whose base is legal.
for (const sp of dex.species.all()) {
	if (!sp.battleOnly) continue;
	const bases = Array.isArray(sp.battleOnly) ? sp.battleOnly : [sp.battleOnly];
	const baseIds = bases.map(b => dex.species.get(b).id).filter(id => legalSpecies[id]);
	if (!baseIds.length) continue;
	const item = sp.requiredItem || (sp.requiredItems ? sp.requiredItems[0] : '');
	let ok = true, problem = null;
	if (item) {
		const base = legalSpecies[baseIds[0]];
		const set = mkSet(base.name, base.abilities[0].name, item, [dex.moves.get(base.learnable[0]).name]);
		const probs = validator.validateSet(set, {});
		if (probs) { ok = false; problem = probs.join(' | '); }
	}
	if (!ok) { rejected[sp.id] = problem; continue; }
	battleFormes[sp.id] = {
		name: sp.name, baseSpecies: sp.baseSpecies, from: baseIds, isMega: !!sp.isMega, isPrimal: !!sp.isPrimal,
		requiredItem: item || null, requiredMove: sp.requiredMove || null, requiredAbility: sp.requiredAbility || null,
		types: sp.types, baseStats: sp.baseStats, abilities: Object.values(sp.abilities).map(a => dex.abilities.get(a).id),
	};
}

// Items
const legalItems = {};
for (const it of dex.items.all()) {
	const set = mkSet('Rillaboom', 'Grassy Surge', it.name, ['Fake Out']);
	const prob = validator.checkItem(set, it, {});
	if (prob) continue;
	// species-restricted items: keep if any legal species can use it
	let users = null;
	if (it.megaStone) {
		const ms = typeof it.megaStone === 'object' ? Object.values(it.megaStone) : [it.megaStone];
		users = ms.map(n => dex.species.get(n).id);
		if (!users.some(u => battleFormes[u])) continue;
	}
	legalItems[it.id] = {
		name: it.name, isMegaStone: !!it.megaStone, megaFormes: users, isBerry: !!it.isBerry,
		itemUser: it.itemUser || null, isChoice: !!it.isChoice,
	};
}

// Moves: union of learnable over legal species
const moveUsers = {};
for (const [sid, s] of Object.entries(legalSpecies)) {
	for (const m of s.learnable) (moveUsers[m] = moveUsers[m] || []).push(sid);
}
const legalMoves = {};
for (const [mid, users] of Object.entries(moveUsers)) {
	const mv = dex.moves.get(mid);
	legalMoves[mid] = {name: mv.name, type: mv.type, category: mv.category, nUsers: users.length};
}
// Abilities: union of set-level abilities + battle-forme abilities
const legalAbilities = {};
for (const [sid, s] of Object.entries(legalSpecies)) {
	for (const a of s.abilities) {
		(legalAbilities[a.id] = legalAbilities[a.id] || {name: a.name, users: [], battleFormeOnly: true}).users.push(sid);
		legalAbilities[a.id].battleFormeOnly = false;
	}
}
for (const [fid, f] of Object.entries(battleFormes)) {
	for (const a of f.abilities) {
		if (!legalAbilities[a]) legalAbilities[a] = {name: dex.abilities.get(a).name, users: [], battleFormeOnly: true};
		legalAbilities[a].users.push(fid);
	}
}

const out = {
	format: format.name, formatId: format.id, mod, ps_commit: psCommit(root),
	ruleset: format.ruleset, banlist: format.banlist,
	counts: {
		setSpecies: Object.keys(legalSpecies).length,
		battleFormes: Object.keys(battleFormes).length,
		megaFormes: Object.values(battleFormes).filter(f => f.isMega).length,
		moves: Object.keys(legalMoves).length,
		abilities: Object.keys(legalAbilities).length,
		abilitiesSetLevel: Object.values(legalAbilities).filter(a => !a.battleFormeOnly).length,
		items: Object.keys(legalItems).length,
		megaStones: Object.values(legalItems).filter(i => i.isMegaStone).length,
	},
	species: legalSpecies, battleFormes, moves: legalMoves, abilities: legalAbilities, items: legalItems,
	rejectedSample: Object.fromEntries(Object.entries(rejected).filter(([k, v]) => v && !/does not exist|not obtainable/.test(v)).slice(0, 400)),
};
writeJSON(outPath, out);
console.log(JSON.stringify(out.counts));

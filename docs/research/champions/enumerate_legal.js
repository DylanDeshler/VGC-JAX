// Enumerate legal species / megas / items / abilities / moves for a PS format using PS's own TeamValidator.
// Usage: node enumerate_legal.js /tmp/psrepo/ps gen9championsvgc2026regmc /path/to/outdir
'use strict';
const path = require('path');
const fs = require('fs');
const psRoot = process.argv[2];
const formatId = process.argv[3] || 'gen9championsvgc2026regmc';
const outDir = process.argv[4] || '.';
const { Dex, TeamValidator, toID } = require(path.join(psRoot, 'dist/sim'));

const format = Dex.formats.get(formatId);
if (!format.exists) throw new Error('format not found ' + formatId);
const ruleTable = Dex.formats.getRuleTable(format);
const dex = Dex.forFormat(format);
const validator = new TeamValidator(format);

// ---- ruleset chain ----
const rules = [...ruleTable.entries()].map(([k, v]) => (v ? `${k} = ${v}` : k));
const ruleInfo = {
  format: format.name, id: format.id, mod: format.mod, gameType: format.gameType,
  ruleTableKeys: [...ruleTable.keys()],
  valueRules: Object.fromEntries(ruleTable.valueRules),
  pickedTeamSize: ruleTable.pickedTeamSize, minTeamSize: ruleTable.minTeamSize, maxTeamSize: ruleTable.maxTeamSize,
  adjustLevel: ruleTable.adjustLevel, maxLevel: ruleTable.maxLevel, defaultLevel: ruleTable.defaultLevel,
  evLimit: ruleTable.evLimit, maxMoveCount: ruleTable.maxMoveCount, minSourceGen: ruleTable.minSourceGen,
  timer: ruleTable.timer, tagRules: ruleTable.tagRules,
  complexBans: ruleTable.complexBans, complexTeamBans: ruleTable.complexTeamBans,
};

function baseSet(species, extra = {}) {
  return Object.assign({
    name: species.baseSpecies, species: species.name, item: '', ability: species.abilities['0'],
    moves: [], nature: 'Hardy', gender: '', evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
    ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50,
  }, extra);
}

// ---- candidate moves of a species: walk the learnset chain exactly like the validator would ----
function candidateMoves(species) {
  const ids = new Set();
  let s = species;
  const seen = new Set();
  while (s && s.exists && !seen.has(s.id)) {
    seen.add(s.id);
    const ld = dex.species.getLearnsetData(s.id);
    if (ld && ld.learnset) for (const m in ld.learnset) ids.add(m);
    s = dex.species.learnsetParent(s, true);
  }
  return [...ids];
}

// ---- species ----
const legalSpecies = [];
const speciesProblems = {};
const allSpecies = dex.species.all();
for (const sp of allSpecies) {
  if (sp.isMega || sp.isPrimal || sp.battleOnly) continue; // handled via their out-of-battle forme
  // choose a learnable move to make a complete set
  const cands = candidateMoves(sp);
  let move = null;
  for (const m of cands) {
    const mv = dex.moves.get(m);
    if (!mv.exists) continue;
    const set = baseSet(sp, { moves: [mv.name] });
    if (!validator.checkMove(set, mv, {}) && !validator.checkCanLearn(mv, sp, validator.allSources(sp), set)) { move = mv.name; break; }
  }
  if (!move) { speciesProblems[sp.name] = ['no learnable legal move / no learnset']; continue; }
  // try each ability slot; species is legal if any set validates
  let ok = false; let probs = null;
  for (const slot of ['0', '1', 'H', 'S']) {
    const ab = sp.abilities[slot];
    if (!ab) continue;
    const set = baseSet(sp, { moves: [move], ability: ab });
    if (sp.requiredItem) set.item = sp.requiredItem;
    if (sp.requiredItems) set.item = sp.requiredItems[0];
    const p = validator.validateSet(set, {});
    if (!p) { ok = true; break; } else probs = p;
  }
  if (ok) legalSpecies.push(sp); else speciesProblems[sp.name] = probs;
}

// ---- items (generic legality) ----
const legalItems = [];
for (const it of dex.items.all()) {
  const set = baseSet(dex.species.get('Pikachu'));
  set.name = 'X';
  if (validator.checkItem(set, it, {}) === null) legalItems.push(it);
}
const legalItemIds = new Set(legalItems.map(i => i.id));

// ---- megas reachable: legal species + legal stone, validated by a full set ----
const megas = [];
for (const it of legalItems) {
  if (!it.megaStone) continue;
  for (const [baseName, megaName] of Object.entries(it.megaStone)) {
    const base = dex.species.get(baseName);
    const mega = dex.species.get(megaName);
    const baseLegal = legalSpecies.some(s => s.id === base.id);
    let problems = null;
    if (baseLegal) {
      const cands = candidateMoves(base);
      let mv = null;
      for (const m of cands) {
        const mo = dex.moves.get(m);
        const set = baseSet(base, { moves: [mo.name] });
        if (!validator.checkMove(set, mo, {}) && !validator.checkCanLearn(mo, base, validator.allSources(base), set)) { mv = mo.name; break; }
      }
      const set = baseSet(base, { moves: [mv], item: it.name });
      problems = validator.validateSet(set, {});
    }
    megas.push({
      stone: it.name, stoneId: it.id, base: base.name, mega: mega.name, megaExists: mega.exists,
      baseLegal, validates: baseLegal && !problems, problems,
      megaTypes: mega.types, megaAbility: mega.abilities, megaBaseStats: mega.baseStats, megaWeightkg: mega.weightkg,
      baseTypes: base.types, baseBaseStats: base.baseStats,
    });
  }
}

// ---- abilities and moves reachable through legal species ----
const abilitySpecies = {};
const moveSpecies = {};
const perSpecies = {};
for (const sp of legalSpecies) {
  const abil = [];
  for (const slot of ['0', '1', 'H', 'S']) {
    const ab = sp.abilities[slot];
    if (!ab) continue;
    const abObj = dex.abilities.get(ab);
    const set = baseSet(sp, { ability: ab });
    if (validator.checkAbility(set, abObj, {}) !== null) continue;
    abil.push(abObj.name);
    (abilitySpecies[abObj.name] = abilitySpecies[abObj.name] || []).push(sp.name);
  }
  const mvs = [];
  for (const m of candidateMoves(sp)) {
    const mo = dex.moves.get(m);
    if (!mo.exists) continue;
    const set = baseSet(sp, { moves: [mo.name] });
    if (validator.checkMove(set, mo, {}) !== null) continue;
    if (validator.checkCanLearn(mo, sp, validator.allSources(sp), set)) continue;
    mvs.push(mo.name);
    (moveSpecies[mo.name] = moveSpecies[mo.name] || []).push(sp.name);
  }
  perSpecies[sp.name] = { num: sp.num, types: sp.types, baseStats: sp.baseStats, abilities: abil, moves: mvs.sort(), weightkg: sp.weightkg };
}
// mega abilities count as reachable abilities in battle
for (const m of megas) if (m.validates) {
  for (const ab of Object.values(m.megaAbility)) (abilitySpecies[ab] = abilitySpecies[ab] || []).push(m.mega + ' (mega)');
}

// ---- legal-but-unreachable moves (pass checkMove but nobody learns them) ----
const legalMovesGeneric = dex.moves.all().filter(mo => validator.checkMove(baseSet(dex.species.get('Pikachu'), { name: 'X' }), mo, {}) === null && !mo.realMove);

const write = (name, obj) => fs.writeFileSync(path.join(outDir, name), JSON.stringify(obj, null, 1));
// battle-distinct de-duplication: Gmax placeholders (validator rewrites them to the base forme) and
// formes identical in battle (cosmetic) collapse onto one key.
const keyOf = s => JSON.stringify([s.types, s.baseStats, perSpecies[s.name].abilities, perSpecies[s.name].moves, s.weightkg]);
const seenKeys = new Map();
const speciesOut = legalSpecies.map(s => {
  const gmaxPlaceholder = s.forme === 'Gmax';
  const k = keyOf(s);
  let duplicateOf = null;
  if (!gmaxPlaceholder) { if (seenKeys.has(k)) duplicateOf = seenKeys.get(k); else seenKeys.set(k, s.name); }
  return { name: s.name, id: s.id, num: s.num, forme: s.forme, types: s.types, baseStats: s.baseStats, weightkg: s.weightkg,
    abilities: perSpecies[s.name].abilities, tags: s.tags, isCosmeticForme: !!s.isCosmeticForme, gmaxPlaceholder, battleDuplicateOf: duplicateOf };
});
const distinct = speciesOut.filter(s => !s.gmaxPlaceholder && !s.battleDuplicateOf);
write('legal_species.json', {
  format: format.name, countValidatorAccepted: legalSpecies.length,
  countExcludingGmaxPlaceholders: speciesOut.filter(s => !s.gmaxPlaceholder).length,
  countBattleDistinct: distinct.length, countDistinctDexNumbers: new Set(distinct.map(s => s.num)).size,
  species: speciesOut,
});
console.error('battle-distinct species:', distinct.length, 'dex numbers:', new Set(distinct.map(s => s.num)).size, 'excl gmax:', speciesOut.filter(s => !s.gmaxPlaceholder).length);
write('legal_megas.json', { format: format.name, countValid: megas.filter(m => m.validates).length, megas });
write('legal_items.json', {
  format: format.name, count: legalItems.length,
  items: legalItems.map(i => ({ name: i.name, id: i.id, megaStone: i.megaStone || null, isBerry: !!i.isBerry, fling: i.fling ? i.fling.basePower : null, num: i.num })),
});
write('legal_abilities.json', { format: format.name, count: Object.keys(abilitySpecies).length, abilities: Object.fromEntries(Object.entries(abilitySpecies).sort()) });
write('legal_moves.json', {
  format: format.name, countReachable: Object.keys(moveSpecies).length, countGenericLegal: legalMovesGeneric.length,
  movesReachable: Object.fromEntries(Object.entries(moveSpecies).sort().map(([k, v]) => [k, v.length])),
  genericLegalButUnlearnable: legalMovesGeneric.map(m => m.name).filter(n => !moveSpecies[n]).sort(),
});
write('legal_learnsets.json', perSpecies);
write('ruleset_resolved.json', ruleInfo);
write('illegal_species_reasons.json', speciesProblems);

console.log(JSON.stringify({
  format: format.name, rules,
  pickedTeamSize: ruleTable.pickedTeamSize, minTeamSize: ruleTable.minTeamSize, maxTeamSize: ruleTable.maxTeamSize,
  adjustLevel: ruleTable.adjustLevel, evLimit: ruleTable.evLimit, timer: ruleTable.timer,
  species: legalSpecies.length, megasValid: megas.filter(m => m.validates).length, megaStonesLegal: legalItems.filter(i => i.megaStone).length,
  items: legalItems.length, nonMegaItems: legalItems.filter(i => !i.megaStone).length,
  abilities: Object.keys(abilitySpecies).length, movesReachable: Object.keys(moveSpecies).length,
  movesGenericLegal: legalMovesGeneric.length,
}, null, 1));

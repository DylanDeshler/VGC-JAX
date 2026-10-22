// Targeted runtime checks of Champions mechanics in PS.
// Usage: node ps_checks.js /tmp/psrepo/ps
'use strict';
const path = require('path');
const psRoot = process.argv[2];
const { Battle, Dex, Teams, TeamValidator } = require(path.join(psRoot, 'dist/sim'));

function mk(species, extra = {}) {
  return Object.assign({
    name: '', species, item: '', ability: '', moves: ['protect'], nature: 'Hardy', gender: '',
    evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50,
  }, extra);
}
function newBattle(formatid, t1, t2, seed = [1, 2, 3, 4]) {
  const b = new Battle({ formatid, seed });
  b.setPlayer('p1', { name: 'A', team: t1 });
  b.setPlayer('p2', { name: 'B', team: t2 });
  return b;
}
const sep = s => console.log('\n==== ' + s + ' ====');

// 1. Stat Points stat calc + PP
sep('1. stat calc and PP (gen9championsvgc2026regmc)');
{
  const chomp = mk('Garchomp', { ability: 'Rough Skin', item: 'Life Orb', nature: 'Jolly', evs: { hp: 32, atk: 32, def: 0, spa: 0, spd: 0, spe: 2 },
    moves: ['protect', 'earthquake', 'dragonclaw', 'scaleshot'] });
  const fill = ['Incineroar', 'Rillaboom', 'Amoonguss', 'Kingambit', 'Whimsicott'].map(s => mk(s));
  const t = [chomp, mk('Incineroar', { ability: 'Intimidate', moves: ['fakeout', 'quickattack'] }), mk('Sinistcha'), mk('Kingambit'), mk('Whimsicott'), mk('Pelipper')];
  const v = new TeamValidator('gen9championsvgc2026regmc');
  console.log('validator problems for team:', v.validateTeam(t));
  const b = newBattle('gen9championsvgc2026regmc', t, t);
  const g = b.p1.pokemon[0];
  console.log('Garchomp base', g.species.baseStats, 'stored', g.storedStats, 'maxhp', g.maxhp);
  console.log('expected HP 108+32+75=215, Atk 130+32+20=182, Spe floor((102+2+20)*1.1)=136, SpA (80+0+20)*0.9=90');
  console.log('PP:', g.baseMoveSlots.map(m => `${m.id}:${m.pp}/${m.maxpp} (dex pp ${b.dex.moves.get(m.id).pp}, base gen9 pp ${Dex.mod('gen9').moves.get(m.id).pp})`).join(', '));
  const inc = b.p1.pokemon[1];
  console.log('Incineroar PP:', inc.baseMoveSlots.map(m => `${m.id}:${m.pp}/${m.maxpp} (base gen9 pp ${Dex.mod('gen9').moves.get(m.id).pp})`).join(', '));
  // Base-mod comparison of a level-50 calc
  const sv = new Battle({ formatid: 'gen9vgc2025regi', seed: [1, 2, 3, 4] });
  console.log('gen9 base statModify(Garchomp atk, 252EV/31IV/Jolly L50)=', sv.statModify(Dex.species.get('Garchomp').baseStats, { evs: { atk: 252 }, ivs: { atk: 31 }, level: 50, nature: 'Jolly' }, 'atk'),
    ' champions SP32 =', b.statModify(Dex.species.get('Garchomp').baseStats, { evs: { atk: 32 }, ivs: { atk: 31 }, level: 50, nature: 'Jolly' }, 'atk'));
  // SP -> EV equivalence check across all SP 0..32 for a few base stats (non-HP and HP), vs mainline with EV = 8*SP-4 (SP>=1)
  let mism = 0, n = 0;
  for (const base of [5, 45, 50, 80, 100, 101, 130, 150, 200, 255]) for (let sp = 0; sp <= 32; sp++) {
    const ev = sp === 0 ? 0 : 8 * sp - 4;
    for (const st of ['hp', 'atk']) {
      const bs = { hp: base, atk: base, def: base, spa: base, spd: base, spe: base };
      const c = b.statModify(bs, { evs: { [st]: sp }, ivs: { [st]: 31 }, level: 50, nature: 'Hardy' }, st);
      const m = sv.statModify(bs, { evs: { [st]: ev }, ivs: { [st]: 31 }, level: 50, nature: 'Hardy' }, st);
      n++; if (c !== m) { mism++; if (mism < 5) console.log('mismatch', st, base, sp, c, m); }
    }
  }
  console.log(`SP vs mainline(EV=8SP-4, IV31, L50) identical in ${n - mism}/${n} cases`);
}

// 2. Open team sheets output in Bo3 (Force OTS) format
sep('2. Force Open Team Sheets (Bo3) team preview log');
{
  const t = [mk('Garchomp', { ability: 'Rough Skin', item: 'Garchompite', nature: 'Jolly', evs: { hp: 2, atk: 32, def: 0, spa: 0, spd: 0, spe: 32 }, moves: ['protect', 'earthquake'] }),
    mk('Incineroar', { ability: 'Intimidate', item: 'Sitrus Berry', moves: ['fakeout'] }), mk('Sinistcha', { moves: ['ragepowder'] }), mk('Kingambit'), mk('Whimsicott'), mk('Pelipper')];
  const b = newBattle('gen9championsvgc2026regmcbo3', t, t);
  const lines = b.log.filter(l => /showteam|poke\||teampreview|clearpoke|rule|gametype|player|teamsize/.test(l));
  console.log(lines.join('\n'));
  console.log('request p1 teamPreview wait:', JSON.stringify(b.p1.activeRequest && { teamPreview: b.p1.activeRequest.teamPreview, maxChosenTeamSize: b.p1.activeRequest.maxChosenTeamSize }));
}

// 3. Mega Evolution flow + dynamic speed re-sort in the same turn
sep('3. Mega flow: Camerupt (Spe 40+20=60 -> Mega 20+20=40) vs foe at 50 Spe');
{
  // p1: Camerupt w/ Cameruptite and a filler; p2: Torkoal? (base 20) -> Spe 40; need ~50: base 30 -> 50: e.g. Slowbro (30)
  const p1 = [mk('Camerupt', { ability: 'Solid Rock', item: 'Cameruptite', moves: ['earthpower', 'protect'] }), mk('Torkoal', { ability: 'Drought', item: 'Charcoal', moves: ['protect'] }),
    mk('Kingambit'), mk('Whimsicott'), mk('Pelipper'), mk('Garchomp')];
  const p2 = [mk('Slowbro', { ability: 'Oblivious', item: 'Leftovers', moves: ['scald', 'protect'] }), mk('Hippowdon', { ability: 'Sand Stream', moves: ['protect'] }),
    mk('Kingambit'), mk('Whimsicott'), mk('Pelipper'), mk('Garchomp')];
  const b = newBattle('gen9championsvgc2026regmcbo3', p1, p2);
  b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234');
  console.log('speeds before:', b.getAllActive().map(p => `${p.name}:${p.getStat('spe')}`).join(' '));
  b.choose('p1', 'move earthpower 1 mega, move protect'); b.choose('p2', 'move scald 1, move protect');
  const start = b.log.lastIndexOf('|turn|1');
  console.log(b.log.slice(start).filter(l => !l.startsWith('|t:|') && l !== '|').join('\n'));
  console.log('speeds after:', b.getAllActive().map(p => `${p.name}(${p.species.name}):${p.getStat('spe')}`).join(' '));
  console.log('canMegaEvo flags p1:', b.p1.pokemon.map(p => `${p.name}:${p.canMegaEvo}`).join(' '));
}

// 4. Mega switch-in-like ability on Mega Evolution (Charizard-Mega-Y Drought) and Mega persistence after faint
sep('4. Mega Charizard Y: Drought fires on mega evolution; species after faint');
{
  const p1 = [mk('Charizard', { ability: 'Blaze', item: 'Charizardite Y', moves: ['heatwave', 'protect'] }), mk('Torkoal', { ability: 'Drought', moves: ['protect'] }),
    mk('Kingambit'), mk('Whimsicott'), mk('Pelipper'), mk('Garchomp')];
  const p2 = [mk('Tyranitar', { ability: 'Sand Stream', item: 'Tyranitarite', nature: 'Adamant', evs: { hp: 0, atk: 32, def: 0, spa: 0, spd: 0, spe: 32 }, moves: ['rockslide', 'protect'] }),
    mk('Garchomp', { ability: 'Rough Skin', nature: 'Jolly', evs: { hp: 0, atk: 32, def: 0, spa: 0, spd: 0, spe: 32 }, moves: ['rockslide', 'protect'] }),
    mk('Kingambit'), mk('Whimsicott'), mk('Pelipper'), mk('Incineroar')];
  const b = newBattle('gen9championsvgc2026regmcbo3', p1, p2, [5, 6, 7, 8]);
  b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234');
  console.log(b.log.slice(b.log.lastIndexOf('|start')).filter(l => !l.startsWith('|t:|')).join('\n'));
  b.choose('p1', 'move heatwave mega, move protect'); b.choose('p2', 'move rockslide mega, move rockslide');
  console.log(b.log.slice(b.log.lastIndexOf('|turn|1')).filter(l => !l.startsWith('|t:|') && l !== '|').join('\n'));
  // force-faint the mega Charizard to inspect persistence
  const zard = b.p1.active[0];
  if (zard.hp) { zard.faint(); b.resolveFaints({}); }
  console.log('after faint: species', zard.species.name, 'baseSpecies', zard.baseSpecies.name, 'details', zard.details, 'ability', zard.ability, 'baseAbility', zard.baseAbility);
}

// 5. Base gen9 (SV) for comparison: faint reverts mega? (use gen9 natdex-like custom game with mega allowed)
sep('5. Base-mod comparison: gen9 National Dex Doubles mega reverts on faint');
{
  const p1 = [mk('Charizard', { ability: 'Blaze', item: 'Charizardite Y', moves: ['heatwave', 'protect'] }), mk('Torkoal', { ability: 'Drought', moves: ['protect'] })];
  const p2 = [mk('Garchomp', { ability: 'Rough Skin', moves: ['protect'] }), mk('Incineroar', { ability: 'Intimidate', moves: ['protect'] })];
  try {
    const b = newBattle('gen9nationaldexdoubles', p1.concat([mk('Kingambit'), mk('Pelipper')]), p2.concat([mk('Kingambit'), mk('Pelipper')]));
    b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234');
    b.choose('p1', 'move heatwave mega, move protect'); b.choose('p2', 'move protect, move protect');
    const zard = b.p1.active[0];
    console.log('before faint species', zard.species.name);
    zard.faint(); b.resolveFaints({});
    console.log('gen9 natdex after faint: species', zard.species.name, 'baseSpecies', zard.baseSpecies.name, 'details', zard.details);
  } catch (e) { console.log('natdex comparison failed:', e.message); }
}

// 6. Trick Room action speed: champions vs base
sep('6. getActionSpeed under Trick Room');
{
  const t = [mk('Garchomp', { moves: ['protect'] }), mk('Incineroar', { moves: ['protect'] }), mk('Kingambit'), mk('Whimsicott'), mk('Pelipper'), mk('Sinistcha')];
  const b = newBattle('gen9championsvgc2026regmcbo3', t, t);
  b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234');
  const g = b.p1.active[0];
  b.field.addPseudoWeather("trickroom", b.p2.active[0]);
  for (const s of [60, 1808, 1809, 2000]) { g.storedStats.spe = s; console.log(`champions TR: spe ${s} -> actionSpeed ${g.getActionSpeed()}`); }
  const sv = newBattle('gen9vgc2025regi', [mk('Garchomp', { moves: ['protect'] }), mk('Incineroar', { moves: ['protect'] }), mk('Kingambit'), mk('Whimsicott')],
    [mk('Garchomp', { moves: ['protect'] }), mk('Incineroar', { moves: ['protect'] }), mk('Kingambit'), mk('Whimsicott')]);
  sv.choose('p1', 'team 1234'); sv.choose('p2', 'team 1234');
  const g2 = sv.p1.active[0];
  sv.field.addPseudoWeather("trickroom", sv.p2.active[0]);
  for (const s of [60, 1808, 1809, 2000]) { g2.storedStats.spe = s; console.log(`base gen9 TR: spe ${s} -> actionSpeed ${g2.getActionSpeed()}`); }
}

// 7. HP display
sep('7. HP shared display (champions) for hp/maxhp samples');
{
  const t = [mk('Garchomp', { moves: ['protect'] }), mk('Incineroar', { moves: ['protect'] }), mk('Kingambit'), mk('Whimsicott'), mk('Pelipper'), mk('Sinistcha')];
  const b = newBattle('gen9championsvgc2026regmcbo3', t, t);
  const g = b.p1.pokemon[0];
  for (const hp of [g.maxhp, g.maxhp - 1, Math.ceil(g.maxhp / 2), Math.floor(g.maxhp / 2), Math.floor(g.maxhp / 5) + 1, 1]) { g.hp = hp; console.log(hp + '/' + g.maxhp, '->', g.getHealth().shared); }
}

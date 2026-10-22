// Scripted Champions Reg M-C scenarios used to confirm turn-flow claims in ps-turn-loop-spec.md.
// Usage: node scenarios.js [scenarioName]
const PS = process.env.PS_DIST || '/tmp/psrepo/ps/dist';
const { Battle } = require(PS + '/sim/battle');
const { State } = require(PS + '/sim/state');

function mk(species, item, ability, moves, extra = {}) {
  return Object.assign({
    name: '', species, item, ability, moves, nature: 'Serious', gender: '',
    evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50,
  }, extra);
}
const filler = [
  mk('Garchomp', 'Leftovers', 'Rough Skin', ['Protect', 'Earthquake', 'Dragon Claw', 'Rock Slide']),
  mk('Corviknight', 'Rocky Helmet', 'Pressure', ['Protect', 'Brave Bird', 'Roost', 'Iron Defense']),
  mk('Milotic', 'Sitrus Berry', 'Competitive', ['Protect', 'Scald', 'Recover', 'Ice Beam']),
  mk('Kingambit', 'Focus Sash', 'Defiant', ['Protect', 'Kowtow Cleave', 'Sucker Punch', 'Iron Head']),
];
function newBattle(t1, t2, seed = [1, 2, 3, 4]) {
  const b = new Battle({ formatid: 'gen9championsvgc2026regmc', seed, p1: { name: 'A', team: t1 }, p2: { name: 'B', team: t2 } });
  return b;
}
let printed = 0;
function flush(b, label) {
  const lines = b.log.slice(printed).filter(l => !l.startsWith('|t:|') && !l.startsWith('|split|') );
  printed = b.log.length;
  // drop duplicated split lines (exact HP + percentage): keep the first of consecutive pairs differing only in HP
  console.log(`---- ${label}`);
  let prev = null;
  for (const l of lines) { if (prev && l.split('|')[1] === prev.split('|')[1] && l.split('|')[2] === prev.split('|')[2] && /\/100/.test(l)) continue; console.log(l); prev = l; }
}
function req(b) {
  for (const s of b.sides) {
    const r = s.activeRequest;
    if (!r) { console.log(`  ${s.id} request: null`); continue; }
    const short = r.wait ? 'wait' : r.teamPreview ? 'teamPreview' : r.forceSwitch ? 'forceSwitch=' + JSON.stringify(r.forceSwitch) :
      'move ' + JSON.stringify(r.active.map(a => ({ trapped: a.trapped, maybeTrapped: a.maybeTrapped, canMegaEvo: a.canMegaEvo, moves: a.moves.map(m => m.id + (m.disabled ? '(X)' : '')).join(',') })));
    console.log(`  ${s.id} request: ${short}`);
  }
}
function queue(b) { console.log('  queue: ' + b.queue.list.map(a => `${a.choice}${a.pokemon ? ':' + a.pokemon.name : ''}${a.move ? ':' + a.move.id : ''}[o${a.order},p${a.priority},s${a.speed}]`).join(' | ')); }
const sc = {};

sc.uturn = () => {
  const b = newBattle(
    [mk('Talonflame', 'Sharp Beak', 'Flame Body', ['U-turn', 'Protect', 'Tailwind', 'Brave Bird']), mk('Milotic', 'Leftovers', 'Competitive', ['Protect', 'Scald', 'Recover', 'Ice Beam']), ...filler.slice(0, 2), filler[3], mk('Pelipper', 'Damp Rock', 'Drizzle', ['Protect', 'Hurricane', 'Tailwind', 'Hydro Pump'])],
    [mk('Incineroar', 'Sitrus Berry', 'Intimidate', ['Fake Out', 'Flare Blitz', 'Parting Shot', 'Knock Off']), mk('Rillaboom', 'Leftovers', 'Grassy Surge', ['Fake Out', 'Grassy Glide', 'Protect', 'Wood Hammer']), ...filler]);
  b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234'); flush(b, 'start'); req(b);
  b.choose('p1', 'move uturn 1, move scald 2'); b.choose('p2', 'move partingshot 2, move protect');
  flush(b, 'after commit (paused?)'); console.log('  requestState=' + b.requestState + ' midTurn=' + b.midTurn); req(b); queue(b);
  const snap = JSON.stringify(b.toJSON());
  b.choose('p1', 'switch 3'); flush(b, 'after p1 switch'); console.log('  requestState=' + b.requestState); req(b); queue(b);
  if (b.requestState === 'switch') { for (const s of b.sides) if (s.activeRequest && s.activeRequest.forceSwitch) b.choose(s.id, 'switch 4'); flush(b, 'after second switch'); req(b); }
  // replay from snapshot through deserialization and compare log tail
  const b2 = State.deserializeBattle(snap); b2.restart(() => {});
  const startLen = b2.log.length;
  b2.choose('p1', 'switch 3'); if (b2.requestState === 'switch') for (const s of b2.sides) if (s.activeRequest && s.activeRequest.forceSwitch) b2.choose(s.id, 'switch 4');
  const tailA = b.log.slice(startLen).filter(l => !l.startsWith('|t:|')).join('\n');
  const tailB = b2.log.slice(startLen).filter(l => !l.startsWith('|t:|')).join('\n');
  console.log('  deserialized-replay identical log tail: ' + (tailA === tailB));
};

sc.megaspeed = () => {
  // Gengar (base 110 spe) vs Garchomp (102): give Garchomp more speed SP so it outspeeds base Gengar but not Mega Gengar (130)
  const g = mk('Gengar', 'Gengarite', 'Cursed Body', ['Shadow Ball', 'Protect', 'Sludge Bomb', 'Icy Wind']);
  const chomp = mk('Garchomp', 'Leftovers', 'Rough Skin', ['Protect', 'Dragon Claw', 'Earthquake', 'Rock Slide'], { evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 20 } });
  const b = newBattle([g, ...filler.slice(1), filler[0]], [chomp, mk('Milotic', 'Leftovers', 'Competitive', ['Protect', 'Scald', 'Recover', 'Ice Beam']), ...filler.slice(1)]);
  b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234'); flush(b, 'start');
  console.log('  speeds: ' + b.getAllActive().map(p => p.name + '=' + p.getStat('spe')).join(' '));
  b.choose('p1', 'move shadowball 2 mega, move protect'); b.choose('p2', 'move dragonclaw 1, move protect');
  flush(b, 'turn 1 (mega then re-sort)');
  console.log('  speeds after: ' + b.getAllActive().map(p => p.name + '=' + p.getStat('spe')).join(' '));
};

sc.icywind = () => {
  // Icy Wind lowers foe speed mid-turn; slower ally should now outspeed the foe in the same turn
  const b = newBattle(
    [mk('Froslass', 'Focus Sash', 'Cursed Body', ['Icy Wind', 'Protect', 'Shadow Ball', 'Will-O-Wisp'], { evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 32 } }),
     mk('Milotic', 'Leftovers', 'Competitive', ['Protect', 'Scald', 'Recover', 'Ice Beam']), ...filler.slice(0, 2), filler[3], filler[2]],
    [mk('Kingambit', 'Focus Sash', 'Inner Focus', ['Protect', 'Kowtow Cleave', 'Sucker Punch', 'Iron Head'], { evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 32 } }), mk('Corviknight', 'Leftovers', 'Pressure', ['Protect', 'Brave Bird', 'Roost', 'Iron Defense']), ...filler]);
  b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234'); flush(b, 'start');
  console.log('  speeds: ' + b.getAllActive().map(p => p.name + '=' + p.getStat('spe')).join(' '));
  b.choose('p1', 'move icywind, move scald 1'); b.choose('p2', 'move ironhead 2, move protect');
  flush(b, 'turn 1');
};

sc.fewbench = () => {
  // p2 (all level 1) loses one mon on turn 1, then both actives on turn 2 with a single bench mon left.
  const lv1 = (sp) => mk(sp, 'Leftovers', 'Static', ['Protect', 'Thunderbolt', 'Growl', 'Volt Switch'], { level: 1 });
  const b = newBattle(
    [mk('Garchomp', 'Leftovers', 'Rough Skin', ['Earthquake', 'Protect', 'Dragon Claw', 'Rock Slide']), mk('Kingambit', 'Leftovers', 'Defiant', ['Kowtow Cleave', 'Protect', 'Iron Head', 'Sucker Punch']), ...filler.slice(1, 3), mk('Pelipper', 'Damp Rock', 'Drizzle', ['Protect', 'Hurricane', 'Tailwind', 'Hydro Pump']), mk('Talonflame', 'Sharp Beak', 'Flame Body', ['U-turn', 'Protect', 'Tailwind', 'Brave Bird'])],
    [lv1('Pikachu'), lv1('Raichu'), lv1('Jolteon'), lv1('Ampharos'), filler[1], filler[2]]);
  b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234'); flush(b, 'start');
  const script = { p1: ['move protect, move kowtowcleave 2', 'move earthquake, move protect'], p2: ['move growl, move growl', 'move growl, move growl'] };
  for (let t = 0; t < 6 && !b.ended; t++) {
    for (const s of b.sides) {
      if (s.isChoiceDone()) continue;
      const r = s.activeRequest;
      if (r.forceSwitch) {
        console.log(`  ${s.id} forceSwitch=${JSON.stringify(r.forceSwitch)} forcedSwitchesLeft=${s.choice.forcedSwitchesLeft} forcedPassesLeft=${s.choice.forcedPassesLeft} canSwitch=${b.canSwitch(s)}`);
        for (const c of ['switch 4, switch 3', 'pass, pass', 'pass, switch 4', 'switch 4, pass', 'switch 4']) { s.clearChoice(); const ok = s.choose(c); console.log(`    try '${c}' -> ${ok} => ${s.getChoice()} ${s.choice.error}`); }
        s.clearChoice(); s.choose('default'); console.log(`  ${s.id} auto choice = ${s.getChoice()}`);
      } else {
        if (s.id === 'p2' && b.turn === 2) for (const m of s.active) m.hp = 1; // force a double KO this turn
        const c = script[s.id].shift() || 'default';
        if (!s.choose(c)) { s.clearChoice(); s.choose('default'); }
      }
    }
    if (b.allChoicesDone()) b.commitChoices();
    flush(b, 'step ' + t);
  }
};

sc.shadowtag = () => {
  const b = newBattle(
    [mk('Gengar', 'Gengarite', 'Cursed Body', ['Shadow Ball', 'Protect', 'Sludge Bomb', 'Icy Wind']), ...filler.slice(0, 3), filler[3], mk('Pelipper', 'Damp Rock', 'Drizzle', ['Protect', 'Hurricane', 'Tailwind', 'Hydro Pump'])],
    [mk('Milotic', 'Leftovers', 'Competitive', ['Protect', 'Scald', 'Recover', 'Ice Beam']), mk('Kingambit', 'Focus Sash', 'Defiant', ['Protect', 'Kowtow Cleave', 'Sucker Punch', 'Iron Head']), ...filler.slice(0, 2), filler[2], filler[1]]);
  b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234');
  b.choose('p1', 'move protect mega, move protect'); b.choose('p2', 'move protect, move protect');
  flush(b, 'turn 1'); req(b);
  const s2 = b.sides[1];
  for (const c of ['switch 3, move protect', 'move protect, switch 3']) {
    s2.clearChoice(); const ok = s2.choose(c); console.log(`  try p2 '${c}' -> ${ok} ${s2.choice.error}`);
  }
};

sc.instruct = () => {
  const b = newBattle(
    [mk('Oranguru', 'Sitrus Berry', 'Inner Focus', ['Instruct', 'Trick Room', 'Protect', 'Psychic']), mk('Garchomp', 'Leftovers', 'Rough Skin', ['Rock Slide', 'Protect', 'Dragon Claw', 'Earthquake'], {evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:32}}), ...filler.slice(1), filler[0]],
    [mk('Milotic', 'Leftovers', 'Competitive', ['Protect', 'Scald', 'Recover', 'Ice Beam']), mk('Corviknight', 'Leftovers', 'Pressure', ['Protect', 'Brave Bird', 'Roost', 'Iron Defense']), ...filler]);
  b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234');
  b.choose('p1', 'move instruct -2, move rockslide'); b.choose('p2', 'move scald 1, move irondefense');
  flush(b, 'turn 1 (instruct)');
};

sc.preview = () => {
  const b = newBattle([...filler, mk('Pelipper', 'Damp Rock', 'Drizzle', ['Protect', 'Hurricane', 'Tailwind', 'Hydro Pump']), mk('Milotic', 'Leftovers', 'Competitive', ['Protect', 'Scald', 'Recover', 'Ice Beam'])],
    [...filler, mk('Pelipper', 'Damp Rock', 'Drizzle', ['Protect', 'Hurricane', 'Tailwind', 'Hydro Pump']), mk('Talonflame', 'Leftovers', 'Gale Wings', ['Protect', 'Brave Bird', 'Tailwind', 'U-turn'])]);
  console.log('  before: requestState=' + b.requestState + ' queue=' + b.queue.list.map(a => a.choice).join(','));
  for (const c of ['team 12', 'team 6513', 'team 1,1,2,3', 'team 123', 'team 7123', 'team [6,5]']) { const s = b.sides[0]; s.clearChoice(); const ok = s.choose(c); console.log(`  try '${c}' -> ${ok} actions=${s.choice.actions.map(a => a.pokemon.name + '@' + a.index).join(',')} ${s.choice.error}`); }
  b.sides[0].clearChoice();
  b.choose('p1', 'team 6513'); b.choose('p2', 'team 2');
  console.log('  p1 pokemon after preview: ' + b.sides[0].pokemon.map(p => p.name + '#' + p.position).join(', ') + ' pokemonLeft=' + b.sides[0].pokemonLeft);
  console.log('  p2 pokemon after preview: ' + b.sides[1].pokemon.map(p => p.name + '#' + p.position).join(', ') + ' pokemonLeft=' + b.sides[1].pokemonLeft);
  flush(b, 'start');
};

sc.doubleswitch = () => {
  // Start-of-turn switches are not batched: each switch's SwitchIn handlers fire before the next switch action.
  const b = newBattle(
    [mk('Talonflame', 'Sharp Beak', 'Flame Body', ['U-turn', 'Protect', 'Tailwind', 'Brave Bird']), mk('Milotic', 'Leftovers', 'Competitive', ['Protect', 'Scald', 'Recover', 'Ice Beam']), mk('Incineroar', 'Sitrus Berry', 'Intimidate', ['Fake Out', 'Flare Blitz', 'Parting Shot', 'Knock Off']), filler[0], filler[1], filler[3]],
    [mk('Kingambit', 'Focus Sash', 'Defiant', ['Protect', 'Kowtow Cleave', 'Sucker Punch', 'Iron Head']), filler[1], mk('Arcanine', 'Sitrus Berry', 'Intimidate', ['Protect', 'Flare Blitz', 'Snarl', 'Extreme Speed']), filler[0], filler[2], filler[3]]);
  b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234'); flush(b, 'start');
  b.choose('p1', 'switch 3, move protect'); b.choose('p2', 'switch 3, move protect');
  flush(b, 'turn 1: two simultaneous switches');
};

const which = process.argv[2];
for (const k of Object.keys(sc)) { if (which && which !== k) continue; printed = 0; console.log(`\n######## ${k}`); sc[k](); }

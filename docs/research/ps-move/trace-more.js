// Trace event dispatches for: battle start (switch-ins), a turn with Protect/Will-O-Wisp/Icy Wind/Earthquake,
// and a turn with Parting Shot (mid-turn switch) + crit. Writes trace-*.txt and prints distinct events.
const fs = require('fs');
const H = require('./harness');
const filler = [
  H.set('Milotic','Leftovers','Competitive',['Scald','Protect']),
  H.set('Metagross','Leftovers','Clear Body',['Iron Head','Protect']),
];
const p1 = [
  H.set('Incineroar','Sitrus Berry','Intimidate',['Fake Out','Will-O-Wisp','Protect','Parting Shot'],'Adamant',{hp:32,atk:32}),
  H.set('Garchomp','Life Orb','Rough Skin',['Rock Slide','Earthquake','Protect','Dragon Claw'],'Jolly',{atk:32,spe:32}),
  H.set('Corviknight','Leftovers','Pressure',['Brave Bird','Protect','Tailwind','Roost']),
  H.set('Tyranitar','Leftovers','Unnerve',['Rock Slide','Protect']),
  ...filler];
const p2 = [
  H.set('Whimsicott','Focus Sash','Prankster',['Moonblast','Tailwind','Protect','Encore'],'Timid',{spa:32,spe:32}),
  H.set('Salamence','Leftovers','Intimidate',['Dragon Claw','Icy Wind','Protect','Draco Meteor'],'Adamant',{hp:32,atk:32}),
  H.set('Milotic','Leftovers','Competitive',['Scald','Protect']),
  H.set('Metagross','Leftovers','Clear Body',['Iron Head','Protect']),
  ...filler];
let T = H.traceEvents();
const b = H.makeBattle(p1, p2, { prePreview: (bb) => {} });
T.restore();
const dump = (name, log) => {
  fs.writeFileSync(__dirname + '/' + name, log.map(e => `${'  '.repeat(e.depth)}${e.kind}(${e.eventid}) tgt=${e.target} src=${e.source} eff=${e.effect}${e.handlers.length ? '  H=[' + e.handlers.join(',') + ']' : ''}`).join('\n') + '\n');
  const seen = new Map();
  for (const e of log) { const k = `${e.kind}:${e.eventid}`; if (!seen.has(k)) seen.set(k, 0); seen.set(k, seen.get(k) + 1); }
  console.log(`== ${name}: ${log.length} dispatches, ${seen.size} distinct`);
  console.log([...seen.keys()].join(' | '));
};
dump('trace-start.txt', T.log);
let crit = false;
H.instrumentPRNG(b, d => {
  const c = d.callers.join(' ');
  if (/getDamage/.test(c) && d.from === 24) return crit ? 0 : 23;
  if (/hitStepAccuracy/.test(c)) return 0;
  if (/secondaries/.test(c)) return 0; // all secondaries proc
  if (/randomizer/.test(c)) return 0;
  if (/onStallMove|StallMove/.test(c)) return 0;
  return undefined;
});
// Turn 1: p1a Incineroar Will-O-Wisp -> p2b Salamence ; p1b Garchomp Earthquake (hits ally too)
//         p2a Whimsicott Tailwind (prankster) ; p2b Salamence Icy Wind (spread, -1 spe secondary)
T = H.traceEvents();
b.makeChoices('move 2 2, move 2', 'move 2, move 2');
T.restore();
dump('trace-turn-b1.txt', T.log);
console.log(b.log.slice(-45).join('\n'));
// Turn 2: p1a Protect ; p1b Dragon Claw -> p2b (crit) ; p2a Encore -> p1a ; p2b Dragon Claw -> p1b
crit = true;
T = H.traceEvents();
b.makeChoices('move 3, move 4 2', 'move 4 1, move 1 2');
T.restore();
dump('trace-turn-b2.txt', T.log);
console.log(b.log.slice(-40).join('\n'));
// Replace fainted Garchomp with Tyranitar (slot 4 of the brought team)
if (b.requestState === 'switch') b.makeChoices('switch 4', 'pass');
console.log(b.log.slice(-8).join('\n'));
// Turn 3: p1a Parting Shot -> p2a (mid-turn switch to Corviknight), p1b Tyranitar Rock Slide
T = H.traceEvents();
b.makeChoices('move 4 1, move 1', 'move 1 1, move 1 2');
console.log('REQ', b.requestState, b.sides.map(s => JSON.stringify(s.activeRequest && s.activeRequest.forceSwitch)));
if (b.requestState === 'switch') { try { b.makeChoices('switch 3', ''); } catch (e) { console.log('ERR', e.message, b.sides.map(s=>s.choice.error)); } }
T.restore();
dump('trace-turn-b3.txt', T.log);
console.log(b.log.slice(-40).join('\n'));

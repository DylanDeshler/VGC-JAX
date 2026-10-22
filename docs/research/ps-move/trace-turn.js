// Trace every event dispatch during one representative Champions doubles turn.
const fs = require('fs');
const H = require('./harness');
const filler = [
  H.set('Milotic','Leftovers','Competitive',['Scald','Protect']),
  H.set('Metagross','Leftovers','Clear Body',['Iron Head','Protect']),
  H.set('Corviknight','Leftovers','Pressure',['Brave Bird','Protect']),
  H.set('Tyranitar','Leftovers','Unnerve',['Rock Slide','Protect']),
];
const p1 = [
  H.set('Incineroar','Sitrus Berry','Intimidate',['Fake Out','Flare Blitz','Protect','Parting Shot'],'Adamant',{hp:32,atk:32}),
  H.set('Garchomp','Life Orb','Rough Skin',['Rock Slide','Earthquake','Protect','Dragon Claw'],'Jolly',{atk:32,spe:32}),
  ...filler];
const p2 = [
  H.set('Pelipper','Focus Sash','Drizzle',['Hurricane','Weather Ball','Protect','Tailwind'],'Modest',{spa:32,spe:32}),
  H.set('Rillaboom','Leftovers','Grassy Surge',['Wood Hammer','Grassy Glide','Fake Out','Protect'],'Adamant',{hp:32,atk:32}),
  ...filler];
const mode = process.argv[2] || 'turn1';
const b = H.makeBattle(p1, p2);
// Force: all accuracy hits, no crits, no secondary procs (so the turn is "plain"), max damage roll.
H.instrumentPRNG(b, d => {
  const c = d.callers.join(' ');
  if (/getDamage/.test(c) && d.from === 24) return 23;      // no crit
  if (/hitStepAccuracy/.test(c)) return 0;                   // hit
  if (/secondaries/.test(c)) return 99;                      // no secondary
  if (/randomizer/.test(c)) return 0;                        // max roll
  return undefined;
});
const T = H.traceEvents();
// Turn 1: P1a Incineroar Fake Out -> P2b Rillaboom; P1b Garchomp Rock Slide (spread);
//         P2a Pelipper Hurricane -> P1b Garchomp; P2b Rillaboom Wood Hammer -> P1a Incineroar
b.makeChoices('move 1 2, move 1', 'move 1 2, move 1 1');
T.restore();
const lines = T.log.map(e => `${'  '.repeat(e.depth)}${e.kind}(${e.eventid}) tgt=${e.target} src=${e.source} eff=${e.effect}${e.handlers.length ? '  H=[' + e.handlers.join(',') + ']' : ''}`);
fs.writeFileSync(__dirname + '/trace-turn1.txt', lines.join('\n') + '\n\n' + b.log.slice(-60).join('\n'));
// condensed: top-level-ish ordered unique names
const seen = new Map();
for (const e of T.log) {
  const k = `${e.kind}:${e.eventid}`;
  if (!seen.has(k)) seen.set(k, { first: seen.size, count: 0, withHandlers: 0 });
  const s = seen.get(k); s.count++; if (e.handlers.length) s.withHandlers++;
}
console.log('distinct (kind:event) in first-occurrence order:');
for (const [k, v] of seen) console.log(String(v.first).padStart(3), k.padEnd(40), 'calls=' + v.count, 'withHandlers=' + v.withHandlers);
console.log('total dispatches', T.log.length);
console.log(b.log.slice(-40).join('\n'));

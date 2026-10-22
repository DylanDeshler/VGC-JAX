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
const b = H.makeBattle(p1, p2);
const rlog = H.instrumentPRNG(b);
console.log(H.lastLines(b, 30));
b.makeChoices('move 1 1, move 1', 'move 3, move 1 2');
console.log(H.lastLines(b, 60));
for (const d of rlog) console.log(d.i, d.turn, d.from, d.to, '=>', d.result, d.callers.slice(0,4).join(' < '));

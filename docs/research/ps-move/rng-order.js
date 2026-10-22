// Print the RNG draw sequence for a doubles turn containing spread moves with secondaries and a self-drop move.
const H = require('./harness');
const filler = [H.set('Milotic','Leftovers','Competitive',['Scald']), H.set('Metagross','Leftovers','Clear Body',['Iron Head'])];
const p1 = [
  H.set('Garchomp','Leftovers','Rough Skin',['Rock Slide'],'Jolly',{atk:32,spe:32}),
  H.set('Salamence','Leftovers','Moxie',['Draco Meteor'],'Modest',{spa:32}),
  ...filler];
const p2 = [
  H.set('Incineroar','Leftovers','Blaze',['Bulk Up'],'Careful',{hp:32,def:32}),
  H.set('Rillaboom','Leftovers','Overgrow',['Icy Wind'],'Adamant',{hp:32,atk:32, spe: 10}),
  ...filler];
const b = H.makeBattle(p1, p2);
const log = H.instrumentPRNG(b, d => (/getDamage/.test(d.callers.join(' ')) && d.from === 24) ? 23 : undefined);
b.makeChoices('move 1, move 1 1', 'move 1, move 1');
for (const d of log) console.log(`#${d.i} random(${d.from}${d.to ? ',' + d.to : ''}) -> ${d.result}  ${d.callers.slice(0, 4).join(' < ')}`);
console.log(b.log.filter(l => /\|move\||-unboost|-boost|\|-miss|flinch/.test(l)).join('\n'));

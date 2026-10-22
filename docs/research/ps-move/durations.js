// Verify duration semantics: Tailwind (4), Trick Room (5), Reflect (5), weather from ability (5), Protect chain draws.
const H = require('./harness');
const filler = [H.set('Milotic','Leftovers','Competitive',['Scald']), H.set('Metagross','Leftovers','Clear Body',['Iron Head'])];
const p1 = [H.set('Whimsicott','Leftovers','Infiltrator',['Tailwind','Protect','Reflect','Moonblast'],'Timid',{spe:32}),
            H.set('Farigiraf','Leftovers','Armor Tail',['Trick Room','Protect','Psychic','Bulk Up'],'Quiet',{hp:32}), ...filler];
const p2 = [H.set('Torkoal','Leftovers','Drought',['Protect','Iron Defense'],'Bold',{hp:32,def:32}),
            H.set('Corviknight','Leftovers','Pressure',['Iron Defense','Roost'],'Impish',{hp:32,def:32}), ...filler];
const b = H.makeBattle(p1, p2);
const log = H.instrumentPRNG(b);
const show = t => b.log.filter(l => /sidestart|sideend|fieldstart|fieldend|\|-weather|\|turn\||-activate.*Protect|-singleturn|\|move\|/.test(l));
b.makeChoices('move 1, move 1', 'move 1, move 1');      // T1: Tailwind, Trick Room, Torkoal Protect, Corv Iron Defense
b.makeChoices('move 3, move 2', 'move 1, move 1');      // T2: Reflect, Farigiraf Protect, Torkoal Protect again (stall check)
for (let t = 3; t <= 7; t++) b.makeChoices('move 2, move 4', 'move 2, move 1');
console.log(show().join('\n'));
console.log('stall draws:', log.filter(d => d.callers.join(' ').includes('onStallMove')).map(d => `turn${d.turn} random(${d.from}) -> ${d.result}`));

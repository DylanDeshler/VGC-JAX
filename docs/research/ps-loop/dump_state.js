// Dump the shape of State.serializeBattle output for a Champions Reg M-C battle (top-level, side, pokemon, field, queue).
// Usage: PS_DIST=/path/to/ps/dist node dump_state.js
const PS = process.env.PS_DIST || '/tmp/psrepo/ps/dist';
const { Battle } = require(PS + '/sim/battle');
const mk = (species, item, ability, moves) => ({ name: '', species, item, ability, moves, nature: 'Serious', gender: '', evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50 });
const team = [mk('Talonflame', 'Sharp Beak', 'Flame Body', ['U-turn', 'Protect', 'Tailwind', 'Brave Bird']), mk('Milotic', 'Leftovers', 'Competitive', ['Protect', 'Scald', 'Recover', 'Ice Beam']), mk('Garchomp', 'Choice Scarf', 'Rough Skin', ['Protect', 'Earthquake', 'Dragon Claw', 'Rock Slide']), mk('Corviknight', 'Rocky Helmet', 'Pressure', ['Protect', 'Brave Bird', 'Roost', 'Iron Defense']), mk('Kingambit', 'Focus Sash', 'Defiant', ['Protect', 'Kowtow Cleave', 'Sucker Punch', 'Iron Head']), mk('Pelipper', 'Damp Rock', 'Drizzle', ['Protect', 'Hurricane', 'Tailwind', 'Hydro Pump'])];
const b = new Battle({ formatid: 'gen9championsvgc2026regmc', seed: [1, 2, 3, 4], p1: { name: 'A', team }, p2: { name: 'B', team } });
b.choose('p1', 'team 1263'); b.choose('p2', 'team 6123');
b.choose('p1', 'move uturn 1, move scald 2'); b.choose('p2', 'move tailwind, move protect');
const s = b.toJSON();
const show = (label, o) => console.log(label + ': ' + Object.keys(o).map(k => `${k}=${JSON.stringify(o[k]).slice(0, 60)}`).join('\n    '));
const { field, sides, log, queue, ...top } = s;
show('BATTLE', top);
show('FIELD', field);
const { pokemon, ...side } = sides[0];
show('SIDE p1', side);
show('POKEMON p1a', pokemon[0]);
console.log('QUEUE: ' + JSON.stringify(queue).slice(0, 1500));
console.log('JSON size bytes: ' + JSON.stringify(s).length + ' (log lines ' + log.length + ')');

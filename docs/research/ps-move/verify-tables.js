// Check integer formulations against PS for: stat-stage multiplier (Pokemon.calculateStat) and Battle.modify/chainModify.
const H = require('./harness');
const filler = [H.set('Milotic','Leftovers','Competitive',['Scald']), H.set('Metagross','Leftovers','Clear Body',['Iron Head'])];
const t = [H.set('Garchomp','Leftovers','Rough Skin',['Earthquake']), H.set('Corviknight','Leftovers','Pressure',['Roost']), ...filler];
const b = H.makeBattle(t, t);
const p = b.p1.active[0];
let bad = 0;
for (let s = 1; s <= 2000; s++) {
  p.storedStats.atk = s;
  for (let bo = -6; bo <= 6; bo++) {
    const ps = p.calculateStat('atk', bo);
    const ref = bo >= 0 ? Math.floor(s * (2 + bo) / 2) : Math.floor(s * 2 / (2 - bo));
    if (ps !== ref) { bad++; if (bad < 5) console.log('stage mismatch', s, bo, ps, ref); }
  }
}
console.log('stat stage: integer floor(s*(2+b)/2) | floor(2s/(2-b)) matches PS for s=1..2000, b=-6..6:', bad === 0);
// modify(): integer form
const u32 = x => x >>> 0;
const modI = (v, m) => u32((u32(v * m) + 2047) / 4096);
let bad2 = 0;
for (let v = 0; v <= 5000; v += 1) for (const m of [1024, 2048, 2732, 3072, 3686, 4505, 4915, 5120, 5324, 5325, 6144, 8192]) {
  if (b.modify(v, m, 4096) !== modI(v, m)) bad2++;
}
console.log('modify(v, m/4096) == floor((v*m + 2047)/4096):', bad2 === 0);
// chain commutativity / associativity sample
const ch = (a, c) => (a * c + 2048) >> 12;
const mods = [4915, 6144, 5325, 2732, 5324, 3072, 4505];
let nonAssoc = [];
for (const a of mods) for (const c of mods) for (const d of mods) if (ch(ch(a, c), d) !== ch(ch(a, d), c)) nonAssoc.push([a, c, d]);
console.log('chain is commutative pairwise; 3-term order-dependent cases among sample mods:', nonAssoc.length, JSON.stringify(nonAssoc.slice(0, 4)));

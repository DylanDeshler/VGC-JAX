// Usage: node diff_moves.js /tmp/psrepo/ps
// Compares champions mod moves against base gen9 for every move overridden in data/mods/champions/moves.ts
const psRoot = process.argv[2];
const {Dex} = require(psRoot + '/dist/sim');
const base = Dex.mod('gen9');
const champ = Dex.mod('champions');
const regmb = Dex.mod('championsregmb');
const fs=require('fs');
const modSrc=fs.readFileSync(psRoot + '/data/mods/champions/moves.ts','utf8');
const modIds=[...modSrc.matchAll(/^\t([a-z0-9]+): \{/gm)].map(m=>m[1]);
const modMoves={}; for (const id of modIds) modMoves[id]={};
const fields = ['basePower','accuracy','pp','type','target','priority','category','isNonstandard','flags','secondary','boosts','self','volatileStatus','noPPBoosts'];
const out = [];
for (const id of Object.keys(modMoves)) {
  const b = base.moves.get(id), c = champ.moves.get(id);
  const diffs = [];
  for (const f of fields) {
    const bv = JSON.stringify(b[f]), cv = JSON.stringify(c[f]);
    if (bv !== cv) diffs.push(`${f}: ${bv} -> ${cv}`);
  }
  // function overrides
  const fnKeys=[];
  out.push({id, name: c.name, diffs, fnKeys});
}
for (const o of out) console.log(o.id.padEnd(18), '|', o.diffs.join(' ; '), o.fnKeys.length ? ' | FN: ' + o.fnKeys.join(',') : '');
// PP cap effect on all moves (not in mod file)
let capped = [];
for (const m of base.moves.all()) {
  const c = champ.moves.get(m.id);
  if (m.pp > 20 && !modMoves[m.id]) capped.push(`${m.id}:${m.pp}->${c.pp}`);
}
console.log('\nPP>20 capped to 20 by init():', capped.length);
console.log(capped.join(' '));
// regmb differences vs champions for moves
let rd = [];
for (const m of champ.moves.all()) { const r = regmb.moves.get(m.id); for (const f of fields) { if (JSON.stringify(m[f]) !== JSON.stringify(r[f])) rd.push(`${m.id}.${f}: MC=${JSON.stringify(m[f])} MB=${JSON.stringify(r[f])}`);} }
console.log('\nRegMB vs RegMC move field diffs:', rd.length); console.log(rd.join('\n'));

// Usage: node diff_nonstandard.js /tmp/psrepo-delta/ps
const { Dex } = require(process.argv[2] + '/dist/sim');
const b = Dex.mod('gen9'), c = Dex.mod('champions');
for (const kind of ['items', 'abilities']) {
  const toLegal = [], toPast = [];
  for (const x of b[kind].all()) {
    const y = c[kind].get(x.id);
    if (x.isNonstandard && !y.isNonstandard) toLegal.push(`${x.name}(${x.isNonstandard})`);
    if (!x.isNonstandard && y.isNonstandard) toPast.push(x.name);
  }
  console.log(`${kind}: base-nonstandard -> champions-standard (${toLegal.length}): ${toLegal.join(', ')}`);
  console.log(`${kind}: base-standard -> champions-nonstandard (${toPast.length})`);
}
const mv = JSON.parse(require('fs').readFileSync('legal_moves.json'));
for (const m of ['Double Shock', 'Revival Blessing', 'Knock Off', 'Fake Out', 'Trick Room', 'Tailwind', 'Protect', 'Detect', 'Wide Guard', 'Quick Guard', 'Follow Me', 'Rage Powder', 'Helping Hand', 'Spore', 'Parting Shot', 'U-turn', 'Encore', 'Curse', 'Salt Cure', 'Make It Rain', 'Freeze-Dry', 'Rage Fist', 'Milk Drink', 'Electro Shot', 'Weather Ball', 'Last Respects', 'Shed Tail', 'Ally Switch', 'Perish Song', 'Sleep Powder', 'Yawn', 'Icy Wind', 'Electroweb', 'Snarl', 'Heat Wave', 'Dazzling Gleam', 'Rock Slide', 'Earthquake'])
  console.log(m.padEnd(16), mv.movesReachable[m] ? 'reachable by ' + mv.movesReachable[m] + ' species' : 'NOT reachable');

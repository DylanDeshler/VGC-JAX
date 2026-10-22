// Drive a Champions VGC Reg M-C battle with random-doubles teams and random legal choices.
// Prints the protocol log, requests and (optionally) a serialized state.
// Usage: node run_battle.js [seed] [--requests] [--state-at=TURN] [--quiet]
const PS = process.env.PS_DIST || '/tmp/psrepo/ps/dist';
const { Battle } = require(PS + '/sim/battle');
const { Teams } = require(PS + '/sim/teams');
const { PRNG } = require(PS + '/sim/prng');
const args = process.argv.slice(2);
const seedNum = Number(args.find(a => /^\d+$/.test(a)) || 1);
const showReq = args.includes('--requests');
const quiet = args.includes('--quiet');
const stateAt = Number((args.find(a => a.startsWith('--state-at=')) || '=0').split('=')[1]);
const seed = [0, 0, 0, seedNum].map(n => n & 0xffff);
const gen = Teams.getGenerator('gen9championsrandomdoublesbattle', seed);
const t1 = gen.getTeam(), t2 = gen.getTeam();
const rng = new PRNG([1, 2, 3, seedNum]);
const requests = { p1: null, p2: null };
const battle = new Battle({
  formatid: 'gen9championsvgc2026regmc', seed: [seedNum, 7, 11, 13],
  send: (type, data) => {
    if (type === 'sideupdate') {
      const [sid, line] = data.split('\n');
      if (line.startsWith('|request|')) {
        requests[sid] = JSON.parse(line.slice(9));
        if (showReq) console.log(`REQUEST ${sid}: ${line.slice(9)}`);
      } else if (line.startsWith('|error|')) console.log(`ERROR ${sid}: ${line}`);
    }
  },
  p1: { name: 'A', team: t1 }, p2: { name: 'B', team: t2 },
});
function choose(side) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  if (req.teamPreview) {
    const idx = [0, 1, 2, 3, 4, 5]; rng.shuffle(idx);
    return 'team ' + idx.slice(0, 4).map(i => i + 1).join('');
  }
  if (req.forceSwitch) {
    const used = new Set(); const parts = [];
    const bench = req.side.pokemon.map((p, i) => [p, i]).filter(([p, i]) => !p.active && !p.condition.endsWith(' fnt'));
    req.forceSwitch.forEach((f) => {
      if (!f) { parts.push('pass'); return; }
      const opts = bench.filter(([p, i]) => !used.has(i));
      if (!opts.length) { parts.push('pass'); return; }
      const [, i] = rng.sample(opts); used.add(i); parts.push('switch ' + (i + 1));
    });
    return parts.join(', ');
  }
  // move request
  const parts = []; const used = new Set(); let megaUsed = false;
  req.active.forEach((a, slot) => {
    const mon = req.side.pokemon[slot];
    if (!a || mon.condition.endsWith(' fnt') || mon.commanding) { parts.push('pass'); return; }
    const moves = a.moves.map((m, i) => [m, i]).filter(([m]) => !m.disabled);
    const bench = req.side.pokemon.map((p, i) => [p, i]).filter(([p, i]) => !p.active && !p.condition.endsWith(' fnt') && !used.has(i));
    if (!a.trapped && bench.length && rng.randomChance(1, 6)) {
      const [, i] = rng.sample(bench); used.add(i); parts.push('switch ' + (i + 1)); return;
    }
    const [m, i] = rng.sample(moves);
    let s = 'move ' + (i + 1);
    const t = m.target;
    if (['normal', 'any', 'adjacentFoe'].includes(t)) s += ' ' + rng.sample([1, 2]);
    else if (t === 'adjacentAlly') s += ' ' + (slot === 0 ? -2 : -1);
    else if (t === 'adjacentAllyOrSelf') s += ' ' + rng.sample([-1, -2]);
    if (a.canMegaEvo && !megaUsed && rng.randomChance(1, 2)) { s += ' mega'; megaUsed = true; }
    parts.push(s);
  });
  return parts.join(', ');
}
let guard = 0;
while (!battle.ended && guard++ < 400) {
  if (stateAt && battle.turn === stateAt && battle.requestState === 'move') {
    console.log('STATE ' + JSON.stringify(battle.toJSON()));
    break;
  }
  for (const side of battle.sides) {
    if (side.isChoiceDone()) continue;
    if (showReq) console.log(`REQUEST ${side.id} turn=${battle.turn} state=${battle.requestState}: ${JSON.stringify(side.activeRequest)}`);
    const c = choose(side);
    if (showReq) console.log(`CHOICE ${side.id}: ${c}`);
    if (c === null) continue;
    if (!battle.choose(side.id, c)) {
      // fall back to auto if our random choice was illegal (e.g. maybeTrapped)
      console.log(`CHOICE-REJECTED ${side.id}: ${c} :: ${side.choice.error}`);
      side.choose('default'); if (battle.allChoicesDone()) battle.commitChoices();
    }
    if (battle.ended) break;
  }
}
if (!quiet) console.log(battle.log.join('\n'));

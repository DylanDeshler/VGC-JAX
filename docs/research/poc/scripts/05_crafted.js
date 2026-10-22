#!/usr/bin/env node
'use strict';
/**
 * Crafted mid-battle scenarios: Trick Room (3 turns left), Tailwind on p2 (3 turns left),
 * p1a at 30% HP holding a Sitrus Berry.
 *   A) build a battle, finish team preview, mutate live objects with PS's own APIs, refresh requests
 *   B) serialize the same pre-turn battle, edit the JSON, deserialize
 * Then play one identical turn from A and B, check both produce the same state, and check the mechanics.
 *   node scripts/05_crafted.js   -> results/05_crafted.json
 */
const fs = require('fs');
const path = require('path');
const { Battle, State, FORMAT } = require('../lib/ps');
const E = require('../lib/experiment');
const T = require('../lib/teams');
const { randomChoice } = require('../lib/player');
const { projectBattle, diffProjection } = require('../lib/project');

const sp = E.spec(Number(process.argv[2] || 3));

function preTurnBattle() {
	const b = new Battle({ formatid: FORMAT, seed: sp.seed, strictChoices: true });
	b.setPlayer('p1', { name: 'p1', team: sp.p1team });
	b.setPlayer('p2', { name: 'p2', team: sp.p2team });
	b.choose('p1', 'team 1234'); b.choose('p2', 'team 1234');
	return b; // turn 1, move requests pending
}

// ---------------------------------------------------------------- A: live mutation
const A = preTurnBattle();
const logMarkA = A.log.length;
{
	const setter = A.sides[1].active[0];
	A.field.addPseudoWeather('trickroom', setter); // runs onFieldStart (adds a -fieldstart line)
	A.field.pseudoWeather.trickroom.duration = 3;
	A.sides[1].addSideCondition('tailwind', setter);
	A.sides[1].sideConditions.tailwind.duration = 3;
	const mon = A.sides[0].active[0];
	mon.setItem('sitrusberry');
	mon.sethp(Math.floor(mon.maxhp * 0.3));
	A.makeRequest('move'); // requests embed HP/items: regenerate them
}
const mutationLines = A.log.slice(logMarkA);

// ---------------------------------------------------------------- B: edit serialized JSON
const B0 = preTurnBattle();
const st = JSON.parse(JSON.stringify(State.serializeBattle(B0)));
{
	const ref = '[Pokemon:p2a]'; // State refs are side id + position letter
	let eo = st.effectOrder;
	st.field.pseudoWeather.trickroom = { id: 'trickroom', effectOrder: ++eo, source: ref, sourceSlot: 'p2a', duration: 3 };
	st.sides[1].sideConditions.tailwind = { id: 'tailwind', effectOrder: ++eo, source: ref, sourceSlot: 'p2a', duration: 3 };
	st.effectOrder = eo;
	const p = st.sides[0].pokemon.find(x => x.isActive && x.position === 0);
	p.item = 'sitrusberry';
	p.itemState = { id: 'sitrusberry', effectOrder: ++st.effectOrder, target: '[Pokemon:p1a]' };
	p.hp = Math.floor(p.maxhp * 0.3);
	// activeRequest is recomputed on deserialize when it was not null; null it so it is rebuilt from new state
	for (const s of st.sides) delete s.activeRequest;
}
const B = State.deserializeBattle(st);

const projA = projectBattle(A); const projB = projectBattle(B);
const preDiff = diffProjection(projA, projB);

// ---------------------------------------------------------------- play one identical turn
function playTurn(b) {
	const rng = new T.HarnessRNG(4242);
	const choices = {};
	for (const side of b.sides) choices[side.id] = randomChoice(side.activeRequest, rng, { megaProb: 0, switchProb: 0 });
	const speeds = Object.fromEntries(b.getAllActive().map(p => [`${p.side.id}${'ab'[p.position]}`, { spe: p.getStat('spe'), tailwind: !!p.side.sideConditions.tailwind }]));
	const mark = b.log.length;
	for (const side of b.sides) b.choose(side.id, choices[side.id]);
	return { choices, speeds, lines: b.log.slice(mark).filter(l => !l.startsWith('|t:|')) };
}
const tA = playTurn(A); const tB = playTurn(B);
const postDiff = diffProjection(projectBattle(A), projectBattle(B));

// mechanics checks on A
const moveLines = tA.lines.filter(l => l.startsWith('|move|'));
const order = moveLines.map(l => l.split('|')[2].slice(0, 3));
const sitrusEaten = tA.lines.find(l => l.includes('Sitrus Berry')) || null;
const mon = A.sides[0].pokemon.find(p => p.baseSpecies.id === projA.sides[0].active[0].split('=')[1]);
const out = {
	teams: { p1: sp.p1team.map(s => s.species), p2: sp.p2team.map(s => s.species) },
	approachA_mutationLogLines: mutationLines,
	preTurnProjectionDiff_A_vs_B: preDiff,
	turn: { choices: tA.choices, speedsAtChoice: tA.speeds, moveOrder: order, sitrusLine: sitrusEaten, firstLines: tA.lines.slice(0, 40) },
	postTurnProjectionDiff_A_vs_B: postDiff,
	postTurnLogsEqual: JSON.stringify(tA.lines) === JSON.stringify(tB.lines),
	afterTurn: {
		trickroom: A.field.pseudoWeather.trickroom ? A.field.pseudoWeather.trickroom.duration : null,
		tailwind: A.sides[1].sideConditions.tailwind ? A.sides[1].sideConditions.tailwind.duration : null,
		p1aItem: mon ? mon.item : null, p1aHp: mon ? `${mon.hp}/${mon.maxhp}` : null,
	},
};
fs.mkdirSync(path.join(__dirname, '..', 'results'), { recursive: true });
fs.writeFileSync(path.join(__dirname, '..', 'results', '05_crafted.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));

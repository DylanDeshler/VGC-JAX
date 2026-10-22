'use strict';
// Acceptance-enumerated legal sets (support.legalForBattle) vs request-derived options (player.legalOptions):
// they must agree except where PS rejects with [Unavailable choice] (hidden info, 07 §6).
const test = require('node:test');
const assert = require('node:assert');
const tr = require('../lib/trace');
const support = require('../lib/support');
const { legalOptions } = require('../lib/player');
const { jointChoices } = require('../lib/policy');

/** request choice string -> engine slot action (04 §4.3), canonicalising mega-on-locked to plain. */
function toAction(str, req, slot, battle, sideIdx) {
	if (str === 'pass') return 28;
	const side = battle.sides[sideIdx];
	let m = /^switch (\d)$/.exec(str);
	if (m) return 24 + battle.__ident.b(side.pokemon[Number(m[1]) - 1]);
	m = /^move (\d)(?: (-?\d))?( mega)?$/.exec(str);
	if (!m) throw new Error(`unparsed ${str}`);
	const mi = Number(m[1]) - 1; const loc = m[2] === undefined ? null : Number(m[2]);
	const act = req.active[slot]; const mv = act.moves[mi];
	const locked = act.moves.length === 1 && (mv.target === undefined || mv.id === 'struggle');
	if (locked) return 0;
	let t = 0;
	if (loc !== null) t = loc === 1 ? 0 : loc === 2 ? 1 : (mv.target === 'adjacentAllyOrSelf' && loc === -(slot + 1)) ? 0 : 2;
	return 6 * mi + 2 * t + (m[3] ? 1 : 0);
}

test('legal RPC equals request-derived joint sets (modulo hidden info)', () => {
	let decisions = 0; let unavailable = 0;
	for (let i = 0; i < 30; i++) {
		const recs = tr.runBattle('legal-test', i, { teams: i % 2 ? 'randdoubles' : 'chaos', policy: 'random-weighted' });
		const h = recs[0]; const dec = recs.filter(r => r.type === 'decision');
		const b = tr.newBattle({ seed: h.seed });
		const { Teams } = require('../lib/ps');
		b.battle.setPlayer('p1', { name: 'p1', team: Teams.unpack(h.teams[0]) });
		b.battle.setPlayer('p2', { name: 'p2', team: Teams.unpack(h.teams[1]) });
		for (let k = 0; k < dec.length; k++) {
			for (let s = 0; s < 2; s++) {
				const req = b.battle.sides[s].activeRequest;
				if (!req || req.wait || req.teamPreview) continue;
				const acc = support.legalForBattle(b.battle, s);
				const lo = legalOptions(req);
				const derived = new Set(jointChoices(lo).map(c => c.map((x, slot) => toAction(x, req, slot, b.battle, s)).join(',')));
				const accepted = new Set(acc.pairs.map(p => p.join(',')));
				const unav = new Set(acc.unavailable.map(p => p.join(',')));
				for (const a of accepted) assert.ok(derived.has(a), `accepted but not derived: ${a} (battle ${i} k ${k} side ${s})`);
				for (const d of derived) assert.ok(accepted.has(d) || unav.has(d), `derived but rejected: ${d} (battle ${i} k ${k} side ${s})`);
				unavailable += unav.size; decisions++;
			}
			tr.driveTraced({ ...b, choose: null, script: [dec[k].choices], stopAfter: 1 });
		}
	}
	assert.ok(decisions > 500, `only ${decisions} decisions`);
	console.log(`legal parity over ${decisions} side-decisions; ${unavailable} hidden-info rejections`);
});

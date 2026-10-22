#!/usr/bin/env node
'use strict';
/**
 * Legal-action mask check: the action set derived from PS's request JSON (lib/player.legalOptions,
 * standing in for the JAX mask) versus what PS actually accepts.
 * At every move/switch decision, the battle is copied (State round trip) and every joint choice is
 * submitted to side.choose() on the copy:
 *   - enumerated options that PS rejects   => mask too permissive (or hidden info: [Unavailable choice])
 *   - non-enumerated probes that PS accepts => mask too strict
 *   node scripts/08_masks.js [N=60]  -> results/08_masks.json
 */
const fs = require('fs');
const path = require('path');
const { State } = require('../lib/ps');
const E = require('../lib/experiment');
const { legalOptions, jointOk } = require('../lib/player');

const N = Number(process.argv[2] || 60);
const res = { N, decisionPoints: 0, jointTested: 0, jointAccepted: 0, rejectedInvalid: 0, rejectedUnavailable: 0,
	probes: 0, probesAccepted: 0, invalidByReason: {}, examples: { invalid: [], unavailable: [], probeAccepted: [] }, sampled: 0 };

function product(slots, cap) {
	let out = [[]];
	for (const opts of slots) {
		const next = [];
		for (const p of out) for (const o of opts) next.push(p.concat([o]));
		out = next;
		if (out.length > cap * 4) break;
	}
	return out;
}

function tryChoice(side, choice) {
	try {
		const ok = side.choose(choice);
		const err = side.choice.error;
		side.clearChoice();
		return ok ? 'ok' : (err && err.includes('Unavailable') ? 'unavailable' : 'invalid:' + err);
	} catch (e) {
		side.clearChoice();
		return e.message.includes('[Unavailable choice]') ? 'unavailable' : 'invalid:' + e.message.split('\n')[0];
	}
}

for (let i = 0; i < N; i++) {
	const sp = E.spec(i);
	E.run(sp, {
		onDecision: (battle, sideid, req) => {
			if (req.teamPreview || req.wait) return;
			res.decisionPoints++;
			const copy = State.deserializeBattle(JSON.stringify(State.serializeBattle(battle)));
			const side = copy.sides.find(s => s.id === sideid);
			const lo = legalOptions(req);
			let joint = product(lo.slots, 300).filter(c => jointOk(c, lo));
			if (joint.length > 300) { res.sampled++; joint = joint.filter((_, k) => k % Math.ceil(joint.length / 300) === 0); }
			for (const c of joint) {
				const r = tryChoice(side, c.join(', '));
				res.jointTested++;
				if (r === 'ok') res.jointAccepted++;
				else if (r === 'unavailable') { res.rejectedUnavailable++; if (res.examples.unavailable.length < 5) res.examples.unavailable.push({ battle: i, turn: battle.turn, side: sideid, choice: c.join(', ') }); }
				else { res.rejectedInvalid++; const reason = r.replace(/Your .* (doesn't|can't|must)/, 'Your X $1').replace(/replace .*/, 'replace X'); res.invalidByReason[reason] = (res.invalidByReason[reason] || 0) + 1; if (res.examples.invalid.length < 8) res.examples.invalid.push({ battle: i, turn: battle.turn, side: sideid, choice: c.join(', '), err: r }); }
			}
			// probes outside the enumerated set (per slot, other slot fixed to its first legal option)
			if (lo.kind === 'move') {
				lo.slots.forEach((opts, si) => {
					if (opts[0] === 'pass') return;
					const other = lo.slots.map(o => o.find(x => !x.startsWith('switch')) || o[0]);
					const set = new Set(opts);
					const cands = [];
					for (let m = 1; m <= 4; m++) for (const loc of ['', ' 1', ' 2', ' -1', ' -2']) cands.push(`move ${m}${loc}`);
					for (let s = 1; s <= 6; s++) cands.push(`switch ${s}`);
					for (const c of cands) {
						if (set.has(c)) continue;
						const ch = other.slice(); ch[si] = c;
						const sw = ch.filter(x => x.startsWith('switch'));
						if (new Set(sw).size !== sw.length) continue;
						res.probes++;
						const r = tryChoice(side, ch.join(', '));
						if (r === 'ok') { res.probesAccepted++; if (res.examples.probeAccepted.length < 8) res.examples.probeAccepted.push({ battle: i, turn: battle.turn, side: sideid, choice: ch.join(', '), slotOptions: opts.slice(0, 12) }); }
					}
				});
			}
		},
	});
}
fs.mkdirSync(path.join(__dirname, '..', 'results'), { recursive: true });
fs.writeFileSync(path.join(__dirname, '..', 'results', '08_masks.json'), JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));

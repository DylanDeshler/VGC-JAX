'use strict';
/**
 * Battle drivers.
 *  runDirect(opts)  - synchronous, drives `Battle` directly via side.activeRequest (fast path)
 *  runStream(opts)  - async, drives `BattleStream` exactly like a client (|request| JSON over the stream)
 *
 * Both take {p1team, p2team, seed, choose(req, sideid, battle) -> string}, an optional `prng`
 * (instrumented proxy), an optional `script` (list of recorded [sideid, choice] for replay), and
 * an optional `onDecision(battle, sideid, req)` hook called at every decision point.
 */
const { Battle, BattleStream, Teams, FORMAT } = require('./ps');

const MAX_TURNS = 300;

function runDirect({
	p1team, p2team, seed, prng, choose, script, onDecision, onBattle, formatid = FORMAT, maxTurns = MAX_TURNS,
	strictChoices = true, // strict: rejections throw with the "[Unavailable choice]" vs "[Invalid choice]" tag
}) {
	const battle = new Battle({ formatid, seed, prng, strictChoices });
	if (onBattle) onBattle(battle); // e.g. install context hooks before teams are created
	battle.setPlayer('p1', { name: 'p1', team: p1team });
	battle.setPlayer('p2', { name: 'p2', team: p2team });
	return drive(battle, { choose, script, onDecision, maxTurns });
}

/** Drive an already-constructed (or deserialized) battle to completion. */
function drive(battle, { choose, script, onDecision, maxTurns = MAX_TURNS, stopAfterDecisions = Infinity }) {
	const choices = []; let decisions = 0; let scriptPos = 0; let diverged = null;
	let unavailable = 0; const rejected = []; let lastReqSeen = null;
	while (!battle.ended && battle.turn <= maxTurns) {
		let acted = false;
		for (const side of battle.sides) {
			const req = side.activeRequest;
			if (!req || req.wait || side.isChoiceDone()) continue;
			if (decisions >= stopAfterDecisions) return { battle, choices, decisions, stopped: true, scriptPos };
			// once per decision point (an [Unavailable choice] retry re-presents the same request object)
			if (onDecision && req !== lastReqSeen) { onDecision(battle, side.id, req); lastReqSeen = req; }
			let choice;
			if (script) {
				const s = script[scriptPos++];
				if (!s || s[0] !== side.id) { diverged = `script mismatch at decision ${decisions}`; break; }
				choice = s[1];
			} else {
				choice = choose(req, side.id, battle);
			}
			decisions++;
			let ok;
			try {
				ok = battle.choose(side.id, choice);
			} catch (e) {
				ok = false; side.choice.error = e.message.split('\n')[0];
			}
			if (!ok) {
				const err = side.choice.error || 'unknown';
				// Hidden-information rejections (maybeTrapped / maybeDisabled): PS refreshes the request; retry.
				if (!script && err.includes('[Unavailable choice]') && unavailable < 20) {
					unavailable++; rejected.push([side.id, choice, err]); side.clearChoice(); acted = true; break;
				}
				if (!script) throw new Error(`invalid choice ${side.id} "${choice}": ${err}`);
				diverged = `choice rejected at decision ${decisions}: ${choice}: ${err}`; break;
			}
			choices.push([side.id, choice]);
			acted = true;
			break; // re-read requests: committing may have changed both sides' requests
		}
		if (diverged) break;
		if (!acted && !battle.ended) {
			if (script && scriptPos >= script.length) break;
			throw new Error(`stuck: no side can act (turn ${battle.turn}, requestState ${battle.requestState})`);
		}
	}
	return { battle, choices, decisions, diverged, scriptPos, winner: battle.winner, turns: battle.turn, rejected };
}

/** BattleStream driver (async). Returns { log, choices, decisions, turns, winner }. */
async function runStream({ p1team, p2team, seed, choose, formatid = FORMAT, onStream }) {
	const stream = new BattleStream({ noCatch: true });
	const choices = []; let decisions = 0; const log = []; let winner = null; const rejected = [];
	await stream.write(`>start ${JSON.stringify({ formatid, seed })}`);
	if (onStream) onStream(stream);
	await stream.write(`>player p1 ${JSON.stringify({ name: 'p1', team: Teams.pack(p1team) })}\n` +
		`>player p2 ${JSON.stringify({ name: 'p2', team: Teams.pack(p2team) })}`);
	for await (const chunk of stream) {
		const nl = chunk.indexOf('\n');
		const type = chunk.slice(0, nl);
		if (type === 'sideupdate') {
			const [sideid, data] = [chunk.slice(nl + 1, nl + 3), chunk.slice(nl + 4)];
			if (data.startsWith('|request|')) {
				const req = JSON.parse(data.slice(9));
				if (req.wait) continue;
				const choice = choose(req, sideid);
				decisions++;
				choices.push([sideid, choice]);
				void stream.write(`>${sideid} ${choice}`);
			} else if (data.startsWith('|error|')) {
				// [Unavailable choice] is followed by a refreshed |request|; anything else is a player bug
				if (!data.startsWith('|error|[Unavailable choice]')) throw new Error(`stream error for ${sideid}: ${data}`);
				choices.pop(); rejected.push([sideid, data]);
			}
		} else if (type === 'update') {
			log.push(chunk.slice(nl + 1));
		} else if (type === 'end') {
			winner = JSON.parse(chunk.slice(nl + 1)).winner;
			break;
		}
		if (stream.battle && stream.battle.turn > MAX_TURNS) break;
	}
	return { log: log.join('\n'), choices, decisions, turns: stream.battle?.turn, winner, battle: stream.battle, rejected };
}

/** Log normalised for comparison (drop wall-clock timestamps). */
function normLog(battleOrLog) {
	const log = Array.isArray(battleOrLog) ? battleOrLog : battleOrLog.log;
	return log.filter(l => !l.startsWith('|t:|')).join('\n');
}

module.exports = { runDirect, runStream, drive, normLog, MAX_TURNS };

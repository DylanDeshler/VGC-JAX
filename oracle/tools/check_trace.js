#!/usr/bin/env node
'use strict';
/**
 * Corpus sanity check (M0.4): every draw maps to a contract DrawKind (contract/drawkey.md §5) or is
 * don't-care (§6) / degenerate (§7); no unreachable kinds (§8); keys have actors where the contract
 * requires one; classifiers agree. Prints a kind histogram as JSON. Exit 1 on any violation.
 *   node oracle/tools/check_trace.js <dir-with-shards>
 */
const fs = require('fs');
const path = require('path');
const { readShard } = require('../lib/trace');

const FORCED = new Set(['ACCURACY', 'CRIT', 'DAMAGE_ROLL', 'CONFUSION_SELFHIT_DAMAGE_ROLL', 'SECONDARY', 'SECONDARY_STATUS_PICK',
	'SELF_EFFECT_ROLL', 'MULTIHIT_COUNT', 'RANDOM_TARGET', 'FRAC_PRIO', 'PARALYSIS', 'SLEEP_DURATION', 'FREEZE_THAW',
	'CONFUSION_DURATION', 'CONFUSION_SELFHIT', 'PARTIAL_TRAP_DURATION', 'LOCKEDMOVE_DURATION', 'PROTECT_STALL',
	'ALLY_SWITCH_STALL', 'INFATUATION', 'FORCED_SWITCH_PICK', 'ABILITY_ONHIT_PROC', 'ABILITY_RESIDUAL_PROC',
	'ABILITY_TRACE_TARGET', 'ABILITY_FOREWARN_PICK', 'ABILITY_PICKUP_TARGET', 'ABILITY_MOODY_STAT', 'ITEM_FOCUS_BAND',
	'MOVE_FICKLE_BEAM', 'MOVE_SHELLSIDEARM_TIE', 'MOVE_ACUPRESSURE_STAT', 'MOVE_CALL_PICK']);
const DONTCARE = new Set(['RANDOM_TARGET_NOMINAL', 'RANDOM_TARGET_NOEFFECT', 'QUEUE_INSERT_TIE', 'TEAM_SORT', 'GENDER']);
const DONTCARE_IF_NO_CHANCE = new Set(['SECONDARY', 'SELF_EFFECT_ROLL']);
const TIE_KINDS = new Set(['SPEED_TIE_SELECT', 'SPEED_TIE_SWITCHIN', 'SPEED_TIE_FIELD_EVENT', 'SPEED_TIE_WEATHER_EACH',
	'SPEED_TIE_HANDLER_MUL', 'SPEED_TIE_ABILITY']);
const NO_ACTOR_OK = new Set(['FORCED_SWITCH_PICK']); // dragged-out mon may be -1 only if PS bails out

function main() {
	const dir = process.argv[2];
	const files = fs.readdirSync(dir).filter(f => f.endsWith('.ndjson.zst')).sort();
	const hist = {}; const ties = {}; const problems = []; let battles = 0; let draws = 0;
	let dropped = 0; let unrecorded = 0;
	for (const f of files) {
		for (const r of readShard(path.join(dir, f))) {
			if (r.type === 'header') { battles++; continue; }
			if (r.type === 'final') {
				if (r.diag) { dropped += r.diag.droppedTies; unrecorded += r.diag.unrecordedSelectTies; }
				continue;
			}
			for (const d of r.draws) {
				draws++;
				const tag = d.kind + (d.dontcare ? '*' : '') + (d.degenerate ? '~' : '');
				hist[tag] = (hist[tag] || 0) + 1;
				const where = `${f} k=${r.k} ${JSON.stringify(d.key)}`;
				if (d.classifierDisagree) problems.push(`classifier disagreement ${d.kind} vs ${d.classifierDisagree} ${where}`);
				if (FORCED.has(d.kind)) {
					if (d.dontcare && !DONTCARE_IF_NO_CHANCE.has(d.kind)) problems.push(`forced kind marked dontcare ${where}`);
					if (!d.key) problems.push(`no key ${d.kind} ${where}`);
					else if (d.key.actor < 0 && !NO_ACTOR_OK.has(d.kind)) problems.push(`no actor ${d.kind} ${where}`);
				} else if (DONTCARE.has(d.kind)) {
					if (!d.dontcare) problems.push(`don't-care kind not flagged ${where}`);
				} else {
					problems.push(`unmapped/unreachable kind ${d.kind} ${where}`);
				}
			}
			for (const t of r.ties) {
				ties[t.kind] = (ties[t.kind] || 0) + 1;
				if (!TIE_KINDS.has(t.kind)) problems.push(`bad tie kind ${t.kind}`);
				if (t.tied.length < 2) problems.push(`tie with <2 members ${JSON.stringify(t)}`);
			}
		}
	}
	const out = { battles, draws, perBattle: +(draws / Math.max(1, battles)).toFixed(1), kinds: hist, ties,
		droppedDontCareTies: dropped, unrecordedSelectTies: unrecorded, problems: problems.length };
	console.log(JSON.stringify(out, null, 1));
	for (const p of problems.slice(0, 30)) console.error(p);
	process.exitCode = problems.length ? 1 : 0;
}
main();

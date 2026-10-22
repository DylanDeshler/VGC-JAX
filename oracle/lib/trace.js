'use strict';
/**
 * Traced battle runner + trace writer (07 §3).
 *
 * One battle = header, one `decision` record per decision point, and a `final` record. `draws`/`ties` of a
 * decision hold everything between that decision's commit and the next decision request; the next record's
 * `projection` is the state they produced.
 */
const fs = require('fs');
const zlib = require('zlib');
const { Battle, Teams, State, FORMAT } = require('./ps');
const T = require('./teams');
const { InstrumentedPRNG } = require('./rng');
const instrument = require('./instrument');
const { projectBattle } = require('./project');
const { getPolicy } = require('./policy');
const seeds = require('./seeds');

const TRACE_VERSION = 1;

/** Structured team sets for consumers that do not parse PS's packed format (ids are toID'd). */
function teamSets(packedOrSets) {
	const { toID } = require('./ps');
	const team = typeof packedOrSets === 'string' ? Teams.unpack(packedOrSets) : packedOrSets;
	return team.map(s => ({
		species: toID(s.species || s.name), item: toID(s.item), ability: toID(s.ability), moves: s.moves.map(toID),
		nature: toID(s.nature || 'serious'), evs: ['hp', 'atk', 'def', 'spa', 'spd', 'spe'].map(k => (s.evs && s.evs[k]) || 0),
		gender: s.gender || '',
	}));
}
const MAX_DECISIONS = 3000;

function phaseOf(reqs) {
	for (const r of reqs) {
		if (!r || r.wait) continue;
		if (r.teamPreview) return 'preview';
		if (r.forceSwitch) return 'switch';
		return 'move';
	}
	return 'none';
}

/** Builds teams for battle i of run S from a team source. Returns {teams:[set[],set[]], source:[str,str]}. */
function makeTeams(source, sd) {
	const rng = new T.HarnessRNG(sd.teamSeed);
	if (source === 'chaos') return { teams: [T.chaosTeam(rng).team, T.chaosTeam(rng).team], source: ['chaos', 'chaos'] };
	if (source === 'randdoubles') {
		const a = T.randomDoubles('sodium,' + sd.teamHex.slice(0, 32));
		const b = T.randomDoubles('sodium,' + sd.teamHex.slice(32, 64));
		for (const t of [a, b]) for (const s of t) if (!s.gender) s.gender = 'M'; // never let PS draw gender
		return { teams: [a, b], source: ['randdoubles', 'randdoubles'] };
	}
	if (source.startsWith('stage:')) {
		const { stageTeam } = require('./stageteams');
		const stage = source.slice(6);
		return { teams: [stageTeam(stage, rng), stageTeam(stage, rng)], source: [source, source] };
	}
	throw new Error(`unknown team source ${source}`);
}

/** Creates an instrumented battle. `classifier`: markers | stack | both. */
function newBattle({ seed, formatid = FORMAT, classifier = 'markers', prngOpts = {} }) {
	const prng = new InstrumentedPRNG({
		seed, mode: 'record', ctx: true, stack: classifier !== 'markers', markers: true, classifier, ...prngOpts,
	});
	const battle = new Battle({ formatid, seed, prng, strictChoices: true });
	prng.attach(battle);
	const ident = instrument.install(battle, prng);
	require('./coverage').install(battle);
	return { battle, prng, ident };
}

/**
 * Drive `battle` to the end (or `stopAfter` decisions), choosing with `choose(req, sideid)`, emitting records.
 * Returns {decisions:[...], final, rejected}.
 */
function driveTraced({ battle, prng, ident, choose, legal = null, stopAfter = Infinity, script = null }) {
	const ctx = battle.__ctx;
	const decisions = []; const rejected = [];
	let k = 0;
	while (!battle.ended && k < stopAfter) {
		if (k >= MAX_DECISIONS) throw new Error(`exceeded ${MAX_DECISIONS} decisions`);
		const reqs = battle.sides.map(s => s.activeRequest);
		const need = battle.sides.filter(s => s.activeRequest && !s.activeRequest.wait && !s.isChoiceDone());
		if (!need.length) throw new Error(`stuck: no side can act (turn ${battle.turn})`);
		const rec = {
			type: 'decision', k, turn: battle.turn, phase: phaseOf(reqs),
			requests: reqs.map(r => (r ? JSON.parse(JSON.stringify(r)) : null)),
			projection: projectBattle(battle),
			legal: legal ? legal(battle) : null,
			choices: [null, null],
		};
		const d0 = prng.draws.length; const t0 = ctx.ties.length;
		for (const side of need) {
			let tries = 0;
			for (;;) {
				const req = side.activeRequest;
				const choice = script ? script[k][side.n] : choose(req, side.id, battle);
				let ok;
				try { ok = battle.choose(side.id, choice); } catch (e) { ok = false; side.choice.error = e.message.split('\n')[0]; }
				if (ok) { rec.choices[side.n] = choice; break; }
				const err = side.choice.error || 'unknown';
				rejected.push({ k, side: side.id, choice, err });
				if (!script && err.includes('[Unavailable choice]') && ++tries < 20) { side.clearChoice(); continue; }
				throw new Error(`choice rejected k=${k} ${side.id} "${choice}": ${err}`);
			}
			if (battle.ended) break;
		}
		rec.draws = prng.draws.slice(d0).filter(d => d.method !== 'shuffle').map(d => instrument.drawRecord(d, ident));
		rec.ties = ctx.ties.slice(t0);
		decisions.push(rec);
		k++;
	}
	const final = battle.ended ? {
		type: 'final', winner: battle.winner || '', turns: battle.turn, projection: projectBattle(battle),
		diag: { droppedTies: ctx.droppedTies, unrecordedSelectTies: ctx.unrecordedSelectTies, rejected: rejected.length },
		coverage: battle.__cov ? { handlers: battle.__cov.handlers, pairs: battle.__cov.pairs } : null,
	} : null;
	return { decisions, final, rejected };
}

/** Run battle i of run S; returns the list of trace records (header, decisions..., final). */
function runBattle(S, i, opts = {}) {
	const {
		teams: teamSource = 'chaos', policy = 'random', stage = null, classifier = 'markers', legal = null,
		formatid = FORMAT,
	} = opts;
	const sd = seeds.battleSeeds(S, i);
	const { teams, source } = makeTeams(teamSource, sd);
	const { battle, prng, ident } = newBattle({ seed: sd.psSeed, formatid, classifier });
	battle.setPlayer('p1', { name: 'p1', team: teams[0] });
	battle.setPlayer('p2', { name: 'p2', team: teams[1] });
	const rng = new T.HarnessRNG(sd.policySeed);
	const choose = getPolicy(policy);
	// --legal on|off|<fraction>: acceptance-enumerated legal sets at every decision of sampled battles (07 §6)
	let legalFn = null;
	const frac = legal === 'on' || legal === true ? 1 : legal === 'off' || !legal ? 0 : Number(legal);
	if (frac > 0 && (frac >= 1 || seeds.u32(seeds.sha256('legal', S, i)) / 4294967296 < frac)) {
		const support = require('./support');
		legalFn = b => [0, 1].map(k => support.legalForBattle(b, k));
	}
	const header = {
		type: 'header', trace_version: TRACE_VERSION, contract_version: seeds.CONTRACT_VERSION,
		ps_commit: seeds.psCommit(), oracle_git: seeds.oracleGit(), format: formatid,
		battle_id: sd.battleId, run_seed: S, index: i, seed: sd.psSeed, stage, team_source: source,
		teams: teams.map(t => Teams.pack(t)), team_sets: teams.map(t => teamSets(Teams.pack(t))), scenario: null, policy,
		classifier,
		canonical_dontcare: Object.keys(prng.canonical || {}).sort(),
	};
	const { decisions, final } = driveTraced({ battle, prng, ident, choose: (req, sid) => choose(req, rng, battle, sid), legal: legalFn });
	return [header, ...decisions, final];
}

/** Serialize a battle with the identity info needed to rebuild DrawKeys after deserialization. */
function snapshotOf(battle) {
	return { state: State.serializeBattle(battle), ident: battle.__ident.toJSON(), turn_state: battle.__ident.saveTurnState() };
}
/** Restore an instrumented battle from snapshotOf() output. */
function fromSnapshot(snap, { classifier = 'markers', prngOpts = {} } = {}) {
	const battle = State.deserializeBattle(JSON.parse(JSON.stringify(snap.state)));
	const prng = new InstrumentedPRNG({
		seed: battle.prng.getSeed(), mode: 'record', ctx: true, stack: classifier !== 'markers', markers: true, classifier,
		...prngOpts,
	});
	battle.prng = prng;
	prng.attach(battle);
	const ident = instrument.install(battle, prng);
	snap.ident.forEach((list, n) => { if (list) ident.set(n, list); });
	ident.wrapPokemon();
	ident.loadTurnState(snap.turn_state);
	return { battle, prng, ident };
}

// ------------------------------------------------------------------ shard writer
// A shard is a concatenation of zstd frames, one per battle (zstd decoders read multi-frame files).
const compressFrame = text => zlib.zstdCompressSync(Buffer.from(text, 'utf8'),
	{ params: { [zlib.constants.ZSTD_c_compressionLevel]: 6 } });
class ShardWriter {
	constructor(file) {
		this.file = file;
		this.out = fs.createWriteStream(file);
	}
	writeFrame(buf) { this.out.write(buf); }
	writeBattle(records) { this.writeFrame(compressFrame(records.map(r => JSON.stringify(r)).join('\n') + '\n')); }
	close() {
		return new Promise((resolve, reject) => { this.out.on('error', reject); this.out.end(resolve); });
	}
}

/** Split a buffer of concatenated zstd frames (RFC 8878 §3.1.1) into per-frame buffers. */
function splitFrames(buf) {
	const frames = []; let pos = 0;
	while (pos < buf.length) {
		const start = pos;
		const magic = buf.readUInt32LE(pos); pos += 4;
		if ((magic & 0xFFFFFFF0) === 0x184D2A50) { pos += 4 + buf.readUInt32LE(pos); continue; } // skippable frame
		if (magic !== 0xFD2FB528) throw new Error(`bad zstd magic at ${start}`);
		const fhd = buf[pos++];
		const fcsFlag = fhd >> 6; const single = (fhd >> 5) & 1; const checksum = (fhd >> 2) & 1; const didFlag = fhd & 3;
		if (!single) pos += 1;
		pos += [0, 1, 2, 4][didFlag];
		pos += [single ? 1 : 0, 2, 4, 8][fcsFlag];
		for (;;) {
			const bh = buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16); pos += 3;
			const last = bh & 1; const type = (bh >> 1) & 3; const size = bh >>> 3;
			pos += type === 1 ? 1 : size;
			if (last) break;
		}
		if (checksum) pos += 4;
		frames.push(buf.subarray(start, pos));
	}
	return frames;
}

/** All records of a shard (one zstd frame per battle). */
function readShard(file) {
	const out = [];
	for (const fr of splitFrames(fs.readFileSync(file))) {
		for (const l of zlib.zstdDecompressSync(fr).toString('utf8').split('\n')) if (l) out.push(JSON.parse(l));
	}
	return out;
}

module.exports = {
	teamSets, runBattle, driveTraced, newBattle, makeTeams, snapshotOf, fromSnapshot, ShardWriter, readShard, splitFrames, compressFrame, phaseOf, TRACE_VERSION,
};

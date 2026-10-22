#!/usr/bin/env node
'use strict';
/**
 * Oracle CLI + JSON-RPC server (02 §4.1-4.2).
 *   oracle.js gen --n N --seed S [--workers W] [--teams chaos|randdoubles] [--policy random|random-weighted]
 *                 [--classifier markers|stack|both] [--shard-size 10000] [--stage core|v0|v1|v2] [--legal on|off|f]
 *                 [--require kind:id,...] [--events] --out DIR
 *     --events: decision records carry `events`, the PS protocol reduced to PublicEvents (oracle/lib/protocol.js)
 *   oracle.js serve                 line-delimited JSON-RPC on stdin/stdout
 *   oracle.js check-callsites       RNG call-site drift alarm (non-zero exit on change)
 */
const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

function parseArgs(argv) {
	const out = { _: [] };
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (a.startsWith('--')) {
			const k = a.slice(2);
			const v = argv[i + 1] !== undefined && !argv[i + 1].startsWith('--') ? argv[++i] : 'true';
			out[k] = v;
		} else {
			out._.push(a);
		}
	}
	return out;
}

// ------------------------------------------------------------------ worker
if (!isMainThread) {
	const tr = require('../lib/trace');
	const zlib = require('zlib');
	const { S, opts } = workerData;
	parentPort.on('message', msg => {
		if (msg.type === 'battle') {
			let lines; let error = null;
			try {
				// one zstd frame per battle: byte-identical shards for any worker count / timing
				lines = tr.compressFrame(tr.runBattle(S, msg.i, opts).map(r => JSON.stringify(r)).join('\n') + '\n');
			} catch (e) {
				error = `${e.message}\n${e.stack}`;
			}
			parentPort.postMessage({ type: 'done', i: msg.i, lines, error });
		} else if (msg.type === 'exit') {
			process.exit(0);
		}
	});
	parentPort.postMessage({ type: 'ready' });
	return;
}

// ------------------------------------------------------------------ gen
async function gen(args) {
	const tr = require('../lib/trace');
	const seeds = require('../lib/seeds');
	const N = Number(args.n || 100); const S = String(args.seed ?? '1');
	const W = Math.max(1, Number(args.workers || 1));
	const shardSize = Number(args['shard-size'] || 10000);
	const outDir = args.out; if (!outDir) throw new Error('--out DIR required');
	// --require kind:id[,kind:id] with --stage S  ->  pool D teams (07 §9.2)
	const teamsArg = args.require ? `poold:${args.stage || 'v0'}:${String(args.require).split(',').join('+')}` : (args.teams || 'chaos');
	const opts = {
		teams: teamsArg, policy: args.policy || 'random', stage: args.stage || null,
		classifier: args.classifier || 'markers', legal: args.legal || 'off', events: args.events === 'true',
	};
	fs.mkdirSync(outDir, { recursive: true });
	const meta = {
		run_seed: S, n: N, argv: process.argv.slice(2), ps_commit: seeds.psCommit(), oracle_git: seeds.oracleGit(),
		contract_version: seeds.CONTRACT_VERSION, trace_version: tr.TRACE_VERSION, opts, shard_size: shardSize,
	};
	const t0 = Date.now();
	const pending = new Map(); let nextWrite = 0; let shard = null; let shardIdx = -1;
	const errors = [];
	let chain = Promise.resolve();
	const writeReady = () => (chain = chain.then(flush)); // serialised: flush awaits shard.close()
	const flush = async () => {
		while (pending.has(nextWrite)) {
			const s = Math.floor(nextWrite / shardSize);
			if (s !== shardIdx) {
				if (shard) await shard.close();
				shardIdx = s;
				shard = new tr.ShardWriter(path.join(outDir, `shard-${String(s).padStart(5, '0')}.ndjson.zst`));
			}
			const r = pending.get(nextWrite); pending.delete(nextWrite);
			if (r.error) errors.push({ i: nextWrite, error: r.error });
			else shard.writeFrame(Buffer.from(r.lines));
			nextWrite++;
		}
	};
	await new Promise((resolve, reject) => {
		let done = 0;
		const workers = [];
		for (let w = 0; w < W; w++) {
			const mine = []; for (let i = w; i < N; i += W) mine.push(i);
			if (!mine.length) continue;
			const wk = new Worker(__filename, { workerData: { S, opts } });
			let pos = 0;
			const sendNext = () => {
				if (pos < mine.length) wk.postMessage({ type: 'battle', i: mine[pos++] });
				else wk.postMessage({ type: 'exit' });
			};
			wk.on('message', async msg => {
				if (msg.type === 'ready') { sendNext(); return; }
				pending.set(msg.i, msg); done++;
				sendNext();
				await writeReady();
				if (done === N) resolve();
			});
			wk.on('error', reject);
			workers.push(wk);
		}
		if (N === 0) resolve();
	});
	await writeReady();
	if (shard) await shard.close();
	meta.seconds = (Date.now() - t0) / 1000;
	meta.battles_per_sec = N / meta.seconds;
	meta.errors = errors.length;
	fs.writeFileSync(path.join(outDir, 'meta.json'), JSON.stringify(meta, null, 1));
	if (errors.length) fs.writeFileSync(path.join(outDir, 'errors.json'), JSON.stringify(errors, null, 1));
	console.error(`gen: ${N} battles in ${meta.seconds.toFixed(1)}s (${meta.battles_per_sec.toFixed(1)}/s), ${errors.length} errors -> ${outDir}`);
	return errors.length ? 1 : 0;
}

// ------------------------------------------------------------------ RPC
function serve() {
	const rpc = require('../lib/rpc');
	const readline = require('readline');
	const rl = readline.createInterface({ input: process.stdin, terminal: false });
	rl.on('line', line => {
		if (!line.trim()) return;
		let req;
		try { req = JSON.parse(line); } catch (e) {
			process.stdout.write(JSON.stringify({ id: null, error: `bad json: ${e.message}` }) + '\n');
			return;
		}
		process.stdout.write(JSON.stringify(rpc.handle(req)) + '\n');
	});
}

function checkCallsites() {
	const { execFileSync } = require('child_process');
	try {
		const out = execFileSync(process.execPath, [path.join(__dirname, '..', 'tools', 'scan_callsites.js'), '--check'], { encoding: 'utf8' });
		process.stdout.write(out);
		return 0;
	} catch (e) {
		process.stdout.write(e.stdout || ''); process.stderr.write(e.stderr || '');
		return e.status || 1;
	}
}

async function main() {
	const args = parseArgs(process.argv.slice(2));
	const cmd = args._[0];
	switch (cmd) {
	case 'gen': return gen(args);
	case 'serve': serve(); return null;
	case 'check-callsites': return checkCallsites();
	default:
		console.error('usage: oracle.js gen|serve|check-callsites [...]');
		return 2;
	}
}

main().then(code => { if (code !== null) process.exitCode = code; }, e => { console.error(e.stack || e); process.exitCode = 1; });

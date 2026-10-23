'use strict';
/**
 * PS protocol -> PublicEvents records (03 §4.7; engine side: vgcjax/battle/events.py, same kinds / fields / flags).
 * Reads the *public* half of each |split| block (the line after the omniscient one), maps idents ("p2a: Goodra",
 * "p1: Rotom") to refs (side*4 + brought index) through the battle's Ident, and returns
 * [{kind, actor, target, move, v1, v2, flags}] in protocol order. Lines with no PublicEvents counterpart are skipped;
 * an unknown message type becomes {kind: 'UNKNOWN', raw} so the conformance test can flag it.
 */
const STATS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const IGNORE = new Set(['', 'upkeep', 'teamsize', 'start', 'gametype', 'player', 'gen', 'tier', 'rule', 'clearpoke',
	'poke', 'teampreview', 'j', 'l', 'win', 'tie', '-anim', 'uhtmlchange', 'uhtml', 't:', 'debug', 'html', 'raw',
	'inactive', 'inactiveoff', 'hint', 'message', 'bigerror', 'error', 'c', 'chat', 'done', 'request', 'sideupdate']);
const F = { DRAG: 1, MISS: 1, NOTARGET: 2, SPREAD: 4, STILL: 8, RECOIL: 1, FROM: 2, DRAIN: 1, LEVITATE: 1, OHKO: 2,
	HEAL: 1, FAILSTATUS: 2, UNBOOST: 1, FATIGUE: 8, FROM_MOVE: 1, FROM_ABILITY: 2, FROM_ITEM: 4, MSG: 1, SILENT: 2, FROM_MOVE_CURE: 4 };
/** "[from] move: X" / "ability: X" / "item: X" / "X" -> [kind, id] */
const fromEffect = s => {
	const m = /^(move|ability|item): (.*)$/.exec(s || '');
	return m ? [m[1], toID(m[2])] : ['', toID(s)];
};

function publicLines(lines) {
	const out = [];
	for (let i = 0; i < lines.length; i++) {
		if (lines[i].startsWith('|split|')) { out.push(lines[i + 2]); i += 2; } else out.push(lines[i]);
	}
	return out;
}

function hp(s) {
	const t = String(s).split(' ')[0];
	if (t === '0') return [0, 0];
	const m = /^(\d+)\/100([a-z]?)$/.exec(t);
	if (!m) throw new Error(`unparsed HP "${s}"`);
	const pct = Number(m[1]);
	const colour = pct === 20 ? (m[2] === 'y' ? 1 : 2) : pct === 50 ? (m[2] === 'g' ? 1 : 2) : 0;
	return [pct, colour];
}

function reduce(lines, battle) {
	const ident = battle.__ident;
	const refOf = s => {
		if (!s || s === 'null' || s === 'undefined') return -1;
		const m = /^p([12])[a-z]?: (.*)$/.exec(s);
		if (!m) return -1;
		const side = Number(m[1]) - 1;
		const p = battle.sides[side].pokemon.find(x => x.name === m[2]);
		return p ? ident.ref(p) : -1;
	};
	const slotBit = s => { const m = /^p([12])([ab])$/.exec(s); return m ? 1 << ((Number(m[1]) - 1) * 2 + (m[2] === 'a' ? 0 : 1)) : 0; };
	const out = [];
	let lastMove = null;
	const ev = (kind, o = {}) => { const r = { kind, actor: -1, target: -1, move: '', v1: 0, v2: 0, flags: 0, ...o }; out.push(r); return r; };
	for (const line of publicLines(lines)) {
		if (!line || line[0] !== '|') continue;
		const parts = line.split('|').slice(1);
		const type = parts[0];
		const tags = parts.filter(x => x.startsWith('['));
		const from = (tags.find(x => x.startsWith('[from]')) || '').slice(7);
		if (type === 'switch' || type === 'drag') {
			const [pct, colour] = hp(parts[3]);
			ev('SWITCH', { actor: refOf(parts[1]), move: type === 'switch' && from ? toID(from) : '', v1: pct, v2: colour,
				flags: type === 'drag' ? F.DRAG : 0 });
		} else if (type === 'move') {
			let flags = 0; let v1 = 0;
			if (tags.includes('[miss]')) flags |= F.MISS;
			if (tags.includes('[notarget]')) flags |= F.NOTARGET;
			if (tags.includes('[still]')) flags |= F.STILL;
			const sp = tags.find(x => x.startsWith('[spread]'));
			if (sp) { flags |= F.SPREAD; for (const s of sp.slice(9).split(',')) v1 |= slotBit(s.trim()); }
			lastMove = ev('MOVE', { actor: refOf(parts[1]), target: refOf(parts[3]), move: toID(parts[2]), v1, flags });
		} else if (type === '-damage' || type === '-heal') {
			const [pct, colour] = hp(parts[2]);
			const of = (tags.find(x => x.startsWith('[of]')) || '').slice(5);
			let flags = 0; let move = '';
			if (type === '-damage' && /^recoil$/i.test(from)) flags = F.RECOIL;
			else if (type === '-heal' && from === 'drain') flags = F.DRAIN;
			else if (from) { flags = F.FROM; move = fromEffect(from)[1]; }
			ev(type === '-damage' ? 'DAMAGE' : 'HEAL', { actor: refOf(parts[1]), target: of && flags !== F.FROM ? refOf(of) : -1,
				move, v1: pct, v2: colour, flags });
		} else if (type === '-crit') {
			ev('CRIT', { actor: refOf(parts[1]) });
		} else if (type === '-supereffective' || type === '-resisted') {
			ev(type === '-supereffective' ? 'SUPEREFFECTIVE' : 'RESISTED', { actor: refOf(parts[1]), v1: Number(parts[2] || 0) });
		} else if (type === '-immune') {
			ev('IMMUNE', { actor: refOf(parts[1]), flags: from === 'ability: Levitate' ? F.LEVITATE : tags.includes('[ohko]') ? F.OHKO : 0 });
		} else if (type === '-miss') {
			ev('MISS', { actor: refOf(parts[1]), target: refOf(parts[2]) });
		} else if (type === '-fail') {
			const why = parts[2] && !parts[2].startsWith('[') ? parts[2] : '';
			ev('FAIL', { actor: refOf(parts[1]), move: why && why !== 'heal' ? toID(why) : '',
				flags: why === 'heal' ? F.HEAL : why ? F.FAILSTATUS : 0 });
		} else if (type === '-status') {
			const [k, id] = fromEffect(from);
			ev('STATUS', { actor: refOf(parts[1]), move: parts[2], v1: from ? id : '',
				flags: k === 'move' ? F.FROM_MOVE : k === 'ability' ? F.FROM_ABILITY : k === 'item' ? F.FROM_ITEM : 0 });
		} else if (type === '-curestatus') {
			const [, id] = fromEffect(from);
			ev('CURESTATUS', { actor: refOf(parts[1]), move: parts[2], v1: from ? id : '',
				flags: from ? F.FROM_MOVE_CURE : tags.includes('[silent]') ? F.SILENT : tags.includes('[msg]') ? F.MSG : 0 });
		} else if (type === 'cant') {
			ev('CANT', { actor: refOf(parts[1]), move: fromEffect(parts[2])[1] });
		} else if (type === '-boost' || type === '-unboost') {
			ev('BOOST', { actor: refOf(parts[1]), v1: STATS.indexOf(parts[2]), v2: Number(parts[3]),
				flags: type === '-unboost' ? F.UNBOOST : 0 });
		} else if (type === 'faint') {
			ev('FAINT', { actor: refOf(parts[1]) });
		} else if (type === '-hitcount') {
			ev('HITCOUNT', { actor: refOf(parts[1]), v1: Number(parts[2]) });
		} else if (type === '-ohko') {
			ev('OHKO');
		} else if (type === '-activate') {
			ev('ACTIVATE', { actor: refOf(parts[1]), move: fromEffect(parts[2])[1] });
		} else if (type === '-start' || type === '-end') {
			const [k, id] = fromEffect(from);
			ev(type === '-start' ? 'START' : 'END', { actor: refOf(parts[1]), move: fromEffect(parts[2])[1],
				v1: from ? id : '', flags: tags.includes('[fatigue]') ? F.FATIGUE : k === 'ability' ? F.FROM_ABILITY : 0 });
		} else if (type === 'turn') {
			ev('TURN', { v1: Number(parts[1]) });
		} else if (!IGNORE.has(type)) {
			ev('UNKNOWN', { raw: line });
		}
	}
	void lastMove;
	return out;
}

module.exports = { reduce, publicLines };

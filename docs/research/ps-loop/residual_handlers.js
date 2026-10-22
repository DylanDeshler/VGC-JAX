// List every non-Past effect in the champions mod that has an onResidual / onSideResidual / onFieldResidual handler,
// with its Order / SubOrder / Priority. Usage: PS_DIST=/path/to/ps/dist node residual_handlers.js
const PS = process.env.PS_DIST || '/tmp/psrepo/ps/dist';
const { Dex } = require(PS + '/sim/dex');
const dex = Dex.mod('champions');
const rows = [];
const add = (kind, name, e, nonstd) => {
  if (nonstd) return;
  for (const cb of ['onResidual', 'onSideResidual', 'onFieldResidual']) {
    if (typeof e[cb] !== 'function') continue;
    rows.push({ kind, name, cb, order: e[cb + 'Order'] ?? '-', sub: e[cb + 'SubOrder'] ?? '-', pri: e[cb + 'Priority'] ?? '-' });
  }
};
for (const a of dex.abilities.all()) add('ability', a.name, a, a.isNonstandard);
for (const i of dex.items.all()) add('item', i.name, i, i.isNonstandard);
for (const m of dex.moves.all()) if (m.condition) add('move-cond', m.name, m.condition, m.isNonstandard);
for (const id in dex.data.Conditions) { const c = dex.conditions.get(id); add('condition', c.name || id, c, false); }
rows.sort((x, y) => (x.order === '-' ? 1e9 : x.order) - (y.order === '-' ? 1e9 : y.order) || String(x.sub).localeCompare(String(y.sub)));
for (const r of rows) console.log([r.order, r.sub, r.pri, r.cb, r.kind, r.name].join('\t'));
// Duration-only handlers (no callback, but collected because fieldEvent('Residual') passes getKey='duration').
console.log('--- duration effects (counted down in Residual) ---');
const dur = [];
const addD = (kind, name, e, nonstd) => {
  if (nonstd || !(e.duration || e.durationCallback)) return;
  const ord = ['onResidualOrder', 'onSideResidualOrder', 'onFieldResidualOrder'].map(k => e[k] !== undefined ? `${k}=${e[k]}` : '').filter(Boolean).join(' ');
  const sub = ['onResidualSubOrder', 'onSideResidualSubOrder', 'onFieldResidualSubOrder'].map(k => e[k] !== undefined ? `${k}=${e[k]}` : '').filter(Boolean).join(' ');
  dur.push([kind, name, 'duration=' + (e.duration ?? 'cb'), ord || '(no order -> 4294967296, last)', sub].join('\t'));
};
for (const m of dex.moves.all()) if (m.condition) addD('move-cond', m.name, m.condition, m.isNonstandard);
for (const id in dex.data.Conditions) { const c = dex.conditions.get(id); addD('condition', c.name || id, c, false); }
for (const r of dur) console.log(r);

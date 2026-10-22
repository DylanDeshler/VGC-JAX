// List every effect with an onSwitchIn / onAnySwitchIn / onStart (abilities & items run onStart during SwitchIn)
// handler and its priority, for the champions mod. Usage: node switchin_handlers.js
const PS = process.env.PS_DIST || '/tmp/psrepo/ps/dist';
const { Dex } = require(PS + '/sim/dex');
const dex = Dex.mod('champions');
const rows = [];
const add = (kind, e) => {
  if (e.isNonstandard) return;
  const has = k => typeof e[k] === 'function';
  if (has('onSwitchIn') || has('onAnySwitchIn') || ((kind === 'ability' || kind === 'item') && has('onStart'))) {
    rows.push([kind, e.name,
      has('onSwitchIn') ? 'onSwitchIn' : (has('onAnySwitchIn') ? 'onAnySwitchIn' : 'onStart(as SwitchIn)'),
      e.onSwitchInPriority ?? e.onAnySwitchInPriority ?? 0, e.onSwitchInSubOrder ?? '']);
  }
};
for (const a of dex.abilities.all()) add('ability', a);
for (const i of dex.items.all()) add('item', i);
for (const m of dex.moves.all()) { if (m.condition) add('move-cond', Object.assign({ name: m.name + ' (cond)', isNonstandard: m.isNonstandard }, m.condition)); }
for (const id in dex.data.Conditions) { const c = dex.conditions.get(id); add('condition', c); }
rows.sort((a, b) => b[3] - a[3]);
for (const r of rows) console.log(r.join('\t'));

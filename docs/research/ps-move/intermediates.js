// Print step-by-step intermediates of the reference formula for the verified examples (roll r=0 and r=15).
const u32 = x => x >>> 0, tr16 = x => (x >>> 0) % 65536;
const modify = (v, m) => u32((u32(v * m) + 2047) / 4096), chain = (a, b) => (a * b + 2048) >> 12;
const boosted = (s, b) => b >= 0 ? Math.floor(s * (2 + b) / 2) : Math.floor(s * 2 / (2 - b));
function trace(p) {
  const out = [];
  let bpM = 4096; for (const m of p.bpMods || []) bpM = chain(bpM, m);
  let bp = Math.max(1, modify(p.bp, bpM)); out.push(`BP=${p.bp} chainM=${bpM} -> ${bp}`);
  let ab = p.atkBoost || 0, db = p.defBoost || 0; if (p.crit && ab < 0) ab = 0; if (p.crit && db > 0) db = 0;
  const A = boosted(p.atk, ab), D = boosted(p.def, db); out.push(`A=${A} D=${D}`);
  const t1 = u32(22 * bp * A), t2 = u32(t1 / D), base = u32(t2 / 50);
  out.push(`22*BP*A=${t1} /D=${t2} /50=${base}`);
  let d = base + 2; out.push(`+2=${d}`);
  if (p.spread) { d = modify(d, 3072); out.push(`spread=${d}`); }
  if (p.pb2) { d = modify(d, 1024); out.push(`PBond2=${d}`); }
  if (p.weather) { d = modify(d, p.weather); out.push(`weather=${d}`); }
  if (p.crit) { d = u32(d * 1.5); out.push(`crit=${d}`); }
  const r0 = d; d = u32(u32(d * (100 - p.roll)) / 100); out.push(`roll(${100 - p.roll})=${d}`);
  if (p.stab) { d = modify(d, p.stab); out.push(`STAB=${d}`); }
  for (let i = 0; i < p.tm; i++) d *= 2; for (let i = 0; i > p.tm; i--) d = u32(d / 2); out.push(`type(${p.tm})=${d}`);
  if (p.burn) { d = modify(d, 2048); out.push(`burn=${d}`); }
  if (p.fin) { d = modify(d, p.fin); out.push(`final(${p.fin})=${d}`); }
  return out.join(' | ');
}
const ex = {
  ex1_incin: { bp: 100, atk: 182, def: 130, spread: true, stab: 6144, tm: 1, fin: 5324 },
  ex1_rilla: { bp: 100, atk: 182, def: 112, spread: true, stab: 6144, tm: -1, fin: 5324 },
  ex2: { bp: 40, atk: 200, def: 189, atkBoost: -1, defBoost: 2, crit: true, stab: 6144, tm: 2, burn: true, fin: 5324 },
  ex3: { bp: 60, bpMods: [4915, 6144], atk: 161, def: 156, weather: 6144, stab: 6144, tm: 1, fin: 2732 },
  ex4_h1: { bp: 120, atk: 194, def: 115, stab: 6144, tm: 0 },
  ex4_h2: { bp: 120, atk: 194, def: 115, stab: 6144, tm: 0, pb2: true },
};
for (const [k, p] of Object.entries(ex)) for (const roll of [0, 15]) console.log(k, 'r=' + roll, trace({ ...p, roll }));

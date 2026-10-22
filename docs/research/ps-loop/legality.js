// Check which turn-flow-relevant moves/items/abilities are usable in Champions Reg M-C.
// Usage: node legality.js
const PS = process.env.PS_DIST || '/tmp/psrepo/ps/dist';
const { Dex } = require(PS + '/sim/dex');
const fmt = Dex.formats.get('gen9championsvgc2026regmc', true);
const dex = Dex.forFormat(fmt);
const rt = Dex.formats.getRuleTable(fmt);
// Species legal in format: not nonstandard, not banned by tag.
const legalSpecies = dex.species.all().filter(s => !s.isNonstandard && !rt.isBannedSpecies(s) && s.exists);
const learnable = new Map();
for (const s of legalSpecies) {
  const ls = dex.species.getLearnsetData(s.id);
  if (!ls || !ls.learnset) continue;
  for (const m in ls.learnset) {
    if (!learnable.has(m)) learnable.set(m, []);
    learnable.get(m).push(s.name);
  }
}
const moves = ['uturn','voltswitch','flipturn','partingshot','teleport','batonpass','shedtail','chillyreception','revivalblessing','roar','whirlwind','dragontail','circlethrow','allyswitch','instruct','afteryou','quash','round','firepledge','grasspledge','waterpledge','shelltrap','focuspunch','beakblast','encore','trickroom','tailwind','icywind','electroweb','fakeout','followme','ragepowder','pursuit','perishsong','destinybond','futuresight','doomdesire','wish','healingwish','lunardance','mefirst','copycat','metronome','sleeptalk','snatch','magiccoat','dragondarts','struggle','imprison','meanlook','block','spiderweb','jawlock','anchorshot','spiritshackle','thousandwaves','octolock','gravity','magicroom','wonderroom','sucker punch','suckerpunch','upperhand','thunderclap','grassyglide','allyswitch','skydrop','solarbeam','meteorbeam','hyperbeam','gigaimpact','outrage','petaldance','ragingfury','uproar','rollout','iceball','bide','dynamaxcannon','curse','pollenpuff','terastarstorm','expandingforce','psychicterrain','burningjealousy','lashout','strengthsap','teatime','courtchange'];
console.log('== moves (isNonstandard / #legal species learning)');
for (const id of [...new Set(moves)]) {
  const m = dex.moves.get(id);
  if (!m.exists) { console.log(id, 'DOES NOT EXIST'); continue; }
  console.log(m.name.padEnd(18), String(m.isNonstandard).padEnd(10), (learnable.get(m.id) || []).length, 'pri', m.priority, 'target', m.target, m.selfSwitch ? 'selfSwitch=' + m.selfSwitch : '', m.forceSwitch ? 'forceSwitch' : '');
}
const items = ['ejectbutton','ejectpack','redcard','quickclaw','custapberry','boosterenergy','choicescarf','choiceband','choicespecs','laggingtail','fullincense','shedshell','mentalherb','roomservice','whiteherb','mirrorherb','adrenalineorb','throatspray','focussash','sitrusberry','lifeorb','clearamulet','covertcloak','safetygoggles','assaultvest','leftovers','blacksludge','rockyhelmet','lightclay','weaknesspolicy','abilityshield','loadeddice','protectivepads','punchingglove'];
console.log('== items (isNonstandard)');
for (const id of items) { const it = dex.items.get(id); console.log(it.name.padEnd(16), it.exists ? String(it.isNonstandard) : 'NOEXIST'); }
const megaStones = dex.items.all().filter(i => i.megaStone && !i.isNonstandard);
console.log('legal mega stones:', megaStones.length, megaStones.map(i=>i.name).join(', '));
const abilities = ['emergencyexit','wimpout','dancer','commander','intimidate','drought','drizzle','sandstream','snowwarning','electricsurge','psychicsurge','grassysurge','mistysurge','orichalcumpulse','hadronengine','protosynthesis','quarkdrive','beadsofruin','swordofruin','tabletsofruin','vesselofruin','prankster','galewings','triage','quickdraw','stall','myceliummight','shadowtag','arenatrap','magnetpull','runaway','neutralizinggas','unnerve','asoneglastrier','asonespectrier','pressure','moldbreaker','trace','imposter','illusion','zerotohero','regenerator','naturalcure','harvest','speedboost','moody','slowstart','truant','download','frisk','anticipation','forewarn','screencleaner','curiousmedicine','pastelveil','hospitality','supersweetsyrup','embodyaspectteal','teraformzero','toxicchain','costar','windrider','sharpness','thermalexchange','opportunist','lingeringaroma','poisonpuppeteer','mirrorarmor','guarddog','defiant','competitive','rattled','innerfocus','owntempo','oblivious','scrappy','analytic','stalwart','propellertail','lightningrod','stormdrain','sapsipper','telepathy','wonderguard','perishbody','poisontouch','cursedbody','cutecharm','flamebody','static','effectspore','roughskin','ironbarbs','aftermath','innardsout','receiver','powerofalchemy','healer','hydration','shedskin','raindish','dryskin','icebody','solarpower','poisonheal','baddreams','pickup','ballfetch','megalauncher','tough claws','toughclaws','parentalbond','aerilate','pixilate','refrigerate','galvanize','dragonize'];
console.log('== abilities (isNonstandard / #legal species having it)');
const abilHolders = new Map();
for (const s of legalSpecies) for (const k in s.abilities) { const a = dex.abilities.get(s.abilities[k]).id; if (!abilHolders.has(a)) abilHolders.set(a, []); abilHolders.get(a).push(s.name); }
for (const id of [...new Set(abilities)]) { const a = dex.abilities.get(id); if (!a.exists) { console.log(id, 'NOEXIST'); continue; } console.log(a.name.padEnd(20), String(a.isNonstandard).padEnd(10), (abilHolders.get(a.id)||[]).length, (abilHolders.get(a.id)||[]).slice(0,6).join('/')); }
console.log('legal species count', legalSpecies.length);

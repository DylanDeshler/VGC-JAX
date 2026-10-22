// Dump the resolved rule table for the Champions VGC Reg M-C format.
const PS = process.env.PS_DIST || '/tmp/psrepo/ps/dist';
const { Dex } = require(PS + '/sim/dex');
const fmt = Dex.formats.get('gen9championsvgc2026regmc', true);
const rt = Dex.formats.getRuleTable(fmt);
console.log('format', fmt.name, 'mod', fmt.mod, 'gameType', fmt.gameType);
console.log('rules', [...rt.keys()].join(', '));
console.log('pickedTeamSize', rt.pickedTeamSize, 'minTeamSize', rt.minTeamSize, 'maxTeamSize', rt.maxTeamSize);
console.log('endlessbattleclause', rt.has('endlessbattleclause'), 'cancelmod', rt.has('cancelmod'), 'teampreview', rt.has('teampreview'));
console.log('adjustLevel', rt.adjustLevel, 'timer', JSON.stringify(rt.timer));

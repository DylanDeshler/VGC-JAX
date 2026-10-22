# Usage: python3 -I compare_bulba.py <bulba M-C wiki> <legal_species.json> <legal_megas.json>
import re, json, sys
wiki = open(sys.argv[1], encoding='utf8').read()
legal = json.load(open(sys.argv[2]))
megas = json.load(open(sys.argv[3]))
def norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())
cards = re.findall(r'\{\{ChampsCard\|(\d+)\|([^|}]+)(?:\|ig=([^|}]*))?', wiki)
names = set()
for num, name, ig in cards:
    names.add(norm(name + (ig or '')))
ps = set()
for s in legal['species']:
    if s['gmaxPlaceholder'] or s['isCosmeticForme']: continue
    ps.add(norm(s['name']))
for m in megas['megas']:
    if m['validates']: ps.add(norm(m['mega']))
print('bulba cards', len(cards), 'unique normalized', len(names), 'ps', len(ps))
print('in bulba not PS:', sorted(names - ps))
print('in PS not bulba:', sorted(ps - names))

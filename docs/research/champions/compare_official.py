# Usage: python3 -I compare_official.py <official mc_pokemon.html> <legal_species.json> <legal_megas.json>
import re, json, sys
html = open(sys.argv[1], encoding='utf8').read()
legal = json.load(open(sys.argv[2])); megas = json.load(open(sys.argv[3]))
arr = re.search(r'const pokemons = (\[.*?\]);', html, re.S).group(1)
rows = json.loads(arr)
from collections import Counter
print('official rows', len(rows), Counter(r[1] for r in rows))
def norm(s):
    s = s.replace('’', "'").lower()
    for a, b in [('(alolan form)', 'alola'), ('(galarian form)', 'galar'), ('(hisuian form)', 'hisui'), ('(paldean form (combat breed))', 'paldeacombat'),
                 ('(paldean form (blaze breed))', 'paldeablaze'), ('(paldean form (aqua breed))', 'paldeaaqua'), ('female', 'f'), ('male', 'm')]:
        s = s.replace(a, b)
    return re.sub(r'[^a-z0-9]', '', s)
off = {}
for r in rows: off[norm(r[2])] = r
ps = {}
for s in legal['species']:
    if s['gmaxPlaceholder'] or s['battleDuplicateOf']: continue
    ps[re.sub(r'[^a-z0-9]', '', s['name'].lower())] = s['name']
for m in megas['megas']:
    if m['validates']: ps[re.sub(r'[^a-z0-9]', '', m['mega'].lower())] = m['mega']
print('official not in PS:', sorted(set(off) - set(ps)))
print('PS not in official:', sorted(set(ps) - set(off)))
print('sample of official names with forms:', [r for r in rows if r[0][-3:] != '000'][:60])

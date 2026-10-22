# Usage: python3 -I html2txt.py <file> [marker] [maxchars]
import re, sys, html
t = open(sys.argv[1], encoding='utf8', errors='replace').read()
t = re.sub(r'(?s)<script.*?</script>|<style.*?</style>|<!--.*?-->', '', t)
t = re.sub(r'<br\s*/?>|</tr>|</p>|</h\d>|</li>|</div>', '\n', t)
t = re.sub(r'</td>', ' | ', t)
t = re.sub(r'<[^>]+>', ' ', t); t = html.unescape(t)
t = re.sub(r'[ \t]+', ' ', t); t = re.sub(r'\n\s*\n+', '\n', t)
marker = sys.argv[2] if len(sys.argv) > 2 else None
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
if marker:
    idx = [m.start() for m in re.finditer(re.escape(marker), t)]
    if idx: t = t[idx[-1] if len(idx) < 3 else idx[1]:]
end = t.find('All Content is ©')
if end > 0: t = t[:end]
print(t[:mx])

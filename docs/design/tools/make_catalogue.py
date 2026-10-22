"""Generate the mechanics catalogue (design/catalogue.csv) from research outputs.

Usage: python3 -I make_catalogue.py <research_dir> <out_csv>
One row per legal entity (move/ability/item/condition) + ENGINE/SYSTEM rows.
stage: v0 | v1 | v2 (approximation of research/dex cost-aware staging; regenerate with
tools/datagen/dex_stages.py in the implementation repo for the authoritative lists).
"""
import csv, json, sys, os

rd, out = sys.argv[1], sys.argv[2]
census = json.load(open(os.path.join(rd, 'dex/handler_census_legal.json')))['entities']
v1 = {}
for kind, fn in (('move', 'v1_moves.json'), ('ability', 'v1_abilities.json'), ('item', 'v1_items.json')):
    d = json.load(open(os.path.join(rd, 'dex', fn)))
    for e in d['entries']:
        v1[(kind, e['id'])] = e
cx = {(r['kind'], r['id']): r for r in json.load(open(os.path.join(rd, 'dex/out/complexity_v1.json')))}

NOTORIOUS = {
    'illusion': 'identity spoofing in obs/protocol', 'imposter': 'transformInto core', 'transform': 'transformInto core',
    'allyswitch': 'swapPosition + stall chance', 'instruct': 'queue insert', 'afteryou': 'prioritizeAction',
    'quash': 'order 201', 'encore': '[C] queue changeAction', 'revivalblessing': 'extra decision variant',
    'emergencyexit': 'mid-turn switch [C]', 'ejectbutton': 'mid-turn switch [C]', 'redcard': 'drag-in',
    'partingshot': 'pivot only if drop succeeded', 'uturn': 'pivot', 'voltswitch': 'pivot', 'flipturn': 'pivot',
    'batonpass': 'copy volatiles', 'shedtail': 'substitute + pivot', 'substitute': 'H12 interception',
    'magicbounce': 're-useMove reflection', 'trace': 'ability copy + random target', 'skillswap': 'ability swap',
    'disguise': 'forme change', 'stancechange': 'forme change', 'zerotohero': 'forme change on switch-out',
    'hungerswitch': 'forme change', 'unburden': 'item-consumption events', 'fakeout': '[C] mask after first action',
    'protect': 'stall counter', 'followme': 'redirection', 'ragepowder': 'redirection + powder',
    'lightningrod': 'redirection + absorb', 'trickroom': 'speed negation [C] + toggle', 'tailwind': 'side cond',
    'intimidate': 'TryBoost/AfterEachBoost web', 'megasol': '[C] user-only sun', 'eelevate': '[C] isGrounded + boost',
    'piercingdrill': '[C] protect bypass x0.25', 'unseenfist': '[C] protect bypass x0.25', 'firemane': '[C] disputed (ADR-019)',
    'dragonize': '[C] -ate', 'spicyspray': '[C]', 'auraguard': '[C] breakable TODO in PS', 'parentalbond': 'hit loop',
    'futuresight': 'slot condition', 'wish': 'slot condition', 'perishsong': 'counter', 'curse': '[C] targeting',
    'knockoff': '[C] AfterHit after faint', 'rapidspin': '[C] AfterHit after faint', 'sheerforce': '[C] suppression changes',
    'healer': '[C] 1/2', 'runaway': '[C] trap immunity', 'saltcure': '[C] 1/16', 'ragefist': '[C] reset on switch',
}

rows = []
for kind in ('move', 'ability', 'item', 'condition'):
    for eid, e in sorted(census.get(kind, {}).items()):
        v = v1.get((kind, eid))
        cls = e.get('class', '')
        if kind == 'condition':
            stage = 'dep'  # implemented when first dependent entity lands
        elif v is None:
            stage = 'v2'
        else:
            u = v.get('usage', {})
            maxp = max(u.get('bo3_1760', {}).get('p_team', 0), u.get('bo1_0', {}).get('p_team', 0),
                       u.get('replays_bo3_p_team', 0) or 0)
            stage = 'v0' if (cls != 'CUSTOM' or maxp >= 0.01 or v.get('inclusion') == 'zero_cost') else 'v1'
            if kind == 'species':
                stage = 'v0'
        c = cx.get((kind, eid), {})
        rows.append({
            'kind': kind.upper(), 'id': eid, 'name': e.get('name', eid), 'class': cls, 'stage': stage,
            'hooks': ' '.join(sorted(set(e.get('hooks', []) + ['cond.' + h for h in e.get('condition_hooks', [])]))),
            'n_fns': len(e.get('fns', [])), 'loc': e.get('loc', 0),
            'core_refs': len(e.get('core_refs', {}) or {}), 'complexity': c.get('score', ''),
            'p_team_bo3_1760': (v or {}).get('usage', {}).get('bo3_1760', {}).get('p_team', ''),
            'p_team_bo1_all': (v or {}).get('usage', {}).get('bo1_0', {}).get('p_team', ''),
            'notes': NOTORIOUS.get(eid, ''), 'status': 'todo', 'ps_commit': '', 'scenario_tests': '', 'fuzz_hits': '',
        })

ENGINE = [
    ('ENGINE_DELTA', 'stat_formula_sp', 'Champions stat formula B+SP+20 / HP B+SP+75, nature floor', 'v0'),
    ('ENGINE_DELTA', 'pp_formula', 'PP cap 20 then (pp/5+1)*4', 'v0'),
    ('ENGINE_DELTA', 'trickroom_negation', 'Trick Room speed key = -spe, no wrap', 'v0'),
    ('ENGINE_DELTA', 'par_1_8', 'Full paralysis 1/8', 'v0'),
    ('ENGINE_DELTA', 'sleep_233', 'Sleep startTime sample([2,3,3])', 'v0'),
    ('ENGINE_DELTA', 'freeze_cap3', 'Freeze 3-attempt counter, 1/4 thaw', 'v0'),
    ('ENGINE_DELTA', 'mega_persist_faint', 'Mega persists through faint', 'v0'),
    ('ENGINE_DELTA', 'no_tera', 'No Terastallization anywhere', 'v0'),
    ('ENGINE_DELTA', 'sheer_force_aftermove', 'Sheer Force does not suppress AfterMoveSecondary/EE', 'v1'),
    ('ENGINE_DELTA', 'afterhit_after_faint', 'onAfterHit runs even if attacker fainted', 'v0'),
    ('ENGINE_DELTA', 'eject_button_pivot', 'Eject Button does not cancel pivot; 1 per action', 'v0'),
    ('ENGINE_DELTA', 'ee_no_clear_flags', 'Emergency Exit does not clear other switch flags', 'v0'),
    ('ENGINE_DELTA', 'encore_requeue', 'Encore changeAction re-queue with own priority, random target', 'v0'),
    ('ENGINE_DELTA', 'fakeout_disable', 'Fake Out/First Impression unselectable after first action', 'v0'),
    ('ENGINE_DELTA', 'protect_bypass_quarter', 'Unseen Fist/Piercing Drill x0.25 through protection', 'v1'),
    ('ENGINE_DELTA', 'hp_display_rule', 'Opponent HP floor(100*hp/maxhp)||1 + colour bits (obs only)', 'v0'),
    ('SYSTEM', 'team_preview', 'Bring 4 of 6, ordered, leads', 'v0'),
    ('SYSTEM', 'queue_orders', 'Order classes, commit-time keys, fractional priority at commit', 'v0'),
    ('SYSTEM', 'dynamic_speed', 'Re-key all moves before each move (Gen 8+)', 'v0'),
    ('SYSTEM', 'switch_batching', 'Batched vs unbatched SwitchIn resolution', 'v0'),
    ('SYSTEM', 'midturn_switch', 'Pause/resume SWITCH phase for pivots/EE/Eject Button', 'v0'),
    ('SYSTEM', 'eot_replacement', 'End-of-turn replacements incl. hazard re-faint loops', 'v0'),
    ('SYSTEM', 'mega_evolution', 'Mega flow: order 104, ability Start, side lock, same-turn speed', 'v0'),
    ('SYSTEM', 'spread_targeting', 'Spread flag before filtering, ally-first order, retargeting', 'v0'),
    ('SYSTEM', 'redirection', 'Follow Me/Rage Powder/Lightning Rod, Stalwart/Snipe Shot', 'v0'),
    ('SYSTEM', 'residual_walk', 'fieldEvent Residual ordering + duration ticks + faint checks', 'v0'),
    ('SYSTEM', 'faint_win', 'resolveFaints, simultaneous wipe last-faint rule', 'v0'),
    ('SYSTEM', 'hook_framework', 'Holder enumeration, ordering, MUL/GATE/REPL folds, suppression', 'v0'),
    ('SYSTEM', 'damage_formula', 'Exact integer damage pipeline', 'v0'),
    ('SYSTEM', 'accuracy_evasion_ohko', 'Accuracy stages, evasion, OHKO rules (legal in Reg M-C)', 'v0'),
    ('SYSTEM', 'legal_masks', 'Exact PS-acceptance-equivalent masks + joint constraints', 'v0'),
    ('SYSTEM', 'turn_cap', 'max_turns + tiebreak; PS 1000-turn tie in conformance mode', 'v0'),
]
for kind, eid, name, stage in ENGINE:
    rows.append({'kind': kind, 'id': eid, 'name': name, 'class': 'SYSTEM', 'stage': stage, 'hooks': '', 'n_fns': '',
                 'loc': '', 'core_refs': '', 'complexity': '', 'p_team_bo3_1760': '', 'p_team_bo1_all': '',
                 'notes': '', 'status': 'todo', 'ps_commit': '', 'scenario_tests': '', 'fuzz_hits': ''})

with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
from collections import Counter
print(len(rows), Counter((r['kind'], r['stage']) for r in rows))
print(Counter((r['kind'], r['class']) for r in rows if r['stage'] in ('v0',)))

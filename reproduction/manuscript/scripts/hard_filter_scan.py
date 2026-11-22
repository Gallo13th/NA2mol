#!/usr/bin/env python3
"""Scan documented hard-filter combinations and report the quinoline scaffold's rank by repeat count."""
import csv, itertools
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

rows = []
for r in csv.DictReader(open('docs/clivia_8hze/cold10k_score.csv')):
    try:
        r.update(mw=float(r['mw']), logp=float(r['logp']), tpsa=float(r['tpsa']), rot=float(r['rot']),
                 sa=float(r['sa_score']), plan=float(r['pi_planarity']))
    except ValueError:
        continue
    r['n_arom'] = int(r['n_arom']); r['pi'] = int(r['pi_atoms'])
    r['da'] = int(r['da_flag']); r['don'] = int(r['has_donor']); r['acc'] = int(r['has_acceptor'])
    r['ok_chem'] = int(r['radicals']) == 0 and abs(int(r['charge'])) <= 1
    m = Chem.MolFromSmiles(r['smiles'])
    if m is None:
        continue
    r['scaffold'] = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    rows.append(r)
QUIN = 'c1ccc(-c2cc(-c3ccccc3)c3ccccc3n2)cc1'

FILTERS = {
    'pi>=12': lambda r: r['pi'] >= 12,
    'pi>=16': lambda r: r['pi'] >= 16,
    'pi>=18': lambda r: r['pi'] >= 18,
    'pi>=20': lambda r: r['pi'] >= 20,
    'pi>=22': lambda r: r['pi'] >= 22,
    'arom>=2': lambda r: r['n_arom'] >= 2,
    'arom==2..4': lambda r: 2 <= r['n_arom'] <= 4,
    'planar<=0.25': lambda r: r['plan'] <= 0.25,
    'planar<=0.20': lambda r: r['plan'] <= 0.20,
    'da==1': lambda r: r['da'] == 1,
    'donor+acceptor': lambda r: r['don'] and r['acc'],
    'mw 250-600': lambda r: 250 <= r['mw'] <= 600,
    'mw 280-450': lambda r: 280 <= r['mw'] <= 450,
    'mw 300-550': lambda r: 300 <= r['mw'] <= 550,
    'logp<=7': lambda r: r['logp'] <= 7.0,
    'logp 2-7': lambda r: 2.0 <= r['logp'] <= 7.0,
    'tpsa<=120': lambda r: r['tpsa'] <= 120,
    'rot<=9': lambda r: r['rot'] <= 9,
    'sa<=4': lambda r: r['sa'] <= 4.0,
    'sa<=3': lambda r: r['sa'] <= 3.0,
}

BASE = ['pi>=12', 'arom>=2', 'planar<=0.25', 'mw 250-600', 'tpsa<=120', 'rot<=9', 'sa<=4']
def run(keys):
    sub = [r for r in rows if r['ok_chem'] and all(FILTERS[k](r) for k in keys)]
    cnt = Counter(r['scaffold'] for r in sub if r['scaffold'] not in ('', 'C', 'c1ccccc1', 'C1CCCCC1'))
    if not cnt:
        return None
    ranked = cnt.most_common()
    rank = next((i for i, (s, _) in enumerate(ranked, 1) if s == QUIN), None)
    return len(sub), len(cnt), cnt.get(QUIN, 0), rank, ranked[:6]

print(f'base filters: {BASE}')
for k in BASE:
    sub = [r for r in rows if r['ok_chem'] and FILTERS[k](r)]
    n_q = sum(1 for r in sub if r['scaffold'] == QUIN)
    print(f'  alone {k:14s} -> {len(sub):5d} molecules, {n_q:2d} quinoline')

cands = []
for extra in ([], ['da==1'], ['donor+acceptor'], ['logp<=7'], ['logp 2-7'], ['logp<=7', 'da==1'],
              ['logp 2-7', 'da==1'], ['da==1', 'mw 280-450'], ['logp<=7', 'mw 280-450'],
              ['sa<=3'], ['sa<=3', 'da==1'], ['planar<=0.20'], ['pi>=16'], ['pi>=16', 'da==1'],
              ['pi>=16', 'logp<=7'], ['pi>=18'], ['pi>=18', 'da==1'], ['logp 2-7', 'donor+acceptor']):
    keys = BASE + extra
    res = run(keys)
    if res:
        n, ns, q, rank, top = res
        print(f'{"+".join(extra) or "base":28s} -> mol {n:5d} scaff {ns:4d} quinoline {q:2d} rank {rank}')

print('\n=== reference-anchored filters (crystal dyes have a 20-atom pi system) ===')
for extra in (['pi>=20'], ['pi>=20', 'sa<=4'], ['pi>=20', 'sa<=4', 'mw 250-600'],
              ['pi>=20', 'sa<=4', 'mw 250-600', 'tpsa<=120'],
              ['pi>=20', 'sa<=4', 'mw 250-600', 'tpsa<=120', 'rot<=9'],
              ['pi>=20', 'sa<=4', 'mw 250-600', 'tpsa<=120', 'rot<=9', 'logp<=7'],
              ['pi>=20', 'sa<=4', 'mw 250-600', 'tpsa<=120', 'rot<=9', 'logp<=7', 'da==1'],
              ['pi>=20', 'sa<=4', 'mw 300-550', 'tpsa<=120', 'rot<=9', 'logp<=7'],
              ['pi>=22'], ['pi>=22', 'sa<=4'], ['pi>=22', 'sa<=4', 'mw 250-600']):
    res = run(extra)
    if res:
        n, ns, q, rank, top = res
        print(f'{"+".join(extra):58s} -> mol {n:5d} scaff {ns:4d} quinoline {q:2d} rank {rank}')
        if rank and rank <= 3:
            for i, (s, c) in enumerate(top[:6], 1):
                marker = '  <== quinoline' if s == QUIN else ''
                print(f'      #{i} n={c:3d} {s}{marker}')

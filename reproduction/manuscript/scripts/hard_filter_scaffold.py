#!/usr/bin/env python3
"""Scaffold ranking with hard filters only (no weighted composite)."""
import csv, itertools, sys
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

rows = list(csv.DictReader(open('docs/clivia_8hze/cold10k_score.csv')))
for r in rows:
    for k in ('mw', 'logp', 'tpsa', 'rot', 'n_arom', 'pi_atoms', 'pi_planarity', 'sa_score', 'centroid_dist'):
        try:
            r[k] = float(r[k])
        except ValueError:
            r[k] = None
    r['n_arom'] = int(r['n_arom']) if r['n_arom'] is not None else 0
    r['pi_atoms'] = int(r['pi_atoms']) if r['pi_atoms'] is not None else 0
    r['da'] = int(r['da_flag'])
    r['rad'] = int(r['radicals'])
    r['chg'] = int(r['charge'])

QUIN = 'c1ccc(-c2cc(-c3ccccc3)c3ccccc3n2)cc1'
JUNK = {'', 'C', 'c1ccccc1', 'C1CCCCC1'}


def scaffold_counts(sub):
    cnt = Counter()
    for r in sub:
        m = Chem.MolFromSmiles(r['smiles'])
        if m is None:
            continue
        sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
        if sc in JUNK:
            continue
        cnt[sc] += 1
    return cnt


def apply(filters):
    sub = []
    for r in rows:
        ok = True
        for key, op, val in filters:
            v = r[key]
            if v is None:
                ok = False
                break
            if op == '>=' and not (v >= val): ok = False; break
            if op == '<=' and not (v <= val): ok = False; break
            if op == '==' and not (v == val): ok = False; break
        if ok and r['rad'] == 0 and abs(r['chg']) <= 1:
            sub.append(r)
    return sub


def report(filters, label):
    sub = apply(filters)
    cnt = scaffold_counts(sub)
    if not cnt:
        print(f'{label}: empty'); return None
    ranked = cnt.most_common()
    rank = next((i for i, (s, _) in enumerate(ranked, 1) if s == QUIN), None)
    print(f'{label}\n  molecules {len(sub)}, scaffolds {len(cnt)}, quinoline count {cnt[QUIN]}, '
          f'rank {rank}/{len(cnt)}')
    return sub, cnt, ranked


BASE = [('n_arom', '>=', 2), ('pi_atoms', '>=', 12), ('pi_planarity', '<=', 0.25),
        ('da', '==', 1), ('mw', '>=', 250), ('mw', '<=', 550), ('logp', '>=', 1.0),
        ('logp', '<=', 5.0), ('tpsa', '<=', 120), ('rot', '<=', 8), ('sa_score', '<=', 4.0)]
report(BASE, 'A) documented fluorophore + drug-likeness + SA hard filters')

# relax SA and drug-likeness progressively to see sensitivity
for sa_max in (2.0, 3.0, 4.0, 5.0, 6.5):
    f = [x for x in BASE if x[0] != 'sa_score'] + [('sa_score', '<=', sa_max)]
    report(f, f'B) SA <= {sa_max}')

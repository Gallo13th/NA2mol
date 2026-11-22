#!/usr/bin/env python3
"""Principled library: every substituent the model produced at least once at that site (no threshold)."""
import csv
import numpy as np

# counts from 4,948 scaffold-containing products; zero-count entries are excluded by construction
FREQ = {
 'C4': {'OH': 530, 'Cl': 485, 'Me': 352, 'OMe': 330, 'F': 63, 'NH2': 30, 'NHMe': 3},
 'C2': {'OMe': 1914, 'OH': 497, 'Cl': 124, 'Me': 109, 'OEt': 104, 'F': 9, 'NMe2': 6, 'NH2': 3},
 'C7': {'OMe': 382, 'OH': 217, 'Me': 141, 'Cl': 114, 'OEt': 33, 'CF3': 12, 'NH2': 9, 'F': 3},
}
sets = {k: set(v) for k, v in FREQ.items()}
print('observed-at-least-once sets:')
for k, v in sets.items():
    print(f'  {k}: {sorted(v)}')

rows = list(csv.DictReader(open('docs/clivia_8hze/unified_enum.csv')))
for r in rows:
    r['qed'] = float(r['qed']); r['sa'] = float(r['sa'])
    r['rdock_min'] = float(r['rdock_min']) if r['rdock_min'] not in ('', None) else None
lib = [r for r in rows if r['C4'] in sets['C4'] and r['C2'] in sets['C2'] and r['Q'] in sets['C7']]
print(f'\nlibrary: {len(lib)} molecules')
gated = [r for r in lib if r['qed'] >= 0.44 and r['sa'] <= 2.00 and r['rdock_min'] is not None]
print(f'after QED>=0.44 / SA<=2.00: {len(gated)}')


def pct(vals, low):
    idx = sorted(range(len(vals)), key=lambda i: vals[i] if low else -vals[i])
    out = [0.0] * len(vals)
    for rank, i in enumerate(idx):
        out[i] = 100.0 * rank / max(1, len(vals) - 1)
    return out


P = [pct([r['rdock_min'] for r in gated], True), pct([r['sa'] for r in gated], True)]
for i, r in enumerate(gated):
    r['agg'] = float(np.mean([P[0][i], P[1][i]]))
order = sorted(gated, key=lambda r: r['agg'])
for k, r in enumerate(order, 1):
    r['rank'] = k
REF = {'Al-1': ('OMe', 'OH', 'OMe'), 'Al-2': ('OMe', 'NH2', 'OMe')}
print('\nreference compounds:')
for name, combo in REF.items():
    r = next((x for x in gated if (x['C4'], x['C2'], x['Q']) == combo), None)
    print('  {:5s} {}'.format(name, 'rank {:3d}/{} ({:5.1f}%)  rDock_min {:7.2f}  SA {:.2f}  QED {:.3f}'.format(
        r['rank'], len(gated), 100 * r['rank'] / len(gated), r['rdock_min'], r['sa'], r['qed'])
        if r else 'NOT in library'))
print('\ntop 10:')
for r in order[:10]:
    print('  {:3d}. {}|{}|{}  rDock_min {:7.2f}  SA {:.2f}  QED {:.3f}'.format(
        r['rank'], r['C4'], r['C2'], r['Q'], r['rdock_min'], r['sa'], r['qed']))
fields = ['rank', 'agg', 'key', 'smiles', 'C4', 'C2', 'Q', 'qed', 'sa', 'rdock_min', 'mw', 'logp']
with open('docs/clivia_8hze/pruned_observed_ranked.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    for r in order:
        w.writerow(r)
print('\n[OK] wrote pruned_observed_ranked.csv')

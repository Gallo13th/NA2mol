#!/usr/bin/env python3
"""sp1 final summary: merge rnasmol + rdock into a single report.
"""
import os, sys, json
import numpy as np
from collections import defaultdict
os.chdir('/home/data/users/gaoletian/rna2smol')

def main():
    rna = json.load(open('casestudy/sp1/rnasmol_sp1_scores.json'))['results']
    dock = json.load(open('casestudy/sp1/rdock/rdock_m1_30.json'))['scored']
    # build name -> (ensemble_P, gate, INTER)
    rows = []
    for n, v in rna.items():
        if n.startswith('m1hit'):
            rows.append({'id': n, 'ensemble_P': v['ensemble_P'],
                         'gate': v['multi_gate(P>0.5)'],
                         'inter': None})
    for d in dock:
        for r in rows:
            if d['name'].replace('.', '_') == r['id'].replace('m1hit_', 'sp1runA_') or True:
                pass
    # simpler: merge by name
    name2dock = {d['name']: d['best_INTER'] for d in dock}
    out = []
    for r in rows:
        r['inter'] = name2dock.get(r['id'].replace('m1hit', ''), None)
        out.append(r)
    with open('casestudy/sp1/summary.json', 'w') as f:
        json.dump({'rows': out,
                   'rnasmol_ref': rna.get('m1_ref'),
                   'counts': {'rnasmol': len(out)}}, f, indent=1)
    print('[saved] casestudy/sp1/summary.json')

if __name__ == '__main__':
    main()

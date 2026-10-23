#!/usr/bin/env python3
"""Aggregate analysis of the 10,000-per-pocket free-generation runs.

Combines the two seeds per pocket and reports, against mol1-mol5:
  - ECFP4 distribution
  - quinoline / 2,4-diaryl-quinoline retention (generic definition: quinoline core
    with an aromatic ring on both C2 and C4)
  - co-occurrence of the target substituent pattern (>=2 methoxy + >=1 hydroxyl)
  - the closest molecules, written to CSV
"""
import argparse
import csv
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
QUIN = Chem.MolFromSmarts('c1ccc2ncccc2c1')
OME = Chem.MolFromSmarts('[OX2][CH3]')
OH = Chem.MolFromSmarts('[OX2H]')

TARGETS = {
    'mol1': ('C23H19NO3', 357.41, 'COc1ccc(-c2cc(-c3ccc(O)cc3)nc3ccc(OC)cc23)cc1'),
    'mol2': ('C23H20N2O2', 356.43, 'COc1ccc(-c2cc(-c3ccc(N)cc3)nc3ccc(OC)cc23)cc1'),
    'mol3': ('C25H24N2O2', 384.48, 'COc1ccc(-c2cc(-c3ccc(N(C)C)cc3)nc3ccc(OC)cc23)cc1'),
    'mol4': ('C22H18N2O2', 342.40, 'COc1ccc(-c2cc(-c3ccncc3)nc3ccc(OC)cc23)cc1'),
    'mol5': ('C23H19NO4', 373.41,
             'COc1ccc2c(c1)C(-c1ccc(O)cc1)C=C(-c1cccc(C(=O)O)c1)N2'),
}
REF = {k: GEN.GetFingerprint(Chem.MolFromSmiles(v[2])) for k, v in TARGETS.items()}


def diaryl_positions(m):
    """Positions among quinoline C2/C4 that carry an aromatic substituent."""
    out = set()
    for mt in m.GetSubstructMatches(QUIN):
        for lab, i in (('C2', 5), ('C4', 7)):
            a = m.GetAtomWithIdx(mt[i])
            for nb in a.GetNeighbors():
                if nb.GetIsAromatic() and nb.GetIdx() not in mt and nb.GetSymbol() in 'CN':
                    out.add(lab)
    return out


RUNS = {
    'clivia': ['output/clivia_10k_s1/clivia_8HZE_dummy/clivia_8HZE_dummy.sdf',
               'output/clivia_10k_s2/clivia_8HZE_dummy/clivia_8HZE_dummy.sdf'],
    'squash': ['output/squash_10k_s1/squash_7KVU_dummy/squash_7KVU_dummy.sdf',
               'output/squash_10k_s2/squash_7KVU_dummy/squash_7KVU_dummy.sdf'],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='case_study/clivia_squash/reports/ten_k_summary.csv')
    a = ap.parse_args()

    for pocket, paths in RUNS.items():
        n = quin = dd = both = 0
        sims_all = []
        rows = []
        for path in paths:
            p = Path(path)
            if not p.is_file():
                print(f'missing {p}')
                continue
            for m in Chem.SDMolSupplier(str(p)):
                if m is None:
                    continue
                n += 1
                if m.HasSubstructMatch(QUIN):
                    quin += 1
                pos = diaryl_positions(m)
                if pos == {'C2', 'C4'}:
                    dd += 1
                n_ome = len(m.GetSubstructMatches(OME))
                n_oh = len(m.GetSubstructMatches(OH))
                if n_ome >= 2 and n_oh >= 1:
                    both += 1
                fp = GEN.GetFingerprint(m)
                s = {k: DataStructs.TanimotoSimilarity(fp, v) for k, v in REF.items()}
                b = max(s, key=s.get)
                sims_all.append(s[b])
                if s[b] >= 0.40:
                    rows.append(dict(pocket=pocket, smiles=Chem.MolToSmiles(m),
                                     heavy=m.GetNumHeavyAtoms(),
                                     mw=round(Descriptors.MolWt(m), 2),
                                     best_target=b, ecfp4=round(s[b], 4),
                                     n_ome=n_ome, n_oh=n_oh,
                                     diaryl=bool(pos == {'C2', 'C4'}),
                                     quinoline=m.HasSubstructMatch(QUIN)))

        sims_all.sort(reverse=True)
        print(f'\n===== {pocket}: {n} molecules (10k run) =====')
        print(f'  max ECFP4            {sims_all[0]:.3f}')
        for q in (0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4):
            c = sum(1 for x in sims_all if x >= q)
            print(f'  >= {q:.2f}: {c:6d}  ({100*c/n:5.2f}%)')
        print(f'  quinoline core       {quin:6d}  ({100*quin/n:5.2f}%)')
        print(f'  2,4-diaryl quinoline {dd:6d}  ({100*dd/n:5.2f}%)')
        print(f'  >=2 OMe + >=1 OH     {both:6d}  ({100*both/n:5.2f}%)')

        rows.sort(key=lambda r: -r['ecfp4'])
        # the two categories that matter: right scaffold vs right substituents
        print(f'  best with 2,4-diaryl scaffold:')
        for r in [x for x in rows if x['diaryl']][:5]:
            print(f"    {r['ecfp4']:.3f} {r['best_target']} hv={r['heavy']:2d} "
                  f"OMe={r['n_ome']} OH={r['n_oh']}  {r['smiles']}")
        print(f'  best with >=2 OMe + >=1 OH (target substituents):')
        for r in [x for x in rows if x['n_ome'] >= 2 and x['n_oh'] >= 1][:5]:
            print(f"    {r['ecfp4']:.3f} {r['best_target']} hv={r['heavy']:2d} "
                  f"OMe={r['n_ome']} OH={r['n_oh']} diaryl={r['diaryl']}  {r['smiles']}")
        print(f'  overall top 5:')
        for r in rows[:5]:
            print(f"    {r['ecfp4']:.3f} {r['best_target']} hv={r['heavy']:2d} "
                  f"OMe={r['n_ome']} OH={r['n_oh']} diaryl={r['diaryl']}  {r['smiles']}")

        outp = Path(a.out).with_name(f"ten_k_{pocket}.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with outp.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
            print(f'  wrote {outp} ({len(rows)} rows with ECFP4 >= 0.40)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

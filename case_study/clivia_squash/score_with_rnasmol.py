#!/usr/bin/env python3
"""Score generated and reference molecules with RNAsmol (github.com/hongli-ma/RNAsmol).

RNAsmol is an RNA-ligand interaction classifier: it takes a ligand SMILES and the RNA
sequence, builds a ligand graph plus a sequence encoding, and outputs the probability that
the pair is a binder. It does not use a 3D pose, so the generated molecules and the reference
compounds are scored on exactly the same footing - which makes it a useful independent check
on the smina docking numbers.

Inputs are assembled here:
  sequence  the full RNA chain from the deposited PDB (not just the pocket residues)
  ligand    the repaired SMILES of each molecule

All six supplied checkpoints are run and their probabilities averaged, with the individual
values kept.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/data/users/gaoletian/rna2smol')
HERE = ROOT / 'case_study/clivia_squash'
RNASMOL = Path('/home/data/users/gaoletian/RNAsmol')
sys.path.insert(0, str(RNASMOL / 'rnasmol'))
os.chdir(RNASMOL)

import torch                                                       # noqa: E402
from torch_geometric.data import DataLoader                        # noqa: E402
from rdkit import Chem, DataStructs, RDLogger                      # noqa: E402
from rdkit.Chem import rdFingerprintGenerator                      # noqa: E402
from preprocessing_demo import GNNDataset, mol_to_graph            # noqa: E402
from model import MCNN_GCN                                         # noqa: E402

RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

REFERENCES = {
    'mol1': 'COc1ccc(-c2cc(-c3ccc(O)cc3)nc3ccc(OC)cc23)cc1',
    'mol2': 'COc1ccc(-c2cc(-c3ccc(N)cc3)nc3ccc(OC)cc23)cc1',
    'mol3': 'COc1ccc(-c2cc(-c3ccc(N(C)C)cc3)nc3ccc(OC)cc23)cc1',
    'mol4': 'COc1ccc(-c2cc(-c3ccncc3)nc3ccc(OC)cc23)cc1',
    'mol5': 'COc1ccc2c(c1)C(-c1ccc(O)cc1)C=C(-c1cccc(C(=O)O)c1)N2',
}
PDB_FOR_SEQ = {'squash': HERE / 'pdb_all' / '7KVU.pdb',
               'clivia': HERE / 'pdb_all' / '8HZE.pdb'}
CHAIN = {'squash': 'G', 'clivia': 'A'}
CHECKPOINTS = ['pdb_bothaug_netperturbation.pt', 'pdb_bothaug_molperturbation.pt',
               'pdb_bothaug_rnaperturbation.pt', 'robin_bothaug_netperturbation.pt',
               'robin_bothaug_molperturbation.pt', 'robin_bothaug_rnaperturbation.pt']
CODE = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U'}


def rna_sequence(pdb, chain):
    """Full RNA sequence of one chain, in residue order."""
    seen = {}
    for line in Path(pdb).read_text().splitlines():
        if not line.startswith('ATOM'):
            continue
        if line[21].strip() != chain:
            continue
        res = line[17:20].strip()
        if res not in CODE:
            continue
        seen[int(line[22:26])] = CODE[res]
    return ''.join(seen[k] for k in sorted(seen))


def load_generated():
    gen = {}
    for f in sorted(HERE.glob('output_traj/st*/molecules.sdf')):
        if f.stat().st_size == 0:
            continue
        for m in Chem.SDMolSupplier(str(f), sanitize=False):
            if m is None:
                continue
            try:
                Chem.SanitizeMol(m)
            except Exception:
                continue
            smi = Chem.MolToSmiles(m)
            gen.setdefault(smi, None)
    return list(gen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pocket', choices=list(PDB_FOR_SEQ), default='squash')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=str(HERE / 'reports' / 'rnasmol_scores.csv'))
    a = ap.parse_args()

    seq = rna_sequence(PDB_FOR_SEQ[a.pocket], CHAIN[a.pocket])
    print(f'{a.pocket} RNA sequence: length {len(seq)}')
    print(f'  {seq[:80]}{"..." if len(seq) > 80 else ""}')

    gen = load_generated()
    if a.limit:
        gen = gen[:a.limit]
    print(f'generated molecules: {len(gen)}   references: {len(REFERENCES)}')

    # one CSV for everything; label 0 is a placeholder (the model predicts it)
    rows = ([dict(compound_iso_smiles=s, target_sequence=seq, affinity=1, tag=f'ref:{k}')
             for k, s in REFERENCES.items()] +
            [dict(compound_iso_smiles=s, target_sequence=seq, affinity=1, tag='generated')
             for s in gen])
    df = pd.DataFrame(rows)
    # PyG InMemoryDataset expects <root>/raw/data_test.csv and writes <root>/processed/
    import shutil
    work = HERE / 'reports' / f'rnasmol_input_{a.pocket}'
    if work.exists():
        shutil.rmtree(work)
    (work / 'raw').mkdir(parents=True, exist_ok=True)
    df[['compound_iso_smiles', 'target_sequence', 'affinity']].to_csv(
        work / 'raw' / 'data_test.csv', index=False)

    ds = GNNDataset(str(work), types='test')
    print(f'graphs built: {len(ds)} of {len(df)}')
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    # torch 1.7.1 in the rnasmol env predates sm_89, so an RTX 4090 cannot be used;
    # the model is small enough that CPU inference over ~1.2k molecules is fine.
    device = torch.device('cpu')
    print(f'device: {device} (torch 1.7.1 cannot target sm_89)')
    probs = {}
    for ck in CHECKPOINTS:
        path = RNASMOL / 'saved_model' / ck
        if not path.is_file():
            print(f'  missing {ck}')
            continue
        model = MCNN_GCN(3, 26, embedding_size=96, filter_num=32, out_dim=2, ban_heads=2).to(device)
        model.load_state_dict(torch.load(str(path), map_location=device))
        model.eval()
        out = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                _, _, _, pred = model(data)
                out.append(torch.softmax(pred, dim=1)[:, 1].cpu().numpy())
        probs[ck] = np.concatenate(out)
        print(f'  {ck}: mean binder prob {probs[ck].mean():.3f}', flush=True)

    n = len(next(iter(probs.values())))
    used = list(probs)
    keep = df.iloc[:n].copy()          # GNNDataset may drop unparsable ligands at the head
    keep['prob'] = np.mean([probs[c] for c in used], axis=0)
    for c in used:
        keep[c.replace('.pt', '')] = probs[c]

    ref_fps = {k: GEN.GetFingerprint(Chem.MolFromSmiles(v)) for k, v in REFERENCES.items()}
    keep['ecfp4_to_ref'] = [
        round(max(DataStructs.TanimotoSimilarity(GEN.GetFingerprint(Chem.MolFromSmiles(s)), v)
                  for v in ref_fps.values()), 4) for s in keep.compound_iso_smiles]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(out, index=False)
    print(f'\nwrote {out} ({len(keep)} rows, {len(used)} checkpoints averaged)')

    ref = keep[keep.tag.str.startswith('ref:')]
    g = keep[keep.tag == 'generated']
    print(f'\n=== {a.pocket}: RNAsmol binder probability ===')
    print('  references:')
    for _, r in ref.iterrows():
        print(f'    {r.tag[4:]:5s} {r.prob:.3f}')
    if len(g):
        print(f'  generated: n={len(g)}  max {g.prob.max():.3f}  '
              f'median {g.prob.median():.3f}  mean {g.prob.mean():.3f}')
        if len(ref):
            print(f'    above the best reference ({ref.prob.max():.3f}): '
                  f'{int((g.prob > ref.prob.max()).sum())}/{len(g)}')
            print(f'    above the mean reference ({ref.prob.mean():.3f}): '
                  f'{int((g.prob > ref.prob.mean()).sum())}/{len(g)}')
        print('  top 10 generated:')
        for _, t in g.nlargest(10, 'prob').iterrows():
            print(f'    {t.prob:.3f}  ECFP4 {t.ecfp4_to_ref:.3f}  {t.compound_iso_smiles[:58]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""sp1 final scoring: rnasmol 6-model ensemble for m1 (and m2/m3 path-prepared).

Sequence: 20-nt 'AGGUGAGCUUAGGGGAAUAA' (extracted from sp1 pocket PDB).
Scores 3 groups:
  - m1 exact hits (up to 200 representatives from gen_runA)
  - reference = m1 minimized conformer (self-score sanity)
  - (optional) top Tc >=0.5 flavone-like hits from runA for comparison
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
PROJ = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJ)
sys.path.insert(0, PROJ)
FMN_DIR = os.path.join(PROJ, 'casestudy/fmn')
RNASMOL_ROOT = os.path.join(FMN_DIR, 'RNAsmol')
RNASMOL_PKG = os.path.join(RNASMOL_ROOT, 'rnasmol')
for p in (RNASMOL_PKG, RNASMOL_ROOT, FMN_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rnasmol_score import build_data_from_mol, load_model
from rnasmol_score_gpu import score_data

SP1_SEQ = 'AGGUGAGCUUAGGGGAAUAA'
MODELS = [
    ('pdb_mol', 'pdb_bothaug_molperturbation.pt'),
    ('pdb_net', 'pdb_bothaug_netperturbation.pt'),
    ('pdb_rna', 'pdb_bothaug_rnaperturbation.pt'),
    ('rob_mol', 'robin_bothaug_molperturbation.pt'),
    ('rob_net', 'robin_bothaug_netperturbation.pt'),
    ('rob_rna', 'robin_bothaug_rnaperturbation.pt'),
]

device = torch.device('cuda')

def draw_mol(name, path_or_smi):
    if path_or_smi.endswith('.sdf'):
        m = next(Chem.SDMolSupplier(path_or_smi, removeHs=True, sanitize=True), None)
    else:
        m = Chem.MolFromSmiles(path_or_smi)
    assert m is not None, path_or_smi
    print(f'{name}: {Chem.MolToSmiles(m)}')
    return m

def main():
    # loads m1 exact hits from gen_runA results.json (first 200)
    m1_smi = 'O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12'
    recs = json.load(open('casestudy/sp1/gen_runA/results.json'))['results']
    m1hits = [r for r in recs if r['exact']]
    print(f'[sp1] {len(m1hits)} exact m1 hits in runA')

    mols = {}
    mols['m1_ref'] = draw_mol('m1_ref', m1_smi)
    for i, r in enumerate(m1hits[:200]):
        m = Chem.MolFromSmiles(r['smiles'])
        if m is not None:
            m.SetProp('_Name', r['name'])
            mols[f'm1hit_{i:03d}'] = m
    top_tc = sorted(recs, key=lambda x: -max(x['tc'].values()))[:60]
    for i, r in enumerate(top_tc):
        m = Chem.MolFromSmiles(r['smiles'])
        if m is not None:
            m.SetProp('_Name', r['name'])
            mols[f'top_{i:03d}'] = m
    print(f'[sp1] scored mols: {len(mols)}')

    graphs = {n: build_data_from_mol(m, SP1_SEQ) for n, m in mols.items()}
    results = {n: {} for n in mols}
    synced = False
    for key, fname in MODELS:
        path = os.path.join(RNASMOL_ROOT, 'saved_model', fname)
        model = load_model(path, device)
        for n in mols:
            p, pred, logits = score_data(model, graphs[n], device)
            results[n][key] = {'P_binder': p, 'logit_diff': float(logits[1] - logits[0])}
        torch.cuda.synchronize()
        del model
        torch.cuda.empty_cache()
        print(f'  [{key}] done', flush=True)
    summary = {}
    for n in mols:
        ps = [results[n][k]['P_binder'] for k, _ in MODELS]
        summary[n] = {'per_model': results[n],
                      'ensemble_P': float(np.mean(ps)),
                      'multi_gate(P>0.5)': int(sum(1 for p in ps if p > 0.5))}
        if n.startswith('m1'):
            print(f"[{n}] ensemble_P={summary[n]['ensemble_P']:.4f} gate={summary[n]['multi_gate(P>0.5)']}/6")
    json.dump({'rna_seq': SP1_SEQ, 'results': summary},
              open('casestudy/sp1/rnasmol_sp1_scores.json', 'w'), indent=2)
    print('[saved] casestudy/sp1/rnasmol_sp1_scores.json')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""RNAsmol 6-model ensemble for the 128 blind-enumerated variants + report
with rDock; combined ranking vs references (berberine, real m1/m2/m3)."""
import os, sys, json
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
FMN = os.path.join(os.path.abspath('.'), 'casestudy/fmn')
ROOT = os.path.join(FMN, 'RNAsmol')
for p in (os.path.join(ROOT, 'rnasmol'), ROOT, FMN):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
from rnasmol_score import build_data_from_mol, load_model
from rnasmol_score_gpu import score_data

OUT = 'casestudy/sp1/m1_route/blind_enum'
SEQ = 'AGGUGAGCUUAGGGGAAUAA'
MODELS = [('pdb_mol', 'pdb_bothaug_molperturbation.pt'),
          ('pdb_net', 'pdb_bothaug_netperturbation.pt'),
          ('pdb_rna', 'pdb_bothaug_rnaperturbation.pt'),
          ('rob_mol', 'robin_bothaug_molperturbation.pt'),
          ('rob_net', 'robin_bothaug_netperturbation.pt'),
          ('rob_rna', 'robin_bothaug_rnaperturbation.pt')]
device = torch.device('cuda')

mols = {}
for m in Chem.SDMolSupplier(f'{OUT}/enumerated.sdf', removeHs=True):
    if m is not None:
        mols[m.GetProp('_Name')] = m
refs = {
    'berberine_ref': 'casestudy/kras_g4/t2_denovo/phaseG_elem_scan/berberine_ref.sdf',
    'real_m1': 'casestudy/sp1/m1_route/mutg0_m1.sdf',
    'real_m2': 'casestudy/sp1/m1_route/mutg0_m2.sdf',
    'real_m3': 'casestudy/sp1/m1_route/mutg0_m3.sdf',
}
graphs = {}
for n, m in mols.items():
    graphs[n] = build_data_from_mol(m, SEQ)
for n, p in refs.items():
    graphs[f'ref_{n}'] = build_data_from_mol(next(Chem.SDMolSupplier(p, removeHs=True)), SEQ)

res = {k: {} for k in graphs}
for key, fname in MODELS:
    model = load_model(os.path.join(ROOT, 'saved_model', fname), device)
    for k in graphs:
        p, pred, logits = score_data(model, graphs[k], device)
        res[k][key] = {'P': round(float(p), 4), 'logit': float(logits[1] - logits[0])}
    del model
    torch.cuda.empty_cache()
    print(f'  [{key}] done', flush=True)

rd = json.load(open(f'{OUT}/rdock_scores.json'))
rows = []
for k in mols:
    ps = [res[k][m]['P'] for m, _ in MODELS]
    rows.append({'name': k, 'ensemble_P': round(float(np.mean(ps)), 4),
                 'gate': int(sum(1 for p in ps if p > 0.5)),
                 'logit': float(np.mean([res[k][m]['logit'] for m, _ in MODELS])),
                 'rdock': rd.get(k),
                 'smiles': Chem.MolToSmiles(mols[k])})
ref_rows = []
for n in refs:
    k = f'ref_{n}'
    ps = [res[k][m]['P'] for m, _ in MODELS]
    ref_rows.append({'name': n, 'ensemble_P': round(float(np.mean(ps)), 4),
                     'gate': int(sum(1 for p in ps if p > 0.5)),
                     'logit': float(np.mean([res[k][m]['logit'] for m, _ in MODELS])),
                     'rdock': None,
                     'smiles': Chem.MolToSmiles(next(Chem.SDMolSupplier(refs[n], removeHs=True)))})

# combined score
def minmax(xs):
    xs = np.array(xs, dtype=float)
    ok = np.isfinite(xs)
    if ok.sum() == 0 or xs[ok].min() == xs[ok].max():
        return np.where(ok, 0.5, np.nan)
    return (xs - xs[ok].min()) / (xs[ok].max() - xs[ok].min())
all_logit = [r['logit'] for r in rows] + [r['logit'] for r in ref_rows]
all_rd = [(-(r['rdock'] if r['rdock'] is not None else -1e9)) for r in rows] + [-1e9] * len(ref_rows)
lm = minmax(all_logit); rm = minmax(all_rd)
for i, r in enumerate(rows):
    r['combined'] = round(0.5 * float(lm[i]) + 0.5 * float(rm[i]), 4)
for j, r in enumerate(ref_rows):
    r['combined'] = round(0.5 * float(lm[-len(ref_rows) + j]) + 0.5 * 0.0, 4)
json.dump({'variants': rows, 'references': ref_rows},
          open(f'{OUT}/ranking.json', 'w'), indent=2)
print('\n=== 排名 Top15 (盲枚举, 分数驱动) ===')
for r in sorted(rows, key=lambda x: -(x['combined'] or 0))[:15]:
    print(f"  {r['name']} comb={r['combined']:.3f} rnA={r['ensemble_P']:.3f}({r['gate']}/6) "
          f"rdock={r['rdock']} | {r['smiles'][:58]}")
print('\n=== 参照 ===')
for r in ref_rows:
    print(f"  {r['name']}: P={r['ensemble_P']}({r['gate']}/6)")
print('[saved] ranking.json')

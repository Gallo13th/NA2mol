#!/usr/bin/env python3
"""t2-DENOVO Phase E3: 6-version RNAsmol ensemble for the de-novo-path t2 product.

Ligand = phaseE_bridge/t2_replaced.sdf (N->C bridge replacement product of the
D1 model product). References for context: t2_repro final model + berberine.
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

KRAS_G4_SEQ = 'GAGGGCGGTGTGGGAATAGGGAA'
MODELS = [
    ('pdb_mol', 'pdb_bothaug_molperturbation.pt'),
    ('pdb_net', 'pdb_bothaug_netperturbation.pt'),
    ('pdb_rna', 'pdb_bothaug_rnaperturbation.pt'),
    ('rob_mol', 'robin_bothaug_molperturbation.pt'),
    ('rob_net', 'robin_bothaug_netperturbation.pt'),
    ('rob_rna', 'robin_bothaug_rnaperturbation.pt'),
]
LIGANDS = [
    ('t2_denovo', 'casestudy/kras_g4/t2_denovo/phaseE_bridge/t2_replaced.sdf'),
    ('t2_final_ref', 'casestudy/kras_g4/t2_repro/final_t2_model.sdf'),
    ('berberine_ref', 'casestudy/kras_g4/_archive_ber101_era/BER101_berberine.sdf'),
]
device = torch.device('cuda')
mols = {}
for name, path in LIGANDS:
    m = next(Chem.SDMolSupplier(path, removeHs=True, sanitize=True), None)
    assert m is not None, path
    mols[name] = m
    print(f'{name}: {Chem.MolToSmiles(m)} (charge {Chem.GetFormalCharge(m)})')
graphs = {n: build_data_from_mol(m, KRAS_G4_SEQ) for n, m in mols.items()}
results = {n: {} for n in mols}
for key, fname in MODELS:
    path = os.path.join(RNASMOL_ROOT, 'saved_model', fname)
    model = load_model(path, device)
    _ = score_data(model, graphs['t2_denovo'], device)
    torch.cuda.synchronize()
    for n in mols:
        p, pred, logits = score_data(model, graphs[n], device)
        results[n][key] = {'P_binder': p, 'logit_diff': logits[1] - logits[0]}
    del model
    torch.cuda.empty_cache()
    print(f'  [{key}] ' + '  '.join(f"{n}: P={results[n][key]['P_binder']:.4f}" for n in mols))
summary = {}
for n in mols:
    ps = [results[n][k]['P_binder'] for k, _ in MODELS]
    summary[n] = {'per_model': results[n], 'ensemble_P': float(np.mean(ps)),
                  'multi_gate(P>0.5)': int(sum(1 for p in ps if p > 0.5))}
    print(f'[{n}] ensemble_P={summary[n]["ensemble_P"]:.4f} gate={summary[n]["multi_gate(P>0.5)"]}/6')
json.dump({'rna_seq': KRAS_G4_SEQ, 'results': summary},
          open('casestudy/kras_g4/t2_denovo/phaseE_bridge/rnasmol_t2_denovo.json', 'w'),
          indent=2)
print('[saved] rnasmol_t2_denovo.json')

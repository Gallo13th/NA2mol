#!/usr/bin/env python3
"""RNAsmol ensemble scoring for the sp1 m1-route mutation ladder."""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
PROJ = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJ); sys.path.insert(0, PROJ)
FMN = os.path.join(PROJ, 'casestudy/fmn')
ROOT = os.path.join(FMN, 'RNAsmol')
for p in (os.path.join(ROOT, 'rnasmol'), ROOT, FMN):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np, torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rnasmol_score import build_data_from_mol, load_model
from rnasmol_score_gpu import score_data

SEQ = 'AGGUGAGCUUAGGGGAAUAA'
MODELS = [('pdb_mol', 'pdb_bothaug_molperturbation.pt'),
          ('pdb_net', 'pdb_bothaug_netperturbation.pt'),
          ('pdb_rna', 'pdb_bothaug_rnaperturbation.pt'),
          ('rob_mol', 'robin_bothaug_molperturbation.pt'),
          ('rob_net', 'robin_bothaug_netperturbation.pt'),
          ('rob_rna', 'robin_bothaug_rnaperturbation.pt')]
device = torch.device('cuda')
tags = ['mut_seed_sat', 'mut_ox', 'mut_ox_dehyd', 'mut_ox_dehyd_3OH',
        'mut_m1', 'mut_m2', 'mut_m3']
mols = {t: next(Chem.SDMolSupplier(f'casestudy/sp1/m1_route/{t}.sdf', removeHs=True))
        for t in tags}
graphs = {t: build_data_from_mol(m, SEQ) for t, m in mols.items()}
results = {t: {} for t in tags}
for key, fname in MODELS:
    model = load_model(os.path.join(ROOT, 'saved_model', fname), device)
    for t in tags:
        p, pred, logits = score_data(model, graphs[t], device)
        results[t][key] = round(float(p), 4)
    del model
    torch.cuda.empty_cache()
rd = json.load(open('/tmp/opencode/nr_rdock.json'))
out = {}
print(f"{'mutant':<20}{'ensemble P':<11}{'gate':<7}{'rDock INTER'}")
for t in tags:
    ps = list(results[t].values())
    out[t] = {'per_model': results[t], 'ensemble_P': round(float(np.mean(ps)), 4),
              'gate': int(sum(1 for p in ps if p > 0.5)), 'rdock_INTER': rd[t]}
    print(f"{t:<20}{out[t]['ensemble_P']:<11.4f}{out[t]['gate']}/6{'   ':<3}{rd[t]}")
json.dump(out, open('casestudy/sp1/m1_route/mutation_scores.json', 'w'), indent=2)
print('[saved] mutation_scores.json')

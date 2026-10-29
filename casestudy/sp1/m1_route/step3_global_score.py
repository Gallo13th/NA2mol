#!/usr/bin/env python3
"""Uniform dual scoring of all global-scan candidates (g0/g1/g2 ladders):
rDock (-n 10 -s 42, sp1 system) + RNAsmol ensemble -> ranking + save."""
import os, sys, json, subprocess
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

G = 'casestudy/sp1/m1_route/global_scan'
RBT = os.environ.get('RBT_ROOT', '/home/data/users/gaoletian/.conda/envs/rna2smol/share/rdock-2013.1-1')
tags = [f'mutg{k}_{x}' for k in ['0', '1', '2'] for x in
        ['seed_sat', 'ox', 'ox_dehyd', 'ox_dehyd_3OH', 'm1', 'm2', 'm3']]

# rDock
rd = {}
for t in tags:
    path = f'casestudy/sp1/m1_route/{t}.sdf'
    if not os.path.exists(path):
        continue
    subprocess.run(['rbdock', '-r', 'casestudy/sp1/rdock/sp1_system.prm', '-i', path,
                    '-o', f'/tmp/opencode/gs_{t}', '-p', f'{RBT}/data/scripts/dock_solv.prm',
                    '-n', '10', '-s', '42'], capture_output=True)
    rows = []
    sd = f'/tmp/opencode/gs_{t}.sd'
    if os.path.exists(sd):
        for m in Chem.SDMolSupplier(sd, removeHs=False, sanitize=False):
            if m is not None and m.HasProp('SCORE.INTER'):
                rows.append(float(m.GetProp('SCORE.INTER')))
    rd[t] = round(min(rows), 2) if rows else None
    print(f'  rdock {t}: {rd[t]}', flush=True)
json.dump(rd, open(f'{G}/rdock_all.json', 'w'), indent=2)

# RNAsmol
FMN = os.path.join(os.path.abspath('.'), 'casestudy/fmn')
ROOT = os.path.join(FMN, 'RNAsmol')
for p in (os.path.join(ROOT, 'rnasmol'), ROOT, FMN):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
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
mols = {t: next(Chem.SDMolSupplier(f'casestudy/sp1/m1_route/{t}.sdf', removeHs=True))
        for t in tags if os.path.exists(f'casestudy/sp1/m1_route/{t}.sdf')}
graphs = {t: build_data_from_mol(m, SEQ) for t, m in mols.items()}
res = {t: {} for t in mols}
for key, fname in MODELS:
    model = load_model(os.path.join(ROOT, 'saved_model', fname), device)
    for t in mols:
        p, pred, logits = score_data(model, graphs[t], device)
        res[t][key] = round(float(p), 4)
    del model
    torch.cuda.empty_cache()
out = {}
print(f'\n{"molecule":<24}{"RNAsmol":<10}{"gate":<6}{"rDock"}')
for t in tags:
    if t not in mols:
        continue
    ps = list(res[t].values())
    out[t] = {'per_model': res[t], 'ensemble_P': round(float(np.mean(ps)), 4),
              'gate': int(sum(1 for p in ps if p > 0.5)), 'rdock_INTER': rd[t]}
    print(f'{t:<24}{out[t]["ensemble_P"]:<10.4f}{out[t]["gate"]}/6{"":<2}{rd[t]}')
json.dump(out, open(f'{G}/global_scores.json', 'w'), indent=2)
print('\n[saved] global_scores.json')

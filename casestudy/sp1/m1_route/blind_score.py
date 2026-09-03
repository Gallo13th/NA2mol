#!/usr/bin/env python3
"""Batch dual scoring of the 128 blind-enumerated variants:
rDock -n 5 (parallel), then RNAsmol 6-model ensemble. Rank by combined score
(0.5*minmax(negative rdock) + 0.5*minmax(rnasmol logit), references included)."""
import os, sys, json, subprocess, glob, shutil
import numpy as np
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
os.chdir('/home/data/users/gaoletian/rna2smol')
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

OUT = 'casestudy/sp1/m1_route/blind_enum'
RBT = os.environ.get('RBT_ROOT', '/home/data/users/gaoletian/.conda/envs/rna2smol/share/rdock-2013.1-1')
mols = list(Chem.SDMolSupplier(f'{OUT}/enumerated.sdf', removeHs=True))
names = [m.GetProp('_Name') for m in mols if m is not None]
print(f'[1/2] rDock {len(names)} molecules...', flush=True)

# parallel rbdock via background jobs on available devices
import concurrent.futures
def dock_job(the_in):
    name = the_in[0]; path = the_in[1]
    subprocess.run(['rbdock', '-r', 'casestudy/sp1/rdock/sp1_system.prm', '-i', path,
                    '-o', f'{OUT}/vd_{name}', '-p', f'{RBT}/data/scripts/dock_solv.prm',
                    '-n', '5', '-s', '42'], capture_output=True)
    sd = f'{OUT}/vd_{name}.sd'
    rows = []
    if os.path.exists(sd):
        for m in Chem.SDMolSupplier(sd, removeHs=False, sanitize=False):
            if m is not None and m.HasProp('SCORE.INTER'):
                rows.append(float(m.GetProp('SCORE.INTER')))
    return name, round(min(rows), 2) if rows else None

jobs = []
for i, m in enumerate(mols):
    if m is None: continue
    fn = f'{OUT}/vmol_{m.GetProp("_Name")}.sdf'
    w = Chem.SDWriter(fn); w.write(m); w.close()
    jobs.append((m.GetProp('_Name'), fn))
rd = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for name, score in ex.map(dock_job, jobs):
        rd[name] = score
    print(f'   done, scored {sum(1 for s in rd.values() if s is not None)}/{len(rd)}')
json.dump(rd, open(f'{OUT}/rdock_scores.json', 'w'), indent=2)
print('[saved] rdock_scores.json')

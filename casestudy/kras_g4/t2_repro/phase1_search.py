#!/usr/bin/env python3
"""t2 Phase1: isoelectronic seed retrieval from ber102_gen2k + library_10k."""
import json
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

T2_SMILES = 'c1ccc2sc(Cc3nc4ccccc4s3)nc2c1'
t2 = Chem.MolFromSmiles(T2_SMILES)
t2_scaf = MurckoScaffold.GetScaffoldForMol(t2)
print('t2 scaffold:', Chem.MolToSmiles(t2_scaf))

mfpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def fp(m): return mfpg.GetFingerprint(m)
def tani(a, b):
    return Chem.rdMolDescriptors.CalcNumAtomStereoCenters  # placeholder no-op

def tanimoto(m1, m2):
    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp(m1), fp(m2))

def load_lib(path, name):
    supp = Chem.SDMolSupplier(path, removeHs=True)
    out = []
    for i, m in enumerate(supp):
        if m is None:
            continue
        n = m.GetProp('_Name') if m.HasProp('_Name') else f'{name}_{i}'
        out.append((n, m))
    print(f'{name}: {len(out)} valid mols')
    return out

ber102 = load_lib('casestudy/kras_g4/ber102_gen2k/valid_mols.sdf', 'ber102')
lib10k = load_lib('casestudy/kras_g4/library_10k.sdf', 'lib10k')

# BM top4 check
print('\n=== BM top4 (from analysis.json) ===')
for name in ['ber102_r01_i032_s43', 'ber102_r03_i047_s45', 'ber102_r17_i017_s59', 'ber102_r19_i049_s61']:
    for n, m in ber102:
        if n == name:
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            print(f'{name}: {Chem.MolToSmiles(m)}')
            print(f'   scaffold: {Chem.MolToSmiles(scaf)}  scaf_tani_vs_t2scaf: {tanimoto(scaf, t2_scaf):.4f}')
            break

# structural searches
QUINAZ = Chem.MolFromSmarts('c1ccc2nc(-[CH2])nc2c1')  # not exact; do broader below
BENZIMID = Chem.MolFromSmarts('c1ccc2[nX2](-[CH2]-c3cccc4[nX2]ccc3-2)c[nX2]2')  # messy; use simpler

print('\n=== CH2-bridged bis-fused-N-heteroaromatics (isoelectronic skeletons) ===')
patt_bridge = Chem.MolFromSmarts('[cR]-[#6H2]-[cR]')
patt_fusedN = Chem.MolFromSmarts('c1ccc2ncc([nX2])c2c1')  # quinazoline-like
patt_anyfused = Chem.MolFromSmarts('c12ccccc1nc2')        # any fused 6-5 N ring on benzene

def is_iso_candidate(m):
    """benzofused diazine/imidazole (all-N analog of benzothiazole) with CH2 bridge, no S."""
    smi = Chem.MolToSmiles(m)
    if 'S' in smi or 'P' in smi or 'F' in smi or 'Cl' in smi or 'Br' in smi:
        return False
    if not m.HasSubstructMatch(patt_bridge):
        return False
    return True

cands = []
for n, m in ber102 + lib10k:
    if not is_iso_candidate(m):
        continue
    # both ends must be benzofused heteroaromatic with >=1 N in 5-ring
    matches = m.GetSubstructMatches(Chem.MolFromSmarts('[#6R2]-[#6H2]-[#6R2]'))
    t = tanimoto(m, t2)
    scaf_t = tanimoto(MurckoScaffold.GetScaffoldForMol(m), t2_scaf)
    cands.append((scaf_t, t, n, Chem.MolToSmiles(m)))

cands.sort(reverse=True)
print(f'total CH2-bridge no-halogen/no-S cands: {len(cands)}')
for scaf_t, t, n, smi in cands[:25]:
    print(f'  scafT={scaf_t:.3f} molT={t:.3f} {n}: {smi}')

# all-S containing mols in ber102 + their similarity
print('\n=== ber102 S-containing mols (top by molTanimoto) ===')
scands = []
for n, m in ber102:
    if any(a.GetSymbol() == 'S' for a in m.GetAtoms()):
        scands.append((tanimoto(m, t2), n, Chem.MolToSmiles(m)))
scands.sort(reverse=True)
print(f'total: {len(scands)}')
for t, n, smi in scands[:10]:
    print(f'  molT={t:.3f} {n}: {smi}')

# t2 exact match check in both libs
t2_canon = Chem.MolToSmiles(t2)
n_exact = sum(1 for n, m in ber102 + lib10k if Chem.MolToSmiles(m) == t2_canon)
print(f'\nexact t2 in libs: {n_exact}')

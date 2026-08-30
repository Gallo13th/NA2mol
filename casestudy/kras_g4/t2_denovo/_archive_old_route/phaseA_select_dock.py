#!/usr/bin/env python3
"""t2-DENONO Phase A1: isostere candidate selection + rDock redock into BER102.

Pure de novo path: candidates come ONLY from model/product libraries
(ber102_gen2k/valid_mols.sdf = model products; library_10k.sdf = docked library).
t2 is used ONLY as a retrieval-ranking criterion (BM-scaffold Tanimoto) —
no t2 topology/conformation/fragment enters any seed or coordinate produced here.

Candidate classes (generic structural queries, no t2 SMARTS):
  1. benzimidazole-type fused 6-5 diaza (pyrrolic-N site is the N->S target)
  2. quinazoline / phenanthroline-type fused diaza (BM-Tanimoto top scaffolds)
  3. aryl-CH2-aryl bridged fused diaza (bridge chemistry prior)

For each candidate: polar-H-only SD -> rbdock (BER102 R6 validated protocol,
-n 10, seed 42) -> rank poses by SCORE.INTER -> keep best pose coordinates
(positionally read, pasted onto chemically-correct heavy-atom copy).

Output: phaseA_redock/<key>_best.sdf + phaseA_redock/summary.json
"""
import os, sys, json, subprocess, argparse

PROJ = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJ)
sys.path.insert(0, PROJ)

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')
import numpy as np

RBT_ROOT = '/home/data/users/gaoletian/.conda/envs/rna2smol/share/rdock-2013.1-1'
SYSTEM_PRM = os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_full_r6_system.prm')
PROTOCOL = os.path.join(RBT_ROOT, 'data/scripts/dock_solv.prm')
OUTDIR = os.path.join(PROJ, 'casestudy/kras_g4/t2_denovo/phaseA_redock')
os.makedirs(OUTDIR, exist_ok=True)

T2 = 'c1ccc2sc(Cc3nc4ccccc4s3)nc2c1'   # retrieval criterion ONLY (ranking/reporting)
ALLOWED_Z = {1, 6, 7, 8, 9, 15, 16, 17}

BENZIMID = [Chem.MolFromSmarts('c1ccc2nc[nX2]c2c1'), Chem.MolFromSmarts('c1ccc2[nX2]cnc2c1')]
QUINAZ = Chem.MolFromSmarts('c1ccc2ncnc2c1')
PHEN = Chem.MolFromSmarts('c1ccc2nc3ccccc3nc2c1')
BRIDGE = Chem.MolFromSmarts('[#6R]-[#6H2]-[#6R]')

mfpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def bm_tanimoto(mol, ref_mol):
    try:
        s1 = MurckoScaffold.GetScaffoldForMol(mol)
        s2 = MurckoScaffold.GetScaffoldForMol(ref_mol)
        if s1.GetNumAtoms() == 0 or s2.GetNumAtoms() == 0:
            return 0.0
        return DataStructs.TanimotoSimilarity(mfpg.GetFingerprint(s1), mfpg.GetFingerprint(s2))
    except Exception:
        return 0.0


def polar_h_only(m):
    """all-H add with coords, then strip non-polar H (rDock input contract)."""
    m2 = Chem.AddHs(Chem.Mol(m), addCoords=True)
    rw = Chem.RWMol(m2)
    for a in list(rw.GetAtoms()):
        if a.GetSymbol() == 'H':
            heavy = [n for n in a.GetNeighbors() if n.GetSymbol() != 'H']
            if heavy and heavy[0].GetSymbol() == 'C':
                rw.RemoveAtom(a.GetIdx())
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def benzimid_sites(m, match):
    """match = 9 atom indices of benzimidazole core. Return (pyrrolic_N_idx, pyridinic_N_idx)."""
    syms = {i: m.GetAtomWithIdx(i).GetSymbol() for i in match}
    pyr, pyd = None, None
    for i in match:
        a = m.GetAtomWithIdx(i)
        if a.GetSymbol() != 'N':
            continue
        nbr_h = sum(1 for n in a.GetNeighbors() if n.GetSymbol() == 'H')
        deg = a.GetDegree()
        # pyrrolic [nH]: degree 2 in-ring, has H, no H-count via GetTotalNumHs
        if a.GetTotalNumHs() > 0 or nbr_h > 0:
            pyr = i
        elif deg == 2:
            pyd = i
    return pyr, pyd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-runs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-d-pocket', type=float, default=7.0,
                    help='core-centroid to BER-centroid cutoff for candidacy (A)')
    ap.add_argument('--cap', type=int, default=24)
    args = ap.parse_args()

    ref = next(Chem.SDMolSupplier('casestudy/kras_g4/ber102_rdock/ber102_ref.sdf', removeHs=True))
    BER = ref.GetConformer().GetPositions().mean(axis=0)
    t2m = Chem.MolFromSmiles(T2)

    sources = [('gen2k', 'casestudy/kras_g4/ber102_gen2k/valid_mols.sdf'),
               ('lib10k', 'casestudy/kras_g4/library_10k.sdf')]

    cands = {}
    seen = set()
    for src, path in sources:
        for i, m in enumerate(Chem.SDMolSupplier(path, removeHs=True)):
            if m is None or not m.GetNumConformers():
                continue
            if any(a.GetAtomicNum() not in ALLOWED_Z for a in m.GetAtoms()):
                continue
            smi = Chem.MolToSmiles(m)
            if smi in seen:
                continue
            entry = None
            for q in BENZIMID:
                for mt in m.GetSubstructMatches(q):
                    pyr, pyd = benzimid_sites(m, mt)
                    if pyr is None:
                        continue
                    c = m.GetConformer().GetPositions()[list(mt)].mean(axis=0)
                    d = float(np.linalg.norm(c - BER))
                    if d > args.max_d_pocket:
                        continue
                    has_bridge = bool(m.HasSubstructMatch(BRIDGE))
                    entry = dict(cls='benzimidazole', key=f'{src}{i}_bim',
                                 src=src, idx=i, smiles=smi, natoms=m.GetNumAtoms(),
                                 sites={'pyrrolicN': pyr, 'pyridinicN': pyd},
                                 d_pocket=d, has_bridge=has_bridge)
                    break
                if entry:
                    break
            if entry is None and m.HasSubstructMatch(PHEN):
                mt = m.GetSubstructMatches(PHEN)[0]
                nsites = [j for j in mt if m.GetAtomWithIdx(j).GetSymbol() == 'N']
                if len(nsites) == 2:
                    c = m.GetConformer().GetPositions()[list(mt)].mean(axis=0)
                    d = float(np.linalg.norm(c - BER))
                    if d <= args.max_d_pocket:
                        entry = dict(cls='phenanthroline', key=f'{src}{i}_phen',
                                     src=src, idx=i, smiles=smi, natoms=m.GetNumAtoms(),
                                     sites={'N1': nsites[0], 'N2': nsites[1]},
                                     d_pocket=d, has_bridge=bool(m.HasSubstructMatch(BRIDGE)))
            if entry is None and m.HasSubstructMatch(QUINAZ) and m.HasSubstructMatch(BRIDGE):
                mt = m.GetSubstructMatches(QUINAZ)[0]
                nsites = [j for j in mt if m.GetAtomWithIdx(j).GetSymbol() == 'N']
                if len(nsites) == 2:
                    c = m.GetConformer().GetPositions()[list(mt)].mean(axis=0)
                    d = float(np.linalg.norm(c - BER))
                    if d <= args.max_d_pocket:
                        entry = dict(cls='quinazoline+bridge', key=f'{src}{i}_qzb',
                                     src=src, idx=i, smiles=smi, natoms=m.GetNumAtoms(),
                                     sites={'N1': nsites[0], 'N2': nsites[1]},
                                     d_pocket=d, has_bridge=True)
            if entry:
                # priority: benzimidazole w/ bridge > benzimidazole > phen > quinaz+bridge
                prio = (0 if entry['cls'] == 'benzimidazole' and entry['has_bridge'] else
                        1 if entry['cls'] == 'benzimidazole' else
                        2 if entry['cls'] == 'phenanthroline' else 3)
                entry['bm_tani_t2'] = round(bm_tanimoto(m, t2m), 3)  # retrieval ranking only
                entry['prio'] = prio
                cands[entry['key']] = entry
                seen.add(smi)

    sel = sorted(cands.values(), key=lambda e: (e['prio'], -e['bm_tani_t2']))[:args.cap]
    print(f'[A1] candidates: {len(cands)} found, {len(sel)} selected')
    for e in sel:
        print(f"  {e['key']:>16} {e['cls']:>17} d={e['d_pocket']:5.2f} bridge={int(e['has_bridge'])} "
              f"BMtani={e['bm_tani_t2']:.3f} sites={e['sites']} {e['smiles'][:70]}")

    # ---- rDock: one polar-H SD, n runs each ----
    keys = [e['key'] for e in sel]
    mols = {}
    for src, path in sources:
        supp = Chem.SDMolSupplier(path, removeHs=True)
        for e in sel:
            if e['src'] == src:
                mols[e['key']] = Chem.Mol(supp[e['idx']])

    inp = os.path.join(OUTDIR, 'dock_input.sdf')
    w = Chem.SDWriter(inp)
    for k in keys:
        mh = polar_h_only(mols[k])
        mh.SetProp('_Name', k)
        w.write(mh)
    w.close()
    print(f'[A1] wrote {len(keys)} polar-H ligands -> {inp}')

    out_base = os.path.join(OUTDIR, 'redock_cands')
    cmd = ['rbdock', '-i', inp, '-o', out_base, '-r', SYSTEM_PRM, '-p', PROTOCOL,
           '-n', str(args.n_runs), '-s', str(args.seed)]
    print('[cmd] ' + ' '.join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, 'RBT_ROOT': RBT_ROOT})

    # ---- parse poses, rank per ligand by SCORE.INTER ----
    recs = {}
    for m in Chem.SDMolSupplier(out_base + '.sd', removeHs=False, sanitize=False):
        if m is None:
            continue
        p = m.GetPropsAsDict()
        name = m.GetProp('_Name')
        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return float('nan')
        recs.setdefault(name, []).append((num(p.get('SCORE.INTER')), m))

    summary = []
    for k in keys:
        lst = recs.get(k, [])
        if not lst:
            print(f'  [FAIL] {k}: no rbdock records')
            continue
        lst.sort(key=lambda t: t[0])
        best_score, pm = lst[0]
        heavy_idx = [j for j, a in enumerate(pm.GetAtoms()) if a.GetSymbol() != 'H']
        syms = [pm.GetAtomWithIdx(j).GetSymbol() for j in heavy_idx]
        ref_syms = [a.GetSymbol() for a in mols[k].GetAtoms()]
        assert syms == ref_syms, f'{k}: atom order mismatch'
        xyz = pm.GetConformer().GetPositions()[heavy_idx]
        bm = Chem.Mol(mols[k])
        conf = Chem.Conformer(bm.GetNumAtoms())
        for j in range(bm.GetNumAtoms()):
            conf.SetAtomPosition(j, Point3D(*xyz[j]))
        bm.RemoveAllConformers()
        bm.AddConformer(conf, assignId=True)
        bm.SetProp('_Name', k)
        outp = os.path.join(OUTDIR, f'{k}_best.sdf')
        ww = Chem.SDWriter(outp)
        ww.write(bm)
        ww.close()
        e = cands[k]
        summary.append({**e, 'best_SCORE_INTER': round(best_score, 2),
                        'n_poses': len(lst), 'best_pose_sdf': outp})
        print(f"  [ok] {k}: best SCORE.INTER = {best_score:.2f} ({len(lst)} poses) -> {outp}")

    with open(os.path.join(OUTDIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[A1] summary -> {OUTDIR}/summary.json')


if __name__ == '__main__':
    main()

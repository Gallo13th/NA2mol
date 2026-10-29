#!/usr/bin/env python3
"""Step2: cyanine target -> align onto BER102 crystal pose -> redock.

1. build 3D conformer from SMILES (ETKDG + MMFF, seed 42)
2. MCS alignment onto ber102_ref.sdf (berberine crystal pose) - the planar
   aromatic/fused ring systems share a quinoline-like/aryl common substructure
3. rbdock -n 20 into the ground-truth-validated BER102 cavity (RADIUS 6)
4. report best SCORE.INTER pose + spatial relation to the berberine pose
"""
import os
import subprocess
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign
from rdkit.Geometry import Point3D

RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
TGT_SMILES = 'CCN1C=C/C(=C/C=C/c2cc[n+]3ccccc3n2)c2ccccc21'
REF_SDF = os.path.join(HERE, 'ber102_ref.sdf')
SYSTEM_PRM = os.path.join(HERE, 'ber102_full_r6_system.prm')
ALIGNED_SDF = os.path.join(HERE, 'cyanine_aligned.sdf')
DOCK_BASE = os.path.join(HERE, 'cyanine_dock')
RBT_ROOT = os.environ.get('RBT_ROOT', '/home/data/users/gaoletian/.conda/envs/rna2smol/share/rdock-2013.1-1')
PROTOCOL = os.path.join(RBT_ROOT, 'data/scripts/dock_solv.prm')
N_RUNS = 20


def main():
    # ---- 1. build 3D ----
    tgt = Chem.MolFromSmiles(TGT_SMILES)
    assert tgt is not None and tgt.GetNumAtoms() == 25, tgt.GetNumAtoms() if tgt else None
    tgt_h = Chem.AddHs(tgt)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    cid = AllChem.EmbedMolecule(tgt_h, ps)
    assert cid == 0, 'ETKDG embed failed'
    res = AllChem.MMFFOptimizeMolecule(tgt_h, maxIters=2000)
    print(f'[3D    ] ETKDG+MMFF (converged={res == 0}), '
          f'{tgt_h.GetNumAtoms()} atoms incl H')
    tgt = Chem.RemoveHs(tgt_h)
    Chem.SanitizeMol(tgt)

    ref = next(Chem.SDMolSupplier(REF_SDF, removeHs=True))
    assert ref is not None

    # ---- 2. MCS alignment ----
    mcs = rdFMCS.FindMCS([tgt, ref], ringMatchesRingOnly=True,
                         completeRingsOnly=True, timeout=60,
                         matchValences=False)
    q = Chem.MolFromSmarts(mcs.smartsString)
    print(f'[MCS   ] {mcs.numAtoms} atoms / {mcs.numBonds} bonds: {mcs.smartsString}')
    t_matches = tgt.GetSubstructMatches(q, uniquify=False)
    r_matches = ref.GetSubstructMatches(q, uniquify=False)
    print(f'[MCS   ] matches: tgt {len(t_matches)}, ref {len(r_matches)}')
    best = None
    for tm in t_matches[:20]:
        for rm in r_matches[:20]:
            amap = list(zip(tm, rm))
            rms = rdMolAlign.AlignMol(tgt, ref, atomMap=amap)
            if best is None or rms < best[0]:
                best = (rms, amap)
    print(f'[align ] best MCS alignment RMSD = {best[0]:.3f} A '
          f'({len(best[1])} atom pairs)')

    aligned = Chem.Mol(tgt)
    conf = Chem.Conformer(25)
    for i, p in enumerate(tgt.GetConformer().GetPositions()):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    aligned.RemoveAllConformers()
    aligned.AddConformer(conf, assignId=True)
    aligned.SetProp('_Name', 'cyanine_aligned_to_BER102')
    w = Chem.SDWriter(ALIGNED_SDF)
    w.write(aligned)
    w.close()

    d_cent = np.linalg.norm(aligned.GetConformer().GetPositions().mean(0)
                            - ref.GetConformer().GetPositions().mean(0))
    print(f'[align ] centroid distance cyanine-vs-berberine = {d_cent:.2f} A '
          f'(aligned input pose)')

    # ---- 3. redock ----
    cmd = ['rbdock', '-i', ALIGNED_SDF, '-o', DOCK_BASE, '-r', SYSTEM_PRM,
           '-p', PROTOCOL, '-n', str(N_RUNS), '-s', '42']
    print('[cmd] ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)

    # ---- 4. analyse ----
    ref_xyz = ref.GetConformer().GetPositions()
    t_sym = [a.GetSymbol() for a in aligned.GetAtoms()]
    rows, poses = [], {}
    for i, m in enumerate(Chem.SDMolSupplier(DOCK_BASE + '.sd', removeHs=False,
                                             sanitize=False)):
        if m is None:
            continue
        syms = [a.GetSymbol() for a in m.GetAtoms()]
        if syms == t_sym:
            xyz = m.GetConformer().GetPositions()
        else:  # safety: element-sequence alignment
            hidx = [j for j, a in enumerate(m.GetAtoms()) if a.GetSymbol() != 'H']
            assert [m.GetAtomWithIdx(j).GetSymbol() for j in hidx] == t_sym, \
                f'pose {i+1} atom order mismatch'
            xyz = m.GetConformer().GetPositions()[hidx]
        props = m.GetPropsAsDict()
        try:
            inter = float(props.get('SCORE.INTER'))
        except (TypeError, ValueError):
            inter = float('nan')
        poses[i + 1] = xyz
        rows.append((inter, i + 1))
    rows.sort()
    best_inter, best_run = rows[0]
    xyz = poses[best_run]

    print('\n===== cyanine redock into BER102 cavity (20 runs) =====')
    for s, r in rows:
        print(f'  run {r:>2}: SCORE.INTER {s:9.3f}')
    print(f'best: run {best_run}, SCORE.INTER = {best_inter:.3f}')

    cent_d = np.linalg.norm(xyz.mean(0) - ref_xyz.mean(0))
    # plane normals (both ~planar chromophores)
    def plane_normal(P):
        c = P.mean(0)
        u, s, vt = np.linalg.svd(P - c)
        return vt[2]
    ang = np.degrees(np.arccos(abs(np.dot(plane_normal(xyz), plane_normal(ref_xyz)))))
    # closest approach between the two ring systems
    dmin = np.linalg.norm(xyz[:, None, :] - ref_xyz[None, :, :], axis=-1).min()

    print(f'\n[pose  ] centroid distance to berberine crystal pose: {cent_d:.2f} A')
    print(f'[pose  ] chromophore plane-plane angle: {ang:.1f} deg')
    print(f'[pose  ] min atom-atom distance to berberine: {dmin:.2f} A')

    out_mol = Chem.Mol(aligned)
    conf = Chem.Conformer(25)
    for j, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(j, Point3D(float(x), float(y), float(z)))
    out_mol.RemoveAllConformers()
    out_mol.AddConformer(conf, assignId=True)
    out_mol.SetProp('_Name', f'cyanine_best_INTER_{best_inter:.2f}')
    out_mol.SetProp('SCORE.INTER', f'{best_inter:.3f}')
    w = Chem.SDWriter(os.path.join(HERE, 'cyanine_best_pose.sdf'))
    w.write(out_mol)
    w.close()
    print(f"[saved ] {ALIGNED_SDF}, {DOCK_BASE}.sd, cyanine_best_pose.sdf")


if __name__ == '__main__':
    main()

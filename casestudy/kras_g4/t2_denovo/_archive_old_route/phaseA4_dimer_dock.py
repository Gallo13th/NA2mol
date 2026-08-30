#!/usr/bin/env python3
"""t2-DENOVO Phase A4: bis-benzimidazole-CH2 dimer (library cores + generic sp3
geometry) -> rDock redock -> aromatic-aware masked probe at the pyrrolic-N sites.

Motivation (A2/A3 evidence): single-isostere contexts cap P[S] at ~1% (redock pose)
/ ~5% (trimmed core). The known high-P[S] context family is bis-fused-diaza + CH2.
No bis-diaza molecule exists in the libraries -> assemble one from model-product
cores with GENERIC chemistry (C-CH2 1.51 A, angle 112 deg, coplanar extended
conformer), then let rDock (validated BER102 R6 protocol) place it in the pocket.
t2 is never used: not for construction, not for coordinates.

Probe contexts at each redocked pose (top-3 poses):
  F4  : full dimer (19 atoms), both pyrrolic-N masked   -> replacement endorsement
  F4b : CH2-deleted (18 atoms, the Phase-C seed shape), both pyrrolic-N masked
        -> the exact context the model sees in Phase-D generation

Gate: P[S] > 10% at any pyrrolic site in any F4/F4b context.

Usage: python phaseA4_dimer_dock.py            (assembly + rbdock)
       CUDA_VISIBLE_DEVICES=1 python phaseA4_dimer_probe.py   (probe)
"""
import os, sys, json, subprocess, itertools

PROJ = '/home/data/users/gaoletian/rna2smol'
os.chdir(PROJ)
sys.path.insert(0, PROJ)

from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')
import numpy as np

RBT_ROOT = '/home/data/users/gaoletian/.conda/envs/rna2smol/share/rdock-2013.1-1'
SYSTEM_PRM = os.path.join(PROJ, 'casestudy/kras_g4/ber102_rdock/ber102_full_r6_system.prm')
PROTOCOL = os.path.join(RBT_ROOT, 'data/scripts/dock_solv.prm')
SUMMARY = 'casestudy/kras_g4/t2_denovo/phaseA_redock/summary.json'
OUTDIR = 'casestudy/kras_g4/t2_denovo/phaseA4_dimer'
os.makedirs(OUTDIR, exist_ok=True)

BIM_T = [Chem.MolFromSmiles('c1ccc2nc[nH]c2c1'), Chem.MolFromSmiles('c1ccc2[nH]cnc2c1')]


def get_bm_core(m):
    for tmpl in BIM_T:
        for mt in m.GetSubstructMatches(tmpl):
            core = Chem.Mol(tmpl)
            conf = Chem.Conformer(9)
            pos = m.GetConformer().GetPositions()
            for i, o in enumerate(mt):
                conf.SetAtomPosition(i, Point3D(*pos[o]))
            core.RemoveAllConformers()
            core.AddConformer(conf, assignId=True)
            pyr, c2 = None, None
            for ring in core.GetRingInfo().AtomRings():
                if len(ring) != 5:
                    continue
                for i in ring:
                    a = core.GetAtomWithIdx(i)
                    if a.GetSymbol() == 'N' and a.GetTotalNumHs() > 0:
                        pyr = i
                    elif a.GetSymbol() == 'C':
                        het = [n for n in a.GetNeighbors() if n.GetIdx() in ring and n.GetSymbol() == 'N']
                        if len(het) == 2:
                            c2 = i
            if pyr is not None and c2 is not None:
                return core, pyr, c2
    return None


def rot_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        ax = np.cross(a, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def build_dimer(core1, core2, c2_1, c2_2, d=2.52):
    """generic anti-coplanar bis-BM-CH2: C2a-CH2=C2b-CH2 1.51 A, angle 112.4 deg,
    rings coplanar, CH2 off-axis in plane. Returns (mol19, ch2_idx, pyr1, pyr2)."""
    P1 = core1.GetConformer().GetPositions()
    P2 = core2.GetConformer().GetPositions()
    n1 = np.linalg.svd(P1 - P1.mean(0), full_matrices=False)[2][2]
    out2 = P2.mean(0) - P2[c2_2]
    out2 /= np.linalg.norm(out2)
    n2 = np.linalg.svd(P2 - P2.mean(0), full_matrices=False)[2][2]
    # target: C2b at +d*x from C2a, ring2 center on far side, CH2 above midpoint in-plane
    v = np.array([1.0, 0.0, 0.0])
    R1 = rot_between(out2, -v)
    n2r = n2 @ R1.T
    n1_ref = np.array([0.0, 0.0, 1.0])
    # align n2r to n1_ref about v axis
    n_perp = n2r - np.dot(n2r, v) * v
    t_perp = n1_ref - np.dot(n1_ref, v) * v
    R2 = np.eye(3) if np.linalg.norm(n_perp) < 1e-6 else rot_between(n_perp, t_perp)
    P2r = (R1 @ (R2 @ P2.T)).T
    # rotate core1 so its normal = n1_ref and C2->outward maps +x
    out1 = P1.mean(0) - P1[c2_1]
    out1 /= np.linalg.norm(out1)
    R1a = rot_between(out1, -v)
    n1r = n1 @ R1a.T
    n1p = n1r - np.dot(n1r, v) * v
    tp = n1_ref - np.dot(n1_ref, v) * v
    R2a = np.eye(3) if np.linalg.norm(n1p) < 1e-6 else rot_between(n1p, tp)
    P1r = (R1a @ (R2a @ P1.T)).T

    r = 1.51
    h = d / 2.0
    off = np.sqrt(max(r * r - h * h, 0.0))
    ch2 = np.array([h, off, 0.0])

    core2r = Chem.Mol(core2)
    conf2 = Chem.Conformer(9)
    for i in range(9):
        conf2.SetAtomPosition(i, Point3D(*(P2r[i] + np.array([d, 0, 0]) - P2r[c2_2] + P1r[c2_1] - P1r[c2_1])))
    # simpler: place core2 C2 at P1r[c2_1] + d*v
    P2p = P2r - P2r[c2_2] + P1r[c2_1] + d * v
    for i in range(9):
        conf2.SetAtomPosition(i, Point3D(*P2p[i]))
    core2r.RemoveAllConformers()
    core2r.AddConformer(conf2, assignId=True)

    core1r = Chem.Mol(core1)
    conf1 = Chem.Conformer(9)
    for i in range(9):
        conf1.SetAtomPosition(i, Point3D(*P1r[i]))
    core1r.RemoveAllConformers()
    core1r.AddConformer(conf1, assignId=True)

    mol = Chem.CombineMols(core1r, core2r)
    rw = Chem.RWMol(mol)
    ch = Chem.Atom(6)
    ch.SetNoImplicit(True)
    ch.SetNumExplicitHs(2)
    rw.AddAtom(ch)
    rw.AddBond(c2_1, 18, Chem.BondType.SINGLE)
    rw.AddBond(18, 9 + c2_2, Chem.BondType.SINGLE)
    for ai in (c2_1, 9 + c2_2):
        a = rw.GetAtomWithIdx(ai)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    out = rw.GetMol()
    conf = Chem.Conformer(19)
    P = out.GetConformer().GetPositions()
    for i in range(18):
        conf.SetAtomPosition(i, Point3D(*P[i]))
    conf.SetAtomPosition(18, Point3D(*ch2))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(out)
    return out, 18


def polar_h_only(m):
    m2 = Chem.AddHs(Chem.Mol(m), addCoords=True)
    rw = Chem.RWMol(m2)
    for a in list(rw.GetAtoms()):
        if a.GetSymbol() == 'H':
            heavy = [n for n in a.GetNeighbors() if n.GetSymbol() != 'H']
            if heavy and heavy[0].GetSymbol() == 'C':
                rw.RemoveAtom(a.GetIdx())
    outmol = rw.GetMol()
    Chem.SanitizeMol(outmol)
    return outmol


def main():
    summary = json.load(open(SUMMARY))
    a2 = json.load(open('casestudy/kras_g4/t2_denovo/phaseA_probe_results.json'))
    bims = [(a2[e['key']]['sites']['pyrrolicN']['redock']['PS'], e)
            for e in summary if e['cls'] == 'benzimidazole']
    bims.sort(key=lambda t: -t[0])
    top = [e['key'] for _, e in bims[:4]]
    print('[A4] cores used:', top)

    cores = {}
    for k in top:
        m = next(Chem.SDMolSupplier(f'casestudy/kras_g4/t2_denovo/phaseA_redock/{k}_best.sdf',
                                    removeHs=True))
        cores[k] = get_bm_core(m)
        assert cores[k] is not None, k

    combos = [(top[0], top[0]), (top[0], top[1]), (top[0], top[2]), (top[1], top[1]), (top[0], top[3])]
    inputs = []
    for i, (k1, k2) in enumerate(combos):
        core1, pyr1, c21 = cores[k1]
        core2, pyr2, c22 = cores[k2]
        mol, ch2i = build_dimer(core1, core2, c21, c22)
        name = f'dimer{i}_{k1.split("_")[0]}x{k2.split("_")[0]}'
        mol.SetProp('_Name', name)
        # pyrrolic sites in combined indexing
        inputs.append((name, mol, pyr1, 9 + pyr2, ch2i, k1, k2))
        smi = Chem.MolToSmiles(mol)
        assert '.' not in smi and smi.count('n') + smi.count('[nH]') >= 4, smi
        print(f'  [dimer] {name}: {smi}')

    inp = os.path.join(OUTDIR, 'dimer_dock_input.sdf')
    w = Chem.SDWriter(inp)
    for name, mol, *_ in inputs:
        mh = polar_h_only(mol)
        mh.SetProp('_Name', name)
        w.write(mh)
    w.close()

    out_base = os.path.join(OUTDIR, 'redock_dimer')
    cmd = ['rbdock', '-i', inp, '-o', out_base, '-r', SYSTEM_PRM, '-p', PROTOCOL,
           '-n', '12', '-s', '42']
    print('[cmd] ' + ' '.join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, 'RBT_ROOT': RBT_ROOT})

    # extract top-3 poses per dimer (SCORE.INTER rank), paste onto heavy-atom copy
    recs = {}
    for m in Chem.SDMolSupplier(out_base + '.sd', removeHs=False, sanitize=False):
        if m is None:
            continue
        p = m.GetPropsAsDict()
        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return float('nan')
        recs.setdefault(m.GetProp('_Name'), []).append((num(p.get('SCORE.INTER')), m))

    manifest = []
    for name, mol, pyr1, pyr2, ch2i, k1, k2 in inputs:
        lst = sorted(recs.get(name, []), key=lambda t: t[0])
        if not lst:
            print(f'  [FAIL] {name} no records')
            continue
        ref_syms = [a.GetSymbol() for a in mol.GetAtoms()]
        for rank, (score, pm) in enumerate(lst[:3]):
            heavy_idx = [j for j, a in enumerate(pm.GetAtoms()) if a.GetSymbol() != 'H']
            syms = [pm.GetAtomWithIdx(j).GetSymbol() for j in heavy_idx]
            assert syms == ref_syms, f'{name} pose order mismatch'
            xyz = pm.GetConformer().GetPositions()[heavy_idx]
            # F4: full 19-atom copy with pose coords
            bm = Chem.Mol(mol)
            conf = Chem.Conformer(19)
            for j in range(19):
                conf.SetAtomPosition(j, Point3D(*xyz[j]))
            bm.RemoveAllConformers()
            bm.AddConformer(conf, assignId=True)
            bm.SetProp('_Name', f'{name}_p{rank}')
            p4 = os.path.join(OUTDIR, f'{name}_p{rank}_F4.sdf')
            ww = Chem.SDWriter(p4); ww.write(bm); ww.close()
            # F4b: delete CH2 -> 18-atom 2-fragment seed shape
            rw = Chem.RWMol(bm)
            rw.RemoveAtom(ch2i)
            f4b = rw.GetMol()
            Chem.SanitizeMol(f4b)
            # pyr indices unchanged (both < ch2i=18)
            p4b = os.path.join(OUTDIR, f'{name}_p{rank}_F4b.sdf')
            ww = Chem.SDWriter(p4b); ww.write(f4b); ww.close()
            manifest.append({'name': name, 'pose': rank, 'score': round(score, 2),
                             'pyr1': pyr1, 'pyr2': pyr2, 'ch2i': ch2i,
                             'k1': k1, 'k2': k2,
                             'F4': p4, 'F4b': p4b,
                             'smiles': Chem.MolToSmiles(bm)})
            print(f'  [ok] {name} pose{rank}: SCORE.INTER={score:.2f}')

    with open(os.path.join(OUTDIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'[A4] manifest -> {OUTDIR}/manifest.json ({len(manifest)} pose contexts)')


if __name__ == '__main__':
    main()

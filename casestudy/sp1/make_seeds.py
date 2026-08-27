#!/usr/bin/env python3
"""Make sp1 flavone inpainting seeds.

Idea (mirrors t2 inpaint_seeds): keep the two phenol rings + ring O positions
(aromatic C*/S-ring atoms that the model supports 86-99%) as FIXED anchors,
release the chromone carbonyl/O/C' positions (model disagreements at 1,2,4,9,
+ risk zone 0/12) as free.

Seed = the m1 methyl scaffold expanded to the 21-atom skeleton, but only the
trusted positions carry real elements; contested positions are masked as C.

Trusted (fix): A/B rings atoms 5-8,10,11,13-15,17,18,20 (model 86-99% C*)
  + ring O 12? (62% O) + OH 3 (81% O) + OH 16/19 (39-44% O weak)
Contested (free): 0 (C=O), 1 (C' spag), 2 (C2), 4 (pyrone C), 9 (OH -> F 80%)
Free-extra: 0 => exact 21-atom inpainting; the model fills the 6 free atoms.

We write several seed variants:
  S1  all 21 atoms as skeleton-C, free={0,1,2,4,9,12}, fix aromatic C rings
      (the "trust then trust" minimal variant)
      -> free atoms are C(not aromatic) = contested core + OH9 position
  S2  same but free also = {16,19} (weak-OH sites), fixing only strong sites
  S3  m1 full skeleton fixed except chromone core free = {0,1,2,4,12}
  (3 seeds give model a short menu; rounds accumulate)
"""
import os, sys, numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog('rdApp.*')

ROOT = '/home/data/users/gaoletian/rna2smol'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

M1 = 'O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12'
SKEL = 'casestudy/sp1/skeleton_flavone_sp1.sdf'
OUTDIR = 'casestudy/sp1/inpaint_seeds'
os.makedirs(OUTDIR, exist_ok=True)

m1 = Chem.MolFromSmiles(M1)
skel = Chem.SDMolSupplier(SKEL)[0]
coords = skel.GetConformer().GetPositions()
assert skel.GetNumAtoms() == m1.GetNumAtoms() == 21

# trusted aromatic C* ring atoms (from probe: 5,6,7,8,10,11 = A ring? mapping:
# see probe table: 6,7,8,10,11 -> 86-99% ; 5 -> 47%; ring atoms)
STRONG_C = [6,7,8,10,11,13,15,18,20]
WEAK_C = [5,14,17]
STRONG_O = [3,12]   # 81%/62%
WEAK_O = [16,19]    # 39%/44%
CONTEST_C = [0,1,2,4]  # carbonyl C0->O? In skeleton all C (smeared); target says
                        # 0=O=C... (target has O at 0!) -- check m1atom 0 = O!
CONTEST_OH = [9]    # model wants F (80%)

def write_seed(fname, elems_by_idx, free_atoms, desc):
    rw = Chem.RWMol(skel)
    for i in range(21):
        a = rw.GetAtomWithIdx(i)
        el = elems_by_idx[i]
        a.SetAtomicNum(6 if el == 'C' else 8)
        a.SetFormalCharge(0)
        a.SetNoImplicit(False)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    conf = Chem.Conformer(21)
    for i in range(21):
        conf.SetAtomPosition(i, Point3D(*[float(x) for x in coords[i]]))
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    m.SetProp('_Name', desc)
    m.SetProp('_free_atoms', ','.join(str(i) for i in free_atoms))
    w = Chem.SDWriter(f'{OUTDIR}/{fname}')
    w.write(m)
    w.close()
    print(f'[seed] {fname}: free={free_atoms}  {desc}', flush=True)

# S1: minimal trust (strong rings + strong O3/O12 fixed), rest free-C
elems1 = {i: 'C' for i in range(21)}
for i in STRONG_O + [16, 19]:
    elems1[i] = 'O'
for i in STRONG_C + WEAK_C:
    elems1[i] = 'C'
free1 = [0, 1, 2, 4, 5, 9, 12, 14, 17]
write_seed('seed_s1_flavone.sdf', elems1, free1,
           f'S1: fix strongC+C/O, free={free1}')

# S2: strong-only trusted (like S1 but also free 16,19 weak O)
elems2 = dict(elems1)
for i in [16, 19]:
    elems2[i] = 'C'
free2 = [0, 1, 2, 4, 5, 9, 12, 14, 16, 17, 19]
write_seed('seed_s2_flavone.sdf', elems2, free2,
           f'S2: free also weak-O 16,19 -> {free2}')

# S3: everything fixed except chromone core: 0(C=O),1,2,4,12 free
elems3 = {i: 'C' for i in range(21)}
for i in [0, 3, 9, 12, 16, 19]:
    elems3[i] = 'O'
free3 = [1, 2, 4]
write_seed('seed_s3_flavone.sdf', elems3, free3,
           f'S3: m1-like fixed, core free={free3}')

# S4: weaker variant - don't even anchor ring O12 (model 62%): free 12 too
elems4 = dict(elems3)
for i in [12]:
    elems4[i] = 'C'  # actually free: S3 has it fixed O; S4 free
free4 = [1, 2, 4, 12, 9]
write_seed('seed_s4_flavone.sdf', elems4, free4,
           f'S4: core+O12+OH9 free={free4}')

print(f'\n[ok] wrote seeds to {OUTDIR}/')

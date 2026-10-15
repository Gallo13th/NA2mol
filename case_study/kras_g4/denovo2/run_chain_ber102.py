#!/usr/bin/env python3
"""BER102 full protocol downstream of cold start (v3 port-tracing chain).

Usage: python case_study/kras_g4/denovo2/run_chain_ber102.py [--lib LIB.sdf]
Stages (with t2 checkpoints):
  A2 cut + conformation dedup (ports recorded later at pool level)
  2c ProLIF + smina vs BER102 pocket (parallel)
  2d gates: smina<=-2.5 & n_total>=3 & heavy<=14  -> QED>0.5 & SAS<2.0 pool
  2e port records; 3 mutation expansion (KEEP) -> QED/SAS -> variants
  3s rDock score-only INTER (ber102_full_r6_system)
  4  port matching (elem same, ghost<=1.0) + clash assembly -> products [t2?]
  5  QED>=0.5 & SAS<=median filter -> multi-seed minimise (3x5, RMSD<=2)
  6  RNAsmol 6-model (G4 SEQ) -> drop dead dims -> rankers
  7  two-axis majority tiers -> MinRank top-20 [t2?]
"""
import os, sys, json, argparse, subprocess, itertools
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem, QED
RDLogger.DisableLog('rdApp.*')
try:
    from rdkit.Contrib.SA_Score import sascorer
except Exception:
    sys.path.insert(0, 'util'); import sascorer

PROJ = Path(__file__).resolve().parents[3]  # repository root
os.chdir(PROJ); sys.path.insert(0, PROJ)
OUT = 'case_study/kras_g4/denovo2'
T2 = 'c1ccc2sc(Cc3nc4ccccc4s3)nc2c1'
BI = 'c1ccc2[nH]cnc2c1'
BT = 'c1ccc2scnc2c1'
POK = 'case_study/kras_g4/ber102_rdock/ber102_pocket.pdb'
POK_ADDH = 'case_study/kras_g4/ber102_rdock/ber102_pocket_addh.pdb'
SMINA = 'util/smina.static'
RDOCK_SYS = 'case_study/kras_g4/ber102_rdock/ber102_full_r6_system.prm'
SCORE_ONLY = 'case_study/kras_g4/rdock_r4/score_only.prm'
RBT = os.environ.get('RBT_ROOT', '/usr/local/share/rdock-2013.1-1')
SEQ = 'GAGGGCGGTGTGGGAATAGGGAA'

def _prolif_one(args):
    frag, pok_path = args
    import prolif as plf
    global _PROT_CACHE
    try:
        prot = _PROT_CACHE
    except NameError:
        from rdkit import Chem as _C
        prot = plf.Molecule.from_rdkit(_C.MolFromPDBFile(pok_path, removeHs=False))
        globals()['_PROT_CACHE'] = prot
    fp = plf.Fingerprint()
    try:
        lig = plf.Molecule.from_rdkit(Chem.AddHs(frag, addCoords=True))
        fp.run_from_iterable([lig], prot, progress=False)
        df = fp.to_dataframe()
        ints = [(str(resid), itype) for col in df.columns if df[col].iloc[0] for _, resid, itype in [col]]
        types = sorted({t for _, t in ints if t != 'VdWContact'})
        return {'n_types': len(types), 'n_total': len(ints), 'n_res': len({r for r, _ in ints})}
    except Exception:
        return None

def _smina_one(args):
    idx, path, addh = args
    r = subprocess.run([SMINA, '--minimize', '-r', addh,
                        '-l', f'/tmp/opencode/chainfrag/f{idx:05d}.sdf',
                        '-o', f'/tmp/opencode/chainfrag/s{idx}.sdf'], capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith('Affinity:'):
            return idx, float(line.split()[1])
    return idx, None

def stop(msg):
    print('\n[STOP] ' + msg, flush=True)
    sys.exit(2)

def ring_systems(mol):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    idx = list(range(len(rings))); systems = []
    while idx:
        seed = rings[idx.pop(0)]; grown = True
        while grown:
            grown = False
            for i in list(idx):
                if rings[i] & seed:
                    seed |= rings[i]; idx.remove(i); grown = True
        systems.append(seed)
    return systems

def extract_system(mol, atoms):
    rw = Chem.RWMol(); o2n = {}
    conf = Chem.Conformer(len(atoms)); src = mol.GetConformer()
    for new, old in enumerate(sorted(atoms)):
        a = mol.GetAtomWithIdx(old)
        na = Chem.Atom(a.GetSymbol())
        if a.GetIsAromatic(): na.SetIsAromatic(True)
        if a.GetNumExplicitHs():
            na.SetNoImplicit(True); na.SetNumExplicitHs(a.GetNumExplicitHs()); na.SetNoImplicit(False)
        rw.AddAtom(na); o2n[old] = new
        p = src.GetAtomPosition(old)
        conf.SetAtomPosition(new, Point3D(p.x, p.y, p.z))
    for b in mol.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if s in o2n and e in o2n:
            rw.AddBond(o2n[s], o2n[e], b.GetBondType())
    m = rw.GetMol(); m.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(m)
    return m

def inPlaceRMSD(mA, mB, match):
    pA = mA.GetConformer().GetPositions()[list(match)]
    pB = mB.GetConformer().GetPositions()
    return float(np.sqrt(((pA - pB) ** 2).sum(1).mean()))

def parse_pocket(path):
    pts = []
    for line in open(path):
        if line.startswith(('ATOM', 'HETATM')):
            el = (line[76:78].strip() or line[12:16].strip()[0])
            if el.upper() != 'H':
                pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts)

def mut_sites(m):
    sites = []
    ri = m.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) not in (5, 6): continue
        for i in ring:
            a = m.GetAtomWithIdx(i)
            if a.GetSymbol() in ('N', 'O', 'S') and a.GetIsAromatic() and a.GetDegree() == 2:
                sites.append(i)
    return sorted(set(sites))

def apply_mut(m, subs):
    rw = Chem.RWMol(m)
    for i, elem in subs.items():
        a = rw.GetAtomWithIdx(i)
        a.SetAtomicNum({'N': 7, 'O': 8, 'S': 16, 'C': 6}[elem])
        if elem == 'C': a.SetIsAromatic(True)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(1 if elem == 'N' else 0)
        a.SetNoImplicit(False)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', default='case_study/kras_g4/ber102_gen10k/valid_mols.sdf')
    ap.add_argument('--skip-minimise', action='store_true')
    ap.add_argument('--skip-rnasmol', action='store_true')
    args = ap.parse_args()
    pocket = parse_pocket(POK)

    # ---- A2 cut + dedup ----
    groups = defaultdict(list); n_cut = 0
    for mi, m in enumerate(Chem.SDMolSupplier(args.lib, removeHs=True)):
        if m is None or not m.GetNumConformers(): continue
        par = f'k{mi:05d}'
        for atoms in ring_systems(m):
            if not (5 <= len(atoms) <= 20): continue
            try:
                frag = extract_system(m, atoms); smi = Chem.MolToSmiles(frag)
            except Exception: continue
            groups[smi].append((par, frag)); n_cut += 1
    reps = []
    for smi, lst in groups.items():
        keep = []
        for par, frag in lst:
            merged = False
            for k, (kp, kf, kn) in enumerate(keep):
                mt = kf.GetSubstructMatch(frag)
                if not mt:
                    mt2 = frag.GetSubstructMatch(kf)
                    if not mt2: continue
                    mt = tuple(mt2.index(i) for i in range(frag.GetNumAtoms()))
                if inPlaceRMSD(kf, frag, mt) < 0.5:
                    keep[k] = (kp, kf, kn + 1); merged = True; break
            if not merged:
                keep.append((par, frag, 1))
        reps.extend((smi, kp, kf, kn) for kp, kf, kn in keep)
    print(f'[A2] {n_cut} instances -> {len(groups)} systems -> {len(reps)} fragments', flush=True)
    frags = []
    w = Chem.SDWriter(f'{OUT}/chain_frag_dedup.sdf')
    for i, (smi, par, frag, kn) in enumerate(reps):
        m = Chem.Mol(frag)
        m.SetProp('_Name', f'cfrag{i:05d}'); m.SetProp('smi', smi)
        m.SetProp('parent', par); m.SetProp('n_merged', str(kn))
        w.write(m); frags.append(m)
    w.close()
    nbi = sum(1 for s, *_ in reps if s == BI)
    if nbi == 0: stop(f'no BI fragments in library ({len(reps)} frags)')
    print(f'[A2] BI conformers: {nbi}', flush=True)
    by_i = {int(m.GetProp('_Name').replace('cfrag', '')): m for m in frags}
    recs0 = [{'i': i, 'smi': m.GetProp('smi'), 'parent': m.GetProp('parent')} for i, m in enumerate(frags)]

    # ---- 2c ProLIF + smina ----
    os.makedirs('/tmp/opencode/chainfrag', exist_ok=True)
    for i, m in enumerate(frags):
        ww = Chem.SDWriter(f'/tmp/opencode/chainfrag/f{i:05d}.sdf'); ww.write(m); ww.close()
    with ProcessPoolExecutor(max_workers=12) as ex:
        prolif_res = list(ex.map(_prolif_one, [(m, os.path.join(PROJ, POK)) for m in frags], chunksize=16))
    smina_aff = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        for idx, aff in ex.map(_smina_one, [(i, '', os.path.join(PROJ, POK_ADDH)) for i in range(len(frags))], chunksize=16):
            if aff is not None: smina_aff[idx] = aff
    stats = []
    for i in range(len(frags)):
        r = dict(recs0[i]); r.update(prolif_res[i] or {}); r['smina'] = smina_aff.get(i)
        stats.append(r)
    json.dump(stats, open(f'{OUT}/chain_frag_stats.json', 'w'), indent=1)
    nbi = sum(1 for r in stats if r['smi'] == BI and r['smina'] and r['smina'] <= -2.5 and (r.get('n_total') or 0) >= 3)
    print(f'[2c] BI passing smina&n_total>=3: {nbi}', flush=True)
    if nbi == 0:
        biall = [r for r in stats if r['smi'] == BI]
        stop('BI died at gates: ' + json.dumps([{k: r.get(k) for k in ("smina", "n_total")} for r in biall][:8]))

    # ---- 2d gates + pool ----
    sel = [r for r in stats if r['smina'] and r['smina'] <= -2.5
           and (r.get('n_total') or 0) >= 3 and by_i[r['i']].GetNumAtoms() <= 14]
    pool = []
    for r in sel:
        m = by_i[r['i']]
        q = QED.qed(m); sa = sascorer.calculateScore(m)
        if q > 0.5 and sa < 2.0:
            pool.append({'i': r['i'], 'smi': r['smi'], 'parent': r['parent'],
                         'smina': r['smina'], 'n_total': r['n_total'],
                         'QED': round(q, 3), 'SAS': round(sa, 2)})
    print(f'[2d] pool: {len(pool)} | BI: {sum(1 for r in pool if r["smi"]==BI)}', flush=True)
    if not any(r['smi'] == BI for r in pool): stop('BI not in QED/SAS pool')
    json.dump(pool, open(f'{OUT}/chain_pool.json', 'w'), indent=1)

    # ---- 2e ports ----
    lib = {}
    for mi, m in enumerate(Chem.SDMolSupplier(args.lib, removeHs=True)):
        if m is not None and m.GetNumConformers(): lib[f'k{mi:05d}'] = m
    def find_in_parent(frag, parent):
        pf = frag.GetConformer().GetPositions(); pp = parent.GetConformer().GetPositions()
        mp = {}
        for i in range(len(pf)):
            d = np.linalg.norm(pp - pf[i], axis=1)
            j = int(d.argmin())
            if d[j] < 0.01: mp[i] = j
        return mp if len(mp) == len(pf) else None
    port_db = {}
    for r in pool:
        frag = by_i[r['i']]; par = lib.get(r['parent'])
        ports = []
        if par is not None:
            mp = find_in_parent(frag, par)
            if mp:
                infrag = set(mp.values())
                for i, j in mp.items():
                    for nb in par.GetAtomWithIdx(j).GetNeighbors():
                        nj = nb.GetIdx()
                        if nj in infrag: continue
                        ports.append({'frag_atom': i, 'ext_elem': nb.GetSymbol(),
                                      'ext_xyz': [round(float(x), 3) for x in par.GetConformer().GetPositions()[nj]]})
        port_db[r['i']] = ports
    json.dump({str(k): v for k, v in port_db.items()}, open(f'{OUT}/chain_ports.json', 'w'))

    # ---- 3 mutation ----
    variants = []
    for r in pool:
        m = by_i[r['i']]
        sites = mut_sites(m)
        site_opts = [[None] + [e for e in ['N','O','S','C'] if e != m.GetAtomWithIdx(i).GetSymbol()] for i in sites]
        if not sites: combos = [()]
        elif len(sites) <= 5: combos = list(itertools.product(*site_opts))
        else:
            combos = [(None,)*len(sites)]
            for k in (1,2,3):
                for sub in itertools.combinations(range(len(sites)), k):
                    for vals in itertools.product(*[site_opts[j][1:] for j in sub]):
                        combo = [None]*len(sites)
                        for jj, j in enumerate(sub): combo[j] = vals[jj]
                        combos.append(tuple(combo))
            combos = list(dict.fromkeys(combos))[:300]
        for combo in combos:
            subs = {i: e for i, e in zip(sites, combo) if e is not None}
            vm = Chem.Mol(m) if not subs else None
            if subs:
                try: vm = apply_mut(m, subs)
                except Exception: continue
            try: smi = Chem.MolToSmiles(vm)
            except Exception: continue
            variants.append({'frag_i': r['i'], 'family': r['i'], 'smiles': smi,
                             'subs': json.dumps({str(k): v for k, v in subs.items()})})
    keep = []
    for v in variants:
        m2 = Chem.MolFromSmiles(v['smiles'])
        if m2 is None: continue
        q = QED.qed(m2); sa = sascorer.calculateScore(m2)
        v['QED'] = round(q, 3); v['SAS'] = round(sa, 2)
        if q > 0.5 and sa < 2.0: keep.append(v)
    for i, v in enumerate(keep): v['variant_id'] = i
    nbt = sum(1 for v in keep if v['smiles'] == BT)
    print(f'[3] variants: {len(keep)} | BT: {nbt}', flush=True)
    if nbt < 2: stop(f'BT variants < 2 ({nbt})')
    json.dump(keep, open(f'{OUT}/chain_variants.json', 'w'), indent=1)
    os.makedirs('/tmp/opencode/chainvar', exist_ok=True)
    w = Chem.SDWriter('/tmp/opencode/chainvar/var.sdf')
    for v in keep:
        m = by_i[v['frag_i']]
        subs = json.loads(v['subs'])
        if subs: m = apply_mut(m, {int(k): e for k, e in subs.items()})
        m.SetProp('_Name', f"var{v['variant_id']:04d}_f{v['family']}")
        w.write(m)
    w.close()

    # ---- 3s rDock score-only ----
    subprocess.run(['rbdock', '-r', os.path.join(PROJ, RDOCK_SYS), '-p', os.path.join(PROJ, SCORE_ONLY),
                    '-i', '/tmp/opencode/chainvar/var.sdf', '-o', '/tmp/opencode/chainvar/score',
                    '-n', '1', '-s', '42', '-H'], cwd='case_study/kras_g4/ber102_rdock',
                   env={**os.environ, 'RBT_ROOT': RBT}, capture_output=True)
    scores = {}
    for m in Chem.SDMolSupplier('/tmp/opencode/chainvar/score.sd', removeHs=False):
        if m is not None and m.HasProp('Name') and m.HasProp('SCORE.INTER'):
            try: scores[m.GetProp('Name')] = float(m.GetProp('SCORE.INTER'))
            except Exception: pass
    for v in keep:
        v['rdock_INTER'] = scores.get(f"var{v['variant_id']:04d}_f{v['family']}")
    json.dump(keep, open(f'{OUT}/chain_variants.json', 'w'), indent=1)
    print(f"[3s] scored {sum(1 for v in keep if v['rdock_INTER'] is not None)}/{len(keep)}", flush=True)

    # ---- 4 port matching + assembly ----
    from scipy.spatial import cKDTree
    varmols = {}
    for m in Chem.SDMolSupplier('/tmp/opencode/chainvar/var.sdf', removeHs=True):
        if m is not None:
            varmols[int(m.GetProp('_Name').split('_')[0].replace('var',''))] = m
    recs = []
    for v in keep:
        vm = varmols[v['variant_id']]
        for p in port_db.get(v['family'], []):
            pa = vm.GetAtomWithIdx(p['frag_atom'])
            if pa.GetSymbol() == 'N' and pa.GetIsAromatic() and pa.GetTotalNumHs() == 0: continue
            recs.append({'vid': v['variant_id'], 'family': v['family'], 'pa': p['frag_atom'],
                         'elem': p['ext_elem'], 'g': np.array(p['ext_xyz']),
                         'rd': v['rdock_INTER'], 'smi': v['smiles']})
    cands = []
    for elem in ('C', 'N', 'O', 'S'):
        pts = [x for x in recs if x['elem'] == elem]
        if len(pts) < 2: continue
        arr = np.array([x['g'] for x in pts])
        for i, j in cKDTree(arr).query_pairs(1.0):
            if pts[i]['vid'] != pts[j]['vid']: cands.append((pts[i], pts[j]))
    print(f'[4] candidates: {len(cands)}', flush=True)
    def check_assemble(a, b):
        A, B = varmols[a['vid']], varmols[b['vid']]
        ia, jb = a['pa'], b['pa']
        na, nb = A.GetNumAtoms(), B.GetNumAtoms()
        pA, pB = A.GetConformer().GetPositions(), B.GetConformer().GetPositions()
        mid = (a['g'] + b['g']) / 2
        d2 = np.linalg.norm(pA[:,None]-pB[None],axis=-1)
        dd = d2.copy(); dd[ia, jb] = 99
        if dd.min() < 2.5: return None
        allat = np.vstack([pA, pB])
        dm = np.linalg.norm(allat - mid, axis=1)
        keepm = np.ones(na+nb, bool); keepm[ia] = False; keepm[na+jb] = False
        if dm[keepm].min() < 1.6: return None
        if np.linalg.norm(pocket[:,None]-allat[None],axis=-1).min() < 2.55: return None
        if np.linalg.norm(pocket - mid, axis=1).min() < 2.55: return None
        if not (1.2 <= dm[ia] <= 1.9 and 1.2 <= dm[na+jb] <= 1.9): return None
        rw = Chem.RWMol(); conf = Chem.Conformer(na+nb+1)
        for src in (A, B):
            for at in src.GetAtoms():
                x = Chem.Atom(at.GetSymbol())
                if at.GetIsAromatic(): x.SetIsAromatic(True)
                if at.GetNumExplicitHs():
                    x.SetNoImplicit(True); x.SetNumExplicitHs(at.GetNumExplicitHs()); x.SetNoImplicit(False)
                rw.AddAtom(x)
        rw.AddAtom(Chem.Atom({'C':6,'N':7,'O':8,'S':16}[a['elem']]))
        for t in range(na): conf.SetAtomPosition(t, Point3D(*pA[t]))
        for t in range(nb): conf.SetAtomPosition(na+t, Point3D(*pB[t]))
        conf.SetAtomPosition(na+nb, Point3D(*mid))
        for src, off in ((A,0),(B,na)):
            for bd in src.GetBonds():
                rw.AddBond(off+bd.GetBeginAtomIdx(), off+bd.GetEndAtomIdx(), bd.GetBondType())
        rw.AddBond(ia, na+nb, Chem.BondType.SINGLE)
        rw.AddBond(na+jb, na+nb, Chem.BondType.SINGLE)
        mol = rw.GetMol(); mol.AddConformer(conf, assignId=True)
        try:
            Chem.SanitizeMol(mol); return mol
        except Exception:
            return None
    feasible = []; seen = set()
    for a, b in cands:
        key = (min(a['vid'], b['vid']), max(a['vid'], b['vid']), a['pa'], b['pa'], a['elem'])
        if key in seen: continue
        seen.add(key)
        mol = check_assemble(a, b)
        if mol is None: continue
        feasible.append({'vids': (a['vid'], b['vid']), 'elem': a['elem'],
                         'sum_rdock': a['rd'] + b['rd'], 'mol': mol, 'fams': (a['family'], b['family'])})
    feasible.sort(key=lambda x: x['sum_rdock'])
    uniq = {}
    for f in feasible:
        smi = Chem.MolToSmiles(f['mol'])
        if smi not in uniq: uniq[smi] = f
    nt2 = sum(1 for s in uniq if s == T2)
    print(f'[4] assemblies: {len(feasible)} | unique: {len(uniq)} | EXACT t2: {nt2}', flush=True)
    w = Chem.SDWriter(f'{OUT}/chain_products.sdf')
    meta = []
    for i, (smi, f) in enumerate(uniq.items(), 1):
        m = Chem.Mol(f['mol']); m.SetProp('_Name', f'cp_{i:04d}')
        m.SetProp('sum_rdock', f"{f['sum_rdock']:.2f}")
        w.write(m)
        meta.append({'rank': i, 'smiles': smi, 'sum_rdock': round(f['sum_rdock'], 2),
                     'fams': f['fams'], 'elem': f['elem']})
    w.close()
    json.dump(meta, open(f'{OUT}/chain_products.json', 'w'), indent=1)
    if nt2 == 0:
        btrecs = [x for x in recs if x['smi'] == BT and x['elem'] == 'C']
        info = []
        for i in range(len(btrecs)):
            for j in range(i+1, len(btrecs)):
                d = float(np.linalg.norm(btrecs[i]['g'] - btrecs[j]['g']))
                if d < 1.5: info.append(round(d, 2))
        stop(f'no t2 among {len(uniq)} products; BT ghost dists(<1.5A): {info[:10]}')

    # ---- 5 filter + minimise ----
    sm_med = None
    recs5 = []
    for r in meta:
        m2 = Chem.MolFromSmiles(r['smiles'])
        q = QED.qed(m2); sa = sascorer.calculateScore(m2)
        recs5.append({**r, 'QED': round(q, 3), 'SAS': round(sa, 2)})
    sm_med = float(np.median([x['SAS'] for x in recs5]))
    surv = [x for x in recs5 if x['QED'] >= 0.5 and x['SAS'] <= sm_med]
    int2 = sum(1 for x in surv if x['smiles'] == T2)
    print(f'[5] filter QED>=0.5 & SAS<={sm_med:.2f}: {len(surv)} | t2 IN: {int2 > 0}', flush=True)
    if int2 == 0:
        t2r = next(x for x in recs5 if x['smiles'] == T2)
        stop(f"t2 failed filter: QED {t2r['QED']} SAS {t2r['SAS']} vs median {sm_med:.2f}")
    json.dump(surv, open(f'{OUT}/chain_finalset.json', 'w'), indent=1)
    prods = {m.GetProp('_Name'): m for m in Chem.SDMolSupplier(f'{OUT}/chain_products.sdf', removeHs=True) if m}
    os.makedirs('case_study/kras_g4/rdock_r4/chainfin', exist_ok=True)
    w = Chem.SDWriter('case_study/kras_g4/rdock_r4/chainfin/in.sdf')
    for r in surv:
        m = prods[f"cp_{r['rank']:04d}"]
        mh = Chem.AddHs(m, addCoords=True); mh.SetProp('_Name', f"final_{r['rank']:04d}")
        w.write(mh)
    w.close()
    if not args.skip_minimise:
        for s in (42, 142, 242):
            subprocess.run(['rbdock', '-r', os.path.join(PROJ, RDOCK_SYS),
                            '-p', os.path.join(RBT, 'data/scripts/minimise.prm'),
                            '-i', os.path.join(PROJ, 'case_study/kras_g4/rdock_r4/chainfin/in.sdf'),
                            '-o', os.path.join(PROJ, f'case_study/kras_g4/rdock_r4/chainfin/out_s{s}'),
                            '-n', '5', '-s', str(s), '-H'],
                           cwd='case_study/kras_g4/ber102_rdock',
                           env={**os.environ, 'RBT_ROOT': RBT}, capture_output=True)
        ref = {m.GetProp('_Name'): m for m in Chem.SDMolSupplier('case_study/kras_g4/rdock_r4/chainfin/in.sdf', removeHs=True) if m}
        def heavy_rmsd(a, b):
            mt = a.GetSubstructMatch(b)
            if not mt:
                mt2 = b.GetSubstructMatch(a)
                if not mt2: return None
                mt = tuple(mt2.index(i) for i in range(a.GetNumAtoms()))
            pa = a.GetConformer().GetPositions()[list(mt)]
            pb = b.GetConformer().GetPositions()
            return float(np.sqrt(((pa-pb)**2).sum(1).mean()))
        best = {}
        for s in (42, 142, 242):
            for m in Chem.SDMolSupplier(f'case_study/kras_g4/rdock_r4/chainfin/out_s{s}.sd', removeHs=True):
                if m is None or not m.HasProp('Name'): continue
                nm = m.GetProp('Name')
                if nm not in ref: continue
                try: inter = float(m.GetProp('SCORE.INTER'))
                except Exception: continue
                r = heavy_rmsd(m, ref[nm])
                if r is None or r > 2.0: continue
                if nm not in best or inter < best[nm]: best[nm] = inter
        out5 = []
        for r in surv:
            r2 = dict(r); r2['ms_INTER'] = best.get(f"final_{r['rank']:04d}")
            out5.append(r2)
        have = [x for x in out5 if x.get('ms_INTER') is not None]
        t2 = next((x for x in have if x['smiles'] == T2), None)
        if t2 is None and any(x['smiles'] == T2 for x in out5):
            stop('t2 lost at minimise: all poses drifted >2.0 A')
        rk = sorted([x['ms_INTER'] for x in have]).index(t2['ms_INTER']) + 1 if t2 else -1
        print(f"[5c] minimise: {len(have)} resolved | t2 rank #{rk} (INTER {t2['ms_INTER'] if t2 else '-'})", flush=True)
        json.dump(out5, open(f'{OUT}/chain_finalset_ms.json', 'w'), indent=1)

    # ---- 6 RNAsmol ----
    if not args.skip_rnasmol:
        import torch
        FMN = os.path.join(PROJ, 'case_study/fmn')
        for p_ in (FMN + '/RNAsmol/rnasmol', FMN + '/RNAsmol', FMN):
            if p_ not in sys.path: sys.path.insert(0, p_)
        from rnasmol_score import build_data_from_mol, load_model
        from rnasmol_score_gpu import score_data
        MODELS = ['pdb_bothaug_molperturbation.pt', 'pdb_bothaug_netperturbation.pt',
                  'pdb_bothaug_rnaperturbation.pt', 'robin_bothaug_molperturbation.pt',
                  'robin_bothaug_netperturbation.pt', 'robin_bothaug_rnaperturbation.pt']
        SHORT = ['pdb_mol', 'pdb_net', 'pdb_rna', 'robin_mol', 'robin_net', 'robin_rna']
        base = json.load(open(f'{OUT}/chain_finalset_ms.json')) if not args.skip_minimise else json.load(open(f'{OUT}/chain_finalset.json'))
        recs6 = [x for x in base if x.get('ms_INTER') is not None or args.skip_minimise]
        graphs = []
        for r in recs6:
            graphs.append(build_data_from_mol(Chem.MolFromSmiles(r['smiles']), SEQ))
        device = torch.device('cuda')
        res = [[] for _ in graphs]
        for fn in MODELS:
            model = load_model(os.path.join(FMN, 'RNAsmol', 'saved_model', fn), device)
            _ = score_data(model, graphs[0], device); torch.cuda.synchronize()
            for i, g in enumerate(graphs):
                p, _, _ = score_data(model, g, device)
                res[i].append(float(p))
            del model; torch.cuda.empty_cache()
        M = np.array(res)
        out6 = []
        for i, r in enumerate(recs6):
            r2 = dict(r)
            for j, s in enumerate(SHORT): r2['P_' + s] = round(M[i, j], 4)
            out6.append(r2)
        json.dump(out6, open(f'{OUT}/chain_rnasmol6.json', 'w'), indent=1)
        alive = [j for j in range(6) if M[:, j].std() > 1e-6]
        dead = [SHORT[j] for j in range(6) if j not in alive]
        print(f'[6] RNAsmol done | dead dims: {dead}', flush=True)

        # ---- 7 tiers ----
        from scipy.stats import rankdata
        SHORTA = [SHORT[j] for j in alive]
        P = np.array([[x['P_' + s] for s in SHORTA] for x in out6])
        I = np.array([x['ms_INTER'] for x in out6])
        n = len(out6)
        Rp = np.vstack([rankdata(-P[:, j]) for j in range(len(SHORTA))])
        Ri = rankdata(I)
        cut = n / 2
        mvotes = (Rp <= cut).sum(0)
        dvote = (Ri <= cut).astype(int)
        nmaj = len(SHORTA) // 2 + 1
        tier1 = (mvotes >= nmaj) & (dvote == 1)
        t2idx = next((i for i, x in enumerate(out6) if x['smiles'] == T2), None)
        print(f'[7] Tier-1: {tier1.sum()}/{n} (model>={nmaj}/{len(SHORTA)} & rdock top-half) | t2 IN: {bool(t2idx is not None and tier1[t2idx])}', flush=True)
        if t2idx is not None and not tier1[t2idx]:
            stop(f"t2 not Tier-1: model votes {mvotes[t2idx]}/{len(SHORTA)}, rdock rank {int(Ri[t2idx])}/{n}")
        R = np.vstack([Rp, Ri])
        minrank = R.min(0)
        t1idx = np.where(tier1)[0]
        order = sorted(t1idx, key=lambda i: minrank[i])
        out7 = []
        for i, x in enumerate(out6):
            x2 = dict(x); x2['model_votes'] = int(mvotes[i]); x2['rdock_rank'] = int(Ri[i])
            x2['tier1'] = bool(tier1[i]); x2['minrank'] = int(minrank[i])
            out7.append(x2)
        json.dump(out7, open(f'{OUT}/chain_tiers.json', 'w'), indent=1)
        t2pos = [k for k, i in enumerate(order, 1) if i == t2idx]
        print(f'[7] t2 MinRank position in Tier-1: #{t2pos[0] if t2pos else "-"}/{len(order)}', flush=True)
        print('\n[CHAIN COMPLETE] t2 survived to the end.', flush=True)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""PubChem 2-D similarity search for the round2 top-20 candidates.

For each molecule in top_binders.sdf (top-20 by combined score):
  1. Canonical SMILES -> URL-encode
  2. Query PubChem PUG-REST fastsimilarity_2d at Threshold=85, MaxRecords=10.
     If no hits, fall back to Threshold=70.
  3. Compute exact Tanimoto (Morgan FP, 2048 bits, radius 2) between the query
     and every returned hit for accurate reporting.
  4. Record #hits, top CID, top name (first synonym), top Tanimoto.

Output: casestudy/fmn/screening_round2/pubchem_top20.tsv
"""
import os, sys, time, json, urllib.parse, urllib.request, urllib.error
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import rdkit.RDLogger
rdkit.RDLogger.DisableLog('rdApp.*')

SDF = 'casestudy/fmn/screening_round2/top_binders.sdf'  # top-100, we take top-20
N_TOP = 20
OUT = 'casestudy/fmn/screening_round2/pubchem_top20.tsv'
UA  = {'User-Agent': 'rna2smol/1.0 (academic research)'}


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def pubchem_query(smiles, threshold, max_records=10, timeout=20, retries=2):
    """Return list of CIDs from fastsimilarity_2d, or (None, err) on error."""
    enc = urllib.parse.quote(smiles, safe='')
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
           f"fastsimilarity_2d/smiles/{enc}/JSON?Threshold={threshold}"
           f"&MaxRecords={max_records}")
    last_err = None
    data = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode('utf-8', 'replace'))
            break
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code in (500, 502, 503, 504) and attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            return None, last_err
        except Exception as e:
            last_err = repr(e)[:80]
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            return None, last_err
    cids = []
    for c in (data or {}).get('PC_Compounds', []):
        try:
            cids.append(c['id']['id']['cid'])
        except Exception:
            pass
    return cids, None


def fetch_properties(cids, timeout=30):
    """Batch-fetch Title + CanonicalSMILES for a list of CIDs."""
    if not cids:
        return {}
    cid_str = ','.join(str(c) for c in cids)
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}"
           f"/property/Title,CanonicalSMILES/JSON")
    req = urllib.request.Request(url, headers=UA)
    out = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode('utf-8', 'replace'))
        for p in data.get('PropertyTable', {}).get('Properties', []):
            smi = p.get('CanonicalSMILES') or p.get('ConnectivitySMILES') or ''
            out[p['CID']] = {'title': p.get('Title', ''), 'smiles': smi}
    except Exception as e:
        print(f"  [warn] property fetch failed: {e!r}", flush=True)
    return out


def load_done_ranks(out_path):
    """Return set of rank ints already in the TSV (for resume)."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as fh:
        next(fh, None)  # header
        for ln in fh:
            f = ln.rstrip('\n').split('\t')
            if f and f[0].isdigit():
                done.add(int(f[0]))
    return done


def main():
    suppl = Chem.SDMolSupplier(SDF, removeHs=False, sanitize=False)
    mols = []
    for i, m in enumerate(suppl):
        if m is None:
            continue
        if i >= N_TOP:
            break
        try:
            ms = Chem.Mol(m)
            Chem.SanitizeMol(ms)
        except Exception:
            ms = m
        mols.append((i, ms))
    print(f"Loaded top-{len(mols)} candidates from {SDF}", flush=True)

    done = load_done_ranks(OUT)
    if done:
        print(f"RESUME: {len(done)} ranks already in TSV: {sorted(done)}", flush=True)
    # append mode if resuming, else write fresh header
    mode = 'a' if done else 'w'
    fout = open(OUT, mode)
    if not done:
        fout.write("rank\tgen_idx\tquery_smiles\tn_hits\ttop_cid\ttop_name\t"
                   "top_tanimoto\ttop_hit_smiles\tnote\n")
    fout.flush()
    rows = []
    for rank, mol in mols:
        if (rank + 1) in done:
            continue
        d = mol.GetPropsAsDict()
        gen_idx = d.get('gen_idx', '')
        qsmi_iso = Chem.MolToSmiles(mol)
        mol_nostereo = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol_nostereo)
        qsmi_nostereo = Chem.MolToSmiles(mol_nostereo)
        qfp = fp(mol)
        variants = [('isomeric', qsmi_iso), ('nostereo', qsmi_nostereo)]

        used_thr, used_var, cids, err = None, None, None, None
        for thr in (85, 70):
            for vname, vsmi in variants:
                c, e = pubchem_query(vsmi, thr)
                if c:
                    cids, used_thr, used_var, err = c, thr, vname, None
                    break
                err = e
            if cids:
                break
            time.sleep(0.2)

        scored = []
        if cids:
            props = fetch_properties(cids)
            for cid in cids:
                p = props.get(cid, {})
                hsmi = p.get('smiles', '')
                hmol = Chem.MolFromSmiles(hsmi) if hsmi else None
                t = DataStructs.TanimotoSimilarity(qfp, fp(hmol)) if hmol else None
                scored.append({'cid': cid, 'title': p.get('title', ''),
                               'smiles': hsmi, 'tanimoto': t})
            scored.sort(key=lambda x: (x['tanimoto'] is not None,
                                       x['tanimoto'] or 0.0), reverse=True)

        n_hits = len(scored)
        top = scored[0] if scored else {}
        top_cid    = top.get('cid', '') or ''
        top_name   = top.get('title', '') or ''
        top_tani   = f"{top.get('tanimoto'):.3f}" if top.get('tanimoto') is not None else ''
        top_smiles = top.get('smiles', '') or ''
        note = err if (n_hits == 0 and err) else \
               ('no_hits' if n_hits == 0 else f'thr={used_thr},var={used_var}')
        row = (rank+1, gen_idx, qsmi_iso, n_hits, top_cid, top_name,
               top_tani, top_smiles, note)
        rows.append(row)
        fout.write("\t".join(str(x) for x in row) + "\n")
        fout.flush()
        print(f"[{rank+1:>2}] gen={gen_idx} hits={n_hits:>2} thr={used_thr} "
              f"var={used_var} topCID={top_cid} topT={top_tani} "
              f"{str(top_name)[:42]} | {note}", flush=True)
        time.sleep(0.3)

    fout.close()
    print(f"\nWrote {OUT}")

    with_hits = sum(1 for r in rows if r[3] > 0)
    high_conf = sum(1 for r in rows if r[6] and float(r[6]) >= 0.85)
    med_conf  = sum(1 for r in rows if r[6] and 0.4 <= float(r[6]) < 0.85)
    print(f"\n=== SUMMARY: {with_hits}/{len(rows)} top-20 have >=1 PubChem hit ===")
    print(f"=== {high_conf}/{len(rows)} have Tanimoto >= 0.85 (strong match) ===")
    print(f"=== {med_conf}/{len(rows)} have 0.40 <= Tanimoto < 0.85 (partial match) ===")


if __name__ == '__main__':
    main()

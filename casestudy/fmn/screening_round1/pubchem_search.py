#!/usr/bin/env python3
"""PubChem 2-D similarity search for the 16 neutral candidates.

For each molecule in neutral_16_poses.sdf:
  1. Canonical SMILES -> URL-encode
  2. Query PubChem PUG-REST fastsimilarity_2d at Threshold=85 (Tanimoto>=0.85),
     MaxRecords=5. If no hits, fall back to Threshold=70.
  3. Compute exact Tanimoto (Morgan FP, 2048 bits, radius 2) between the query
     and every returned hit for accurate reporting.
  4. Record #hits, top CID, top name (first synonym), top Tanimoto.

Output: casestudy/fmn/screening_round1/pubchem_search.tsv
"""
import os, sys, time, json, urllib.parse, urllib.request, urllib.error
sys.path.insert(0, '/home/data/users/gaoletian/rna2smol')
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import rdkit.RDLogger
rdkit.RDLogger.DisableLog('rdApp.*')

SDF = 'casestudy/fmn/screening_round1/neutral_16_poses.sdf'
OUT = 'casestudy/fmn/screening_round1/pubchem_search.tsv'
UA  = {'User-Agent': 'rna2smol/1.0 (academic research)'}


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def pubchem_query(smiles, threshold, max_records=5, timeout=14, retries=0):
    """Return list of CIDs from fastsimilarity_2d, or (None, err) on error."""
    enc = urllib.parse.quote(smiles, safe='')
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
           f"fastsimilarity_2d/smiles/{enc}/JSON?Threshold={threshold}&MaxRecords={max_records}")
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
    """Batch-fetch Title + Canonical SMILES for a list of CIDs.
    Returns dict {cid: {'title':, 'smiles':}}."""
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
            # PubChem may return the SMILES under 'CanonicalSMILES' or
            # 'ConnectivitySMILES' depending on service version.
            smi = p.get('CanonicalSMILES') or p.get('ConnectivitySMILES') or ''
            out[p['CID']] = {'title': p.get('Title', ''), 'smiles': smi}
    except Exception:
        pass
    return out


def main():
    suppl = Chem.SDMolSupplier(SDF)
    mols = [(i, m) for i, m in enumerate(suppl) if m is not None]
    print(f"Loaded {len(mols)} neutral candidates from {SDF}")

    # write header immediately, flush after each row so partial progress survives
    fout = open(OUT, 'w')
    fout.write("idx\tquery_smiles\tn_hits\ttop_cid\ttop_name\ttop_tanimoto\ttop_hit_smiles\tnote\n")
    fout.flush()
    rows = []
    for idx, mol in mols:
        qsmi_iso = Chem.MolToSmiles(mol)
        mol_nostereo = Chem.Mol(mol)            # copy
        Chem.RemoveStereochemistry(mol_nostereo)  # in-place
        qsmi_nostereo = Chem.MolToSmiles(mol_nostereo)
        qfp = fp(mol)
        # Ordered attempt list: cheapest/most-likely first. retries handled inside.
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

        # Fetch titles + canonical SMILES for the returned CIDs
        scored = []
        if cids:
            props = fetch_properties(cids)
            for cid in cids:
                p = props.get(cid, {})
                hsmi = p.get('smiles', '')
                hmol = Chem.MolFromSmiles(hsmi) if hsmi else None
                t = DataStructs.TanimotoSimilarity(qfp, fp(hmol)) if hmol else None
                scored.append({'cid': cid, 'title': p.get('title', ''), 'smiles': hsmi, 'tanimoto': t})
            scored.sort(key=lambda x: (x['tanimoto'] is not None, x['tanimoto'] or 0.0), reverse=True)

        n_hits = len(scored)
        top = scored[0] if scored else {}
        top_cid   = top.get('cid', '') or ''
        top_name  = top.get('title', '') or ''
        top_tani  = f"{top.get('tanimoto'):.3f}" if top.get('tanimoto') is not None else ''
        top_smiles = top.get('smiles', '') or ''
        note = err if (n_hits == 0 and err) else ('no_hits' if n_hits == 0 else f'thr={used_thr},var={used_var}')
        rows.append((idx, qsmi_iso, n_hits, top_cid, top_name, top_tani, top_smiles, note))
        fout.write("\t".join(str(x) for x in rows[-1]) + "\n"); fout.flush()
        print(f"[{idx:>2}] hits={n_hits:>2} thr={used_thr} var={used_var} topCID={top_cid} topT={top_tani} {str(top_name)[:42]} | {note}", flush=True)
        time.sleep(0.3)

    fout.close()
    print(f"\nWrote {OUT}")

    # Summary
    with_hits = sum(1 for r in rows if r[2] > 0)
    high_conf = sum(1 for r in rows if r[5] and float(r[5]) >= 0.85)
    print(f"\n=== SUMMARY: {with_hits}/{len(rows)} neutral candidates have >=1 PubChem hit ===")
    print(f"=== {high_conf}/{len(rows)} have a hit at Tanimoto >= 0.85 ===")


if __name__ == '__main__':
    main()

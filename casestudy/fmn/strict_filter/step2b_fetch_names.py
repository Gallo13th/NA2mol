"""Step 2b: Fetch names / InChIKey / formula for all unique CIDs in hits.
PubChem /property/ endpoint accepts up to 100 CIDs at a time.
"""
import json
import time
import urllib.request
import urllib.parse
import sys

PROG = "casestudy/fmn/strict_filter/pubchem_progress.json"
OUT = "casestudy/fmn/strict_filter/pubchem_cid_info.json"

with open(PROG) as f:
    r = json.load(f)

all_cids = set()
for v in r.values():
    if v.get("status") == "hit":
        for c in v["cids"]:
            all_cids.add(c)
all_cids = sorted(all_cids)
print(f"unique CIDs: {len(all_cids)}", flush=True)

# Properties: InChIKey, MolecularFormula, MolecularWeight
# Title via /synonyms/TXT (top one)
url_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
info = {}
BATCH = 100

def fetch_batch(cids):
    url = url_base + ",".join(str(c) for c in cids) + "/property/MolecularFormula,MolecularWeight,InChIKey/JSON"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "rna2smol-research/1.0 (academic)")
    backoff = 1
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            return obj.get("PropertyTable", {}).get("Properties", [])
        except Exception as e:
            sys.stderr.write(f"  err {e}; backoff {backoff}\n")
            time.sleep(backoff); backoff = min(backoff*2, 30)
    return []

for i in range(0, len(all_cids), BATCH):
    batch = all_cids[i:i+BATCH]
    props = fetch_batch(batch)
    for p in props:
        cid = p.get("CID")
        info[cid] = {
            "mf": p.get("MolecularFormula"),
            "mw": p.get("MolecularWeight"),
            "inchikey": p.get("InChIKey"),
            "name": None,  # will fill next
        }
    print(f"  [{i+len(batch)}/{len(all_cids)}] got {len(props)} props", flush=True)
    time.sleep(0.3)

# Now fetch a top synonym for each CID (just top hit CIDs to save calls - 135 total)
# Use /synonyms/TXT and take first line
top_only = sorted(set(v["top_cid"] for v in r.values() if v.get("status") == "hit"))
print(f"\nfetching top synonym for {len(top_only)} top-hit CIDs", flush=True)

def fetch_top_synonym(cid):
    url = url_base + str(cid) + "/synonyms/TXT"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "rna2smol-research/1.0 (academic)")
    backoff = 1
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                txt = resp.read().decode("utf-8", errors="ignore")
            # first line is usually the IUPAC name or preferred name
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                # skip super-long IUPAC names if there's a shorter one
                if len(line) > 200:
                    continue
                return line
            return txt.splitlines()[0].strip() if txt.strip() else None
        except Exception as e:
            time.sleep(backoff); backoff = min(backoff*2, 30)
    return None

for i, cid in enumerate(top_only):
    name = fetch_top_synonym(cid)
    if cid in info:
        info[cid]["name"] = name
    else:
        info[cid] = {"name": name}
    if (i+1) % 25 == 0:
        print(f"  [{i+1}/{len(top_only)}] last: CID={cid} name={(name or '')[:60]}", flush=True)
        # incremental save
        with open(OUT, "w") as f:
            json.dump(info, f, indent=2)
    time.sleep(0.2)

with open(OUT, "w") as f:
    json.dump(info, f, indent=2)

named = sum(1 for v in info.values() if v.get("name"))
print(f"\nDONE. {named}/{len(info)} CIDs have a name")

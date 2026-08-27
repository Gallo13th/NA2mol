"""Step 2c: resume name-fetching for remaining top-hit CIDs."""
import json, time, urllib.request, sys, os.path as op

PROG = "casestudy/fmn/strict_filter/pubchem_progress.json"
INFO = "casestudy/fmn/strict_filter/pubchem_cid_info.json"

with open(PROG) as f:
    r = json.load(f)
with open(INFO) as f:
    info = json.load(f)

top_cids = sorted(set(v["top_cid"] for v in r.values() if v.get("status") == "hit"))
need = [c for c in top_cids if not info.get(c, {}).get("name")]
print(f"need names for {len(need)} CIDs", flush=True)

url_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"

def fetch_top_synonym(cid):
    url = url_base + str(cid) + "/synonyms/TXT"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "rna2smol-research/1.0 (academic)")
    backoff = 1
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                txt = resp.read().decode("utf-8", errors="ignore")
            for line in txt.splitlines():
                line = line.strip()
                if not line or len(line) > 200:
                    continue
                return line
            return txt.splitlines()[0].strip() if txt.strip() else None
        except Exception as e:
            time.sleep(backoff); backoff = min(backoff*2, 30)
    return None

t0 = time.time()
for i, cid in enumerate(need):
    name = fetch_top_synonym(cid)
    info.setdefault(str(cid), {})["name"] = name
    if (i+1) % 5 == 0:
        with open(INFO, "w") as f:
            json.dump(info, f, indent=2)
        elapsed = time.time() - t0
        rate = (i+1) / elapsed if elapsed > 0 else 0
        print(f"  [{i+1}/{len(need)}] CID={cid} name={(name or '')[:55]} elapsed={elapsed:.0f}s rate={rate:.2f}/s", flush=True)
    time.sleep(0.15)

with open(INFO, "w") as f:
    json.dump(info, f, indent=2)
named = sum(1 for v in info.values() if v.get("name"))
print(f"\nDONE. {named}/{len(info)} CIDs named")

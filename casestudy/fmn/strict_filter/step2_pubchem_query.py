"""Step 2: Query PubChem fastsimilarity_2d (Threshold=90) for SAS>=3 subset.

Resumable: saves results to pubchem_hits.json after every batch.
Rate-limit safe: 5 req/sec, exponential backoff on 503/timeout.
"""
import json
import time
import os.path as op
import urllib.parse
import urllib.request
import urllib.error
import sys

IN_JSON = "casestudy/fmn/strict_filter/sas_fail_idx.json"
SAS_JSON = "casestudy/fmn/strict_filter/sas_scores.json"
OUT_JSON = "casestudy/fmn/strict_filter/pubchem_hits.json"
PROGRESS_JSON = "casestudy/fmn/strict_filter/pubchem_progress.json"

THRESHOLD = 90   # PubChem internal Tanimoto >= 0.9
MAX_RECORDS = 3  # only need top-3 CIDs to know "hit/no-hit"
SLEEP_BETWEEN = 0.25   # 4 req/sec - safe under 5/sec limit
BATCH_SAVE = 25        # save progress every N calls
TIMEOUT = 30

with open(IN_JSON) as f:
    fail_idx = set(json.load(f))
with open(SAS_JSON) as f:
    all_records = json.load(f)
to_query = [r for r in all_records if r["idx"] in fail_idx]
print(f"will query PubChem for {len(to_query)} molecules (SAS>=3)")

# resume support
if op.exists(PROGRESS_JSON):
    with open(PROGRESS_JSON) as f:
        results = json.load(f)
    print(f"resuming: {len(results)} already done")
else:
    results = {}

API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/JSON"


def query_one(smiles):
    """Return list of CIDs (empty if no hit / fault)."""
    url = f"{API}?Threshold={THRESHOLD}&MaxRecords={MAX_RECORDS}"
    data = urllib.parse.urlencode({"smiles": smiles}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("User-Agent", "rna2smol-research/1.0 (academic)")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
    obj = json.loads(raw)
    if "PC_Compounds" in obj:
        cids = []
        for c in obj["PC_Compounds"]:
            try:
                cids.append(int(c["id"]["id"]["cid"]))
            except (KeyError, TypeError, ValueError):
                pass
        return cids
    return []  # no hits or fault


def query_with_retry(smiles, idx):
    backoff = 1
    for attempt in range(7):
        try:
            return query_one(smiles)
        except urllib.error.HTTPError as e:
            if e.code == 503:
                sys.stderr.write(f"  idx={idx} 503 (rate-limit). backoff {backoff}s\n")
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)
                continue
            elif e.code == 404:
                return []  # no hits -> 404 sometimes returned
            elif e.code == 400:
                sys.stderr.write(f"  idx={idx} 400 bad request for SMILES={smiles[:60]}\n")
                return []
            else:
                sys.stderr.write(f"  idx={idx} HTTPError {e.code}: {e.reason}\n")
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)
                continue
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
            sys.stderr.write(f"  idx={idx} net err: {e}. backoff {backoff}s\n")
            time.sleep(backoff)
            backoff = min(backoff * 2, 64)
            continue
        except Exception as e:
            sys.stderr.write(f"  idx={idx} err: {type(e).__name__}: {e}\n")
            time.sleep(backoff)
            backoff = min(backoff * 2, 64)
            continue
    sys.stderr.write(f"  idx={idx} FAILED after 7 retries\n")
    return None  # indeterminate


t0 = time.time()
done_in_run = 0
hits_count = 0
for rec in to_query:
    idx = rec["idx"]
    if str(idx) in results:
        continue
    cids = query_with_retry(rec["SMILES"], idx)
    if cids is None:
        results[str(idx)] = {"status": "error", "cids": []}
    elif len(cids) == 0:
        results[str(idx)] = {"status": "no_hit", "cids": []}
    else:
        results[str(idx)] = {"status": "hit", "cids": cids, "top_cid": cids[0]}
        hits_count += 1
    done_in_run += 1
    if done_in_run % BATCH_SAVE == 0:
        with open(PROGRESS_JSON, "w") as f:
            json.dump(results, f)
        elapsed = time.time() - t0
        rate = done_in_run / elapsed if elapsed > 0 else 0
        print(f"  [{done_in_run}/{len(to_query)}] hits={hits_count} elapsed={elapsed:.0f}s rate={rate:.2f}/s")
    time.sleep(SLEEP_BETWEEN)

# final save
with open(PROGRESS_JSON, "w") as f:
    json.dump(results, f)
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - t0
total_hits = sum(1 for v in results.values() if v.get("status") == "hit")
total_nohit = sum(1 for v in results.values() if v.get("status") == "no_hit")
total_err = sum(1 for v in results.values() if v.get("status") == "error")
print(f"\n=== DONE ===")
print(f"  queried:  {len(results)}")
print(f"  hits:     {total_hits}")
print(f"  no_hit:   {total_nohit}")
print(f"  errors:   {total_err}")
print(f"  elapsed:  {elapsed:.0f}s")

"""Step 2 (parallel): Query PubChem fastsimilarity_2d Threshold=90 for SAS>=3.

Resumable from pubchem_progress.json.
Concurrency=4 workers, rate-limited to ~4 req/sec (safe under PubChem 5/sec).
"""
import json
import time
import os.path as op
import urllib.parse
import urllib.request
import urllib.error
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

IN_JSON = "casestudy/fmn/strict_filter/sas_fail_idx.json"
SAS_JSON = "casestudy/fmn/strict_filter/sas_scores.json"
PROGRESS_JSON = "casestudy/fmn/strict_filter/pubchem_progress.json"
OUT_JSON = "casestudy/fmn/strict_filter/pubchem_hits.json"

THRESHOLD = 90
MAX_RECORDS = 3
TIMEOUT = 30
N_WORKERS = 4   # 4 concurrent requests, each request takes 3-5s = ~1 req/sec/worker = ~4 req/sec total
SAVE_EVERY = 50

with open(IN_JSON) as f:
    fail_idx = set(json.load(f))
with open(SAS_JSON) as f:
    all_records = json.load(f)
to_query = [r for r in all_records if r["idx"] in fail_idx]

if op.exists(PROGRESS_JSON):
    with open(PROGRESS_JSON) as f:
        results = json.load(f)
    print(f"resume: {len(results)} already done", flush=True)
else:
    results = {}

pending = [r for r in to_query if str(r["idx"]) not in results]
print(f"pending: {len(pending)}", flush=True)

API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/JSON"
lock = threading.Lock()
done = 0
hits = sum(1 for v in results.values() if v.get("status") == "hit")


def query_one(smiles):
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
    return []


def query_with_retry(smiles, idx):
    backoff = 1
    for _ in range(7):
        try:
            return query_one(smiles), None
        except urllib.error.HTTPError as e:
            if e.code == 503:
                time.sleep(backoff); backoff = min(backoff * 2, 64); continue
            if e.code in (404, 400):
                return [], f"http_{e.code}"
            time.sleep(backoff); backoff = min(backoff * 2, 64); continue
        except Exception as e:
            time.sleep(backoff); backoff = min(backoff * 2, 64); continue
    return None, "failed_7x"


def worker(rec):
    global done, hits
    cids, err = query_with_retry(rec["SMILES"], rec["idx"])
    if cids is None:
        res = {"status": "error", "cids": [], "err": err}
    elif len(cids) == 0:
        res = {"status": "no_hit", "cids": []}
        if err:
            res["err"] = err
    else:
        res = {"status": "hit", "cids": cids, "top_cid": cids[0]}
    with lock:
        results[str(rec["idx"])] = res
        done += 1
        if res["status"] == "hit":
            hits += 1
        if done % SAVE_EVERY == 0:
            with open(PROGRESS_JSON, "w") as f:
                json.dump(results, f)
            print(f"  [{done}/{len(pending)}] hit={hits} (last={rec['idx']} cids={cids[:2]})", flush=True)
    return res


t0 = time.time()
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = [ex.submit(worker, r) for r in pending]
    for f in as_completed(futs):
        try:
            f.result()
        except Exception as e:
            sys.stderr.write(f"  worker exc: {e}\n")

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

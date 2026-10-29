import json, os, subprocess, time

base = "https://raw.githubusercontent.com/hongli-ma/RNAsmol/main/"
d = json.load(open("tree.json"))
blobs = {t["path"]: t.get("size", 0) for t in d["tree"] if t["type"] == "blob"}

EXCLUDE_PREFIXES = ("scripts/robustness_test/",)
EXCLUDE_EXACT = {"scripts/robustness_test/pdb_molperturbation_rnaperturbation.zip"}
SKIP_PYC = "__pycache__/"

def local_size(p):
    dst = os.path.join("RNAsmol", p)
    return os.path.getsize(dst) if os.path.exists(dst) else -1

# Build todo = files that are missing OR wrong-size (partial)
todo = []
for p in sorted(blobs):
    if any(p.startswith(pre) for pre in EXCLUDE_PREFIXES): continue
    if p in EXCLUDE_EXACT: continue
    if SKIP_PYC in p: continue
    sz = local_size(p)
    if sz != blobs[p]:
        todo.append(p)

print(f"need fetch/fix: {len(todo)} files")
fails = []
for i, p in enumerate(todo, 1):
    dst = os.path.join("RNAsmol", p)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) != blobs[p]:
        os.remove(dst)
    ok = False
    for attempt in range(8):
        subprocess.run(["curl","-sSL","--connect-timeout","20","--max-time","120","-o",dst, base+p],
                       capture_output=True, text=True)
        if os.path.exists(dst) and os.path.getsize(dst) == blobs[p]:
            ok = True; break
        time.sleep(2)
    print(f"[{i}/{len(todo)}] {'OK' if ok else 'FAIL'} {p} ({os.path.getsize(dst) if os.path.exists(dst) else 0}/{blobs[p]})", flush=True)
    if not ok:
        fails.append(p)

print("\nfails:", fails)

# final verify across ALL tracked (non-excluded) blobs
bad = []
for p in sorted(blobs):
    if any(p.startswith(pre) for pre in EXCLUDE_PREFIXES): continue
    if p in EXCLUDE_EXACT: continue
    if SKIP_PYC in p: continue
    if local_size(p) != blobs[p]:
        bad.append(p)
print("size-mismatch after run:", bad)

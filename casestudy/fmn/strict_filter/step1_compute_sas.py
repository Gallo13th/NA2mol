"""Step 1: Compute SAS for all 678 molecules in final_binders.sdf."""
import json
import sys
import os.path as op

sys.path.insert(0, "/home/data/users/gaoletian/rna2smol/util")
from rdkit import Chem
from rdkit import RDLogger
from sascorer import calculateScore

RDLogger.DisableLog("rdApp.*")

IN_SDF = "casestudy/fmn/final_binders.sdf"
OUT_SDF = "casestudy/fmn/final_binders_sas.sdf"
OUT_JSON = "casestudy/fmn/strict_filter/sas_scores.json"

sup = Chem.SDMolSupplier(IN_SDF)
mols = [m for m in sup if m is not None]
print(f"loaded {len(mols)} mols")

writer = Chem.SDWriter(OUT_SDF)
records = []
for i, m in enumerate(mols):
    try:
        s = calculateScore(m)
    except Exception as e:
        print(f"SAS fail idx={i}: {e}")
        s = None
    if s is not None:
        m.SetProp("SAS", f"{s:.4f}")
        rec = {
            "idx": i,
            "SMILES": m.GetProp("canonical_smiles"),
            "SAS": round(s, 4),
            "rdock_INTER": m.GetProp("rdock_INTER"),
            "rdock_SCORE": m.GetProp("rdock_SCORE"),
            "rnasmol_p_binder": m.GetProp("rnasmol_p_binder"),
            "rnasmol_logit": m.GetProp("rnasmol_logit"),
            "gen_idx": m.GetProp("gen_idx"),
            "source_round": m.GetProp("source_round"),
            "rnasmol_rank": m.GetProp("rnasmol_rank"),
        }
        records.append(rec)
    writer.write(m)
writer.close()

with open(OUT_JSON, "w") as f:
    json.dump(records, f, indent=2)

sas_list = [r["SAS"] for r in records]
sas_list_sorted = sorted(sas_list)
n = len(sas_list_sorted)
import statistics
print(f"\n=== SAS distribution (n={n}) ===")
print(f"  min:    {sas_list_sorted[0]:.3f}")
print(f"  25%:    {sas_list_sorted[n//4]:.3f}")
print(f"  median: {statistics.median(sas_list):.3f}")
print(f"  75%:    {sas_list_sorted[3*n//4]:.3f}")
print(f"  max:    {sas_list_sorted[-1]:.3f}")
print(f"  mean:   {statistics.mean(sas_list):.3f}")
for thr in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    cnt = sum(1 for s in sas_list if s < thr)
    print(f"  SAS<{thr}: {cnt} ({100*cnt/n:.1f}%)")

# Save SAS-pass list (SAS<3) for next step
sas_pass = [r for r in records if r["SAS"] < 3.0]
sas_fail = [r for r in records if r["SAS"] >= 3.0]
print(f"\n  SAS<3 pass:  {len(sas_pass)}")
print(f"  SAS>=3 -> need PubChem check: {len(sas_fail)}")
with open("casestudy/fmn/strict_filter/sas_pass_idx.json", "w") as f:
    json.dump([r["idx"] for r in sas_pass], f)
with open("casestudy/fmn/strict_filter/sas_fail_idx.json", "w") as f:
    json.dump([r["idx"] for r in sas_fail], f)
print(f"\nwrote {OUT_SDF}")
print(f"wrote {OUT_JSON}")

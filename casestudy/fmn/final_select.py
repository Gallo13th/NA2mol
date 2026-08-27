"""Select final binders: P(binder)>=0.9 AND rdock INTER<-16.79 (better than FMN -16.79).

Merge round2 (docked_top300.sd) + round3 (docked_top500.sd).
Per molecule keep best pose (most negative INTER). Dedup by canonical SMILES.
Write docked 3D poses to casestudy/fmn/final_binders.sdf, sorted by INTER asc.
"""
import os
from rdkit import Chem
from rdkit.Chem import AllChem

ROUNDS = [
    ("round2", "casestudy/fmn/screening_round2/docked_top300.sd"),
    ("round3", "casestudy/fmn/screening_round3/docked_top500.sd"),
]
OUT_SDF = "casestudy/fmn/final_binders.sdf"
P_CUTOFF = 0.9
INTER_CUTO = -16.79  # strictly better than FMN (-16.79)


def get_prop(mol, name, cast=float, default=None):
    if mol.HasProp(name):
        v = mol.GetProp(name)
        try:
            return cast(v)
        except Exception:
            return v
    return default


# ---- 1. parse SD files, collect all poses ----
all_poses = []  # list of dict per pose
total_poses = 0
for round_name, sd_path in ROUNDS:
    suppl = Chem.SDMolSupplier(sd_path, removeHs=False, sanitize=False)
    n_this = 0
    for mol in suppl:
        if mol is None:
            continue
        total_poses += 1
        n_this += 1
        score = get_prop(mol, "SCORE", float)
        inter = get_prop(mol, "SCORE.INTER", float)
        p_binder = get_prop(mol, "rnasmol_p_binder", float)
        logit = get_prop(mol, "rnasmol_logit", float)
        gen_idx = get_prop(mol, "gen_idx", int)
        rnasmol_rank = get_prop(mol, "rnasmol_rank", int, default=None)
        # canonical SMILES from the docked 3D structure (strip Hs for canonical key)
        try:
            mol_nh = Chem.RemoveHs(mol)
            cansmi = Chem.MolToSmiles(mol_nh)
        except Exception:
            continue
        all_poses.append({
            "round": round_name,
            "sd_path": sd_path,
            "mol": mol,
            "cansmi": cansmi,
            "score": score,
            "inter": inter,
            "p_binder": p_binder,
            "logit": logit,
            "gen_idx": gen_idx,
            "rnasmol_rank": rnasmol_rank,
        })
    print(f"[parse] {round_name}: {n_this} poses")

print(f"[parse] total poses = {total_poses}")
print(f"[parse] parsed entries = {len(all_poses)}")

# ---- 2. filter: P>=0.9 AND INTER<-16.79 ----
filt = [p for p in all_poses
        if p["p_binder"] is not None and p["p_binder"] >= P_CUTOFF
        and p["inter"] is not None and p["inter"] < INTER_CUTO]
print(f"[filter] P>=0.9 AND INTER<{INTER_CUTO}: {len(filt)} poses pass")

n2 = sum(1 for p in filt if p["round"] == "round2")
n3 = sum(1 for p in filt if p["round"] == "round3")
print(f"[filter]   round2 poses pass: {n2}")
print(f"[filter]   round3 poses pass: {n3}")

# ---- 3. per-molecule best pose (most negative INTER) ----
# group by (round, gen_idx) first since same mol in same round has multiple poses
from collections import defaultdict
by_mol_key = defaultdict(list)  # key = (round, gen_idx)
for p in filt:
    by_mol_key[(p["round"], p["gen_idx"])].append(p)

best_per_mol = []
for key, poses in by_mol_key.items():
    best = min(poses, key=lambda x: x["inter"])  # most negative INTER
    best_per_mol.append(best)
print(f"[per-mol] unique (round,gen_idx) molecules passing: {len(best_per_mol)}")

# ---- 4. dedup by canonical SMILES across rounds (keep most negative INTER) ----
by_smi = defaultdict(list)
for p in best_per_mol:
    by_smi[p["cansmi"]].append(p)

deduped = []
for smi, mols in by_smi.items():
    best = min(mols, key=lambda x: x["inter"])
    deduped.append(best)

# ---- 5. sort by INTER ascending (most negative first) ----
deduped.sort(key=lambda x: x["inter"])
print(f"[dedup] final unique molecules by canonical SMILES: {len(deduped)}")

# round contributions after dedup
r2_final = sum(1 for p in deduped if p["round"] == "round2")
r3_final = sum(1 for p in deduped if p["round"] == "round3")
print(f"[dedup]   from round2: {r2_final}")
print(f"[dedup]   from round3: {r3_final}")

# ---- 6. write final SDF ----
writer = Chem.SDWriter(OUT_SDF)
for p in deduped:
    mol = p["mol"]
    # clear all props and set canonical set
    for k in list(mol.GetPropsAsDict().keys()):
        mol.ClearProp(k)
    mol.SetProp("rnasmol_p_binder", f"{p['p_binder']:.6f}")
    mol.SetProp("rnasmol_logit", f"{p['logit']:.6f}" if p["logit"] is not None else "")
    mol.SetProp("rdock_SCORE", f"{p['score']:.4f}" if p["score"] is not None else "")
    mol.SetProp("rdock_INTER", f"{p['inter']:.4f}")
    mol.SetProp("gen_idx", str(p["gen_idx"]))
    mol.SetProp("source_round", p["round"])
    mol.SetProp("canonical_smiles", p["cansmi"])
    if p["rnasmol_rank"] is not None:
        mol.SetProp("rnasmol_rank", str(p["rnasmol_rank"]))
    writer.write(mol)
writer.close()
size = os.path.getsize(OUT_SDF)
print(f"[write] {OUT_SDF}: {len(deduped)} mols, {size} bytes")

# ---- 7. stats ----
import statistics
inters = [p["inter"] for p in deduped]
ps = [p["p_binder"] for p in deduped]
print("\n===== SUMMARY =====")
print(f"N final binders: {len(deduped)}")
print(f"INTER  min={min(inters):.3f}  median={statistics.median(inters):.3f}  max={max(inters):.3f}")
print(f"P(binder) min={min(ps):.4f}  median={statistics.median(ps):.4f}  max={max(ps):.4f}")
print(f"All P(binder) >= 0.9: {all(x>=0.9 for x in ps)}")
print(f"All INTER < -16.79:  {all(x<-16.79 for x in inters)}")

print("\n===== TOP-10 (by INTER, most negative first) =====")
for i, p in enumerate(deduped[:10], 1):
    print(f"{i:2d}. INTER={p['inter']:+.3f}  P={p['p_binder']:.4f}  {p['round']}  gen{p['gen_idx']}")
    print(f"     {p['cansmi']}")

# ---- 8. check T14 (1,2,4-triazine phenol) ----
# 1,2,4-triazine ring SMARTS + phenol
triazine_phenol = Chem.MolFromSmarts("c1nncnc1")
phenol = Chem.MolFromSmarts("cO")
print("\n===== T14 (1,2,4-triazin-phenol) check =====")
hits = []
for p in deduped:
    m = Chem.MolFromSmiles(p["cansmi"])
    if m is None:
        continue
    # 1,2,4-triazine: 6-membered aromatic with nncnc or ncncn pattern
    # generic triazine (1,2,4 or 1,3,5) = c1ncncn1
    t124 = Chem.MolFromSmarts("c1ncncn1")  # all triazines (1,2,4 / 1,3,5)
    if m.HasSubstructMatch(t124) and m.HasSubstructMatch(phenol):
        hits.append(p)
        print(f"  HIT: gen{p['gen_idx']} {p['round']} INTER={p['inter']:.3f} P={p['p_binder']:.4f}")
        print(f"    {p['cansmi']}")
if not hits:
    print("  No 1,2,4-triazine-phenol matches found in final set.")

# save summary as TSV
with open("casestudy/fmn/final_binders_summary.tsv", "w") as f:
    f.write("rank\tsource_round\tgen_idx\trnasmol_rank\trdock_INTER\trdock_SCORE\trnasmol_p_binder\trnasmol_logit\tcanonical_smiles\n")
    for i, p in enumerate(deduped, 1):
        f.write(f"{i}\t{p['round']}\t{p['gen_idx']}\t{p['rnasmol_rank']}\t{p['inter']:.4f}\t{p['score']:.4f}\t{p['p_binder']:.6f}\t{p['logit']:.6f}\t{p['cansmi']}\n")
print("\n[done] wrote casestudy/fmn/final_binders_summary.tsv")

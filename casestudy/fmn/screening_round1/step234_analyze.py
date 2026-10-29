"""Steps 2-4: prolif interaction analysis + FMN comparison + drug-likeness.

Outputs (all under casestudy/fmn/):
  - neutral_16_analysis.json    : full structured per-ligand results
  - neutral_16_interactions.tsv : flat interaction table
  - neutral_16_druglike.tsv     : drug-likeness table
  - neutral_16_analysis_summary.txt : short text summary
"""
import os, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/data/users/gaoletian/rna2smol")

from collections import defaultdict, Counter
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
import prolif as plf
from util import sascorer

ROOT = "/home/data/users/gaoletian/rna2smol/casestudy/fmn"
NEU_SDF = os.path.join(ROOT, "neutral_16_poses.sdf")
POCKET_PDB = os.path.join(ROOT, "1FMN_pocket_extracted_addh.pdb")
FMN_REDOCK = os.path.join(ROOT, "rdock/FMN_redock.sd")
FMN_SDF    = os.path.join(ROOT, "FMN_ligand.sdf")
OUT_JSON   = os.path.join(ROOT, "neutral_16_analysis.json")
OUT_INT_TSV= os.path.join(ROOT, "neutral_16_interactions.tsv")
OUT_DRUG_TSV = os.path.join(ROOT, "neutral_16_druglike.tsv")
OUT_SUM    = os.path.join(ROOT, "neutral_16_analysis_summary.txt")

INTERESTING = ["HBDonor", "HBAcceptor", "PiStacking", "Hydrophobic",
               "Anionic", "Cationic", "CationPi", "PiCation", "VdWContact"]


def load_prot():
    m = Chem.MolFromPDBFile(POCKET_PDB, removeHs=False)
    return plf.Molecule.from_rdkit(m)


def add_h_3d(mol):
    """Add explicit H with 3D coords using existing heavy-atom geometry."""
    return Chem.AddHs(mol, addCoords=True)


def run_prolif(lig_mol, prot):
    fp = plf.Fingerprint()  # default curated set
    fp.run_from_iterable([lig_mol], prot, progress=False)
    df = fp.to_dataframe()
    res = []
    for col in df.columns:
        # col = (lig_resname, prot_resid, interaction)
        if df[col].iloc[0]:
            _, resid, itype = col
            res.append({"residue": resid, "interaction": itype})
    return res


def summarize_interactions(int_list):
    by_res = defaultdict(list)
    by_type = Counter()
    for r in int_list:
        by_res[r["residue"]].append(r["interaction"])
        by_type[r["interaction"]] += 1
    # remove VdW-only residues if you want a "key contacts" view, but keep both
    key_residues_no_vdw = {r: v for r, v in by_res.items()
                           if any(t != "VdWContact" for t in v)}
    return {
        "total": len(int_list),
        "by_type": dict(by_type),
        "by_residue": {k: sorted(set(v)) for k, v in by_res.items()},
        "key_residues_no_vdw": key_residues_no_vdw,
        "n_residues": len(by_res),
        "n_residues_no_vdw": len(key_residues_no_vdw),
    }


def drug_likeness(mol_heavy):
    """Compute on a sanitized heavy-atom mol."""
    mw = Descriptors.MolWt(mol_heavy)
    qed = QED.qed(mol_heavy)
    sa = sascorer.calculateScore(mol_heavy)  # lower = easier to synth (1..10)
    nha = mol_heavy.GetNumHeavyAtoms()
    hbd = rdMolDescriptors.CalcNumHBD(mol_heavy)
    hba = rdMolDescriptors.CalcNumHBA(mol_heavy)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol_heavy)
    nring = rdMolDescriptors.CalcNumRings(mol_heavy)
    naro = rdMolDescriptors.CalcNumAromaticRings(mol_heavy)
    tpsa = Descriptors.TPSA(mol_heavy)
    clogp = Descriptors.MolLogP(mol_heavy)
    fchg = int(sum(a.GetFormalCharge() for a in mol_heavy.GetAtoms()))
    return {
        "MW": round(mw, 2), "QED": round(qed, 3), "SA": round(sa, 2),
        "heavy_atoms": nha, "HBD": hbd, "HBA": hba, "rotatable": rot,
        "rings": nring, "aromatic_rings": naro, "TPSA": round(tpsa, 1),
        "cLogP": round(clogp, 2), "formal_charge": fchg,
    }


def main():
    print("loading pocket...")
    prot = load_prot()
    print(f"  residues: {[str(r) for r in prot.residues]}")

    neu = [m for m in Chem.SDMolSupplier(NEU_SDF, removeHs=False)]
    print(f"neutral mols: {len(neu)}")

    out = {"pocket_residues": [str(r) for r in prot.residues], "ligands": []}
    int_rows = []
    drug_rows = []

    # ---- Neutral 16 ----
    for i, m in enumerate(neu, 1):
        d = m.GetPropsAsDict()
        pid = d.get("pose_id", f"N{i:02d}")
        smi = d.get("smiles_sanitized") or d.get("smiles")
        # add H with 3D, build prolif mol
        m_h = add_h_3d(m)
        lig = plf.Molecule.from_rdkit(m_h)
        int_list = run_prolif(lig, prot)
        summ = summarize_interactions(int_list)
        # drug-likeness on heavy-atom sanitized mol
        dl = drug_likeness(m)
        rec = {
            "pose_id": pid, "gen_idx": int(d["gen_idx"]),
            "combined_rank": int(d["combined_rank"]),
            "rnasmol_p_binder": float(d["rnasmol_p_binder"]),
            "rdock_INTER": float(d.get("rdock_INTER", d.get("rdock_pose_INTER"))),
            "rdock_SCORE": float(d.get("rdock_SCORE", d.get("rdock_pose_SCORE"))),
            "smiles": smi,
            "interactions": int_list,
            "interaction_summary": summ,
            "druglikeness": dl,
        }
        out["ligands"].append(rec)
        for it in int_list:
            int_rows.append((pid, int(d["gen_idx"]), it["residue"], it["interaction"]))
        drug_rows.append((pid, int(d["gen_idx"]), dl))

    # ---- FMN reference (primary = crystal pose; secondary = best redock pose) ----
    # Note: rdock self-docking drifted ~5 A from crystal (INTER -21 but pose
    # has only VdW contacts), so we use the crystal pose (FMN_ligand.sdf) as
    # the *binding-mode* reference and report redock scores separately.
    print("loading FMN reference...")
    fmn_crystal = Chem.SDMolSupplier(FMN_SDF, removeHs=False)[0]
    fmn_crystal_lig = plf.Molecule.from_rdkit(fmn_crystal)
    fmn_int = run_prolif(fmn_crystal_lig, prot)
    fmn_summ = summarize_interactions(fmn_int)
    fmn_dl = drug_likeness(Chem.RemoveHs(fmn_crystal))

    # also pick best redock pose for score reporting
    fmnsup = Chem.SDMolSupplier(FMN_REDOCK, removeHs=False, sanitize=False)
    best = None
    for m in fmnsup:
        if m is None:
            continue
        d = m.GetPropsAsDict()
        try:
            inter = float(d.get("SCORE.INTER", 1e9))
        except (TypeError, ValueError):
            inter = 1e9
        try:
            score = float(d.get("SCORE", 1e9))
        except (TypeError, ValueError):
            score = 1e9
        if best is None or inter < best[0]:
            best = (inter, score)
    fmn_inter_redock, fmn_score_redock = best
    print(f"  FMN redock best-INTER pose: INTER={fmn_inter_redock:.3f} SCORE={fmn_score_redock:.3f}")
    print(f"  FMN crystal interactions: {fmn_summ['by_type']}")
    out["fmn_reference"] = {
        "source": "FMN_ligand.sdf (crystal pose, PDB 1FMN chain A ligand)",
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(fmn_crystal)),
        "rdock_INTER_redock": fmn_inter_redock,
        "rdock_SCORE_redock": fmn_score_redock,
        "interactions": fmn_int,
        "interaction_summary": fmn_summ,
        "druglikeness": fmn_dl,
        "rdock_note": ("Self-docking drifted ~5 A from crystal pose; redock INTER "
                       "(-21.0) is more negative than crystal-like pose (-16.8) but "
                       "the pose is geometrically poor, so interactions are not "
                       "biologically meaningful. Crystal pose is used for binding-mode "
                       "comparison."),
    }

    # ---- Compare: residue overlap with FMN ----
    fmn_res = set(fmn_summ["by_residue"].keys())
    fmn_res_no_vdw = set(fmn_summ["key_residues_no_vdw"].keys())
    for L in out["ligands"]:
        lr = set(L["interaction_summary"]["by_residue"].keys())
        lr_no_vdw = set(L["interaction_summary"]["key_residues_no_vdw"].keys())
        L["fmn_overlap_residues"] = sorted(lr & fmn_res)
        L["fmn_overlap_residues_no_vdw"] = sorted(lr_no_vdw & fmn_res_no_vdw)
        L["extra_residues_no_vdw"] = sorted(lr_no_vdw - fmn_res_no_vdw)
        L["fmn_shared_residues_no_vdw_count"] = len(lr_no_vdw & fmn_res_no_vdw)
        # Tanimoto-like over residues (no VdW only)
        union = lr_no_vdw | fmn_res_no_vdw
        L["residue_jaccard_no_vdw"] = (
            round(len(lr_no_vdw & fmn_res_no_vdw) / len(union), 3) if union else 0.0
        )

    # save JSON
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {OUT_JSON}")

    # save interactions tsv
    with open(OUT_INT_TSV, "w") as f:
        f.write("pose_id\tgen_idx\tresidue\tinteraction\n")
        for r in int_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
        # add FMN rows
        for it in fmn_int:
            f.write(f"FMN\t0\t{it['residue']}\t{it['interaction']}\n")
    print(f"-> {OUT_INT_TSV}")

    # save drug-like tsv
    cols = ["pose_id","gen_idx","MW","QED","SA","heavy_atoms","HBD","HBA",
            "rotatable","rings","aromatic_rings","TPSA","cLogP","formal_charge",
            "rdock_INTER"]
    with open(OUT_DRUG_TSV, "w") as f:
        f.write("\t".join(cols) + "\n")
        for pid, gi, dl in drug_rows:
            # pull INTER from corresponding ligand
            inter = next(L["rdock_INTER"] for L in out["ligands"] if L["pose_id"]==pid)
            row = [pid, gi, dl["MW"], dl["QED"], dl["SA"], dl["heavy_atoms"],
                   dl["HBD"], dl["HBA"], dl["rotatable"], dl["rings"],
                   dl["aromatic_rings"], dl["TPSA"], dl["cLogP"],
                   dl["formal_charge"], inter]
            f.write("\t".join(str(x) for x in row) + "\n")
        # FMN row
        dl = fmn_dl
        f.write("\t".join(str(x) for x in
            ["FMN", 0, dl["MW"], dl["QED"], dl["SA"], dl["heavy_atoms"],
             dl["HBD"], dl["HBA"], dl["rotatable"], dl["rings"],
             dl["aromatic_rings"], dl["TPSA"], dl["cLogP"],
             dl["formal_charge"], fmn_inter_redock]) + "\n")
    print(f"-> {OUT_DRUG_TSV}")

    # ---- text summary ----
    lines = []
    lines.append("# 1FMN neutral-16 deep evaluation summary")
    lines.append(f"# pocket residues: {out['pocket_residues']}")
    lines.append("")
    lines.append("== FMN reference (crystal pose) ==")
    lines.append(f"  rdock INTER (self-dock best) = {fmn_inter_redock:.3f}  SCORE = {fmn_score_redock:.3f}")
    lines.append(f"  drug-likeness: {fmn_dl}")
    lines.append(f"  interactions by type: {fmn_summ['by_type']}")
    lines.append(f"  key residues (no VdW): {sorted(fmn_summ['key_residues_no_vdw'].keys())}")
    lines.append("")
    lines.append("== Neutral 16 (sorted by combined_rank) ==")
    for L in out["ligands"]:
        s = L["interaction_summary"]
        lines.append(f"[{L['pose_id']}] gen={L['gen_idx']} rank={L['combined_rank']} "
                     f"P={L['rnasmol_p_binder']:.3f} INTER={L['rdock_INTER']:.2f} "
                     f"MW={L['druglikeness']['MW']} QED={L['druglikeness']['QED']} "
                     f"SA={L['druglikeness']['SA']}")
        lines.append(f"    total_int={s['total']} by_type={s['by_type']}")
        lines.append(f"    key_res(no_vdw)={sorted(s['key_residues_no_vdw'].keys())}")
        lines.append(f"    fmn_shared(no_vdw)={L['fmn_shared_residues_no_vdw_count']} "
                     f"jaccard={L['residue_jaccard_no_vdw']} "
                     f"extra={L['extra_residues_no_vdw']}")
    lines.append("")
    lines.append("== Aggregate interaction-type distribution (neutral 16) ==")
    agg = Counter()
    for L in out["ligands"]:
        for k, v in L["interaction_summary"]["by_type"].items():
            agg[k] += v
    for k, v in agg.most_common():
        lines.append(f"  {k:15s} {v}")
    lines.append("")
    lines.append("== Top residues (neutral 16, non-VdW contact frequency) ==")
    resc = Counter()
    for L in out["ligands"]:
        for r in L["interaction_summary"]["key_residues_no_vdw"].keys():
            resc[r] += 1
    for r, c in resc.most_common():
        lines.append(f"  {r:8s} contacted by {c}/16 neutral ligands")
    with open(OUT_SUM, "w") as f:
        f.write("\n".join(lines))
    print(f"-> {OUT_SUM}")
    print("DONE")


if __name__ == "__main__":
    main()

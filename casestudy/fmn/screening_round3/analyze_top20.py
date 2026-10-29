"""Round-3 deep evaluation of top-20 candidates (all neutral, 1w pool).

Pipeline
--------
1. Read top_binders.sdf (top-100 by combined score), pick top-20
2. For each top-20 ligand, find its best-INTER pose in docked_top500.sd
3. Compute drug-likeness (MW/QED/SA/...) on the heavy-atom mol
4. Run prolif vs 1FMN pocket -> interaction fingerprint
5. Compare each ligand's interaction set with FMN (crystal pose)
6. Save JSON + interactions TSV + drug-likeness TSV + text summary

Outputs (under casestudy/fmn/screening_round3/):
  - top20_analysis.json
  - top20_interactions.tsv
  - top20_druglike.tsv
  - top20_analysis_summary.txt
"""
import os, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/data/users/gaoletian/rna2smol")

from collections import defaultdict, Counter
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
import prolif as plf
from util import sascorer

ROOT    = "/home/data/users/gaoletian/rna2smol/casestudy/fmn"
ROUND2  = os.path.join(ROOT, "screening_round3")
TOPBIND = os.path.join(ROUND2, "top_binders.sdf")        # top-100, has props
DOCKED  = os.path.join(ROUND2, "docked_top500.sd")        # 3 poses x 500
POCKET  = os.path.join(ROUND2, "1FMN_pocket_extracted_addh.pdb")
FMN_SDF = os.path.join(ROOT, "FMN_ligand.sdf")
FMN_REDOCK = os.path.join(ROOT, "rdock", "FMN_redock.sd")
OUT_JSON   = os.path.join(ROUND2, "top20_analysis.json")
OUT_INT_TSV= os.path.join(ROUND2, "top20_interactions.tsv")
OUT_DRUG_TSV = os.path.join(ROUND2, "top20_druglike.tsv")
OUT_SUM    = os.path.join(ROUND2, "top20_analysis_summary.txt")
N_TOP = 20

INTERESTING = ["HBDonor", "HBAcceptor", "PiStacking", "Hydrophobic",
               "Anionic", "Cationic", "CationPi", "PiCation", "VdWContact"]


def load_prot():
    m = Chem.MolFromPDBFile(POCKET, removeHs=False)
    return plf.Molecule.from_rdkit(m)


def add_h_3d(mol):
    return Chem.AddHs(mol, addCoords=True)


def run_prolif(lig_mol, prot):
    fp = plf.Fingerprint()
    fp.run_from_iterable([lig_mol], prot, progress=False)
    df = fp.to_dataframe()
    res = []
    for col in df.columns:
        if df[col].iloc[0]:
            _, resid, itype = col
            res.append({"residue": str(resid), "interaction": itype})
    return res


def summarize_interactions(int_list):
    by_res = defaultdict(list)
    by_type = Counter()
    for r in int_list:
        by_res[r["residue"]].append(r["interaction"])
        by_type[r["interaction"]] += 1
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
    mw = Descriptors.MolWt(mol_heavy)
    qed = QED.qed(mol_heavy)
    sa = sascorer.calculateScore(mol_heavy)
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


def pick_best_inter_poses(docked_sd, wanted_gen, n_runs=3):
    """Scan docked SD, return {gen_idx: (INTER, SCORE, rdkit_mol_pose)} for the
    pose with smallest SCORE.INTER for each wanted gen_idx."""
    sup = Chem.SDMolSupplier(docked_sd, removeHs=False, sanitize=False)
    best = {}
    for m in sup:
        if m is None:
            continue
        d = m.GetPropsAsDict()
        gi = d.get("gen_idx")
        if gi is None:
            continue
        gi = int(gi)
        if gi not in wanted_gen:
            continue
        try:
            inter = float(d.get("SCORE.INTER", 1e9))
        except (TypeError, ValueError):
            inter = 1e9
        try:
            score = float(d.get("SCORE", 1e9))
        except (TypeError, ValueError):
            score = 1e9
        cur = best.get(gi)
        if cur is None or inter < cur[0]:
            best[gi] = (inter, score, m)
    return best


def main():
    print("loading pocket...", flush=True)
    prot = load_prot()
    print(f"  residues: {[str(r) for r in prot.residues]}", flush=True)

    tb = [m for m in Chem.SDMolSupplier(TOPBIND, removeHs=False, sanitize=False)]
    print(f"top_binders mols: {len(tb)}", flush=True)

    # take top-20 (already sorted by combined_rank in SDF)
    top20 = tb[:N_TOP]
    print(f"top-20 selected", flush=True)

    wanted_gen = set()
    for i, m in enumerate(top20, 1):
        d = m.GetPropsAsDict()
        gi = int(d["gen_idx"])
        wanted_gen.add(gi)
        fc = sum(a.GetFormalCharge() for a in m.GetAtoms())
        assert fc == 0, f"rank {i} (gen {gi}) has charge {fc} (round3 should be all-neutral)"

    print("picking best-INTER pose per top-20 ligand from docked SD...", flush=True)
    best_poses = pick_best_inter_poses(DOCKED, wanted_gen)
    print(f"  recovered {len(best_poses)}/{len(wanted_gen)} poses", flush=True)

    out = {"pocket_residues": [str(r) for r in prot.residues], "ligands": []}
    int_rows = []
    drug_rows = []

    for i, m_tb in enumerate(top20, 1):
        d_tb = m_tb.GetPropsAsDict()
        gi = int(d_tb["gen_idx"])
        smi = d_tb.get("smiles", "")
        if gi in best_poses:
            inter, score, m_pose = best_poses[gi]
            m = Chem.Mol(m_pose)
            try:
                Chem.SanitizeMol(m)
            except Exception as e:
                print(f"  [rank {i}] sanitize failed ({e}); using top_binders mol", flush=True)
                m = Chem.Mol(m_tb)
            m = Chem.RemoveHs(m)
        else:
            print(f"  [rank {i}] no docked pose; using top_binders 3D", flush=True)
            inter = float(d_tb.get("rdock_INTER", float("nan")))
            score = float(d_tb.get("rdock_SCORE", float("nan")))
            m = Chem.Mol(m_tb)

        # 3D-H mol for prolif
        try:
            m_h = add_h_3d(m)
            lig = plf.Molecule.from_rdkit(m_h)
            int_list = run_prolif(lig, prot)
        except Exception as e:
            print(f"  [rank {i}] prolif failed: {e}", flush=True)
            int_list = []
        summ = summarize_interactions(int_list)
        dl = drug_likeness(m)

        rec = {
            "rank": i,
            "pose_id": f"T{i:02d}",
            "gen_idx": gi,
            "combined_rank": int(d_tb["combined_rank"]),
            "combined_score": float(d_tb["combined_score"]),
            "rnasmol_p_binder": float(d_tb["rnasmol_p_binder"]),
            "rnasmol_logit": float(d_tb["rnasmol_logit"]),
            "rdock_INTER": float(inter),
            "rdock_SCORE": float(score),
            "smiles": smi,
            "smiles_sanitized": Chem.MolToSmiles(m),
            "interactions": int_list,
            "interaction_summary": summ,
            "druglikeness": dl,
        }
        out["ligands"].append(rec)
        for it in int_list:
            int_rows.append((rec["pose_id"], gi, it["residue"], it["interaction"]))
        drug_rows.append((rec["pose_id"], gi, dl, inter))
        print(f"  [rank {i:>2}] gen={gi} P={rec['rnasmol_p_binder']:.3f} "
              f"INTER={inter:.2f} SCORE={score:.2f} MW={dl['MW']} "
              f"QED={dl['QED']} SA={dl['SA']} int={summ['total']}", flush=True)

    # FMN reference (crystal pose for binding mode, redock for score)
    print("loading FMN reference...", flush=True)
    fmn_crystal = Chem.SDMolSupplier(FMN_SDF, removeHs=False, sanitize=False)[0]
    try:
        fmn_crystal_lig = plf.Molecule.from_rdkit(Chem.Mol(fmn_crystal))
        fmn_int = run_prolif(fmn_crystal_lig, prot)
    except Exception as e:
        print(f"  FMN prolif failed: {e}", flush=True)
        fmn_int = []
    fmn_summ = summarize_interactions(fmn_int)
    fmn_dl = drug_likeness(Chem.RemoveHs(fmn_crystal))

    fmnsup = Chem.SDMolSupplier(FMN_REDOCK, removeHs=False, sanitize=False)
    best = None
    for m in fmnsup:
        if m is None:
            continue
        d = m.GetPropsAsDict()
        try:
            score = float(d.get("SCORE", 1e9))
        except (TypeError, ValueError):
            score = 1e9
        if best is None or score < best[1]:
            try:
                best = (float(d.get("SCORE.INTER", 1e9)), score)
            except (TypeError, ValueError):
                pass
    fmn_inter_redock, fmn_score_redock = best
    print(f"  FMN redock best-SCORE pose: SCORE={fmn_score_redock:.3f} INTER={fmn_inter_redock:.3f}", flush=True)
    print(f"  FMN crystal interactions: {fmn_summ['by_type']}", flush=True)
    out["fmn_reference"] = {
        "source": "FMN_ligand.sdf (crystal pose, PDB 1FMN chain A ligand)",
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(fmn_crystal)),
        "rdock_INTER_redock_bestSCOREpose": fmn_inter_redock,
        "rdock_SCORE_redock": fmn_score_redock,
        "interactions": fmn_int,
        "interaction_summary": fmn_summ,
        "druglikeness": fmn_dl,
    }

    # residue overlap with FMN
    fmn_res = set(fmn_summ["by_residue"].keys())
    fmn_res_no_vdw = set(fmn_summ["key_residues_no_vdw"].keys())
    for L in out["ligands"]:
        lr = set(L["interaction_summary"]["by_residue"].keys())
        lr_no_vdw = set(L["interaction_summary"]["key_residues_no_vdw"].keys())
        L["fmn_overlap_residues"] = sorted(lr & fmn_res)
        L["fmn_overlap_residues_no_vdw"] = sorted(lr_no_vdw & fmn_res_no_vdw)
        L["extra_residues_no_vdw"] = sorted(lr_no_vdw - fmn_res_no_vdw)
        L["fmn_shared_residues_no_vdw_count"] = len(lr_no_vdw & fmn_res_no_vdw)
        union = lr_no_vdw | fmn_res_no_vdw
        L["residue_jaccard_no_vdw"] = (
            round(len(lr_no_vdw & fmn_res_no_vdw) / len(union), 3) if union else 0.0
        )

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {OUT_JSON}", flush=True)

    with open(OUT_INT_TSV, "w") as f:
        f.write("pose_id\tgen_idx\tresidue\tinteraction\n")
        for r in int_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
        for it in fmn_int:
            f.write(f"FMN\t0\t{it['residue']}\t{it['interaction']}\n")
    print(f"-> {OUT_INT_TSV}", flush=True)

    cols = ["pose_id", "gen_idx", "MW", "QED", "SA", "heavy_atoms", "HBD", "HBA",
            "rotatable", "rings", "aromatic_rings", "TPSA", "cLogP",
            "formal_charge", "rdock_INTER", "rnasmol_p_binder"]
    with open(OUT_DRUG_TSV, "w") as f:
        f.write("\t".join(cols) + "\n")
        for pid, gi, dl, inter in drug_rows:
            p = next(L["rnasmol_p_binder"] for L in out["ligands"] if L["pose_id"] == pid)
            row = [pid, gi, dl["MW"], dl["QED"], dl["SA"], dl["heavy_atoms"],
                   dl["HBD"], dl["HBA"], dl["rotatable"], dl["rings"],
                   dl["aromatic_rings"], dl["TPSA"], dl["cLogP"],
                   dl["formal_charge"], inter, p]
            f.write("\t".join(str(x) for x in row) + "\n")
        # FMN row
        dl = fmn_dl
        f.write("\t".join(str(x) for x in
            ["FMN", 0, dl["MW"], dl["QED"], dl["SA"], dl["heavy_atoms"],
             dl["HBD"], dl["HBA"], dl["rotatable"], dl["rings"],
             dl["aromatic_rings"], dl["TPSA"], dl["cLogP"],
             dl["formal_charge"], fmn_inter_redock, 1.0]) + "\n")
    print(f"-> {OUT_DRUG_TSV}", flush=True)

    # text summary
    lines = []
    lines.append("# 1FMN round-3 top-20 deep evaluation summary (all neutral, 1w pool)")
    lines.append(f"# pocket residues: {out['pocket_residues']}")
    lines.append("")
    lines.append("== FMN reference ==")
    lines.append(f"  rdock SCORE (best) = {fmn_score_redock:.3f}  INTER (best-SCORE pose) = {fmn_inter_redock:.3f}")
    lines.append(f"  drug-likeness: {fmn_dl}")
    lines.append(f"  interactions by type: {fmn_summ['by_type']}")
    lines.append(f"  key residues (no VdW): {sorted(fmn_summ['key_residues_no_vdw'].keys())}")
    lines.append("")
    lines.append("== Top-20 (sorted by combined score) ==")
    for L in out["ligands"]:
        s = L["interaction_summary"]
        lines.append(f"[{L['pose_id']}] gen={L['gen_idx']} rank={L['combined_rank']} "
                     f"P={L['rnasmol_p_binder']:.3f} logit={L['rnasmol_logit']:+.2f} "
                     f"INTER={L['rdock_INTER']:.2f} SCORE={L['rdock_SCORE']:.2f} "
                     f"MW={L['druglikeness']['MW']} QED={L['druglikeness']['QED']} "
                     f"SA={L['druglikeness']['SA']}")
        lines.append(f"    total_int={s['total']} by_type={s['by_type']}")
        lines.append(f"    key_res(no_vdw)={sorted(s['key_residues_no_vdw'].keys())}")
        lines.append(f"    fmn_shared(no_vdw)={L['fmn_shared_residues_no_vdw_count']} "
                     f"jaccard={L['residue_jaccard_no_vdw']} "
                     f"extra={L['extra_residues_no_vdw']}")
    lines.append("")
    # stats across 20
    qeds = [L["druglikeness"]["QED"] for L in out["ligands"]]
    sas = [L["druglikeness"]["SA"] for L in out["ligands"]]
    mws = [L["druglikeness"]["MW"] for L in out["ligands"]]
    inters = [L["rdock_INTER"] for L in out["ligands"]]
    lines.append("== Top-20 aggregate ==")
    lines.append(f"  QED: min={min(qeds)} max={max(qeds)} mean={sum(qeds)/len(qeds):.3f}")
    lines.append(f"  SA : min={min(sas)} max={max(sas)} mean={sum(sas)/len(sas):.2f}")
    lines.append(f"  MW : min={min(mws)} max={max(mws)} mean={sum(mws)/len(mws):.1f}")
    lines.append(f"  INTER: min={min(inters):.2f} max={max(inters):.2f} mean={sum(inters)/len(inters):.2f}")
    with open(OUT_SUM, "w") as f:
        f.write("\n".join(lines))
    print(f"-> {OUT_SUM}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

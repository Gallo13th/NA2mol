"""Step 1: extract the 16 neutral molecules' best rdock poses from docked_top300.sd.

Logic:
  - top_binders.sdf holds the top-100 candidates with combined_rank, gen_idx,
    rnasmol_p_binder, rdock_SCORE, rdock_INTER, smiles.
  - 16 of them are formal-charge == 0 (clean candidates).
  - docked_top300.sd holds 3 poses per molecule (900 total), keyed by `gen_idx`
    and rDock `SCORE`/`SCORE.INTER`. We pick the best INTER pose for each neutral
    molecule (INTER is the pure receptor-ligand term; total SCORE is polluted by
    INTRA conformational strain, see SCREENING_REPORT.md note 1).
  - Output: casestudy/fmn/neutral_16_poses.sdf carrying every property from
    top_binders.sdf plus the rdock SCORE/INTER and a pose_id.
"""
import os
from rdkit import Chem

ROOT = "/home/data/users/gaoletian/rna2smol/casestudy/fmn"
TOPBIND = os.path.join(ROOT, "top_binders.sdf")
DOCKED = os.path.join(ROOT, "docked_top300.sd")
OUT    = os.path.join(ROOT, "neutral_16_poses.sdf")


def total_formal_charge(m):
    return int(sum(a.GetFormalCharge() for a in m.GetAtoms()))


def main():
    tb = [m for m in Chem.SDMolSupplier(TOPBIND, removeHs=False)]
    print(f"[top_binders] parsed {len(tb)} molecules")
    neutral = [m for m in tb if total_formal_charge(m) == 0]
    print(f"[neutral] formal_charge==0: {len(neutral)}")

    # Map gen_idx -> props we want to carry over from top_binders.sdf
    neutral_gen = {}
    for m in neutral:
        d = m.GetPropsAsDict()
        neutral_gen[int(d["gen_idx"])] = d
    print(f"[neutral] gen_idx set: {sorted(neutral_gen)}")

    # Scan docked poses; rDock output sometimes breaks RDKit sanitization on
    # exotic valence states of OTHER molecules. Use sanitize=False and skip
    # sanitize cleanly so we can still read every block via gen_idx.
    sup = Chem.SDMolSupplier(DOCKED, removeHs=False, sanitize=False)
    best_pose = {}   # gen_idx -> (INTER, mol)
    n_poses_seen = 0
    for m in sup:
        if m is None:
            continue
        d = m.GetPropsAsDict()
        gi = d.get("gen_idx")
        if gi is None:
            continue
        gi = int(gi)
        if gi not in neutral_gen:
            continue
        n_poses_seen += 1
        try:
            inter = float(d.get("SCORE.INTER", 1e9))
        except (TypeError, ValueError):
            inter = 1e9
        try:
            score = float(d.get("SCORE", 1e9))
        except (TypeError, ValueError):
            score = 1e9
        cur = best_pose.get(gi)
        if cur is None or inter < cur[0]:
            # store mol, keep going
            best_pose[gi] = (inter, score, m)

    print(f"[docked] neutral poses seen: {n_poses_seen}, unique gen: {len(best_pose)}")
    missing = set(neutral_gen) - set(best_pose)
    if missing:
        print(f"[WARN] no docked pose for gen_idx: {sorted(missing)}")

    # Now build the output: combine top_binders props + rdock pose.
    # Order by combined_rank from top_binders (i.e. ranking among top-100).
    ordered = sorted(neutral, key=lambda m: int(m.GetPropsAsDict()["combined_rank"]))
    writer = Chem.SDWriter(OUT)
    written = 0
    for i, m_tb in enumerate(ordered, 1):
        d_tb = m_tb.GetPropsAsDict()
        gi = int(d_tb["gen_idx"])
        if gi not in best_pose:
            print(f"[WARN] skip N{i:02d} gen {gi}: no pose")
            continue
        inter, score, m_pose = best_pose[gi]
        # Use pose mol (has 3D), but switch to heavy-atom rdkit mol; sanitize now
        # (neutral mols must sanitize cleanly).
        smi_tb = m_tb.GetPropsAsDict()["smiles"]
        # Rebuild pose mol with sanitized chem to be safe; pose 3D kept as is.
        m = Chem.Mol(m_pose)
        try:
            Chem.SanitizeMol(m)
        except Exception as e:
            # fall back to top_binders mol (still 3D from generation)
            print(f"[WARN] N{i:02d} gen {gi} sanitize failed: {e}; using top_binders mol")
            m = Chem.Mol(m_tb)
        m = Chem.RemoveHs(m)

        # clear all props on m, then set canonical set
        for k in list(m.GetPropsAsDict().keys()):
            m.ClearProp(k)
        d = dict(d_tb)
        d["pose_id"] = f"N{i:02d}"
        d["neutral_rank"] = i
        d["rdock_pose_SCORE"] = score
        d["rdock_pose_INTER"] = inter
        # ensure formal_charge and smiles are explicit
        d["formal_charge"] = 0
        d["smiles_sanitized"] = Chem.MolToSmiles(m)
        for k, v in d.items():
            try:
                m.SetProp(k, str(v))
            except Exception:
                pass
        writer.write(m)
        written += 1
    writer.close()
    print(f"[done] wrote {written} neutral poses -> {OUT}")


if __name__ == "__main__":
    main()

"""Step 5: build neutral_16_evaluation.pdf.

Layout (per page, ~A4 portrait):
  - Cover page with summary table + interaction-type bar chart
  - FMN reference page (2D + binding mode + properties)
  - One page per neutral ligand: 2D depiction + interaction summary table
    + drug-likeness panel
  - Top-5 highlight page (sorted by a composite binding+drug-likeness score)

Uses matplotlib PdfPages (no reportlab needed).
"""
import os, json, sys, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

ROOT = "/home/data/users/gaoletian/rna2smol/casestudy/fmn"
JSON_IN  = os.path.join(ROOT, "neutral_16_analysis.json")
POSES_IN = os.path.join(ROOT, "neutral_16_poses.sdf")
FMN_SDF  = os.path.join(ROOT, "FMN_ligand.sdf")
OUT_PDF  = os.path.join(ROOT, "neutral_16_evaluation.pdf")


def mol_to_svg_img(mol, size=420):
    """Render 2D depiction to a numpy array via PIL."""
    mol = Chem.RemoveHs(mol)
    AllChem.Compute2DCoords(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 1.6
    opts.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(png)).convert("RGB")
    return np.asarray(img)


def page_header(fig, title, subtitle=None):
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=10, color="#444")


def draw_props_table(ax, dl, ref_dl=None, title="Drug-likeness"):
    ax.axis("off")
    keys = ["MW", "QED", "SA", "heavy_atoms", "HBD", "HBA",
            "rotatable", "rings", "aromatic_rings", "TPSA", "cLogP", "formal_charge"]
    rows = []
    have_ref = ref_dl is not None
    for k in keys:
        v = dl.get(k, "")
        if have_ref:
            rv = ref_dl[k]
            mark = ""
            if k == "QED" and isinstance(v,(int,float)) and v > ref_dl.get("QED",0):
                mark = "  >"
            if k == "SA" and isinstance(v,(int,float)) and v < ref_dl.get("SA",99):
                mark = "  <"
            rows.append([k, str(v), str(rv) + mark])
        else:
            rows.append([k, str(v)])
    col_labels = ["property", "value", "FMN ref"] if have_ref else ["property", "value"]
    col_widths = [0.35, 0.3, 0.35] if have_ref else [0.5, 0.5]
    t = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                 cellLoc="center", colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.0, 1.35)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


def draw_interactions_table(ax, L, fmn_residues_set):
    ax.axis("off")
    summ = L["interaction_summary"]
    by_res = summ["by_residue"]
    # build rows sorted by residue name
    rows = []
    for r in sorted(by_res.keys()):
        ints = by_res[r]
        # indicate which are non-VdW
        non_vdw = [i for i in ints if i != "VdWContact"]
        flag = " *" if r in fmn_residues_set else ""
        rows.append([r, ", ".join(ints), flag])
    if not rows:
        rows.append(["(none)", "", ""])
    t = ax.table(cellText=rows,
                 colLabels=["residue", "interactions", "FMN?"],
                 loc="center", cellLoc="left",
                 colWidths=[0.25, 0.6, 0.15])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.0, 1.25)
    ax.set_title(f"Binding mode  (total {summ['total']}, "
                 f"key residues w/o VdW: "
                 f"{len(summ['key_residues_no_vdw'])})",
                 fontsize=11, fontweight="bold", pad=8)


def composite_score(L, fmn_dl):
    """Composite: combine rdock INTER (binding), QED (drug), SA (synth, lower better).
    All normalized to roughly [0,1]."""
    dl = L["druglikeness"]
    # binding: rdock_INTER (negative); map -10 -> 0, -27 -> 1
    inter = L["rdock_INTER"]
    bind = np.clip((inter - (-10)) / ((-27) - (-10)), 0, 1)
    # drug: QED
    qed = dl["QED"]
    # synth: SA in 1..10 -> normalize lower is better
    sa = np.clip((8 - dl["SA"]) / (8 - 2), 0, 1)
    # weight: binding 0.5, drug 0.3, synth 0.2
    return 0.5*bind + 0.3*qed + 0.2*sa


def main():
    with open(JSON_IN) as f:
        data = json.load(f)
    ligands = data["ligands"]
    fmn = data["fmn_reference"]
    fmn_dl = fmn["druglikeness"]
    fmn_res_set = set(fmn["interaction_summary"]["by_residue"].keys())

    # load pose mols (for 2D depictions)
    poses = {m.GetPropsAsDict().get("pose_id"): m
             for m in Chem.SDMolSupplier(POSES_IN, removeHs=False)}
    fmn_mol = Chem.SDMolSupplier(FMN_SDF, removeHs=False)[0]

    # composite scores
    for L in ligands:
        L["composite"] = composite_score(L, fmn_dl)
    ligands_sorted_by_score = sorted(ligands, key=lambda x: -x["composite"])
    top5 = ligands_sorted_by_score[:5]

    with PdfPages(OUT_PDF) as pdf:
        # ============== COVER PAGE ==============
        fig = plt.figure(figsize=(8.27, 11.69))
        page_header(fig, "1FMN neutral-16 deep evaluation",
                    subtitle="rnasmol + rdock double-screened top-100 → 16 formal-charge==0 candidates")
        # left col: pipeline summary text
        ax_text = fig.add_axes([0.06, 0.50, 0.55, 0.40])
        ax_text.axis("off")
        summary = (
            "PIPELINE\n"
            "  4464 generated ligands (RNA2smol, 20-28 heavy atoms)\n"
            "  → rnasmol GPU scoring (all 4464, P(binder))\n"
            "  → rdock top-300 docking (3 poses each)\n"
            "  → combined rank (0.5 rnasmol_norm + 0.5 rdock_norm)\n"
            "  → top-100 binders\n"
            "       └─ 84 carry net formal charge (phosphonium / oxonium / chloronium\n"
            "          by-products of aromaticity maintenance) — understood but not\n"
            "          promoted as candidates.\n"
            "       └─ 16 neutral (formal_charge==0) — clean candidates analysed here.\n\n"
            "FMN reference (crystal pose)\n"
            f"  rdock INTER (self-dock) = {fmn['rdock_INTER_redock']:.2f}\n"
            f"  MW={fmn_dl['MW']}  QED={fmn_dl['QED']}  SA={fmn_dl['SA']}  "
            f"heavy={fmn_dl['heavy_atoms']}  fchg={fmn_dl['formal_charge']}\n"
            f"  key residues: {sorted(fmn['interaction_summary']['key_residues_no_vdw'].keys())}\n"
            f"  interactions: {fmn['interaction_summary']['by_type']}\n\n"
            "Note on FMN redock pose\n"
            "  rDock self-docking of FMN drifted ~5 A from the crystal pose; the\n"
            "  redock INTER (-21.0) is more negative than the crystal-like pose\n"
            "  (-16.8) but the pose is geometrically poor, so the crystal pose is\n"
            "  used as the binding-mode reference."
        )
        ax_text.text(0.0, 1.0, summary, va="top", ha="left", fontsize=8.6,
                     family="monospace")

        # right col: 2D of FMN
        ax_fmn = fig.add_axes([0.62, 0.62, 0.34, 0.30])
        ax_fmn.imshow(mol_to_svg_img(fmn_mol, size=500))
        ax_fmn.axis("off")
        ax_fmn.set_title("FMN (reference ligand)", fontsize=10, fontweight="bold")

        # bottom: interaction type distribution bar chart for neutral 16
        ax_bar = fig.add_axes([0.10, 0.08, 0.55, 0.34])
        agg = {}
        for L in ligands:
            for k, v in L["interaction_summary"]["by_type"].items():
                agg[k] = agg.get(k, 0) + v
        items = sorted(agg.items(), key=lambda x: -x[1])
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        bars = ax_bar.bar(range(len(names)), vals, color="#3a7ca5", edgecolor="black")
        # overlay FMN counts
        fmn_counts = fmn["interaction_summary"]["by_type"]
        for i, n in enumerate(names):
            fv = fmn_counts.get(n, 0)
            if fv:
                ax_bar.bar(i, fv, color="#e07a5f", edgecolor="black",
                           alpha=0.85, label="FMN" if i == 0 else None)
        ax_bar.set_xticks(range(len(names)))
        ax_bar.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax_bar.set_ylabel("count across 16 ligands")
        ax_bar.set_title("Interaction-type distribution (neutral-16, blue) "
                         "vs FMN (orange)", fontsize=10, fontweight="bold")
        ax_bar.legend(["neutral-16 total", "FMN"], loc="upper right", fontsize=8)
        for i, v in enumerate(vals):
            ax_bar.text(i, v + 0.3, str(v), ha="center", fontsize=8)

        # bottom-right: top residues contacted
        ax_res = fig.add_axes([0.70, 0.08, 0.27, 0.34])
        ax_res.axis("off")
        resc = {}
        for L in ligands:
            for r in L["interaction_summary"]["key_residues_no_vdw"].keys():
                resc[r] = resc.get(r, 0) + 1
        items = sorted(resc.items(), key=lambda x: -x[1])
        rows = [[r, f"{c}/16"] for r, c in items]
        t = ax_res.table(cellText=rows,
                         colLabels=["residue", "ligands"],
                         loc="center", cellLoc="center",
                         colWidths=[0.6, 0.4])
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.0, 1.3)
        ax_res.set_title("Key residues (non-VdW contact frequency)",
                         fontsize=10, fontweight="bold", pad=8)

        pdf.savefig(fig); plt.close(fig)

        # ============== FMN REFERENCE PAGE ==============
        fig = plt.figure(figsize=(8.27, 11.69))
        page_header(fig, "FMN reference binding mode",
                    subtitle="Crystal pose from PDB 1FMN (ground truth) — prolif vs 1FMN pocket")
        ax_img = fig.add_axes([0.12, 0.55, 0.45, 0.35])
        ax_img.imshow(mol_to_svg_img(fmn_mol, size=600))
        ax_img.axis("off")
        ax_img.set_title("FMN 2D structure", fontsize=11, fontweight="bold")

        ax_int = fig.add_axes([0.55, 0.45, 0.40, 0.45])
        fmn_fake_L = {
            "interaction_summary": fmn["interaction_summary"],
            "rdock_INTER": fmn["rdock_INTER_redock"],
        }
        draw_interactions_table(ax_int, fmn_fake_L, set())  # FMN self
        # Re-title to reflect FMN
        ax_int.set_title(f"FMN binding mode  (redock INTER = "
                         f"{fmn['rdock_INTER_redock']:.2f})", fontsize=11,
                         fontweight="bold", pad=8)

        ax_dl = fig.add_axes([0.08, 0.30, 0.40, 0.18])
        draw_props_table(ax_dl, fmn_dl, ref_dl=None, title="FMN properties")

        ax_note = fig.add_axes([0.55, 0.30, 0.40, 0.18])
        ax_note.axis("off")
        note = (
            "INTERPRETATION\n\n"
            "• FMN's binding mode is dominated by 2 H-bond acceptors\n"
            "  (G9, G27 — both on the isoalloxazine/phosphate edge),\n"
            "  plus VdW contacts across the G9-G10-U12-G24-A25-G27-G28\n"
            "  shell, and one Hydrophobic contact with U12.\n\n"
            "• These 3 residues {G9, U12, G27} form the FMN 'signature'\n"
            "  contact set we compare neutral-16 against.\n\n"
            "• FMN itself has very poor drug-likeness (QED 0.18, MW 456,\n"
            "  6 HBD / 10 HBA, TPSA 208) — it is a cofactor, not a drug.\n"
            "  So beating FMN on drug-likeness is a low bar; the real\n"
            "  question is whether neutral ligands preserve the binding\n"
            "  mode while being more drug-like."
        )
        ax_note.text(0.0, 1.0, note, va="top", ha="left", fontsize=9,
                     family="monospace")

        pdf.savefig(fig); plt.close(fig)

        # ============== ONE PAGE PER NEUTRAL LIGAND ==============
        for L in ligands:  # already in combined_rank order
            fig = plt.figure(figsize=(8.27, 11.69))
            pid = L["pose_id"]
            title = (f"{pid}  (gen_idx={L['gen_idx']}, combined_rank={L['combined_rank']}, "
                     f"rnasmol P={L['rnasmol_p_binder']:.3f}, rdock INTER={L['rdock_INTER']:.2f})")
            page_header(fig, title, subtitle=L["smiles"])

            mol = poses.get(pid)
            ax_img = fig.add_axes([0.08, 0.55, 0.45, 0.35])
            if mol is not None:
                ax_img.imshow(mol_to_svg_img(mol, size=600))
            ax_img.axis("off")
            ax_img.set_title(f"{pid} 2D structure", fontsize=11, fontweight="bold")

            ax_int = fig.add_axes([0.55, 0.45, 0.40, 0.45])
            draw_interactions_table(ax_int, L, fmn_res_set)

            ax_dl = fig.add_axes([0.08, 0.30, 0.40, 0.18])
            draw_props_table(ax_dl, L["druglikeness"], ref_dl=fmn_dl,
                             title=f"{pid} properties (vs FMN)")

            ax_extra = fig.add_axes([0.55, 0.08, 0.40, 0.30])
            ax_extra.axis("off")
            summ = L["interaction_summary"]
            extra_text = (
                f"FMN overlap\n"
                f"  shared key residues (no VdW): "
                f"{L['fmn_shared_residues_no_vdw_count']}\n"
                f"  shared: {L['fmn_overlap_residues_no_vdw']}\n"
                f"  extra (not in FMN): {L['extra_residues_no_vdw']}\n"
                f"  residue Jaccard (no VdW): "
                f"{L['residue_jaccard_no_vdw']:.2f}\n\n"
                f"Composite (binding 0.5 + QED 0.3 + SA 0.2)\n"
                f"  score = {L['composite']:.3f}\n"
                f"  rank among 16: "
                f"{ligands_sorted_by_score.index(L)+1}/16\n\n"
                f"Interaction-type totals\n"
            )
            for k, v in sorted(summ["by_type"].items(), key=lambda x: -x[1]):
                extra_text += f"  {k:14s} {v}\n"
            extra_text += (
                f"\nFMN residue set (signature): "
                f"{sorted(fmn_res_set)}\n"
                f"* = residue shared with FMN signature (any interaction)\n"
            )
            ax_extra.text(0.0, 1.0, extra_text, va="top", ha="left",
                          fontsize=8.6, family="monospace")

            pdf.savefig(fig); plt.close(fig)

        # ============== TOP-5 HIGHLIGHT PAGE ==============
        fig = plt.figure(figsize=(8.27, 11.69))
        page_header(fig, "Top-5 overall (binding + drug-likeness)",
                    subtitle="composite = 0.5 normalized(rdock INTER) + 0.3 QED + 0.2 normalized(SA)")
        for i, L in enumerate(top5):
            row = i // 2
            col = i % 2
            left = 0.05 + col * 0.48
            bottom = 0.62 - row * 0.30
            ax = fig.add_axes([left, bottom, 0.20, 0.25])
            mol = poses.get(L["pose_id"])
            if mol is not None:
                ax.imshow(mol_to_svg_img(mol, size=380))
            ax.axis("off")
            dl = L["druglikeness"]
            ax.set_title(f"#{i+1}  {L['pose_id']}  gen={L['gen_idx']}",
                         fontsize=10, fontweight="bold")
            ax_txt = fig.add_axes([left + 0.21, bottom, 0.27, 0.25])
            ax_txt.axis("off")
            summ = L["interaction_summary"]
            txt = (
                f"composite score: {L['composite']:.3f}\n"
                f"rnasmol P  : {L['rnasmol_p_binder']:.3f}\n"
                f"rdock INTER: {L['rdock_INTER']:.2f}\n"
                f"MW / QED / SA: {dl['MW']} / {dl['QED']} / {dl['SA']}\n"
                f"heavy / HBD / HBA: {dl['heavy_atoms']} / {dl['HBD']} / {dl['HBA']}\n"
                f"rings / aromatic / rot: {dl['rings']} / {dl['aromatic_rings']} / {dl['rotatable']}\n"
                f"TPSA / cLogP: {dl['TPSA']} / {dl['cLogP']}\n\n"
                f"interactions: total {summ['total']}\n"
            )
            for k, v in sorted(summ["by_type"].items(), key=lambda x: -x[1]):
                txt += f"   {k:14s} {v}\n"
            txt += (
                f"\nFMN shared: {L['fmn_shared_residues_no_vdw_count']}  "
                f"(Jacc {L['residue_jaccard_no_vdw']:.2f})\n"
                f"key res: {sorted(summ['key_residues_no_vdw'].keys())}\n"
            )
            ax_txt.text(0.0, 1.0, txt, va="top", ha="left", fontsize=7.8,
                        family="monospace")

        # bottom: ranking table of all 16
        ax_full = fig.add_axes([0.05, 0.05, 0.90, 0.22])
        ax_full.axis("off")
        rows = []
        for r, L in enumerate(ligands_sorted_by_score, 1):
            dl = L["druglikeness"]
            summ = L["interaction_summary"]
            rows.append([
                r, L["pose_id"], L["gen_idx"], L["combined_rank"],
                f"{L['rdock_INTER']:.2f}", dl["MW"], dl["QED"], dl["SA"],
                L["fmn_shared_residues_no_vdw_count"],
                f"{L['residue_jaccard_no_vdw']:.2f}",
                f"{L['composite']:.3f}",
            ])
        t = ax_full.table(
            cellText=rows,
            colLabels=["rank","pose","gen","c_rank","INTER","MW","QED","SA",
                       "FMN∩","Jacc","comp"],
            loc="center", cellLoc="center",
        )
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1.0, 1.25)
        ax_full.set_title("Full 16 ranking by composite score",
                          fontsize=10, fontweight="bold", pad=8)

        pdf.savefig(fig); plt.close(fig)

        # metadata
        d = pdf.infodict()
        d["Title"] = "1FMN neutral-16 deep evaluation"
        d["Author"] = "rna2smol pipeline"
    print(f"-> {OUT_PDF}")
    print(f"   pages: cover + FMN + 16 ligands + top5 = 19")


if __name__ == "__main__":
    main()

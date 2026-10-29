"""Round-2 top-20 evaluation PDF builder.

Pages
-----
1. Cover  : pipeline summary + stats + interaction-type bar chart (top-20 vs FMN)
2. FMN reference page (2D + binding mode + properties)
3-22. One page per top-20 ligand (2D + binding mode + properties + PubChem hit)
23. Top-5 highlight page (composite binding+drug score)
24. PubChem overview page (closest known compounds)
25. Full ranking table page

Inputs
------
- top20_analysis.json (from analyze_top20.py)
- top_binders.sdf    (top-100, props preserved)
- FMN_ligand.sdf     (reference)
- pubchem_top20.tsv  (from pubchem_search_top20.py)

Output
------
- top20_evaluation.pdf
"""
import os, json, csv, sys, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

ROOT = "/home/data/users/gaoletian/rna2smol/casestudy/fmn"
R2   = os.path.join(ROOT, "screening_round2")
JSON_IN   = os.path.join(R2, "top20_analysis.json")
TOPB_IN   = os.path.join(R2, "top_binders.sdf")
PUBCH_IN  = os.path.join(R2, "pubchem_top20.tsv")
FMN_SDF   = os.path.join(ROOT, "FMN_ligand.sdf")
OUT_PDF   = os.path.join(R2, "top20_evaluation.pdf")


def mol_to_svg_img(mol, size=420):
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
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=9,
                 color="#444", wrap=True)


def draw_props_table(ax, dl, ref_dl=None, title="Drug-likeness"):
    ax.axis("off")
    keys = ["MW", "QED", "SA", "heavy_atoms", "HBD", "HBA",
            "rotatable", "rings", "aromatic_rings", "TPSA", "cLogP",
            "formal_charge"]
    rows = []
    have_ref = ref_dl is not None
    for k in keys:
        v = dl.get(k, "")
        if have_ref:
            rv = ref_dl[k]
            mark = ""
            if k == "QED" and isinstance(v, (int, float)) and v > ref_dl.get("QED", 0):
                mark = "  >"
            if k == "SA" and isinstance(v, (int, float)) and v < ref_dl.get("SA", 99):
                mark = "  <"
            rows.append([k, str(v), str(rv) + mark])
        else:
            rows.append([k, str(v)])
    col_labels = ["property", "value", "FMN ref"] if have_ref else ["property", "value"]
    col_widths = [0.35, 0.3, 0.35] if have_ref else [0.5, 0.5]
    t = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                 cellLoc="center", colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1.0, 1.30)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)


def draw_interactions_table(ax, L, fmn_residues_set):
    ax.axis("off")
    summ = L["interaction_summary"]
    by_res = summ["by_residue"]
    rows = []
    for r in sorted(by_res.keys()):
        ints = by_res[r]
        flag = " *" if r in fmn_residues_set else ""
        rows.append([r, ", ".join(ints), flag])
    if not rows:
        rows.append(["(none)", "", ""])
    t = ax.table(cellText=rows,
                 colLabels=["residue", "interactions", "FMN?"],
                 loc="center", cellLoc="left",
                 colWidths=[0.22, 0.63, 0.15])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1.0, 1.20)
    ax.set_title(f"Binding mode (total {summ['total']}, key res w/o VdW: "
                 f"{len(summ['key_residues_no_vdw'])})",
                 fontsize=10, fontweight="bold", pad=6)


def composite_score(L, fmn_dl):
    """composite = 0.5 normalized INTER + 0.3 QED + 0.2 normalized(8-SA)."""
    dl = L["druglikeness"]
    inter = L["rdock_INTER"]
    bind = np.clip((inter - (-10)) / ((-30) - (-10)), 0, 1)
    qed = dl["QED"]
    sa = np.clip((8 - dl["SA"]) / (8 - 2), 0, 1)
    return 0.5 * bind + 0.3 * qed + 0.2 * sa


def load_pubchem(path):
    out = {}  # rank -> dict
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for r in rdr:
            try:
                rank = int(r["rank"])
            except (ValueError, KeyError):
                continue
            t = r.get("top_tanimoto", "")
            try:
                t = float(t)
            except ValueError:
                t = None
            out[rank] = {
                "n_hits": int(r.get("n_hits", 0) or 0),
                "top_cid": r.get("top_cid", ""),
                "top_name": r.get("top_name", ""),
                "top_tanimoto": t,
                "top_hit_smiles": r.get("top_hit_smiles", ""),
                "note": r.get("note", ""),
            }
    return out


def main():
    with open(JSON_IN) as f:
        data = json.load(f)
    ligands = data["ligands"]
    fmn = data["fmn_reference"]
    fmn_dl = fmn["druglikeness"]
    fmn_res_set = set(fmn["interaction_summary"]["by_residue"].keys())
    pubchem = load_pubchem(PUBCH_IN)
    print(f"loaded {len(ligands)} ligands, {len(pubchem)} pubchem rows")

    # 2D mol dict by pose_id (T01..T20) — use sanitized heavy-atom mol from SDF
    topb = [m for m in Chem.SDMolSupplier(TOPB_IN, removeHs=False, sanitize=False)]
    mols_by_rank = {}
    for i, m in enumerate(topb[:20]):
        try:
            ms = Chem.Mol(m)
            Chem.SanitizeMol(ms)
            ms = Chem.RemoveHs(ms)
        except Exception:
            ms = Chem.Mol(m)
        mols_by_rank[i + 1] = ms

    fmn_mol = Chem.SDMolSupplier(FMN_SDF, removeHs=False, sanitize=False)[0]

    # composite score
    for L in ligands:
        L["composite"] = composite_score(L, fmn_dl)
    ligands_by_score = sorted(ligands, key=lambda x: -x["composite"])
    top5 = ligands_by_score[:5]

    with PdfPages(OUT_PDF) as pdf:
        # ============== COVER PAGE ==============
        fig = plt.figure(figsize=(8.27, 11.69))
        page_header(fig, "1FMN screening round-2: top-20 evaluation",
                    subtitle="All-neutral 4247 generated ligands -> rnasmol+rdock double screen -> top-20")
        ax_text = fig.add_axes([0.06, 0.50, 0.55, 0.40])
        ax_text.axis("off")
        n_qed_good = sum(1 for L in ligands if L["druglikeness"]["QED"] >= 0.5)
        n_qed_high = sum(1 for L in ligands if L["druglikeness"]["QED"] >= 0.7)
        n_inter_beat_fmn = sum(1 for L in ligands
                               if L["rdock_INTER"] < fmn["rdock_INTER_redock_bestSCOREpose"])
        n_pubch = sum(1 for r, p in pubchem.items() if p["n_hits"] > 0)
        n_pubch_hi = sum(1 for r, p in pubchem.items()
                         if p["top_tanimoto"] is not None
                         and p["top_tanimoto"] >= 0.4)
        max_tani = max((p["top_tanimoto"] for p in pubchem.values()
                        if p["top_tanimoto"] is not None), default=None)

        summary = (
            "PIPELINE (round-2)\n"
            "  Input: 4247 neutral generated ligands (RNA2smol, post-charge-fix)\n"
            "  Step1: rnasmol GPU scoring (4247/4247, ~5.6 ms/mol)\n"
            "  Step2: rdock top-300 docking (3 poses each, 32-shard parallel, 10s)\n"
            "  Step3: combined rank (0.5 rnasmol_norm + 0.5 rdock_norm)\n"
            "  Step4: PubChem fastsimilarity (threshold 85 -> 70 fallback)\n"
            "  Step5: this PDF (deep evaluation of top-20)\n\n"
            "KEY RESULTS (top-20, all neutral)\n"
            f"  All 20 carry formal_charge = 0 (vs round-1 16/100 neutral)\n"
            f"  QED >= 0.5: {n_qed_good}/20 ; QED >= 0.7: {n_qed_high}/20\n"
            f"  rdock INTER < FMN (-16.79): {n_inter_beat_fmn}/20\n"
            f"  PubChem hits: {n_pubch}/20 have >=1 hit at T>=0.70\n"
            f"  Closest known: Tanimoto max = "
            f"{max_tani:.3f} (>= 0.4: {n_pubch_hi} candidates)\n\n"
            "FMN reference\n"
            f"  rdock SCORE = {fmn['rdock_SCORE_redock']:.2f}  "
            f"INTER (best-SCORE pose) = {fmn['rdock_INTER_redock_bestSCOREpose']:.2f}\n"
            f"  rnasmol P(binder) = 1.000\n"
            f"  MW={fmn_dl['MW']}  QED={fmn_dl['QED']}  SA={fmn_dl['SA']}  "
            f"heavy={fmn_dl['heavy_atoms']}\n"
            f"  crystal-pose key residues: "
            f"{sorted(fmn['interaction_summary']['key_residues_no_vdw'].keys())}\n"
            f"  crystal-pose interactions: "
            f"{fmn['interaction_summary']['by_type']}\n\n"
            "COMPARISON VS ROUND-1\n"
            "  Round-1: 4464 gen (84% charged) -> 16/100 neutral top, "
            "median QED ~0.55\n"
            "  Round-2: 4247 gen (100% neutral) -> 100/100 neutral top, "
            "median QED ~0.52\n"
            "  Round-2 top-20 INTER comparable to round-1 neutral-16, with"
            " cleaner chemotypes\n"
            "  (no phosphonium/oxonium by-products)."
        )
        ax_text.text(0.0, 1.0, summary, va="top", ha="left", fontsize=8.4,
                     family="monospace")

        ax_fmn = fig.add_axes([0.62, 0.62, 0.34, 0.30])
        ax_fmn.imshow(mol_to_svg_img(fmn_mol, size=500))
        ax_fmn.axis("off")
        ax_fmn.set_title("FMN (reference ligand)", fontsize=10,
                         fontweight="bold")

        # interaction-type bar chart
        ax_bar = fig.add_axes([0.10, 0.08, 0.55, 0.34])
        agg = {}
        for L in ligands:
            for k, v in L["interaction_summary"]["by_type"].items():
                agg[k] = agg.get(k, 0) + v
        items = sorted(agg.items(), key=lambda x: -x[1])
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        ax_bar.bar(range(len(names)), vals, color="#3a7ca5",
                   edgecolor="black", label="top-20")
        fmn_counts = fmn["interaction_summary"]["by_type"]
        for i, n in enumerate(names):
            fv = fmn_counts.get(n, 0)
            if fv:
                ax_bar.bar(i, fv, color="#e07a5f", edgecolor="black",
                           alpha=0.85, label="FMN" if i == 0 else None)
        ax_bar.set_xticks(range(len(names)))
        ax_bar.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax_bar.set_ylabel("count across 20 ligands")
        ax_bar.set_title("Interaction-type distribution (top-20, blue) "
                         "vs FMN (orange)", fontsize=10, fontweight="bold")
        ax_bar.legend(["top-20 total", "FMN"], loc="upper right", fontsize=8)
        for i, v in enumerate(vals):
            ax_bar.text(i, v + 0.3, str(v), ha="center", fontsize=8)

        # key residues frequency
        ax_res = fig.add_axes([0.70, 0.08, 0.27, 0.34])
        ax_res.axis("off")
        resc = {}
        for L in ligands:
            for r in L["interaction_summary"]["key_residues_no_vdw"].keys():
                resc[r] = resc.get(r, 0) + 1
        items = sorted(resc.items(), key=lambda x: -x[1])
        rows = [[r, f"{c}/20"] for r, c in items]
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
                    subtitle="Crystal pose from PDB 1FMN (ground truth) -- prolif vs 1FMN pocket")
        ax_img = fig.add_axes([0.12, 0.55, 0.45, 0.35])
        ax_img.imshow(mol_to_svg_img(fmn_mol, size=600))
        ax_img.axis("off")
        ax_img.set_title("FMN 2D structure", fontsize=11, fontweight="bold")

        ax_int = fig.add_axes([0.55, 0.45, 0.40, 0.45])
        fmn_fake_L = {
            "interaction_summary": fmn["interaction_summary"],
            "rdock_INTER": fmn["rdock_INTER_redock_bestSCOREpose"],
        }
        draw_interactions_table(ax_int, fmn_fake_L, set())
        ax_int.set_title(f"FMN binding mode (redock INTER on best-SCORE pose "
                         f"= {fmn['rdock_INTER_redock_bestSCOREpose']:.2f})",
                         fontsize=10, fontweight="bold", pad=8)

        ax_dl = fig.add_axes([0.08, 0.30, 0.40, 0.18])
        draw_props_table(ax_dl, fmn_dl, ref_dl=None, title="FMN properties")

        ax_note = fig.add_axes([0.55, 0.30, 0.40, 0.18])
        ax_note.axis("off")
        note = (
            "INTERPRETATION\n\n"
            "• FMN crystal binding mode: 1 HBAcceptor (G9), VdW x7,\n"
            "  Hydrophobic x1 (U12).\n\n"
            "• Signature contact set = {G9, U12} (non-VdW residues).\n\n"
            "• FMN itself has very poor drug-likeness (QED 0.18, MW 456).\n\n"
            "• Best rdock SCORE (self-dock, 5 poses) = -17.16 reproduces the\n"
            "  literature ~-17 kcal/mol, validating the rdock setup.\n"
        )
        ax_note.text(0.0, 1.0, note, va="top", ha="left",
                     fontsize=8.6, family="monospace")
        pdf.savefig(fig); plt.close(fig)

        # ============== ONE PAGE PER TOP-20 LIGAND ==============
        for L in ligands:
            rank = L["rank"]
            fig = plt.figure(figsize=(8.27, 11.69))
            pid = L["pose_id"]
            title = (f"{pid}  (gen_idx={L['gen_idx']}, "
                     f"combined_rank={L['combined_rank']}, "
                     f"P={L['rnasmol_p_binder']:.3f}, "
                     f"INTER={L['rdock_INTER']:.2f})")
            page_header(fig, title, subtitle=L["smiles"])

            mol = mols_by_rank.get(rank)
            ax_img = fig.add_axes([0.08, 0.55, 0.45, 0.35])
            if mol is not None:
                ax_img.imshow(mol_to_svg_img(mol, size=600))
            ax_img.axis("off")
            ax_img.set_title(f"{pid} 2D structure", fontsize=11,
                             fontweight="bold")

            ax_int = fig.add_axes([0.55, 0.45, 0.40, 0.45])
            draw_interactions_table(ax_int, L, fmn_res_set)

            ax_dl = fig.add_axes([0.08, 0.30, 0.40, 0.18])
            draw_props_table(ax_dl, L["druglikeness"], ref_dl=fmn_dl,
                             title=f"{pid} properties (vs FMN)")

            ax_extra = fig.add_axes([0.55, 0.08, 0.40, 0.30])
            ax_extra.axis("off")
            summ = L["interaction_summary"]
            pc = pubchem.get(rank, {})
            tani = pc.get("top_tanimoto")
            tani_s = f"{tani:.3f}" if tani is not None else "n/a"
            extra_text = (
                f"FMN overlap\n"
                f"  shared key res (no VdW): "
                f"{L['fmn_shared_residues_no_vdw_count']}\n"
                f"  shared: {L['fmn_overlap_residues_no_vdw']}\n"
                f"  extra: {L['extra_residues_no_vdw']}\n"
                f"  residue Jaccard (no VdW): "
                f"{L['residue_jaccard_no_vdw']:.2f}\n\n"
                f"Composite (bind 0.5 + QED 0.3 + SA 0.2)\n"
                f"  score = {L['composite']:.3f}\n"
                f"  rank among 20: "
                f"{ligands_by_score.index(L)+1}/20\n\n"
                f"PubChem nearest known compound\n"
                f"  hits: {pc.get('n_hits', 0)}\n"
                f"  top Tanimoto: {tani_s}\n"
                f"  top CID: {pc.get('top_cid', '')}\n"
                f"  top name: {str(pc.get('top_name', ''))[:60]}\n"
            )
            ax_extra.text(0.0, 1.0, extra_text, va="top", ha="left",
                          fontsize=8.2, family="monospace")

            # bottom: PubChem hit 2D structure (if any)
            if pc.get("top_hit_smiles"):
                hit_mol = Chem.MolFromSmiles(pc["top_hit_smiles"])
                if hit_mol is not None:
                    ax_hit = fig.add_axes([0.08, 0.04, 0.30, 0.22])
                    ax_hit.imshow(mol_to_svg_img(hit_mol, size=400))
                    ax_hit.axis("off")
                    ax_hit.set_title(
                        f"PubChem CID {pc.get('top_cid','?')}\n"
                        f"Tanimoto {tani_s}",
                        fontsize=8.5, fontweight="bold")

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
            mol = mols_by_rank.get(L["rank"])
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
                f"composite: {L['composite']:.3f}\n"
                f"rnasmol P  : {L['rnasmol_p_binder']:.3f}\n"
                f"rdock INTER: {L['rdock_INTER']:.2f}\n"
                f"MW / QED / SA: {dl['MW']} / {dl['QED']} / {dl['SA']}\n"
                f"heavy / HBD / HBA: {dl['heavy_atoms']} / {dl['HBD']} / {dl['HBA']}\n"
                f"rings / aro / rot: {dl['rings']} / {dl['aromatic_rings']} / {dl['rotatable']}\n"
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
            ax_txt.text(0.0, 1.0, txt, va="top", ha="left", fontsize=7.6,
                        family="monospace")

        ax_full = fig.add_axes([0.05, 0.05, 0.90, 0.22])
        ax_full.axis("off")
        rows = []
        for r, L in enumerate(ligands_by_score, 1):
            dl = L["druglikeness"]
            pc = pubchem.get(L["rank"], {})
            t = pc.get("top_tanimoto")
            t_s = f"{t:.2f}" if t is not None else "-"
            rows.append([
                r, L["pose_id"], L["gen_idx"], L["combined_rank"],
                f"{L['rdock_INTER']:.2f}", dl["MW"], dl["QED"], dl["SA"],
                L["fmn_shared_residues_no_vdw_count"],
                f"{L['residue_jaccard_no_vdw']:.2f}",
                t_s,
                f"{L['composite']:.3f}",
            ])
        t = ax_full.table(
            cellText=rows,
            colLabels=["rk","pose","gen","cR","INTER","MW","QED","SA",
                       "FMN∩","Jacc","Tani","comp"],
            loc="center", cellLoc="center",
        )
        t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1.0, 1.25)
        ax_full.set_title("Full 20 ranking by composite score",
                          fontsize=10, fontweight="bold", pad=8)
        pdf.savefig(fig); plt.close(fig)

        # ============== PUBCHEM OVERVIEW PAGE ==============
        fig = plt.figure(figsize=(8.27, 11.69))
        page_header(fig, "PubChem nearest-known-compound overview",
                    subtitle="Tanimoto (Morgan FP r=2, 2048 bit) between top-20 query and PubChem hit")
        # sort by Tanimoto desc
        def tani_of(L):
            pc = pubchem.get(L["rank"], {})
            return pc.get("top_tanimoto") or 0.0
        lig_by_tani = sorted(ligands, key=lambda L: -tani_of(L))
        for i, L in enumerate(lig_by_tani):
            row = i // 2
            col = i % 2
            left = 0.05 + col * 0.48
            bottom = 0.85 - row * 0.18
            # query mol
            ax = fig.add_axes([left, bottom, 0.10, 0.14])
            mol = mols_by_rank.get(L["rank"])
            if mol is not None:
                ax.imshow(mol_to_svg_img(mol, size=240))
            ax.axis("off")
            ax.set_title(f"{L['pose_id']} gen={L['gen_idx']}",
                         fontsize=8, fontweight="bold")
            # hit mol + text
            pc = pubchem.get(L["rank"], {})
            hit_smi = pc.get("top_hit_smiles", "")
            hit_mol = Chem.MolFromSmiles(hit_smi) if hit_smi else None
            ax_h = fig.add_axes([left + 0.10, bottom, 0.10, 0.14])
            if hit_mol is not None:
                ax_h.imshow(mol_to_svg_img(hit_mol, size=240))
            ax_h.axis("off")
            tani = pc.get("top_tanimoto")
            t_s = f"{tani:.3f}" if tani is not None else "n/a"
            ax_h.set_title(f"T={t_s}", fontsize=8, fontweight="bold",
                           color="#a32000")
            ax_tx = fig.add_axes([left + 0.21, bottom, 0.27, 0.14])
            ax_tx.axis("off")
            tx = (f"CID {pc.get('top_cid','')}\n"
                  f"{str(pc.get('top_name',''))[:55]}\n"
                  f"hits={pc.get('n_hits',0)}  "
                  f"note={pc.get('note','')}")
            ax_tx.text(0.0, 1.0, tx, va="top", ha="left",
                       fontsize=6.8, family="monospace")
        pdf.savefig(fig); plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "1FMN screening round-2 -- top-20 evaluation"
        d["Author"] = "rna2smol pipeline"

    print(f"-> {OUT_PDF}")
    print(f"   pages: cover + FMN + 20 ligands + top5 + pubchem = 24")


if __name__ == "__main__":
    main()

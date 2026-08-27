"""Generate PDF report for mol4 reproduction products (spec07_iterative).

Inputs:
- casestudy/8hzk/spec07_iterative/mol4_top_hits.sdf : 20 products
- casestudy/8hzk/8HZK_pocket_addh.pdb               : receptor (with H)
- casestudy/8hzk/dummy_ligand.sdf                    : 1-atom ligand marking pocket center
- util/smina.static                                  : docking rescoring

Per-product panel: 2D structure + MW / QED / SA / heavy atoms /
                   docking affinity (smina --minimize) / LBE /
                   Tanimoto vs mol4 / Murcko-same-as-mol4.
Products sorted by Tanimoto descending.

Output: casestudy/8hzk/mol4_reproduction_report.pdf
"""
import os
import sys
import subprocess
import tempfile
from io import BytesIO

# project root on path for util.sascorer
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, QED
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

from util.sascorer import calculateScore as sa_score

CASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CASE))  # rna2smol repo root
INPUT_SDF = os.path.join(CASE, "spec07_iterative", "mol4_top_hits.sdf")
RECEPTOR = os.path.join(CASE, "8HZK_pocket_addh.pdb")
DUMMY_LIG = os.path.join(CASE, "dummy_ligand.sdf")
SMINA = os.path.join(ROOT, "util", "smina.static")
OUT_PDF = os.path.join(CASE, "mol4_reproduction_report.pdf")

MOL4_SMILES = "CN(CCO)c1cccc(/C=C(\\C#N)c2ccc(-c3ccccc3)cc2)c1"


# ------------------------------------------------------------------
def mol_to_image(mol, size=(520, 520), kekulize=True):
    """Render a 2D molecule structure to a numpy image array."""
    mc = Chem.Mol(mol.ToBinary())
    try:
        Chem.Kekulize(mc, clearAromaticFlags=True) if kekulize else None
        AllChem.Compute2DCoords(mc)
    except Exception:
        kekulize = False
        mc = Chem.Mol(mol.ToBinary())
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.padding = 0.08
    opts.baseFontSize = 0.8
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    arr = imread(BytesIO(png))
    return arr


def smina_minimize(lig_sdf, out_sdf):
    """Score a single ligand pose via smina --minimize with autobox on the
    pocket dummy ligand. Returns affinity (float) or None on failure."""
    if not os.access(SMINA, os.X_OK):
        os.chmod(SMINA, 0o755)
    cmd = [
        SMINA, "-r", RECEPTOR, "-l", lig_sdf, "-o", out_sdf,
        "--minimize", "--minimize_iters", "1000",
        "--autobox_ligand", DUMMY_LIG, "--cpu", "1",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    sup = list(Chem.SDMolSupplier(out_sdf))
    if not sup or sup[0] is None:
        return None
    mol = sup[0]
    if mol.HasProp("minimizedAffinity"):
        return float(mol.GetProp("minimizedAffinity"))
    return None


def ensure_3d(mol):
    """Embed a 3D conformer (ETKDG + UFF) if absent."""
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
    return mol


def calc_props(mol):
    return {
        "MW": Descriptors.MolWt(mol),
        "QED": QED.qed(mol),
        "SA": sa_score(mol),
        "heavy": mol.GetNumHeavyAtoms(),
    }


def write_temp_sdf(mol):
    fd, path = tempfile.mkstemp(suffix=".sdf", dir="/tmp/opencode")
    os.close(fd)
    w = Chem.SDWriter(path)
    w.write(mol)
    w.close()
    return path


# ------------------------------------------------------------------
def main():
    print(f"Loading {INPUT_SDF}")
    mols = list(Chem.SDMolSupplier(INPUT_SDF))
    mols = [m for m in mols if m is not None]
    print(f"  {len(mols)} products")

    # ---- reference mol4 ----
    ref = Chem.MolFromSmiles(MOL4_SMILES)
    ref_props = calc_props(ref)

    # ---- products: compute props + dock ----
    records = []
    for i, m in enumerate(mols):
        smi = Chem.MolToSmiles(m)
        props = calc_props(m)
        tani = float(m.GetPropsAsDict().get("Tanimoto_vs_mol4", float("nan")))
        murcko = str(m.GetPropsAsDict().get("Murcko_same_as_mol4", "?"))
        # docking: try minimizedAffinity prop, else run smina
        if m.HasProp("minimizedAffinity"):
            aff = float(m.GetProp("minimizedAffinity"))
            dock_src = "prop"
        else:
            m3d = ensure_3d(Chem.Mol(m))
            in_sdf = write_temp_sdf(m3d)
            out_sdf = in_sdf.replace(".sdf", "_min.sdf")
            aff = smina_minimize(in_sdf, out_sdf)
            dock_src = "smina"
            os.path.exists(in_sdf) and os.remove(in_sdf)
            os.path.exists(out_sdf) and os.remove(out_sdf)
        lbe = (-aff / props["heavy"]) if (aff is not None and props["heavy"]) else float("nan")
        records.append({
            "idx": i,
            "smi": smi,
            "mol": m,
            "props": props,
            "affinity": aff,
            "aff_src": dock_src,
            "lbe": lbe,
            "tani": tani,
            "murcko": murcko,
        })
        print(f"  [{i:>2}] {smi[:60]:<60} tan={tani:.3f} aff={aff} LBE={lbe:.3f}")

    # sort by Tanimoto descending
    records.sort(key=lambda r: r["tani"], reverse=True)

    # overview statistics
    affs = np.array([r["affinity"] for r in records if r["affinity"] is not None])
    lbes = np.array([r["lbe"] for r in records if r["lbe"] is not None])
    tanis = np.array([r["tani"] for r in records])
    stats = {
        "Tanimoto": (np.min(tanis), np.median(tanis), np.max(tanis)),
        "Affinity": (np.min(affs), np.median(affs), np.max(affs)) if len(affs) else None,
        "LBE": (np.min(lbes), np.median(lbes), np.max(lbes)) if len(lbes) else None,
    }
    n_aff_ok = int(np.sum(~np.isnan(affs))) if len(affs) else 0

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------
    with PdfPages(OUT_PDF) as pdf:
        # ===== Page 1: cover + reference + overview =====
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("mol4 Reproduction Report  (8HZK pocket)",
                     fontsize=18, fontweight="bold", y=0.97)

        # reference panel
        ax_ref = fig.add_axes([0.30, 0.55, 0.40, 0.35])
        ax_ref.imshow(mol_to_image(ref))
        ax_ref.set_title("Reference  mol4", fontsize=12, fontweight="bold")
        ax_ref.axis("off")
        ref_txt = (
            f"SMILES:  {MOL4_SMILES}\n\n"
            f"MW          : {ref_props['MW']:.2f}\n"
            f"QED         : {ref_props['QED']:.3f}\n"
            f"SA          : {ref_props['SA']:.2f}\n"
            f"Heavy atoms : {ref_props['heavy']}"
        )
        fig.text(0.08, 0.50, ref_txt, fontsize=9, family="monospace", va="top")

        # overview stats
        ov_lines = ["Overview  (20 products, sorted by Tanimoto desc)", ""]
        ov_lines.append(f"Products scored    : {len(records)}")
        ov_lines.append(f"Docking succeeded  : {n_aff_ok}/{len(records)}")
        ov_lines.append("")
        ov_lines.append("Tanimoto vs mol4")
        if stats["Tanimoto"] is not None:
            mn, md, mx = stats["Tanimoto"]
            ov_lines.append(f"   min={mn:.3f}   median={md:.3f}   max={mx:.3f}")
        ov_lines.append("")
        ov_lines.append("Docking affinity (kcal/mol)")
        if stats["Affinity"] is not None:
            mn, md, mx = stats["Affinity"]
            ov_lines.append(f"   min={mn:.3f}   median={md:.3f}   max={mx:.3f}")
        ov_lines.append("")
        ov_lines.append("LBE  (-affinity / heavy atoms)")
        if stats["LBE"] is not None:
            mn, md, mx = stats["LBE"]
            ov_lines.append(f"   min={mn:.3f}   median={md:.3f}   max={mx:.3f}")
        fig.text(0.08, 0.42, "\n".join(ov_lines), fontsize=10,
                 family="monospace", va="top")
        fig.text(0.08, 0.05,
                 "Tools: RDKit (Draw/Descriptors/QED) | util/sascorer.py | util/smina.static\n"
                 f"Receptor: {os.path.basename(RECEPTOR)}   |   Products: {os.path.basename(INPUT_SDF)}",
                 fontsize=8, color="gray", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # ===== Product pages: 2 per page =====
        per_page = 2
        for k in range(0, len(records), per_page):
            page_records = records[k:k + per_page]
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle(
                f"mol4 Reproduction Products  ({k + 1}–{min(k + per_page, len(records))}/{len(records)})",
                fontsize=14, fontweight="bold", y=0.97,
            )
            for j, rec in enumerate(page_records):
                top = 0.88 - j * 0.48
                # structure
                ax = fig.add_axes([0.06, top - 0.36, 0.45, 0.40])
                ax.imshow(mol_to_image(rec["mol"]))
                ax.set_title(f"#{rec['idx']}   Tanimoto={rec['tani']:.3f}",
                             fontsize=10, fontweight="bold")
                ax.axis("off")
                # property table
                p = rec["props"]
                aff_str = f"{rec['affinity']:.3f}" if rec["affinity"] is not None else "MISSING"
                lbe_str = f"{rec['lbe']:.3f}" if rec["lbe"] is not None else "—"
                aff_note = "" if rec["aff_src"] == "prop" else f"  ({rec['aff_src']})"
                table = (
                    f"SMILES       {rec['smi']}\n\n"
                    f"MW               {p['MW']:.2f}\n"
                    f"QED              {p['QED']:.3f}\n"
                    f"SA score         {p['SA']:.2f}\n"
                    f"Heavy atoms      {p['heavy']}\n"
                    f"Docking affinity {aff_str}{aff_note}\n"
                    f"LBE              {lbe_str}\n"
                    f"Tanimoto vs mol4 {rec['tani']:.3f}\n"
                    f"Murcko == mol4   {rec['murcko']}"
                )
                fig.text(0.55, top, table, fontsize=9, family="monospace", va="top")
            pdf.savefig(fig)
            plt.close(fig)

        metadata = pdf.infodict()
        metadata["Title"] = "mol4 Reproduction Report (8HZK)"
        metadata["Subject"] = "20 mol4 reproduction products — properties, smina docking, LBE, Tanimoto"

    print(f"\nPDF written: {OUT_PDF}")
    n_pages = 1 + (len(records) + per_page - 1) // per_page
    print(f"Pages      : {n_pages}")
    print(f"Docking OK : {n_aff_ok}/{len(records)}")
    print("Stats:")
    for k, v in stats.items():
        if v is None:
            print(f"  {k:10s}: <no data>")
        else:
            mn, md, mx = v
            print(f"  {k:10s}: min={mn:.3f} median={md:.3f} max={mx:.3f}")


if __name__ == "__main__":
    main()

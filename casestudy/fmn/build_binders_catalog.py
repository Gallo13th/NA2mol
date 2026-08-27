"""Build final_binders_catalog.pdf for the 678 FMN binders.

Layout:
  - Cover page: overview stats + INTER/LLE/logP histograms + sorting logic
  - FMN reference page (native cofactor, INTER = -16.79 baseline)
  - Scaffold index pages (all scaffolds, frequency + best INTER)
  - Body: 6 molecules per page (3x2 grid), grouped by Murcko scaffold
          scaffold groups ordered by best (most negative) INTER within group;
          within group: LLE desc -> rdock INTER asc -> rnasmol logit desc.

Output: casestudy/fmn/final_binders_catalog.pdf
"""
import io
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

RDLogger.DisableLog("rdApp.*")

ROOT = "/home/data/users/gaoletian/rna2smol/casestudy/fmn"
SDF_PATH = os.path.join(ROOT, "final_binders.sdf")
FMN_PATH = os.path.join(ROOT, "FMN_ligand.sdf")
OUT_PDF = os.path.join(ROOT, "final_binders_catalog.pdf")
FMN_INTER = -16.79  # baseline: native FMN cofactor
LOGP_FLOOR = 0.5    # avoid div-by-zero / negative logP in LLE

MOLS_PER_PAGE = 6   # 3 cols x 2 rows
PAGE_W, PAGE_H = 14.0, 9.0  # landscape, larger than A4 for legibility


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def mol_to_img(mol, size=360):
    """Render a clean 2D depiction (override any 3D conformer) to RGBA numpy."""
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    AllChem.Compute2DCoords(m)
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.0
    opts.padding = 0.10
    opts.clearBackground = True
    drawer.DrawMolecule(m)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def short(s, n=48):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def lle_of(inter, logp):
    """LLE = -INTER / max(logP, 0.5). Higher is better."""
    return (-inter) / max(logp, LOGP_FLOOR)


# ----------------------------------------------------------------------------
# load + annotate
# ----------------------------------------------------------------------------
def load_records():
    suppl = Chem.SDMolSupplier(SDF_PATH)
    mols = [m for m in suppl if m is not None]
    print(f"  loaded {len(mols)} molecules")
    records = []
    for m in mols:
        d = m.GetPropsAsDict()
        inter = float(d.get("rdock_INTER", 0.0))
        score = float(d.get("rdock_SCORE", 0.0))
        logit = float(d.get("rnasmol_logit", 0.0))
        p = float(d.get("rnasmol_p_binder", 0.0))
        logp = MolLogP(m)
        mw = Descriptors.MolWt(m)
        ha = m.GetNumHeavyAtoms()
        try:
            scaf_mol = MurckoScaffold.GetScaffoldForMol(m)
            scaf_smi = Chem.MolToSmiles(scaf_mol) if scaf_mol.GetNumAtoms() else ""
        except Exception:
            scaf_smi = ""
        if not scaf_smi:
            scaf_smi = "<acyclic>"
        records.append(
            dict(
                mol=m,
                inter=inter,
                score=score,
                logit=logit,
                p=p,
                logp=logp,
                mw=mw,
                ha=ha,
                lle=lle_of(inter, logp),
                scaf=scaf_smi,
                smiles=str(d.get("canonical_smiles", Chem.MolToSmiles(m))).strip(),
                gen_idx=d.get("gen_idx", ""),
                source_round=d.get("source_round", ""),
                rnasmol_rank=d.get("rnasmol_rank", ""),
                rank=None,
                scaf_id=None,
            )
        )
    return records


def group_and_sort(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["scaf"]].append(r)
    scaf_list = []
    for scaf, recs in groups.items():
        best_inter = min(r["inter"] for r in recs)
        scaf_list.append([scaf, recs, best_inter])
    # scaffolds ordered by best INTER asc (most negative = best), then size desc
    scaf_list.sort(key=lambda x: (x[2], -len(x[1])))
    # within group: LLE desc, INTER asc (more negative), logit desc
    for sid, (scaf, recs, best_inter) in enumerate(scaf_list, 1):
        recs.sort(key=lambda r: (-r["lle"], r["inter"], -r["logit"]))
        for r in recs:
            r["scaf_id"] = sid
    # assign global rank in the final viewing order
    rank = 0
    flat = []
    for scaf, recs, best_inter in scaf_list:
        for r in recs:
            rank += 1
            r["rank"] = rank
            flat.append(r)
    return scaf_list, flat


# ----------------------------------------------------------------------------
# pages
# ----------------------------------------------------------------------------
def make_cover_page(pdf, records, scaf_list):
    inters = np.array([r["inter"] for r in records])
    lles = np.array([r["lle"] for r in records])
    logps = np.array([r["logp"] for r in records])
    mws = np.array([r["mw"] for r in records])
    logits = np.array([r["logit"] for r in records])

    # top-5 scaffolds (by frequency)
    freq_top = sorted(scaf_list, key=lambda x: -len(x[1]))[:5]
    # top-5 molecules (by LLE)
    lle_top = sorted(records, key=lambda r: -r["lle"])[:5]

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.suptitle(
        "FMN Binders Catalog  —  rna2smol",
        fontsize=22, fontweight="bold", y=0.965,
    )

    # ---- left: text summary ----
    fig.text(0.03, 0.89, "Overview", fontsize=14, fontweight="bold")
    summary = (
        f"Molecules                 : {len(records)}\n"
        f"Unique Murcko scaffolds   : {len(scaf_list)}\n"
        f"FMN reference rdock INTER : {FMN_INTER:.2f} kcal/mol\n"
        f"Binders better than FMN   : {int((inters < FMN_INTER).sum())} / {len(records)}\n"
        f"\n"
        f"rdock INTER  (more negative = stronger)\n"
        f"   min {inters.min():.2f}   median {np.median(inters):.2f}   max {inters.max():.2f}\n"
        f"\n"
        f"LLE = -INTER / max(logP, {LOGP_FLOOR})   (higher = more efficient)\n"
        f"   min {lles.min():.2f}   median {np.median(lles):.2f}   max {lles.max():.2f}\n"
        f"\n"
        f"logP (Crippen)\n"
        f"   min {logps.min():.2f}   median {np.median(logps):.2f}   max {logps.max():.2f}\n"
        f"   (# with logP<=0 : {int((logps <= 0).sum())})\n"
        f"\n"
        f"MW\n"
        f"   min {mws.min():.1f}   median {np.median(mws):.1f}   max {mws.max():.1f}\n"
        f"\n"
        f"rnasmol logit\n"
        f"   min {logits.min():.2f}   median {np.median(logits):.2f}   max {logits.max():.2f}\n"
    )
    fig.text(0.03, 0.50, summary, fontsize=9.5, family="monospace", va="top")

    # ---- middle: top-5 scaffolds ----
    fig.text(0.34, 0.89, "Top-5 scaffolds (by frequency)", fontsize=11, fontweight="bold")
    lines = [f"{'scaf#':>6}  {'N':>4}  {'best INTER':>10}  Murcko SMILES"]
    lines.append("-" * 70)
    for scaf, recs, bi in freq_top:
        sid = recs[0]["scaf_id"]
        smi = scaf if scaf != "<acyclic>" else "<acyclic / no rings>"
        lines.append(f"#{sid:>5d}  {len(recs):>4d}  {bi:>10.2f}  {short(smi, 56)}")
    fig.text(
        0.34, 0.83, "\n".join(lines), fontsize=8.0, family="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    # ---- middle: top-5 molecules ----
    fig.text(0.34, 0.56, "Top-5 molecules (by LLE)", fontsize=11, fontweight="bold")
    lines = [f"{'rank':>4}  {'scaf#':>5}  {'LLE':>6}  {'INTER':>7}  {'logP':>5}  {'logit':>6}  SMILES"]
    lines.append("-" * 90)
    for r in lle_top:
        lines.append(
            f"#{r['rank']:>3d}  #{r['scaf_id']:>4d}  {r['lle']:>6.2f}  "
            f"{r['inter']:>7.2f}  {r['logp']:>5.2f}  {r['logit']:>6.2f}  "
            f"{short(r['smiles'], 46)}"
        )
    fig.text(
        0.34, 0.50, "\n".join(lines), fontsize=8.0, family="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff7f0", edgecolor="#e0b090"),
    )

    # ---- right column: distributions ----
    def _hist(ax, vals, color, title, ref=None, ref_label=None):
        ax.hist(vals, bins=40, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        if ref is not None:
            ax.axvline(ref, color="red", linestyle="--", linewidth=1.6, label=ref_label)
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("# mols", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    ax1 = fig.add_axes([0.69, 0.62, 0.29, 0.24])
    _hist(ax1, inters, "steelblue", "rdock INTER distribution", FMN_INTER, f"FMN {FMN_INTER:.2f}")
    ax1.set_xlabel("rdock INTER (kcal/mol)", fontsize=9)

    ax2 = fig.add_axes([0.69, 0.34, 0.29, 0.24])
    _hist(ax2, lles, "darkorange", "LLE distribution")
    ax2.set_xlabel("LLE = -INTER / max(logP, 0.5)", fontsize=9)

    ax3 = fig.add_axes([0.69, 0.06, 0.29, 0.24])
    _hist(ax3, logps, "seagreen", "logP distribution")
    ax3.set_xlabel("Crippen logP", fontsize=9)

    # ---- bottom: sorting logic ----
    logic = (
        "SORTING / GROUPING\n"
        "  1. Primary grouping : Murcko scaffold (same skeleton kept together).\n"
        "     Scaffold groups ordered by best (most negative) rdock INTER within group.\n"
        "  2. Within group     : LLE desc  ->  rdock INTER asc  ->  rnasmol logit desc.\n"
        "  LLE = -rdock_INTER / max(MolLogP, 0.5)\n"
        "  (high LLE = strong binding per unit lipophilicity; polar strong binders rank high)\n"
        "  Panels starting a new scaffold are tagged  ' NEW'."
    )
    fig.text(
        0.03, 0.27, logic, fontsize=8.6, family="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef3fb", edgecolor="#9bb4d6"),
    )

    fig.text(
        0.5, 0.012,
        f"rna2smol  |  {len(records)} molecules, {len(scaf_list)} scaffolds  "
        f"|  source: final_binders.sdf",
        ha="center", fontsize=8.5, color="gray",
    )
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


def make_fmn_reference_page(pdf):
    fmn_mol = Chem.SDMolSupplier(FMN_PATH)[0]
    img = mol_to_img(fmn_mol, size=720)
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.suptitle(
        "Reference: FMN (native cofactor, PDB 1FMN)",
        fontsize=18, fontweight="bold", y=0.96,
    )
    ax = fig.add_axes([0.18, 0.18, 0.50, 0.70])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("FMN 2D structure", fontsize=12, fontweight="bold")

    info = (
        "FMN baseline\n"
        "---------------------------\n"
        f"rdock INTER (ref) : {FMN_INTER:.2f} kcal/mol\n"
        f"SMILES            : {Chem.MolToSmiles(fmn_mol)}\n"
        f"MW                : {Descriptors.MolWt(fmn_mol):.1f}\n"
        f"Heavy atoms       : {fmn_mol.GetNumHeavyAtoms()}\n"
        f"logP (Crippen)    : {MolLogP(fmn_mol):.2f}\n"
        f"\n"
        "Context\n"
        "---------------------------\n"
        "FMN is the native cofactor of the 1FMN riboswitch.\n"
        "Its docked INTER = -16.79 kcal/mol is the baseline:\n"
        "candidate binders with INTER < -16.79 are considered\n"
        "potentially stronger than the natural cofactor.\n"
        f"\n"
        f"In this catalog: {678} candidate binders all satisfy\n"
        f"rnasmol P(binder) >= 0.9 AND rdock INTER < {FMN_INTER:.2f}.\n"
    )
    fig.text(0.70, 0.50, info, fontsize=10, family="monospace", va="center")
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


def make_scaffold_index_pages(pdf, scaf_list):
    """Index of all scaffolds (frequency + best INTER) — multi-page table."""
    per_page = 38
    n = len(scaf_list)
    pages = (n + per_page - 1) // per_page
    for pg in range(pages):
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        fig.suptitle(
            f"Scaffold index  (page {pg + 1}/{pages})  "
            f"—  ordered by best rdock INTER within scaffold",
            fontsize=13, fontweight="bold", y=0.96,
        )
        ax = fig.add_axes([0.03, 0.04, 0.94, 0.88])
        ax.axis("off")
        header = f"{'scaf#':>6}  {'N':>4}  {'best INTER':>11}  {'median LLE':>11}  Murcko SMILES"
        lines = [header, "-" * 130]
        start, end = pg * per_page, min(n, pg * per_page + per_page)
        for i in range(start, end):
            scaf, recs, bi = scaf_list[i]
            sid = recs[0]["scaf_id"]
            med_lle = float(np.median([r["lle"] for r in recs]))
            smi = scaf if scaf != "<acyclic>" else "<acyclic / no rings>"
            smi = short(smi, 96)
            lines.append(
                f"#{sid:>5d}  {len(recs):>4d}  {bi:>11.2f}  {med_lle:>11.2f}  {smi}"
            )
        ax.text(
            0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.4, family="monospace", va="top",
        )
        pdf.savefig(fig, dpi=110)
        plt.close(fig)
    return pages


def draw_panel(ax, rec, scaf, n_in_scaf, best_inter, is_new):
    img = mol_to_img(rec["mol"], size=360)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # title above image
    new_flag = "  NEW" if is_new else ""
    title = f"#{rec['rank']}  scaf#{rec['scaf_id']} (x{n_in_scaf}){new_flag}"
    color = "firebrick" if is_new else "black"
    ax.set_title(title, fontsize=8.5, fontweight="bold", color=color, pad=3)

    # Murcko scaffold banner (above title) for new scaffold
    if is_new:
        scaf_show = short(scaf, 50) if scaf != "<acyclic>" else "<acyclic>"
        ax.text(
            0.5, 1.085,
            f"Murcko: {scaf_show}   (best INTER {best_inter:.2f})",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.2, style="italic", color="firebrick",
        )

    # below image: properties (LLE bold on first line)
    props = (
        f"LLE={rec['lle']:.2f}   INTER={rec['inter']:.2f}   SCORE={rec['score']:.2f}\n"
        f"P={rec['p']:.3g}  logit={rec['logit']:.2f}  "
        f"MW={rec['mw']:.0f} HA={rec['ha']} logP={rec['logp']:.1f}\n"
        f"{short(rec['smiles'], 52)}"
    )
    ax.text(
        0.5, -0.085, props, transform=ax.transAxes,
        ha="center", va="top", fontsize=6.5, family="monospace",
    )


def make_body_pages(pdf, scaf_list):
    # flatten with per-record metadata
    flat = []
    for scaf, recs, best_inter in scaf_list:
        for j, r in enumerate(recs):
            flat.append((r, scaf, len(recs), best_inter, j == 0))
    total = len(flat)
    n_pages = (total + MOLS_PER_PAGE - 1) // MOLS_PER_PAGE
    for p in range(n_pages):
        chunk = flat[p * MOLS_PER_PAGE:(p + 1) * MOLS_PER_PAGE]
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        fig.suptitle(
            f"FMN binders catalog  —  page {p + 1}/{n_pages}",
            fontsize=9.5, color="gray", y=0.992,
        )
        gs = GridSpec(
            2, 3, figure=fig, hspace=0.55, wspace=0.04,
            left=0.03, right=0.97, top=0.93, bottom=0.04,
        )
        for idx, (rec, scaf, n_in_scaf, best_inter, is_new) in enumerate(chunk):
            r, c = divmod(idx, 3)
            ax = fig.add_subplot(gs[r, c])
            draw_panel(ax, rec, scaf, n_in_scaf, best_inter, is_new)
        pdf.savefig(fig, dpi=110)
        plt.close(fig)
    return n_pages


# ----------------------------------------------------------------------------
def main():
    print("== loading molecules ==")
    records = load_records()
    print("== grouping by Murcko scaffold + sorting ==")
    scaf_list, flat = group_and_sort(records)

    # diagnostics
    lles = np.array([r["lle"] for r in records])
    print(f"  unique scaffolds : {len(scaf_list)}")
    print(f"  LLE  min/median/max : {lles.min():.2f} / {np.median(lles):.2f} / {lles.max():.2f}")
    print("  top-5 scaffolds (frequency):")
    for scaf, recs, bi in sorted(scaf_list, key=lambda x: -len(x[1]))[:5]:
        print(f"     scaf#{recs[0]['scaf_id']:>3}  n={len(recs):>3}  best INTER={bi:.2f}  {short(scaf, 60)}")
    print("  top-5 molecules (LLE):")
    for r in sorted(records, key=lambda r: -r["lle"])[:5]:
        print(f"     rank#{r['rank']:>3} scaf#{r['scaf_id']:>3}  LLE={r['lle']:.2f}  "
              f"INTER={r['inter']:.2f}  logP={r['logp']:.2f}  logit={r['logit']:.2f}")

    print("== writing PDF ==")
    with PdfPages(OUT_PDF) as pdf:
        make_cover_page(pdf, records, scaf_list)
        make_fmn_reference_page(pdf)
        scaf_pages = make_scaffold_index_pages(pdf, scaf_list)
        body_pages = make_body_pages(pdf, scaf_list)
        d = pdf.infodict()
        d["Title"] = "rna2smol FMN binders catalog"
        d["Author"] = "rna2smol"

    total_pages = 1 + 1 + scaf_pages + body_pages
    sz_mb = os.path.getsize(OUT_PDF) / 1e6
    print(
        f"\n== done ==\n"
        f"  PDF      : {OUT_PDF}\n"
        f"  pages    : {total_pages}  (cover 1 + fmn 1 + scaffold index {scaf_pages} + body {body_pages})\n"
        f"  size     : {sz_mb:.1f} MB\n"
        f"  molecules rendered : {len(records)} / 678\n"
        f"  scaffolds          : {len(scaf_list)}\n"
        f"  LLE min/median/max : {lles.min():.2f} / {np.median(lles):.2f} / {lles.max():.2f}"
    )


if __name__ == "__main__":
    main()

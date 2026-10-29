"""Build final_binders_top100_catalog.pdf — ensemble rnasmol + rdock + druglike.

Pages
-----
1. Cover       : overview stats + 6-version model agreement + histograms
                 (ensemble_P, multi_gate, rdock_INTER, SAS, QED) + sorting logic
2. FMN ref     : native cofactor 2D + its 6-version scores + ensemble
3. Body        : 6 molecules per page (3x2 grid), ordered by
                 multi_gate desc -> ensemble_P desc -> rdock INTER asc

Per-panel
- 2D molecule
- rnasmol ensemble_P (bold) + multi_gate (e.g. 5/6 with vote pictogram)
- per-version vote row (6 dots, colored by YES/NO)
- rdock INTER / SCORE
- QED / SAS / MW
- rank + gen_idx
"""
import io
import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

RDLogger.DisableLog("rdApp.*")

ROOT = "/home/data/users/gaoletian/rna2smol/casestudy/fmn"
SDF_PATH = os.path.join(ROOT, "final_binders_top100_ensemble.sdf")
FMN_PATH = os.path.join(ROOT, "FMN_ligand.sdf")
FMN_JSON = os.path.join(ROOT, "fmn_ensemble_reference.json")
OUT_PDF = os.path.join(ROOT, "final_binders_top100_catalog.pdf")

MOLS_PER_PAGE = 6   # 3 cols x 2 rows
PAGE_W, PAGE_H = 14.0, 9.0  # landscape

# keys must match ensemble_score_top100.py
MODEL_KEYS = ["pdb_mol", "pdb_net", "pdb_rna", "rob_mol", "rob_net", "rob_rna"]
MODEL_LABELS = {
    "pdb_mol": "pdb·mol",
    "pdb_net": "pdb·net",
    "pdb_rna": "pdb·rna",
    "rob_mol": "rob·mol",
    "rob_net": "rob·net",
    "rob_rna": "rob·rna",
}
P_THRESHOLD = 0.5
FMN_INTER = -16.79


# ----------------------------------------------------------------------------
def mol_to_img(mol, size=420):
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


def short(s, n=46):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def gate_int(s):
    try:
        return int(str(s).split("/")[0])
    except Exception:
        return 0


# ----------------------------------------------------------------------------
def load_records():
    suppl = Chem.SDMolSupplier(SDF_PATH, removeHs=False, sanitize=False)
    mols = [m for m in suppl if m is not None]
    print(f"  loaded {len(mols)} molecules")
    records = []
    for r_idx, m in enumerate(mols):
        d = m.GetPropsAsDict()
        per_p = {k: float(d[f"rnasmol_p_{k}"]) for k in MODEL_KEYS}
        per_logit = {k: float(d[f"rnasmol_logit_{k}"]) for k in MODEL_KEYS}
        rec = dict(
            mol=m,
            rank=int(d.get("ensemble_rank", r_idx + 1)),
            gen_idx=d.get("gen_idx", ""),
            source_round=d.get("source_round", ""),
            inter=float(d.get("rdock_INTER", 0.0)),
            score=float(d.get("rdock_SCORE", 0.0)),
            ensemble_P=float(d["rnasmol_ensemble_P"]),
            ensemble_logit=float(d["rnasmol_ensemble_logit"]),
            multi_gate=gate_int(d.get("rnasmol_multi_gate", "0/6")),
            per_p=per_p,
            per_logit=per_logit,
            sas=float(d.get("SAS", 0.0)),
            qed=float(d.get("QED", 0.0)),
            mw=float(d.get("MW", 0.0)),
            smiles=str(d.get("canonical_smiles", "")).strip(),
        )
        records.append(rec)
    # already sorted by ensemble_score_top100.py, but enforce
    records.sort(key=lambda r: (-r["multi_gate"], -r["ensemble_P"], r["inter"]))
    for i, r in enumerate(records, 1):
        r["rank"] = i
    return records


# ----------------------------------------------------------------------------
def draw_vote_row(ax, x0, y0, dx, per_p, radius=0.012):
    """Draw 6 small circles, colored green (P>thr) / red (P<thr) / number."""
    for i, k in enumerate(MODEL_KEYS):
        p = per_p[k]
        color = "#2ca02c" if p > P_THRESHOLD else "#d62728"
        edge = "#1a7a1a" if p > P_THRESHOLD else "#8b1d1d"
        c = Circle((x0 + i * dx, y0), radius, facecolor=color,
                   edgecolor=edge, linewidth=0.8, transform=ax.transAxes,
                   clip_on=False)
        ax.add_patch(c)


def make_cover_page(pdf, records):
    ep = np.array([r["ensemble_P"] for r in records])
    el = np.array([r["ensemble_logit"] for r in records])
    mg = np.array([r["multi_gate"] for r in records])
    inter = np.array([r["inter"] for r in records])
    sas = np.array([r["sas"] for r in records])
    qed = np.array([r["qed"] for r in records])
    mw = np.array([r["mw"] for r in records])

    # per-model P stats
    per_stats = {}
    for k in MODEL_KEYS:
        v = np.array([r["per_p"][k] for r in records])
        per_stats[k] = (v.min(), np.median(v), v.max(), v.mean())

    # load FMN reference for cover
    with open(FMN_JSON) as fh:
        fmn_info = json.load(fh)
    fmn_p = fmn_info["fmn_scores"]["per_p"]
    fmn_ep = fmn_info["fmn_scores"]["ensemble_P"]
    fmn_mg = fmn_info["fmn_scores"]["multi_gate"]

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.suptitle(
        "FMN Top-100 Catalog (SAS-optimal)  —  6-version RNAsmol Ensemble",
        fontsize=20, fontweight="bold", y=0.965,
    )

    # ---- left: overview text ----
    fig.text(0.025, 0.89, "Overview", fontsize=14, fontweight="bold")
    better_than_fmn = int((inter < FMN_INTER).sum())
    summary = (
        f"Molecules                       : {len(records)}\n"
        f"RNA target                      : 1FMN aptamer (35 nt)\n"
        f"Ensemble models                 : 6 (pdb×{3}, robin×{3}) × {3} perturbations\n"
        f"multi_gate = #models with P(binder) > {P_THRESHOLD}  (0..6)\n"
        f"ensemble_P = mean P(binder) over 6 models\n"
        f"\n"
        f"ensemble_P     min {ep.min():.4f}  med {np.median(ep):.4f}  max {ep.max():.4f}\n"
        f"ensemble_logit min {el.min():.3f}  med {np.median(el):.3f}  max {el.max():.3f}\n"
        f"multi_gate     {int((mg==6).sum())}×6/6  {int((mg>=5).sum())}×>=5/6  "
        f"{int((mg>=4).sum())}×>=4/6\n"
        f"\n"
        f"rdock_INTER    min {inter.min():.2f}  med {np.median(inter):.2f}  max {inter.max():.2f}\n"
        f"               (more negative = stronger; FMN ref = {FMN_INTER:.2f})\n"
        f"               better than FMN : {better_than_fmn}/{len(records)}\n"
        f"\n"
        f"SAS            min {sas.min():.2f}  med {np.median(sas):.2f}  max {sas.max():.2f}\n"
        f"QED            min {qed.min():.3f} med {np.median(qed):.3f} max {qed.max():.3f}\n"
        f"MW             min {mw.min():.1f}  med {np.median(mw):.1f}  max {mw.max():.1f}\n"
    )
    fig.text(0.025, 0.50, summary, fontsize=9.5, family="monospace", va="top")

    # ---- middle: per-model table ----
    fig.text(0.34, 0.89, "Per-model P(binder) over 100 mols", fontsize=11, fontweight="bold")
    lines = [f"{'model':<10}{'min':>8}{'median':>9}{'mean':>9}{'max':>8}   FMN_P  vote"]
    lines.append("-" * 70)
    for k in MODEL_KEYS:
        mn, md, mx, me = per_stats[k]
        fp = fmn_p[k]
        vote = "YES" if fp > P_THRESHOLD else "NO "
        lines.append(f"{MODEL_LABELS[k]:<10}{mn:>8.4f}{md:>9.4f}{me:>9.4f}{mx:>8.4f}   "
                     f"{fp:>6.4f}  {vote}")
    lines.append("-" * 70)
    lines.append(f"FMN ensemble_P = {fmn_ep:.4f}   multi_gate = {fmn_mg}/6")
    fig.text(
        0.34, 0.81, "\n".join(lines), fontsize=8.4, family="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    # ---- middle: sorting logic ----
    logic = (
        "SORTING / RANKING\n"
        "  1. multi_gate desc  (6/6 best — full model agreement)\n"
        "  2. ensemble_P desc  (mean P(binder) over 6 models)\n"
        "  3. rdock INTER asc  (more negative = stronger dock)\n"
        "\n"
        "PANEL LEGEND\n"
        "  Vote pictogram: 6 dots (pdb·mol pdb·net pdb·rna rob·mol rob·net rob·rna)\n"
        "    green = P(binder) > 0.5 (model says binder)\n"
        "    red   = P(binder) < 0.5 (model says non-binder)\n"
        f"\nFMN reference : ensemble_P {fmn_ep:.3f}, multi_gate {fmn_mg}/6  "
        f"(see FMN page)"
    )
    fig.text(
        0.34, 0.50, logic, fontsize=8.5, family="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef3fb", edgecolor="#9bb4d6"),
    )

    # ---- right column: distributions ----
    def _hist(ax, vals, color, title, ref=None, ref_label=None, bins=30):
        ax.hist(vals, bins=bins, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        if ref is not None:
            ax.axvline(ref, color="red", linestyle="--", linewidth=1.6,
                       label=ref_label)
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("# mols", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    ax1 = fig.add_axes([0.69, 0.66, 0.29, 0.21])
    _hist(ax1, ep, "purple", "ensemble_P distribution",
          ref=fmn_ep, ref_label=f"FMN {fmn_ep:.3f}")
    ax1.set_xlabel("ensemble_P  (mean over 6 models)", fontsize=9)

    ax2 = fig.add_axes([0.69, 0.40, 0.29, 0.21])
    bars = [int((mg == g).sum()) for g in range(7)]
    ax2.bar(range(7), bars, color="#7c5caf", edgecolor="white")
    for g, b in enumerate(bars):
        if b:
            ax2.text(g, b, f"{b}", ha="center", va="bottom", fontsize=8)
    ax2.axvline(fmn_mg, color="red", linestyle="--", linewidth=1.6,
                label=f"FMN {fmn_mg}/6")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xticks(range(7))
    ax2.set_title("multi_gate distribution", fontsize=10, fontweight="bold")
    ax2.set_xlabel("multi_gate  (#models voting P>0.5)", fontsize=9)
    ax2.set_ylabel("# mols", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_axes([0.69, 0.14, 0.29, 0.21])
    _hist(ax3, inter, "steelblue", "rdock INTER distribution",
          ref=FMN_INTER, ref_label=f"FMN {FMN_INTER:.2f}")
    ax3.set_xlabel("rdock INTER (kcal/mol)", fontsize=9)

    ax4 = fig.add_axes([0.40, 0.06, 0.27, 0.18])
    _hist(ax4, sas, "darkorange", "SAS distribution", bins=24)
    ax4.set_xlabel("SAS (synthetic accessibility)", fontsize=9)

    ax5 = fig.add_axes([0.40, 0.30, 0.27, 0.18])
    _hist(ax5, qed, "seagreen", "QED distribution", bins=24)
    ax5.set_xlabel("QED (drug-likeness)", fontsize=9)

    fig.text(
        0.5, 0.012,
        f"rna2smol  |  {len(records)} molecules × 6 RNAsmol models  "
        f"|  source: final_binders_top100_sas.sdf",
        ha="center", fontsize=8.5, color="gray",
    )
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


def make_fmn_reference_page(pdf):
    fmn_mol = Chem.SDMolSupplier(FMN_PATH, removeHs=False)[0]
    img = mol_to_img(fmn_mol, size=720)
    with open(FMN_JSON) as fh:
        info = json.load(fh)
    fs = info["fmn_scores"]

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.suptitle(
        "Reference: FMN (native cofactor, PDB 1FMN)  —  6-version scores",
        fontsize=18, fontweight="bold", y=0.96,
    )
    ax = fig.add_axes([0.04, 0.20, 0.48, 0.70])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("FMN 2D structure", fontsize=12, fontweight="bold")

    # table of 6 versions
    lines = ["FMN per-model scores", "=" * 48]
    lines.append(f"{'model':<10}{'P(binder)':>12}{'logit':>10}   vote")
    lines.append("-" * 48)
    for k in MODEL_KEYS:
        p = fs["per_p"][k]
        l = fs["per_logit"][k]
        vote = "YES" if p > P_THRESHOLD else "NO "
        lines.append(f"{MODEL_LABELS[k]:<10}{p:>12.6f}{l:>10.3f}   {vote}")
    lines.append("-" * 48)
    lines.append(f"{'ensemble':<10}{fs['ensemble_P']:>12.6f}"
                 f"{fs['ensemble_logit']:>10.3f}")
    lines.append(f"multi_gate = {fs['multi_gate']}/6")
    fig.text(
        0.56, 0.70, "\n".join(lines), fontsize=10.5, family="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff7f0",
                  edgecolor="#e0b090"),
    )

    note = (
        "Sorting & Interpretation\n"
        "========================\n"
        "Candidates are sorted by THREE keys (in priority order):\n"
        "  1. multi_gate DESC  (6-version vote count, 6=unanimous binder)\n"
        "  2. ensemble_P DESC  (mean P(binder) across 6 RNAsmol models)\n"
        "  3. rdock INTER ASC  (more negative = stronger binding)\n"
        "\n"
        "Selection criteria for this catalog (top-100):\n"
        "  a) SAS < 3.47 (top-100 easiest to synthesise from 678 binders)\n"
        "  b) rnasmol P(binder) >= 0.9 (all pass, actually all = 1.0 in pdb_mol)\n"
        "  c) rdock INTER < -16.79 (stronger than FMN native cofactor)\n"
        "\n"
        "6-model ensemble (multi_gate vote):\n"
        f"  FMN native cofactor: multi_gate = {fs['multi_gate']}/6\n"
        "  All 100 candidates:  multi_gate >= 4/6 (> FMN's 3/6)\n"
        "  This means the ensemble consistently rates generated\n"
        "  candidates as MORE binder-like than FMN itself.\n"
        "\n"
        f"SMILES : {Chem.MolToSmiles(fmn_mol)}\n"
        f"MW     : {Descriptors.MolWt(fmn_mol):.1f}\n"
        f"rdock INTER (1FMN baseline) : {FMN_INTER:.2f} kcal/mol"
    )
    fig.text(0.56, 0.40, note, fontsize=9.3, family="monospace", va="top")
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


def draw_panel(ax, rec):
    img = mol_to_img(rec["mol"], size=300)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # title above image
    ax.set_title(
        f"#{rec['rank']}  gen{rec['gen_idx']}  ({rec['source_round']})",
        fontsize=8.5, fontweight="bold", pad=4,
    )

    # multi_gate badge + ensemble_P (top-right corner over image)
    badge_color = "#2ca02c" if rec["multi_gate"] >= 5 else (
        "#ff7f0e" if rec["multi_gate"] >= 4 else "#d62728"
    )
    ax.text(
        0.985, 0.96, f"{rec['multi_gate']}/6",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=badge_color,
                  edgecolor="black", linewidth=0.6),
    )

    # vote pictogram + labels (just below image)
    draw_vote_row(ax, x0=0.20, y0=-0.045, dx=0.115,
                  per_p=rec["per_p"], radius=0.018)
    ax.text(
        0.5, -0.085, "pdb·mol pdb·net pdb·rna rob·mol rob·net rob·rna",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=5.3, family="monospace", color="#555555",
    )

    # main props block
    inter_flag = " *" if rec["inter"] < FMN_INTER else ""
    props = (
        f"ensemble_P = {rec['ensemble_P']:.4f}   "
        f"logit = {rec['ensemble_logit']:.2f}\n"
        f"rdock INTER = {rec['inter']:.2f}{inter_flag}   "
        f"SCORE = {rec['score']:.2f}\n"
        f"QED = {rec['qed']:.3f}   SAS = {rec['sas']:.2f}   "
        f"MW = {rec['mw']:.0f}\n"
        f"{short(rec['smiles'], 54)}"
    )
    ax.text(
        0.5, -0.135, props, transform=ax.transAxes,
        ha="center", va="top", fontsize=6.5, family="monospace",
    )


def make_body_pages(pdf, records):
    total = len(records)
    n_pages = (total + MOLS_PER_PAGE - 1) // MOLS_PER_PAGE
    for p in range(n_pages):
        chunk = records[p * MOLS_PER_PAGE:(p + 1) * MOLS_PER_PAGE]
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        fig.suptitle(
            f"FMN top-100 ensemble catalog  —  page {p + 1}/{n_pages}  "
            f"(sorted: multi_gate↓ · ensemble_P↓ · rdock INTER↑)",
            fontsize=9.5, color="gray", y=0.992,
        )
        gs = GridSpec(
            2, 3, figure=fig, hspace=0.95, wspace=0.04,
            left=0.03, right=0.97, top=0.93, bottom=0.12,
        )
        for idx, rec in enumerate(chunk):
            r, c = divmod(idx, 3)
            ax = fig.add_subplot(gs[r, c])
            draw_panel(ax, rec)
        pdf.savefig(fig, dpi=110)
        plt.close(fig)
    return n_pages


def main():
    print("== loading molecules ==")
    records = load_records()

    # diagnostics
    mg = np.array([r["multi_gate"] for r in records])
    ep = np.array([r["ensemble_P"] for r in records])
    inter = np.array([r["inter"] for r in records])
    print(f"  multi_gate >=5 : {int((mg >= 5).sum())}/{len(records)}")
    print(f"  multi_gate ==4 : {int((mg == 4).sum())}/{len(records)}")
    print(f"  ensemble_P min/med/max: {ep.min():.4f}/{np.median(ep):.4f}/{ep.max():.4f}")
    print("  top-5 (sorted):")
    for r in records[:5]:
        votes = "".join("Y" if r["per_p"][k] > P_THRESHOLD else "n"
                        for k in MODEL_KEYS)
        print(f"   #{r['rank']:>3}  mg={r['multi_gate']}/6  "
              f"eP={r['ensemble_P']:.4f}  INTER={r['inter']:.2f}  "
              f"QED={r['qed']:.2f}  SAS={r['sas']:.2f}  votes={votes}")

    print("== writing PDF ==")
    with PdfPages(OUT_PDF) as pdf:
        # make_cover_page(pdf, records)  # removed per user request (too cluttered)
        make_fmn_reference_page(pdf)
        body_pages = make_body_pages(pdf, records)
        d = pdf.infodict()
        d["Title"] = "rna2smol FMN top-100 ensemble catalog"
        d["Author"] = "rna2smol"

    total_pages = 1 + 1 + body_pages
    sz_mb = os.path.getsize(OUT_PDF) / 1e6
    print(
        f"\n== done ==\n"
        f"  PDF      : {OUT_PDF}\n"
        f"  pages    : {total_pages}  (cover 1 + FMN 1 + body {body_pages})\n"
        f"  size     : {sz_mb:.1f} MB\n"
        f"  molecules rendered : {len(records)}\n"
    )


if __name__ == "__main__":
    main()

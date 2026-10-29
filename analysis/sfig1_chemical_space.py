#!/usr/bin/env python
"""Plot the chemical-space audit figure for the RNA2smol manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "benchmark" / "output" / "chemspace_targeted"
PAPER_FIG_DIR = ROOT / "reproduction" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_STEM = "chemical_space_audit"

SMRT_METHODS = [
    "RNA2smol",
    "RNA2smol (Baseline)",
    "MolCRAFT",
    "Pocket2Mol",
    "ED2Mol",
]
DUDE_METHODS = ["RNA2smol", "MolCRAFT", "Pocket2Mol", "ED2Mol"]
SCAFFOLD_METHODS = ["RNA2smol", "MolCRAFT", "Pocket2Mol", "ED2Mol"]
DISPLAY = {
    "RNA2smol": "RNA2smol",
    "RNA2smol (Baseline)": "RNA2smol-base",
    "MolCRAFT": "MolCRAFT",
    "Pocket2Mol": "Pocket2Mol",
    "ED2Mol": "ED2Mol",
}

COL = {
    "ink": "#202124",
    "muted": "#68717A",
    "grid": "#E4E7EA",
    "rna": "#5F9EA0",
    "intermediate": "#D7DDE2",
    "protein": "#C98255",
    "reference": "#56687D",
    "reference_light": "#A9B8C6",
    "smrt": "#246B70",
    "dude": "#A35F3F",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 600,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COL["ink"],
    )


def stacked_legend(ax: plt.Axes, x0: float = 0.02, y: float = 1.13, dx: float = 0.31) -> None:
    items = [
        ("RNA-leaning", COL["rna"]),
        ("Intermediate", COL["intermediate"]),
        ("Protein-leaning", COL["protein"]),
    ]
    for i, (label, color) in enumerate(items):
        x = x0 + i * dx
        ax.scatter(
            [x],
            [y],
            s=24,
            marker="s",
            color=color,
            transform=ax.transAxes,
            clip_on=False,
            edgecolor="none",
        )
        ax.text(
            x + 0.038,
            y,
            label,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=6.3,
            color=COL["ink"],
            clip_on=False,
        )


def plot_reference_overlap(ax: plt.Axes, ref: pd.DataFrame, label: str = "c") -> None:
    rna = ref[ref["query_class"] == "RNA query"].iloc[0]
    protein = ref[ref["query_class"] == "Protein query"].iloc[0]
    auc = float(rna["balanced_classifier_auc_mean"])
    auc_sd = float(rna["balanced_classifier_auc_sd"])

    labels = [
        "AUC",
        "RNA query\nprotein NN >= 0.30",
        "Protein query\nRNA NN >= 0.30",
        "RNA query\nprotein NN >= 0.50",
        "Protein query\nRNA NN >= 0.50",
    ]
    values = np.array(
        [
            auc * 100,
            rna["cross_ge_0.30_fraction"] * 100,
            protein["cross_ge_0.30_fraction"] * 100,
            rna["cross_ge_0.50_fraction"] * 100,
            protein["cross_ge_0.50_fraction"] * 100,
        ],
        dtype=float,
    )
    y = np.arange(len(labels))[::-1]
    colors = [COL["reference"]] + [COL["reference_light"]] * (len(labels) - 1)
    ax.barh(y, values, color=colors, edgecolor="none", height=0.58)
    ax.errorbar(
        auc * 100,
        y[0],
        xerr=auc_sd * 100,
        fmt="none",
        ecolor=COL["ink"],
        elinewidth=0.8,
        capsize=2,
        zorder=3,
    )
    for yi, v in zip(y, values):
        ax.text(min(v + 2.0, 99.0), yi, f"{v:.1f}%", va="center", ha="left", color=COL["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Value (%)")
    ax.set_title("Reference ligand spaces are separable", loc="left", pad=4)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    panel_label(ax, label)


def plot_control_bias(ax: plt.Axes, control: pd.DataFrame, label: str = "d") -> None:
    order = ["SMRT-Net positive ligands", "DUD-E active ligands"]
    display = {
        "SMRT-Net positive ligands": "SMRT-Net\npositive",
        "DUD-E active ligands": "DUD-E\nactive",
    }
    sub = control.set_index("control_set").reindex(order).dropna(how="all").reset_index()
    y = np.arange(len(sub))[::-1]
    parts = [
        ("rna_leaning_fraction", COL["rna"]),
        ("intermediate_fraction", COL["intermediate"]),
        ("protein_leaning_fraction", COL["protein"]),
    ]

    left = np.zeros(len(sub))
    for col, color in parts:
        vals = sub[col].fillna(0).to_numpy(dtype=float) * 100
        ax.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.5, height=0.48)
        for yi, x0, val in zip(y, left, vals):
            if val >= 9:
                ax.text(x0 + val / 2, yi, f"{val:.1f}%", ha="center", va="center", fontsize=5.6, color=COL["ink"])
        left += vals

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, len(sub) - 0.45)
    ax.set_yticks(y)
    ax.set_yticklabels([display.get(v, v) for v in sub["control_set"]])
    ax.set_xlabel("Nearest-neighbour bias (%)")
    ax.set_title("Benchmark-positive controls", loc="left", pad=4)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    panel_label(ax, label, x=-0.10)


def plot_benchmark_bias(
    ax: plt.Axes,
    bias: pd.DataFrame,
    benchmark: str,
    methods: list[str],
    title: str,
    label: str | None = None,
) -> None:
    sub = bias[(bias["benchmark"] == benchmark) & (bias["set"] == "screened")]
    sub = sub.set_index("method").reindex(methods).reset_index()
    y = np.arange(len(sub))[::-1]
    left = np.zeros(len(sub))
    parts = [
        ("rna_leaning_fraction", COL["rna"]),
        ("intermediate_fraction", COL["intermediate"]),
        ("protein_leaning_fraction", COL["protein"]),
    ]
    for col, color in parts:
        vals = sub[col].fillna(0).to_numpy(dtype=float) * 100
        ax.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.45, height=0.58)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[m] for m in sub["method"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Molecules by ligand-space bias (%)")
    ax.set_title(title, loc="left", pad=4)
    ax.grid(axis="x", color=COL["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    if label:
        panel_label(ax, label)


def export_scaffold_breadth_table(div: pd.DataFrame) -> None:
    screen = div[div["set"] == "screened"].copy()
    method_order = SCAFFOLD_METHODS
    rows = []
    for method in method_order:
        row = {"method": DISPLAY[method]}
        for benchmark, prefix in [("SMRTnet RNA", "smrt"), ("DUD-E protein", "dude")]:
            hit = screen[(screen["benchmark"] == benchmark) & (screen["method"] == method)]
            if hit.empty:
                continue
            vals = hit.iloc[0]
            row[f"{prefix}_unique_smiles"] = int(vals["n_unique_smiles"])
            row[f"{prefix}_bm_scaffolds"] = int(vals["n_unique_bm_scaffolds"])
            row[f"{prefix}_internal_diversity"] = float(vals["internal_ecfp4_diversity"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(DATA_DIR / "scaffold_breadth_table.csv", index=False)


def main() -> None:
    setup_style()
    ref = pd.read_csv(DATA_DIR / "reference_overlap.csv")
    control = pd.read_csv(DATA_DIR / "control_ligand_space_bias.csv")
    bias = pd.read_csv(DATA_DIR / "model_ligand_space_bias.csv")
    div = pd.read_csv(DATA_DIR / "model_diversity.csv")

    export_scaffold_breadth_table(div)

    fig = plt.figure(figsize=(7.2, 4.7), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.22, 0.92],
        left=0.105,
        right=0.985,
        bottom=0.125,
        top=0.850,
        wspace=0.50,
        hspace=0.70,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_benchmark_bias(ax_a, bias, "SMRTnet RNA", SMRT_METHODS, "Generated on SMRT-Net RNA", label="a")
    plot_benchmark_bias(ax_b, bias, "DUD-E protein", DUDE_METHODS, "Generated on DUD-E protein", label="b")
    plot_reference_overlap(ax_c, ref, label="c")
    plot_control_bias(ax_d, control, label="d")

    fig.legend(
        handles=[
            Patch(facecolor=COL["rna"], edgecolor="none", label="RNA-leaning"),
            Patch(facecolor=COL["intermediate"], edgecolor="none", label="Intermediate"),
            Patch(facecolor=COL["protein"], edgecolor="none", label="Protein-leaning"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=3,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=1.1,
    )

    for out_dir in [DATA_DIR, PAPER_FIG_DIR]:
        base = out_dir / OUT_STEM
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".eps"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Wrote {base}.pdf/.svg/.eps/.png/.tiff")


if __name__ == "__main__":
    main()

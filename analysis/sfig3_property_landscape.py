#!/usr/bin/env python
"""Plot property-landscape audit panels for Extended Data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from style import (
    GRID,
    INK,
    METHOD_COLORS,
    METHOD_LABELS,
    panel_label,
    save_all,
    setup_publication_style,
    style_axis,
)


ROOT = Path(__file__).resolve().parent.parent
SMRT_RAW = ROOT / "data" / "benchmark" / "output" / "smrtbench_combined_results.csv"
DUDE_RAW = ROOT / "data" / "benchmark" / "dude_stat_all.csv"
OUT_DIR = ROOT / "data" / "benchmark" / "output" / "property_landscape"
PAPER_DIR = ROOT / "reproduction" / "figures"

SMRT_METHODS = [
    "RNA2smol (Only General Feat)",
    "RNA2smol (Baseline)",
    "MolCRAFT",
    "Pocket2Mol",
    "ED2Mol",
]
DUDE_METHODS = ["RNA2smol", "MolCRAFT", "Pocket2Mol", "ed2mol"]


def color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#606060")


def label_for(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_cmap(method: str) -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        f"{method}_density",
        ["#FFFFFF", "#EEF2F3", color_for(method)],
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    smrt = pd.read_csv(
        SMRT_RAW,
        usecols=["mode", "qed", "sa_score", "affinity", "tpsa", "SEI", "strain"],
    )
    dude = pd.read_csv(
        DUDE_RAW,
        usecols=["mode", "qed", "sa_score", "affinity", "tpsa", "strain"],
    )
    dude["SEI"] = -dude["affinity"] / dude["tpsa"].replace(0, np.nan)
    return smrt, dude


def plot_qed_sa_heatmap(ax: plt.Axes, data: pd.DataFrame, method: str, show_ylabel: bool) -> None:
    sub = data[data["mode"] == method][["qed", "sa_score"]].dropna()
    x_edges = np.linspace(0.2, 1.0, 42)
    y_edges = np.linspace(1.5, 7.0, 42)
    hist, _, _ = np.histogram2d(sub["qed"], sub["sa_score"], bins=[x_edges, y_edges])
    density = np.log10(hist.T + 1.0)
    if density.max() > 0:
        density = density / density.max()

    ax.imshow(
        density,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin="lower",
        cmap=method_cmap(method),
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    ax.axvline(0.6, color=INK, linewidth=0.55, linestyle="--", alpha=0.65)
    ax.axhline(4.0, color=INK, linewidth=0.55, linestyle="--", alpha=0.65)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(7.0, 1.5)
    ax.set_title(label_for(method), pad=3, fontsize=6.4)
    ax.set_xlabel("QED")
    ax.set_xticks([0.3, 0.6, 0.9])
    ax.set_yticks([2, 4, 6])
    if show_ylabel:
        ax.set_ylabel("SA score")
    else:
        ax.set_yticklabels([])
    style_axis(ax, grid_axis=None)


def plot_heatmap_row(
    fig: plt.Figure,
    spec,
    data: pd.DataFrame,
    methods: list[str],
    row_label: str,
    panel: str,
) -> None:
    sub = spec.subgridspec(1, len(methods), wspace=0.18)
    axes = [fig.add_subplot(sub[0, i]) for i in range(len(methods))]
    for i, (ax, method) in enumerate(zip(axes, methods)):
        plot_qed_sa_heatmap(ax, data, method, show_ylabel=i == 0)
    panel_label(axes[0], panel, x=-0.32, y=1.22)
    axes[0].text(
        -0.32,
        1.08,
        row_label,
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=INK,
    )


def plot_tradeoff(
    ax: plt.Axes,
    data: pd.DataFrame,
    methods: list[str],
    title: str,
    panel: str,
) -> None:
    xx, yy = np.mgrid[-13.0:-3.0:100j, 0.0:5.5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    linestyles = ["-", "--", "-.", ":", "--"]

    for method, ls in zip(methods, linestyles):
        sub = data[(data["mode"] == method) & (data["strain"] > 0)][["affinity", "strain"]].dropna()
        sub = sub.assign(log_strain=np.log10(sub["strain"]))
        sub = sub[(sub["log_strain"] >= 0) & (sub["log_strain"] <= 5.5)]
        if len(sub) > 7000:
            sub = sub.sample(n=7000, random_state=830 + methods.index(method))
        values = sub[["affinity", "log_strain"]].to_numpy().T
        kernel = gaussian_kde(values, bw_method=0.26)
        density = np.reshape(kernel(positions).T, xx.shape)
        levels = np.percentile(density[density > 0], [70, 84, 93])
        ax.contour(
            xx,
            yy,
            density,
            levels=levels,
            colors=color_for(method),
            linestyles=ls,
            linewidths=1.0,
            alpha=0.82,
        )

    ax.axvline(-9, color=GRID, linewidth=0.8, linestyle="--", zorder=0)
    ax.axhline(np.log10(200), color=GRID, linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlim(-13, -3)
    ax.set_ylim(0, 5.5)
    ax.set_xlabel("Docking score (kcal mol$^{-1}$)")
    ax.set_ylabel("Strain energy (log$_{10}$ kcal mol$^{-1}$)")
    ax.set_title(title, loc="left", pad=4)
    style_axis(ax, grid_axis="both")
    panel_label(ax, panel, x=-0.08)


def main() -> None:
    setup_publication_style()
    smrt, dude = load_data()

    fig = plt.figure(figsize=(7.2, 7.35), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.05, 1.15],
        left=0.09,
        right=0.985,
        bottom=0.12,
        top=0.94,
        wspace=0.38,
        hspace=0.72,
    )

    plot_heatmap_row(fig, gs[0, :], smrt, SMRT_METHODS, "SMRT-Net RNA benchmark", "a")
    plot_heatmap_row(fig, gs[1, :], dude, DUDE_METHODS, "DUD-E protein benchmark", "b")
    plot_tradeoff(fig.add_subplot(gs[2, 0]), smrt, SMRT_METHODS, "RNA docking-strain landscape", "c")
    plot_tradeoff(fig.add_subplot(gs[2, 1]), dude, DUDE_METHODS, "Protein docking-strain landscape", "d")

    handles = [
        Line2D([0], [0], color=color_for(method), linewidth=1.3, label=label_for(method))
        for method in SMRT_METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.025),
        ncol=5,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    for out_dir in [OUT_DIR, PAPER_DIR]:
        stem = out_dir / "property_landscape_audit"
        save_all(fig, stem)
        print(f"Wrote {stem}.pdf/.svg/.eps/.png/.tiff")


if __name__ == "__main__":
    main()

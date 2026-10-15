#!/usr/bin/env python
"""Assemble the DUD-E manuscript figure directly from source tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from figure_style import (
    GRID,
    INK,
    METHOD_COLORS,
    METHOD_LABELS,
    MUTED,
    group_metrics_by_direction,
    lighten,
    panel_label,
    save_all,
    save_panel,
    setup_publication_style,
    style_axis,
)


ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "notebooks" / "output" / "tables" / "dude"
OUT_DIR = ROOT / "notebooks" / "output" / "dude"
PAPER_DIR = ROOT / "reproduction" / "figures"

METHOD_ORDER = ["RNA2smol", "MolCRAFT", "MolCRAFT (Original)", "Pocket2Mol", "ed2mol"]
PROFILE_METRICS = [
    ("affinity", "Dock."),
    ("rmsd", "Relax.\nRMSD"),
    ("qed", "QED"),
    ("sa_score", "SA"),
    ("SEI", "SEI"),
    ("max_ecfp4_tanimoto", "ECFP4"),
    ("strain", "Strain"),
]
LOWER_BETTER = {"affinity", "rmsd", "sa_score", "strain"}
PROFILE_METRICS_DIR = group_metrics_by_direction(PROFILE_METRICS, LOWER_BETTER)
FUNNEL_STEPS = [
    ("All", "All"),
    ("aff<-8,QED>0.5,SA<5", "Practical"),
    ("+SEI>0.1", "+SEI"),
    ("aff<-9,QED>0.5,SA<5,SEI>0.15", "Stringent"),
    ("aff<-9,QED>0.6,SA<4,SEI>0.2,strain<200", "Strain\nstrict"),
]


def color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#606060")


def label_for(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def read_tables() -> dict[str, pd.DataFrame]:
    return {
        "profile": pd.read_csv(TABLE_DIR / "d8b_radar_rank_normalized.csv"),
        "ecfp4": pd.read_csv(TABLE_DIR / "d2_ecfp4_target_success_by_threshold.csv"),
        "funnel": pd.read_csv(TABLE_DIR / "d7_virtual_screening_funnel.csv"),
        "affinity": pd.read_csv(TABLE_DIR / "d4_heatmap_affinity.csv"),
    }


def plot_profile(ax, profile: pd.DataFrame) -> None:
    data = profile.set_index("method").reindex(METHOD_ORDER)
    y_positions = np.arange(len(METHOD_ORDER))[::-1]
    x_positions = np.arange(len(PROFILE_METRICS_DIR))
    for yi, method in zip(y_positions, METHOD_ORDER):
        row = data.loc[method]
        for xi, (metric, _) in zip(x_positions, PROFILE_METRICS_DIR):
            score = float(row[f"{metric}_rank_norm"])
            size = 24 + 78 * score
            ax.scatter(
                xi,
                yi,
                s=size,
                color=color_for(method),
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
    ax.set_xlim(-0.55, len(PROFILE_METRICS_DIR) - 0.45)
    ax.set_ylim(-0.6, len(METHOD_ORDER) - 0.4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([m[1] for m in PROFILE_METRICS_DIR])
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label_for(m) for m in METHOD_ORDER])
    ax.set_title("Rank-normalized benchmark profile", loc="left", pad=4)
    ax.set_xlabel("Metric; larger circles indicate higher rank")
    style_axis(ax, grid_axis="x")
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "a", x=-0.18)


def plot_ecfp4(ax, ecfp4: pd.DataFrame) -> None:
    for method in METHOD_ORDER:
        sub = ecfp4[ecfp4["method"] == method].sort_values("tau")
        if sub.empty:
            continue
        ax.plot(
            sub["tau"],
            sub["successful_targets_ge1_molecule"],
            marker="o",
            markersize=3.0,
            linewidth=1.15,
            color=color_for(method),
            label=label_for(method),
        )
    ax.set_title("Target-level ECFP4 recovery", loc="left", pad=4)
    ax.set_xlabel("ECFP4-TS threshold")
    ax.set_ylabel("Targets recovered")
    ax.set_xlim(0.195, 0.505)
    ax.set_ylim(0, 105)
    ax.set_xticks([0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    style_axis(ax, grid_axis="y")
    panel_label(ax, "b")


def _selected_funnel(funnel: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        sub = funnel[funnel["method"] == method]
        for step, label in FUNNEL_STEPS:
            hit = sub[sub["filter_step"] == step]
            if hit.empty:
                continue
            rows.append({"method": method, "step": label, value_col: float(hit.iloc[0][value_col])})
    return pd.DataFrame(rows)


def plot_funnel(ax, funnel: pd.DataFrame, value_col: str, ylabel: str, title: str, label: str, log: bool = False) -> None:
    selected = _selected_funnel(funnel, value_col)
    step_labels = [label for _, label in FUNNEL_STEPS]
    x = np.arange(len(step_labels))
    for method in METHOD_ORDER:
        sub = selected[selected["method"] == method].set_index("step").reindex(step_labels)
        if sub[value_col].isna().all():
            continue
        yvals = sub[value_col].to_numpy(dtype=float)
        if log:
            yvals = np.where(yvals > 0, yvals, np.nan)
        ax.plot(
            x,
            yvals,
            marker="o",
            markersize=3.0,
            linewidth=1.1,
            color=color_for(method),
            label=label_for(method),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=4)
    if log:
        ax.set_yscale("log")
        positive_vals = selected.loc[selected[value_col] > 0, value_col].to_numpy(dtype=float)
        if len(positive_vals):
            lower = max(1, 10 ** np.floor(np.log10(positive_vals.min())))
            upper = 10 ** np.ceil(np.log10(positive_vals.max()))
            ax.set_ylim(lower, upper * 1.05)
    else:
        ax.set_ylim(0, 105)
    style_axis(ax, grid_axis="y")
    panel_label(ax, label, x=-0.02, y=1.13)


def plot_affinity_distribution(ax, affinity: pd.DataFrame) -> None:
    data = affinity.set_index("method").reindex(METHOD_ORDER)
    values = [pd.to_numeric(data.loc[m].dropna(), errors="coerce").dropna().to_numpy() for m in METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER))
    box = ax.boxplot(
        values,
        positions=x,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 0.8},
        whiskerprops={"color": MUTED, "linewidth": 0.7},
        capprops={"color": MUTED, "linewidth": 0.7},
    )
    for patch, method in zip(box["boxes"], METHOD_ORDER):
        patch.set_facecolor("white")
        patch.set_edgecolor(color_for(method))
        patch.set_linewidth(0.9)
    for i, (method, vals) in enumerate(zip(METHOD_ORDER, values)):
        if len(vals) == 0:
            continue
        rng = np.random.default_rng(1000 + i)
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(
            i + jitter,
            vals,
            s=7,
            color=lighten(color_for(method), 0.35),
            edgecolor="none",
            zorder=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m) for m in METHOD_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Docking score (kcal mol$^{-1}$)")
    ax.set_title("Per-target docking score", loc="left", pad=4)
    style_axis(ax, grid_axis="y")
    panel_label(ax, "e", x=-0.06)


def render_panels(tables: dict, panels_dir: Path) -> dict:
    assets = {}

    def _one(name: str, plot_fn, figsize, rect=None) -> None:
        fig = plt.figure(figsize=figsize, facecolor="white")
        ax = fig.add_axes(rect if rect else [0.11, 0.15, 0.84, 0.68])
        plot_fn(ax)
        assets[name] = save_panel(fig, panels_dir, name)
        plt.close(fig)

    _one("a", lambda ax: plot_profile(ax, tables["profile"]), (5.4, 2.6))
    _one("b", lambda ax: plot_ecfp4(ax, tables["ecfp4"]), (3.4, 2.6))
    _one("c", lambda ax: plot_funnel(ax, tables["funnel"], "targets_ge5",
                                     "Targets with >=5 candidates",
                                     "Target retention through strain-aware screening", "c"),
         (3.4, 2.6))
    _one("d", lambda ax: plot_funnel(ax, tables["funnel"], "mols_retained",
                                     "Molecules retained",
                                     "Molecule retention through strain-aware screening",
                                     "d", log=True), (3.4, 2.6))
    _one("e", lambda ax: plot_affinity_distribution(ax, tables["affinity"]), (5.4, 2.6))
    return assets


def render_master(tables: dict) -> None:
    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.0, 1.08],
        left=0.105,
        right=0.985,
        bottom=0.115,
        top=0.875,
        wspace=0.42,
        hspace=0.70,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    plot_profile(ax_a, tables["profile"])
    plot_ecfp4(ax_b, tables["ecfp4"])
    plot_funnel(ax_c, tables["funnel"], "targets_ge5", "Targets with >=5 candidates", "Target retention through strain-aware screening", "c")
    plot_funnel(ax_d, tables["funnel"], "mols_retained", "Molecules retained", "Molecule retention through strain-aware screening", "d", log=True)
    plot_affinity_distribution(ax_e, tables["affinity"])

    handles = [
        Line2D([0], [0], color=color_for(m), marker="o", linewidth=1.1, markersize=3.2, label=label_for(m))
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=len(handles),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    for out_dir in [OUT_DIR, PAPER_DIR]:
        stem = out_dir / "test_on_dude"
        save_all(fig, stem)
        print(f"Wrote {stem}.pdf/.svg/.eps/.png/.tiff")


PANELS_DIR = OUT_DIR / "panels" / "test_on_dude"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="DUD-E benchmark figure")
    ap.add_argument("--mode", choices=["panels", "stitch", "all"], default="all")
    args = ap.parse_args()

    setup_publication_style()
    tables = read_tables()

    if args.mode in ("panels", "all"):
        print("Rendering standalone panels ...")
        assets = render_panels(tables, PANELS_DIR)
        for k, files in assets.items():
            print(f"  panel {k}: {', '.join(str(f) for f in files)}")

    if args.mode in ("stitch", "all"):
        print("Rendering stitched figure ...")
        render_master(tables)


if __name__ == "__main__":
    main()

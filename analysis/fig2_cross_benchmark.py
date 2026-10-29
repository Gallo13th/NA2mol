#!/usr/bin/env python
"""Assemble paired RNA/protein benchmark figure for the manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import radar as R
from style import (
    FUNNEL_STEPS,
    INK,
    METHOD_COLORS,
    METHOD_MARKERS,
    METHOD_DASHES,
    place_panel_labels_from_rects,
    METHOD_LABELS,
    detect_text_overlaps,
    group_metrics_by_direction,
    lighten,
    panel_label,
    plot_profile_table,
    profile_value_formatter,
    save_all,
    save_panel,
    setup_publication_style,
    style_axis,
)


ROOT = Path(__file__).resolve().parent.parent
SMRT_TABLE_DIR = ROOT / "data" / "benchmark" / "output" / "tables" / "smrtbench"
DUDE_TABLE_DIR = ROOT / "data" / "benchmark" / "output" / "tables" / "dude"
OUT_DIR = ROOT / "data" / "benchmark" / "output" / "cross_benchmark"
PAPER_DIR = ROOT / "reproduction" / "figures"
SMRT_RAW = ROOT / "data" / "benchmark" / "output" / "smrtbench_combined_results.csv"
DUDE_RAW = ROOT / "data" / "benchmark" / "dude_stat_all.csv"
SCAFFOLD_CSV = ROOT / "data" / "benchmark" / "output" / "chemspace_targeted" / "scaffold_breadth_table.csv"
RADAR_DIR = ROOT / "data" / "benchmark" / "output" / "tables" / "radar"

SMRT_METHODS = [
    "RNA2smol (Only General Feat)",
    "MolCRAFT",
    "Pocket2Mol",
    "ED2Mol",
]
DUDE_METHODS = ["RNA2smol", "MolCRAFT", "Pocket2Mol", "ed2mol"]
LEGEND_METHODS = [
    "RNA2smol",
    "MolCRAFT",
    "Pocket2Mol",
    "ED2Mol",
]
PROFILE_METRICS = [
    ("affinity", "Dock."),
    ("rmsd", "RMSD"),
    ("qed", "QED"),
    ("sa_score", "SA"),
    ("SEI", "SEI"),
    ("max_ecfp4_tanimoto", "ECFP4"),
    ("strain", "Strain"),
]
# Direction-grouped display order (higher-is-better ↑ left, lower-is-better ↓ right).
LOWER_BETTER = {"affinity", "rmsd", "sa_score", "strain"}
PROFILE_METRICS_DIR = group_metrics_by_direction(PROFILE_METRICS, LOWER_BETTER)
# FUNNEL_STEPS is imported from style.



def _read_radar(name: str) -> pd.DataFrame:
    path = RADAR_DIR / name
    if not path.is_file():
        raise SystemExit(
            f"missing {path}\n"
            "The radar panels need the seven-axis score tables. Build them with:\n"
            "  python3 analysis/radar_profiles.py")
    return pd.read_csv(path)


def radar_scores(table: pd.DataFrame, methods: list) -> dict:
    """{method: {axis_key: s_m}} for the methods present, in the given order."""
    sub = table.set_index('method')
    axes = list(R.AXIS_ORDER)
    out = {}
    for m in methods:
        if m in sub.index:
            row = sub.loc[m]
            out[m] = {a: float(row[a]) for a in axes if a in row}
    return out


def color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#606060")


def marker_for(method: str) -> str:
    return METHOD_MARKERS.get(method, "o")


def dash_for(method: str):
    return METHOD_DASHES.get(method, "solid")


def label_for(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def read_tables() -> dict[str, pd.DataFrame]:
    smrt_raw = pd.read_csv(
        SMRT_RAW,
        usecols=["mode", "pdb_id", "qed", "sa_score", "affinity", "tpsa", "SEI", "strain"],
    )
    dude_raw = pd.read_csv(
        DUDE_RAW,
        usecols=["mode", "qed", "sa_score", "affinity", "tpsa", "strain"],
    )
    dude_raw["SEI"] = -dude_raw["affinity"] / dude_raw["tpsa"].replace(0, np.nan)
    return {
        # Radar panels a/d use the manuscript's seven-axis scores (Methods,
        # "Radar-chart normalisation"), produced by analysis/radar_profiles.py.
        # NOT t7b/d8b: those are rank-normalised tables from an earlier code
        # generation and include metrics the manuscript's radar does not plot.
        "smrt_radar": _read_radar("radar_scores_smrtbench.csv"),
        "dude_radar": _read_radar("radar_scores_dude.csv"),
        "smrt_ecfp4": pd.read_csv(SMRT_TABLE_DIR / "t1b_ecfp4_target_success_by_threshold.csv"),
        "smrt_funnel": pd.read_csv(SMRT_TABLE_DIR / "t6_virtual_screening_funnel.csv"),
        "smrt_raw": smrt_raw,
        "dude_profile": pd.read_csv(DUDE_TABLE_DIR / "d8b_radar_rank_normalized.csv"),
        "dude_ecfp4": pd.read_csv(DUDE_TABLE_DIR / "d2_ecfp4_target_success_by_threshold.csv"),
        "dude_funnel": pd.read_csv(DUDE_TABLE_DIR / "d7_virtual_screening_funnel.csv"),
        "dude_raw": dude_raw,
    }


def plot_ecfp4(
    ax: plt.Axes,
    ecfp4: pd.DataFrame,
    methods: list[str],
    title: str,
    label: str,
    y_max: int,
) -> None:
    for method in methods:
        sub = ecfp4[ecfp4["method"] == method].sort_values("tau")
        if sub.empty:
            continue
        ax.plot(
            sub["tau"],
            sub["successful_targets_ge1_molecule"],
            marker=marker_for(method),
            linestyle=dash_for(method),
            markersize=2.8,
            linewidth=1.05,
            color=color_for(method),
            label=label_for(method),
        )
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("ECFP4-TS threshold")
    ax.set_ylabel("Targets")
    ax.set_xlim(0.195, 0.505)
    ax.set_ylim(0, y_max)
    ax.set_xticks([0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    style_axis(ax, grid_axis="y")
    panel_label(ax, label, x=-0.08)


def selected_funnel(funnel: pd.DataFrame, methods: list[str], value_col: str) -> pd.DataFrame:
    rows = []
    for method in methods:
        sub = funnel[funnel["method"] == method]
        for step, step_label in FUNNEL_STEPS:
            hit = sub[sub["filter_step"] == step]
            if hit.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "step": step_label,
                    value_col: float(hit.iloc[0][value_col]),
                }
            )
    return pd.DataFrame(rows)


def plot_funnel(
    ax: plt.Axes,
    funnel: pd.DataFrame,
    methods: list[str],
    value_col: str,
    ylabel: str,
    title: str,
    label: str,
    y_max: float | None = None,
    log: bool = False,
) -> None:
    selected = selected_funnel(funnel, methods, value_col)
    step_labels = [step_label for _, step_label in FUNNEL_STEPS]
    x = np.arange(len(step_labels))
    for method in methods:
        sub = selected[selected["method"] == method].set_index("step").reindex(step_labels)
        if sub[value_col].isna().all():
            continue
        yvals = sub[value_col].to_numpy(dtype=float)
        if log:
            yvals = np.where(yvals > 0, yvals, np.nan)
        ax.plot(
            x,
            yvals,
            marker=marker_for(method),
            linestyle=dash_for(method),
            markersize=2.8,
            linewidth=1.05,
            color=color_for(method),
            label=label_for(method),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=4)
    if log:
        ax.set_yscale("log")
        positive_vals = selected.loc[selected[value_col] > 0, value_col].to_numpy(dtype=float)
        if len(positive_vals):
            lower = max(1, 10 ** np.floor(np.log10(positive_vals.min())))
            upper = 10 ** np.ceil(np.log10(positive_vals.max()))
            ax.set_ylim(lower, upper * 1.05)
    elif y_max is not None:
        ax.set_ylim(0, y_max)
    style_axis(ax, grid_axis="y")
    panel_label(ax, label, x=-0.08)


def strain_filter(d: pd.DataFrame):
    """strain-energy endpoint gate: affinity AND qed AND sa AND SEI AND strain."""
    return (
        (d["affinity"] < -9)
        & (d["qed"] > 0.6)
        & (d["sa_score"] < 4)
        & (d["SEI"] > 0.2)
        & (d["strain"] < 200)
    )


def plot_scaffold_diversity(ax: plt.Axes, label: str) -> None:
    """Grouped bar chart of unique BM scaffolds per method, SMRT-Net vs DUD-E.

    Bars use per-method colours (consistent with the rest of the figure).
    Datasets are distinguished by HATCH PATTERN: solid = SMRT-Net, diagonal
    hatch = DUD-E. The legend uses neutral grey swatches to explain the pattern
    encoding without colour confusion.
    """
    from matplotlib.patches import Patch

    df = pd.read_csv(SCAFFOLD_CSV)
    methods = ["RNA2smol", "MolCRAFT", "Pocket2Mol", "ED2Mol"]
    df = df.set_index("method").reindex(methods)

    # Percentages within each benchmark (generated-molecule counts differ
    # between SMRT-Net and DUD-E, so raw scaffold counts are not comparable
    # across datasets).
    smrt_pct = df["smrt_bm_scaffolds"] / df["smrt_bm_scaffolds"].sum() * 100
    dude_pct = df["dude_bm_scaffolds"] / df["dude_bm_scaffolds"].sum() * 100

    bar_methods = [
        "RNA2smol (Only General Feat)",
        "MolCRAFT",
        "Pocket2Mol",
        "ED2Mol",
    ]
    method_colors_list = [color_for(m) for m in bar_methods]

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2, smrt_pct, width,
        color=method_colors_list, edgecolor="white", linewidth=0.4,
        hatch="",  # solid = SMRT-Net
    )
    bars2 = ax.bar(
        x + width / 2, dude_pct, width,
        color=method_colors_list, edgecolor="white", linewidth=0.4,
        hatch="//",  # diagonal hatch = DUD-E
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Unique scaffolds (%)")
    ax.set_title("Scaffold diversity", loc="left", pad=4)

    # Legend: neutral grey swatches explain the hatch convention.
    legend_handles = [
        Patch(facecolor="#888888", edgecolor="#888888", label="RNA test set"),
        Patch(facecolor="#888888", edgecolor="#aaaaaa", hatch="////", label="DUD-E protein"),
    ]
    ax.legend(handles=legend_handles, fontsize=5.2, loc="upper right", frameon=False)
    style_axis(ax, grid_axis="y")
    # Panel letter at the TOP-LEFT (just outside the axes). Measured against the
    # y-axis label window (x∈[-11,3]px) and the h-panel letter (x∈[254,262]px):
    # x=-0.16 → letter bbox x∈[10,19]px — clear of both.
    panel_label(ax, label, x=-0.16, y=1.06)

    # Value labels on top of bars (ONE label per bar: the percentage within
    # the benchmark — the raw count is readable on the axis; two stacked labels
    # collided on adjacent bars).
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.08, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=4.8, color=INK)


def plot_per_target_coverage(
    ax: plt.Axes,
    raw: pd.DataFrame,
    methods: list[str],
    label: str,
) -> None:
    """Heatmap of molecule counts per (method x target) at the strain-energy endpoint."""
    all_targets: set[str] = set()
    counts: dict[str, pd.Series] = {}
    for m in methods:
        sub = raw[raw["mode"] == m].copy()
        filtered = sub[strain_filter(sub)]
        tc = filtered.groupby("pdb_id").size()
        counts[m] = tc
        all_targets.update(tc.index.tolist())

    targets = sorted(all_targets)
    matrix = np.zeros((len(methods), len(targets)))
    for i, m in enumerate(methods):
        for j, t in enumerate(targets):
            matrix[i, j] = counts[m].get(t, 0)

    log_matrix = np.log1p(matrix)
    im = ax.imshow(log_matrix, aspect="auto", cmap="YlGnBu", interpolation="nearest")

    log_max = log_matrix.max()
    threshold = 0.8 * log_max if log_max > 0 else float("inf")
    for i in range(len(methods)):
        for j in range(len(targets)):
            if matrix[i, j] > 0:
                ax.text(
                    j,
                    i,
                    f"{int(matrix[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    color="white" if log_matrix[i, j] > threshold else INK,
                )

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([label_for(m) for m in methods], fontsize=6.0)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=0, ha="center", fontsize=5.5)
    ax.set_xlabel("RNA test set target (PDB ID)")
    ax.set_title("Per-target coverage", loc="left", pad=4)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Retained molecules (log$_{1+n}$)", fontsize=6.0)
    cbar.ax.tick_params(labelsize=5.5)

    panel_label(ax, label, x=-0.10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def plot_qed_sa_heatmap(
    ax: plt.Axes,
    raw: pd.DataFrame,
    methods: list[str],
    label: str,
) -> None:
    """Side-by-side QED×SA heatmaps (median docking) for comparative methods.

    Each method gets one mini heatmap; a shared colorbar lets the reader
    directly compare which method fills the favourable QED-SA region
    with well-docked molecules.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    qed_bins = np.arange(0, 1.01, 0.1)
    sa_bins = np.arange(1, 8.01, 1.0)
    qed_centers = (qed_bins[:-1] + qed_bins[1:]) / 2
    sa_centers = (sa_bins[:-1] + sa_bins[1:]) / 2

    qed_thresh_idx = np.searchsorted(qed_bins, 0.6) - 1
    sa_thresh_idx = np.searchsorted(sa_bins, 4.0) - 1

    # Turn off parent axes; build sub-grid inside.
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    n_methods = len(methods)
    sub_gs = GridSpecFromSubplotSpec(
        1, n_methods + 1, subplot_spec=ax.get_subplotspec(),
        wspace=0.55,
        width_ratios=[1] * n_methods + [0.06],
    )

    vmin, vmax = -10, -5
    first_im = None

    for idx, method in enumerate(methods):
        sub_ax = ax.figure.add_subplot(sub_gs[0, idx])
        sub = raw[
            (raw["mode"] == method)
            & raw["affinity"].notna()
            & raw["qed"].notna()
            & raw["sa_score"].notna()
        ].copy()

        median_dock = np.full((len(sa_centers), len(qed_centers)), np.nan)
        for i, (sa_lo, sa_hi) in enumerate(zip(sa_bins[:-1], sa_bins[1:])):
            for j, (qed_lo, qed_hi) in enumerate(zip(qed_bins[:-1], qed_bins[1:])):
                cell = sub[
                    (sub["qed"] >= qed_lo)
                    & (sub["qed"] < qed_hi)
                    & (sub["sa_score"] >= sa_lo)
                    & (sub["sa_score"] < sa_hi)
                ]
                if len(cell) >= 10:
                    median_dock[i, j] = cell["affinity"].median()

        im = sub_ax.imshow(
            median_dock, aspect="auto", cmap="RdYlGn",
            interpolation="nearest", vmin=vmin, vmax=vmax,
        )
        if first_im is None:
            first_im = im

        # Threshold lines
        sub_ax.axhline(sa_thresh_idx + 0.5, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        sub_ax.axvline(qed_thresh_idx + 0.5, color="black", linewidth=0.6, linestyle="--", alpha=0.6)

        # Labels
        tick_idx = range(0, len(qed_centers), 3)
        sub_ax.set_xticks(tick_idx)
        sub_ax.set_xticklabels([f"{qed_centers[i]:.1f}" for i in tick_idx], fontsize=4.8)
        tick_idx_y = range(0, len(sa_centers), 2)
        sub_ax.set_yticks(tick_idx_y)
        if idx == 0:
            sub_ax.set_yticklabels([f"{sa_centers[i]:.0f}" for i in tick_idx_y], fontsize=4.8)
            sub_ax.set_ylabel("SA score", fontsize=5.5)
        else:
            sub_ax.set_yticklabels([])
        sub_ax.set_xlabel("QED", fontsize=5.5)
        sub_ax.set_title(label_for(method), fontsize=5.8, color=INK, pad=2)

    # Shared colorbar
    cax = ax.figure.add_subplot(sub_gs[0, n_methods])
    cbar = ax.figure.colorbar(first_im, cax=cax)
    cbar.set_label("Median docking\n(kcal mol$^{-1}$)", fontsize=5.0)
    cbar.ax.tick_params(labelsize=4.5)

    panel_label(ax, label, x=-0.05, y=1.04)


def render_panels(tables: dict, panels_dir: Path) -> dict:
    """Render each panel as a standalone figure for audit. Returns name -> files."""
    lower_better = LOWER_BETTER
    assets = {}

    def _one(name: str, plot_fn, figsize, rect=None, xy_lbl=None,
             polar: bool = False) -> None:
        fig = plt.figure(figsize=figsize, facecolor="white")
        kw = {"projection": "polar"} if polar else {}
        if rect is None:
            ax = fig.add_axes([0.10, 0.15, 0.86, 0.68], **kw)
        else:
            ax = fig.add_axes(rect, **kw)
        plot_fn(ax)
        assets[name] = save_panel(fig, panels_dir, name)
        plt.close(fig)

    # a — DUD-E radar (paper order: DUD-E is panel a, the RNA test set is panel d)
    _one("a", lambda ax: R.draw_radar(
        ax, radar_scores(tables["dude_radar"], DUDE_METHODS), title="",
        colors=METHOD_COLORS, labels=METHOD_LABELS,
        markers=METHOD_MARKERS, linestyles=METHOD_DASHES),
        (2.48, 2.08), [0.13, 0.16, 0.82, 0.62], polar=True)

    # b — Protein ECFP4 recovery (row 0 is the DUD-E control, as in the manuscript)
    _one("b", lambda ax: plot_ecfp4(ax, tables["dude_ecfp4"], DUDE_METHODS,
                                    "Protein ECFP4 recovery", "b", y_max=105), (3.33, 2.08))

    # c — Protein target retention funnel
    _one("c", lambda ax: plot_funnel(ax, tables["dude_funnel"], DUDE_METHODS,
                                     "targets_ge1", "Targets",
                                     "Protein target retention", "c", y_max=105), (3.33, 2.08))

    # d — DUD-E profile table
    _one("d", lambda ax: R.draw_radar(
        ax, radar_scores(tables["smrt_radar"], SMRT_METHODS), title="",
        colors=METHOD_COLORS, labels=METHOD_LABELS,
        markers=METHOD_MARKERS, linestyles=METHOD_DASHES),
        (2.48, 1.84), [0.13, 0.16, 0.82, 0.62], polar=True)

    # e — RNA ECFP4 recovery (row 1 is the RNA test set)
    _one("e", lambda ax: plot_ecfp4(ax, tables["smrt_ecfp4"], SMRT_METHODS,
                                    "RNA ECFP4 recovery", "e", y_max=28), (3.33, 1.84))

    # f — RNA target retention funnel
    _one("f", lambda ax: plot_funnel(ax, tables["smrt_funnel"], SMRT_METHODS,
                                     "targets_ge1", "Targets",
                                     "RNA target retention", "f", y_max=28), (3.33, 1.84))

    # g — scaffold diversity
    _one("g", lambda ax: plot_scaffold_diversity(ax, "g"), (2.48, 1.36))

    # h — per-target coverage (spans 2 cols in master; standalone slightly wider)
    _one("h", lambda ax: plot_per_target_coverage(ax, tables["smrt_raw"], SMRT_METHODS, "h"),
         (7.15, 1.36))

    return assets


def render_master(tables: dict) -> None:
    """Full 8-panel figure in the wide-format layout (papers-ready)."""
    setup_publication_style()
    tables = read_tables()

    # Wide-format layout: 11 × 8.0 in.
    # b=c width (ECFP4 = funnel), row heights shortened.
    # Physical size chosen to match the manuscript slot (~7.1 x 6.4 in of content):
    # the panel geometry below is fractional, so this keeps the same layout while
    # putting the text at the size it is printed at.
    fig = plt.figure(figsize=(7.3, 6.6))

    # Column positions — b and c get equal width
    # Gutters are sized for the 'Targets' y-axis labels of columns 1 and 2, which would
    # otherwise run into the radar of column 0 at the printed width.
    C0_L, C0_R = 0.045, 0.258      # col 0: radars
    C1_L, C1_R = 0.352, 0.622      # col 1: ECFP4
    C2_L, C2_R = 0.672, 0.965      # col 2: funnels
    CW0, CW1, CW2 = C0_R - C0_L, C1_R - C1_L, C2_R - C2_L

    # Row positions — shorter rows, generous 0.09 inter-row gap
    R0_B, R0_T = 0.62, 0.88        # height 0.26 (~2.1 in)
    R1_B, R1_T = 0.30, 0.53        # height 0.23 (~1.8 in)
    R2_B, R2_T = 0.04, 0.21        # height 0.17 (~1.4 in)
    RH0, RH1, RH2 = R0_T - R0_B, R1_T - R1_B, R2_T - R2_B

    axes = {}
    # The radars are inset within their cell: the polar axes inscribes a circle, so at
    # full cell width its own "SA" tick label would meet column 1's "Targets" ylabel.
    AX, AY, AW, AH = 0.055, 0.055, 0.89, 0.89
    axes["a"] = fig.add_axes([C0_L + CW0 * AX, R0_B + RH0 * AY,
                              CW0 * AW, RH0 * AH], projection="polar")
    axes["b"] = fig.add_axes([C1_L, R0_B, CW1, RH0])
    axes["c"] = fig.add_axes([C2_L, R0_B, CW2, RH0])
    axes["d"] = fig.add_axes([C0_L + CW0 * AX, R1_B + RH1 * AY,
                              CW0 * AW, RH1 * AH], projection="polar")
    axes["e"] = fig.add_axes([C1_L, R1_B, CW1, RH1])
    axes["f"] = fig.add_axes([C2_L, R1_B, CW2, RH1])
    # Row 2 is not tied to the radar column: the scaffold-diversity panel needs more
    # width than a radar does, or its rotated category labels collide at print size.
    G_R = 0.330
    # h starts far enough right that panel g's legend cannot run into its y labels
    H_L = 0.406
    axes["g"] = fig.add_axes([C0_L, R2_B, G_R - C0_L, RH2])
    axes["h"] = fig.add_axes([H_L, R2_B, C2_R - H_L, RH2])  # spans what is left

    # All eight letters placed from the cell rectangles, in figure coordinates. Per-axes
    # placement cannot align them: panels a/d are inset polar axes inside a larger cell,
    # and the plot_* helpers each use their own axes-fraction offset.
    place_panel_labels_from_rects(fig, {
        "a": (C0_L, R0_B, CW0, RH0), "b": (C1_L, R0_B, CW1, RH0),
        "c": (C2_L, R0_B, CW2, RH0), "d": (C0_L, R1_B, CW0, RH1),
        "e": (C1_L, R1_B, CW1, RH1), "f": (C2_L, R1_B, CW2, RH1),
        "g": (C0_L, R2_B, G_R - C0_L, RH2), "h": (H_L, R2_B, C2_R - H_L, RH2),
    })

    lower_better = {"affinity", "rmsd", "sa_score", "strain"}

    # Row 0 — DUD-E protein control (manuscript order: a = DUD-E, d = RNA test set)
    R.draw_radar(
        axes["a"], radar_scores(tables["dude_radar"], DUDE_METHODS),
        title="DUD-E protein", colors=METHOD_COLORS, labels=METHOD_LABELS,
        markers=METHOD_MARKERS, linestyles=METHOD_DASHES,
        legend=False)

    plot_ecfp4(axes["b"], tables["dude_ecfp4"], DUDE_METHODS,
              "Protein ECFP4 recovery", None, y_max=105)
    plot_funnel(axes["c"], tables["dude_funnel"], DUDE_METHODS,
              "targets_ge1", "Targets",
              "Protein target retention", None, y_max=105)

    # Row 1 — RNA test set
    R.draw_radar(
        axes["d"], radar_scores(tables["smrt_radar"], SMRT_METHODS),
        title="RNA test set", colors=METHOD_COLORS, labels=METHOD_LABELS,
        markers=METHOD_MARKERS, linestyles=METHOD_DASHES,
        legend=False)

    plot_ecfp4(axes["e"], tables["smrt_ecfp4"], SMRT_METHODS,
              "RNA ECFP4 recovery", None, y_max=28)
    plot_funnel(axes["f"], tables["smrt_funnel"], SMRT_METHODS,
              "targets_ge1", "Targets",
              "RNA target retention", None, y_max=28)

    # Row 2 — Cross-benchmark evidence
    plot_scaffold_diversity(axes["g"], None)
    plot_per_target_coverage(axes["h"], tables["smrt_raw"], SMRT_METHODS, None)

    # Top method legend
    handles = [
        Line2D(
            [0], [0],
            color=color_for(method), marker=marker_for(method),
            linestyle=dash_for(method),
            linewidth=1.05, markersize=3.1,
            label=label_for(method),
        )
        for method in LEGEND_METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=5,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.75,
    )

    detect_text_overlaps(fig)

    for out_dir in [OUT_DIR, PAPER_DIR]:
        stem = out_dir / "cross_benchmark_transfer"
        save_all(fig, stem, fast=True)
        print(f"Wrote {stem}.pdf/.svg/.eps/.png/.tiff")


PANELS_DIR = OUT_DIR / "panels" / "cross_benchmark_transfer"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Cross-benchmark transfer figure")
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
        print("Rendering stitched wide-format figure ...")
        render_master(tables)


if __name__ == "__main__":
    main()

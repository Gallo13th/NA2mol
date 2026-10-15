"""
Assemble ablation comparison figure for RNA2smol manuscript.
Uses shared figure_style.py for consistent publication aesthetics.

Single multi-panel figure (Nature-style compact layout):
  a — Rank-normalized ablation profile (bubble chart)
  b — Incremental contribution waterfall (key metrics)
  c — Virtual screening funnel — targets retained
  d — Virtual screening funnel — molecules (log)
  e — ECFP4 target-level recovery curve
"""

from __future__ import annotations

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

from figure_style import (
    INK, MUTED, GRID, METHOD_LABELS,
    detect_text_overlaps,
    group_metrics_by_direction,
    lighten, panel_label, plot_profile_table, profile_value_formatter,
    save_all, save_panel,
    setup_publication_style, style_axis,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "notebooks" / "output" / "smrtbench"
PAPER_DIR = ROOT / "reproduction" / "figures"
CSV_PATH = ROOT / "notebooks" / "output" / "smrtbench_combined_results.csv"

# ── Ablation-specific palette (extends METHOD_COLORS) ────────────────────
ABL_COLORS = {
    "RNA2smol (Baseline)":          "#B0B6BD",   # light grey — worst
    "Pretrain Only":                "#62A6A6",   # teal — intermediate
    "Pretrain+Constrain":           "#3B7A7A",   # darker teal — stronger
    "RNA2smol (Only General Feat)": "#246B70",   # deepest teal — full model
}
ABL_LABELS = {
    "RNA2smol (Baseline)":          "Baseline",
    "Pretrain Only":                "Pretrain",
    "Pretrain+Constrain":           "Pretrain\n+Constrain",
    "RNA2smol (Only General Feat)": "Full Model\n(+Contrastive)",
}
ABL_ORDER = [
    "RNA2smol (Baseline)",
    "Pretrain Only",
    "Pretrain+Constrain",
    "RNA2smol (Only General Feat)",
]


def color_for(m):
    return ABL_COLORS.get(m, "#606060")


def label_for(m):
    return ABL_LABELS.get(m, m)


# ── Data ──────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df[df["mode"].isin(ABL_ORDER)].copy()
    df["SEI"] = -df["affinity"] / df["tpsa"].replace(0, np.nan)
    df["SEI"] = df["SEI"].replace([float("inf"), float("-inf")], np.nan).fillna(0)
    df["LLE"] = -df["affinity"] - df["logp"]
    return df


def per_target_median(data, col):
    return data.groupby("pdb_id")[col].median()


FUNNEL_STEPS = [
    ("All", "All"),
    ("aff<-8,QED>0.5,SA<5", "Practical"),
    ("+SEI>0.1", "+SEI"),
    ("aff<-9,QED>0.5,SA<5,SEI>0.15", "Stringent"),
    ("aff<-9,QED>0.6,SA<4,SEI>0.2,strain<200", "Strain\nstrict"),
]

FUNNEL_PREDICATES = {
    "All": None,
    "aff<-8,QED>0.5,SA<5": lambda d: (d["affinity"] < -8) & (d["qed"] > 0.5) & (d["sa_score"] < 5),
    "+SEI>0.1": lambda d: (d["affinity"] < -8) & (d["qed"] > 0.5) & (d["sa_score"] < 5) & (d["SEI"] > 0.1),
    "aff<-9,QED>0.5,SA<5,SEI>0.15": lambda d: (d["affinity"] < -9) & (d["qed"] > 0.5) & (d["sa_score"] < 5) & (d["SEI"] > 0.15),
    "aff<-9,QED>0.6,SA<4,SEI>0.2,strain<200": lambda d: (d["affinity"] < -9) & (d["qed"] > 0.6) & (d["sa_score"] < 4) & (d["SEI"] > 0.2) & (d["strain"] < 200),
}


# ══════════════════════════════════════════════════════════════════════════
# Panel (a) — Rank-normalized ablation profile (bubble chart)
# ══════════════════════════════════════════════════════════════════════════
ABL_PROFILE_METRICS = [
    ("affinity", "Dock."),
    ("rmsd", "Relax.\nRMSD"),
    ("qed", "QED"),
    ("sa_score", "SA"),
    ("SEI", "SEI"),
    ("LLE", "LLE"),
    ("max_ecfp4_tanimoto", "ECFP4"),
    ("strain", "Strain"),
]
ABL_FLOOR = 0.25
LOWER_IS_BETTER_METRICS = {"affinity", "rmsd", "sa_score", "strain"}


def _rank_normalize(profile, methods, metrics):
    """Rank-based normalization: worst → FLOOR, best → 1.0."""
    normed = {m: [] for m in methods}
    for col, _ in metrics:
        vals = {m: profile[m].get(col, np.nan) for m in methods}
        valid = {k: v for k, v in vals.items() if not np.isnan(v)}
        n_valid = len(valid)
        if n_valid < 2:
            for m in methods:
                normed[m].append((1.0 + ABL_FLOOR) / 2)
            continue
        ascending = col in LOWER_IS_BETTER_METRICS
        sorted_m = sorted(valid, key=valid.get, reverse=not ascending)
        for m in methods:
            if m in sorted_m:
                rank = sorted_m.index(m) + 1
                normed[m].append(1.0 - (rank - 1) * (1.0 - ABL_FLOOR) / (n_valid - 1))
            else:
                normed[m].append((1.0 + ABL_FLOOR) / 2)
    return normed


def plot_profile(ax, df, title_pad: float | None = None):
    profile = {}
    for m in ABL_ORDER:
        profile[m] = {}
        sub = df[df["mode"] == m]
        for col, _ in ABL_PROFILE_METRICS:
            pt = per_target_median(sub, col)
            profile[m][col] = pt.mean()

    normed = _rank_normalize(profile, ABL_ORDER, ABL_PROFILE_METRICS)

    # Build a tidy DataFrame for ``plot_profile_table`` (chart-enhanced table).
    rows = []
    for m in ABL_ORDER:
        row = {"method": m}
        for i, (col, _) in enumerate(ABL_PROFILE_METRICS):
            row[f"{col}_raw"] = profile[m][col]
            row[f"{col}_rank_norm"] = normed[m][i]
        rows.append(row)
    profile_df = pd.DataFrame(rows)

    # Direction-grouped column order: higher-is-better (↑) together,
    # lower-is-better (↓) together; bars use the rank-normalised score.
    metrics_disp = group_metrics_by_direction(ABL_PROFILE_METRICS, LOWER_IS_BETTER_METRICS)

    plot_profile_table(
        ax,
        profile_df,
        ABL_ORDER,
        metrics_disp,
        title="Rank-normalised ablation profile",
        label="a",
        method_colors=ABL_COLORS,
        method_labels=ABL_LABELS,
        lower_is_better=LOWER_IS_BETTER_METRICS,
        value_fmt=profile_value_formatter,
        bar_mode="rank",
        title_pad=title_pad,
    )


# ══════════════════════════════════════════════════════════════════════════
# Panel (b) — Incremental contribution waterfall
# ══════════════════════════════════════════════════════════════════════════
WATERFALL_METRICS = [
    ("affinity", "Docking\n(kcal mol⁻¹)", -1),
    ("qed", "QED", 1),
    ("sa_score", "SA Score", -1),
    ("SEI", "SEI", 1),
    ("strain", "Strain\n(log₁₀ kcal mol⁻¹)", -1),
]


def _bootstrap_ci(diffs, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def plot_waterfall(ax, df):
    components = ["+Pretrain", "+Constrain", "+Contrastive"]
    pairs = [
        ("Pretrain Only", "RNA2smol (Baseline)"),
        ("Pretrain+Constrain", "Pretrain Only"),
        ("RNA2smol (Only General Feat)", "Pretrain+Constrain"),
    ]

    n_metrics = len(WATERFALL_METRICS)
    x_offsets = np.arange(len(components) + 1)
    width = 0.18
    offsets = np.linspace(-(n_metrics - 1) * width / 2, (n_metrics - 1) * width / 2, n_metrics)

    metric_colors_light = ["#B0C4DE", "#C8E6C9", "#FFE0B2", "#D1C4E9", "#E8D5B7"]

    for mi, (col, label, direction) in enumerate(WATERFALL_METRICS):
        deltas = []
        use_log = (col == "strain")
        for comp_a, comp_b in pairs:
            a_vals = per_target_median(df[df["mode"] == comp_a], col)
            b_vals = per_target_median(df[df["mode"] == comp_b], col)
            if use_log:
                a_vals = np.log10(a_vals.replace(0, np.nan).dropna())
                b_vals = np.log10(b_vals.replace(0, np.nan).dropna())
            common = sorted(set(a_vals.index) & set(b_vals.index))
            if len(common) == 0:
                deltas.append(0.0)
                continue
            diffs = (a_vals[common].values - b_vals[common].values) * direction
            deltas.append(diffs.mean())

        # Draw bars
        cumulative = 0.0
        x_pos = x_offsets + offsets[mi]
        for ci, d in enumerate(deltas):
            bottom = cumulative
            color = "#2E7D32" if d >= 0 else "#C62828"
            bar = ax.bar(x_pos[ci + 1], d, width, bottom=bottom,
                        color=color, edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)
            # Error bar
            comp_a, comp_b = pairs[ci]
            a_vals = per_target_median(df[df["mode"] == comp_a], col)
            b_vals = per_target_median(df[df["mode"] == comp_b], col)
            if use_log:
                a_vals = np.log10(a_vals.replace(0, np.nan).dropna())
                b_vals = np.log10(b_vals.replace(0, np.nan).dropna())
            common = sorted(set(a_vals.index) & set(b_vals.index))
            diffs = (a_vals[common].values - b_vals[common].values) * direction
            if len(diffs) >= 3:
                ci_low, ci_high = _bootstrap_ci(diffs)
                yerr_low = max(0, d - ci_low)
                yerr_high = max(0, ci_high - d)
                ax.errorbar(x_pos[ci + 1], bottom + d,
                           yerr=[[yerr_low], [yerr_high]], color="#333333",
                           capsize=2, linewidth=0.8, zorder=5)
            # Annotate
            if abs(d) > 0.002:
                va = "bottom" if d >= 0 else "top"
                y_anno = bottom + d + (0.01 if d >= 0 else -0.01)
                ax.text(x_pos[ci + 1], y_anno, f"{d:+.2f}", ha="center", va=va,
                       fontsize=6.2, fontweight="bold", color=INK)
            cumulative += d

        # Baseline bar (zero)
        ax.bar(x_pos[0], 0, width, bottom=0, color="#E0E0E0",
              edgecolor="#cccccc", linewidth=0.3, zorder=2)
        # Total bar
        ax.bar(x_pos[-1] + 0.7, cumulative, width, bottom=0,
              color=metric_colors_light[mi], edgecolor=MUTED, linewidth=0.4, zorder=2)
        ax.text(x_pos[-1] + 0.7, cumulative + (0.02 if cumulative >= 0 else -0.02),
               f"{cumulative:+.2f}", ha="center",
               va="bottom" if cumulative >= 0 else "top", fontsize=6.5,
               fontweight="bold", color=INK)

        # Draw line from baseline through components to total
        y_path = [0.0]
        for d in deltas:
            y_path.append(y_path[-1] + d)
        x_path = [x_pos[0]] + list(x_pos[1:-1]) + [x_pos[-1] + 0.7]
        ax.plot(x_path, y_path, "o-", color=metric_colors_light[mi],
               linewidth=1.2, markersize=4, zorder=1, alpha=0.6)

    ax.axhline(y=0, color=INK, linewidth=0.5)
    all_xticks = ["Baseline"] + components + ["Total"]
    x_all = list(x_offsets) + [x_offsets[-1] + 0.7]
    ax.set_xticks(x_all)
    ax.set_xticklabels(all_xticks, rotation=15, ha="right")
    ax.set_ylabel("Δ (better →)", fontsize=7)
    ax.set_title("Incremental contribution (waterfall)", loc="left", pad=4)

    # Legend for metrics
    legend_el = [Line2D([0], [0], color=metric_colors_light[i], linewidth=3,
                        label=WATERFALL_METRICS[i][1].replace("\n", " "))
                 for i in range(n_metrics)]
    ax.legend(handles=legend_el, loc="upper left", fontsize=5.5,
             frameon=False, ncol=2, handlelength=1.0)
    style_axis(ax, grid_axis="y")
    panel_label(ax, "b")


# ══════════════════════════════════════════════════════════════════════════
# Panel (c) / (d) — Virtual screening funnel
# ══════════════════════════════════════════════════════════════════════════
def _funnel_data(df, value_col):
    rows = []
    for m in ABL_ORDER:
        sub = df[df["mode"] == m]
        for step_key, step_label in FUNNEL_STEPS:
            flt = FUNNEL_PREDICATES[step_key]
            if flt is None:
                filtered = sub
            else:
                filtered = sub[flt(sub)]
            if value_col == "targets_ge1":
                val = int((filtered.groupby("pdb_id").size() >= 1).sum())
            else:
                val = len(filtered)
            rows.append({"method": m, "step": step_label, value_col: val})
    return pd.DataFrame(rows)


def plot_funnel(ax, df, value_col, ylabel, title, panel_lbl, log=False):
    selected = _funnel_data(df, value_col)
    step_labels = [lbl for _, lbl in FUNNEL_STEPS]
    x = np.arange(len(step_labels))
    for m in ABL_ORDER:
        sub = selected[selected["method"] == m].set_index("step").reindex(step_labels)
        if sub[value_col].isna().all():
            continue
        yvals = sub[value_col].to_numpy(dtype=float)
        ax.plot(x, yvals, marker="o", markersize=3.0, linewidth=1.1,
               color=color_for(m), label=label_for(m))
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=4)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(5, 80000)
    else:
        ax.set_ylim(0, 28)
    style_axis(ax, grid_axis="y")
    panel_label(ax, panel_lbl)


# ══════════════════════════════════════════════════════════════════════════
# Panel (e) — ECFP4 target-level recovery curve
# ══════════════════════════════════════════════════════════════════════════
def plot_ecfp4(ax, df):
    tau_vals = np.arange(0.15, 0.56, 0.05)
    for m in ABL_ORDER:
        sub = df[df["mode"] == m]
        targets_ge1 = []
        for tau in tau_vals:
            per_target = sub.groupby("pdb_id")["max_ecfp4_tanimoto"].apply(
                lambda x, t=tau: int((x >= t).sum()))
            targets_ge1.append(int((per_target >= 1).sum()))
        ax.plot(tau_vals, targets_ge1, marker="o", markersize=3.0,
               linewidth=1.15, color=color_for(m), label=label_for(m))
    ax.set_title("Target-level ECFP4 recovery", loc="left", pad=4)
    ax.set_xlabel("ECFP4-TS threshold τ")
    ax.set_ylabel("Targets with ≥1 recovered molecule")
    ax.set_xlim(0.145, 0.555)
    ax.set_ylim(0, 28)
    ax.set_xticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    style_axis(ax, grid_axis="y")
    panel_label(ax, "b", x=-0.08)


# ══════════════════════════════════════════════════════════════════════════
# Panel (f) — Per-target docking score distribution
# ══════════════════════════════════════════════════════════════════════════
def plot_affinity_distribution(ax, df):
    per_target = {}
    for m in ABL_ORDER:
        sub = df[df["mode"] == m]
        per_target[m] = sub.groupby("pdb_id")["affinity"].median()

    common = sorted(set.intersection(*[set(per_target[m].index) for m in ABL_ORDER]))

    values = [per_target[m].reindex(common).dropna().to_numpy() for m in ABL_ORDER]
    x = np.arange(len(ABL_ORDER))
    box = ax.boxplot(
        values, positions=x, widths=0.48, patch_artist=True, showfliers=False,
        medianprops={"color": INK, "linewidth": 0.8},
        whiskerprops={"color": MUTED, "linewidth": 0.7},
        capprops={"color": MUTED, "linewidth": 0.7},
    )
    for patch, m in zip(box["boxes"], ABL_ORDER):
        patch.set_facecolor("white")
        patch.set_edgecolor(color_for(m))
        patch.set_linewidth(0.9)
    for i, (m, vals) in enumerate(zip(ABL_ORDER, values)):
        if len(vals) == 0:
            continue
        rng = np.random.default_rng(2000 + i)
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(i + jitter, vals, s=13,
                  color=lighten(color_for(m), 0.35),
                  edgecolor="none", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m) for m in ABL_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Docking score (kcal mol⁻¹)")
    ax.set_title("Per-target docking score", loc="left", pad=4)
    style_axis(ax, grid_axis="y")
    panel_label(ax, "e", x=-0.06)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
PANELS_DIR = OUT_DIR / "panels" / "fig_ablation_smrtnet"


def render_panels(df) -> dict:
    """Each panel as its own standalone figure. Returns name -> list of files."""
    assets = {}

    # Panel a — rank-normalised profile (chart-enhanced table)
    fig = plt.figure(figsize=(5.6, 2.7), facecolor="white")
    ax = fig.add_axes([0.16, 0.16, 0.80, 0.62])
    plot_profile(ax, df)
    assets["a"] = save_panel(fig, PANELS_DIR, "a_profile")
    plt.close(fig)

    # Panel b — ECFP4 recovery
    fig = plt.figure(figsize=(3.4, 2.6), facecolor="white")
    ax = fig.add_axes([0.14, 0.16, 0.82, 0.70])
    plot_ecfp4(ax, df)
    assets["b"] = save_panel(fig, PANELS_DIR, "b_ecfp4")
    plt.close(fig)

    # Panel c — funnel targets
    fig = plt.figure(figsize=(3.4, 2.6), facecolor="white")
    ax = fig.add_axes([0.14, 0.16, 0.82, 0.70])
    plot_funnel(ax, df, "targets_ge1", "Targets with >=1 candidate",
                "Target retention through strain-aware screening", "c")
    assets["c"] = save_panel(fig, PANELS_DIR, "c_funnel_targets")
    plt.close(fig)

    # Panel d — funnel molecules (log)
    fig = plt.figure(figsize=(3.4, 2.6), facecolor="white")
    ax = fig.add_axes([0.14, 0.16, 0.82, 0.70])
    plot_funnel(ax, df, "mols_retained", "Molecules retained",
                "Molecule retention through strain-aware screening", "d", log=True)
    assets["d"] = save_panel(fig, PANELS_DIR, "d_funnel_mols")
    plt.close(fig)

    # Panel e — per-target docking score distribution
    fig = plt.figure(figsize=(5.4, 2.6), facecolor="white")
    ax = fig.add_axes([0.09, 0.14, 0.88, 0.70])
    plot_affinity_distribution(ax, df)
    assets["e"] = save_panel(fig, PANELS_DIR, "e_docking_dist")
    plt.close(fig)

    return assets


def render_master(df) -> None:
    """Original compact 3x2 gridspec layout (rows: profile+ECFP4 / funnels /
    docking distribution) — matches the archived manuscript rendering."""
    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=False)
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.55, 1.0, 1.0],
        left=0.085, right=0.985, bottom=0.08, top=0.855,
        wspace=0.36, hspace=0.72,
    )
    ax_a = fig.add_subplot(gs[0, 0])   # profile
    ax_b = fig.add_subplot(gs[0, 1])   # ECFP4
    ax_c = fig.add_subplot(gs[1, 0])   # funnel targets
    ax_d = fig.add_subplot(gs[1, 1])   # funnel molecules
    ax_e = fig.add_subplot(gs[2, :])   # docking distribution (spanning 2 cols)

    plot_profile(ax_a, df, title_pad=14)
    plot_ecfp4(ax_b, df)
    plot_funnel(ax_c, df, "targets_ge1", "Targets with >=1 candidate",
               "Target retention through strain-aware screening", "c")
    plot_funnel(ax_d, df, "mols_retained", "Molecules retained",
               "Molecule retention through strain-aware screening", "d", log=True)
    plot_affinity_distribution(ax_e, df)

    # Shared legend at top
    handles = [
        Line2D([0], [0], color=color_for(m), marker="o", linewidth=1.1,
               markersize=3.2, label=label_for(m))
        for m in ABL_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=6.0,
    )

    fig.suptitle("Ablation analysis — RNA2smol component contributions",
                fontsize=8.5, fontweight="bold", color=INK, y=0.998)

    detect_text_overlaps(fig)

    for out_dir in [OUT_DIR, PAPER_DIR]:
        stem = out_dir / "fig_ablation_smrtnet"
        save_all(fig, stem)
        print(f"Wrote {stem}.pdf/.svg/.eps/.png/.tiff")

    # Also save a standalone higher-res PNG for quick viewing
    fig.savefig(OUT_DIR / "fig_ablation_smrtnet_2x.png", dpi=1200,
               bbox_inches="tight", facecolor="white")
    print("Wrote 2x PNG")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Ablation comparison figure")
    ap.add_argument("--mode", choices=["panels", "stitch", "all"], default="all")
    args = ap.parse_args()

    setup_publication_style()
    df = load_data()
    print(f"Data: {len(df)} mols, {df['pdb_id'].nunique()} targets")

    if args.mode in ("panels", "all"):
        print("Rendering standalone panels ...")
        panels = render_panels(df)
        for k, files in panels.items():
            print(f"  panel {k}: {', '.join(str(f) for f in files)}")

    # "stitch"/"all" reproduce the ORIGINAL compact layout (not PIL stitching),
    # with the audited column grouping + rank semantics kept inside the panels.
    if args.mode in ("stitch", "all"):
        print("Rendering compact 3x2 master figure ...")
        render_master(df)


if __name__ == "__main__":
    main()

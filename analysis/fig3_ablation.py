"""
Fine-grained ablation analysis figure for RNA2smol manuscript.

Panels directly mirror test_on_smrtnet.png style for consistency:
  a — Rank-normalised ablation profile          (chart-enhanced table)
  b — Target retention through screening        (line)
  c — Per-target docking score                  (boxplot + strip)
  d — Per-target strain energy                  (boxplot + strip)
  e — QED-SA × docking heatmap                  (2×2 small multiples)
  f — Final-tier retention waterfall by configuration

Style: matches the RNA benchmark figure via style.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import radar as R
from style import (
    ABL_COLORS,
    ABL_MARKERS,
    ABL_DASHES,
    METHOD_COLORS,
    METHOD_MARKERS,
    readable_ink,
    FUNNEL_STEPS,
    INK,
    MUTED,
    detect_text_overlaps,
    group_metrics_by_direction,
    lighten,
    panel_label,
    place_panel_labels,
    plot_profile_table,
    profile_value_formatter,
    save_all,
    save_panel,
    setup_publication_style,
    style_axis,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "benchmark" / "output" / "smrtbench"
PAPER_DIR = ROOT / "reproduction" / "figures"
CSV_PATH = ROOT / "data" / "benchmark" / "output" / "smrtbench_combined_results.csv"
RADAR_DIR = ROOT / "data" / "benchmark" / "output" / "tables" / "radar"

# ── Ablation palette ──────────────────────────────────────────────────────
# Method colours are imported from ``style.ABL_COLORS`` (shared with
# other figures) so the manuscript stays consistent. Labels and ordering are
# figure-specific and remain local.
ABL_LABELS = {
    "RNA2smol (Baseline)":          "Baseline",
    "Pretrain Only":                "Pretrain",
    "Pretrain+Constrain":           "Pretrain\n+Constrain",
    "RNA2smol (Only General Feat)": "Full Model",
}
ABL_ORDER = [
    "RNA2smol (Baseline)",
    "Pretrain Only",
    "Pretrain+Constrain",
    "RNA2smol (Only General Feat)",
]


def color_for(m: str) -> str:
    return ABL_COLORS.get(m, "#606060")


def label_for(m: str) -> str:
    return ABL_LABELS.get(m, m)


def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df[df["mode"].isin(ABL_ORDER)].copy()
    df["SEI"] = -df["affinity"] / df["tpsa"].replace(0, np.nan)
    df["SEI"] = df["SEI"].replace([float("inf"), float("-inf")], np.nan).fillna(0)
    return df


def per_target_median(data: pd.DataFrame, col: str) -> pd.Series:
    return data.groupby("pdb_id")[col].median()


# ═══════════════════════════════════════════════════════════════════════════
# Panel (a) — Rank-normalised ablation profile (chart-enhanced table)
# ═══════════════════════════════════════════════════════════════════════════
PROFILE_METRICS = [
    ("affinity", "Dock."),
    ("rmsd", "RMSD"),
    ("qed", "QED"),
    ("sa_score", "SA"),
    ("SEI", "SEI"),
    ("max_ecfp4_tanimoto", "ECFP4"),
    ("strain", "Strain"),
]
ABL_FLOOR = 0.25
LOWER_IS_BETTER_METRICS = {"affinity", "rmsd", "sa_score", "strain"}


def _rank_normalize(profile, methods, metrics):
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


def _radar_table() -> pd.DataFrame:
    path = RADAR_DIR / "radar_scores_smrtbench.csv"
    if not path.is_file():
        raise SystemExit(
            f"missing {path}\n"
            "Panel a needs the seven-axis radar scores. Build them with:\n"
            "  python3 analysis/radar_profiles.py")
    return pd.read_csv(path).set_index("method")


def plot_profile(ax, df, title_pad: float | None = None, panel_lbl: str | None = None,
                 title: str = "Ablation profile"):
    """Panel a — the manuscript's seven-axis radar for the four configurations.

    Uses the same radar scheme as Fig. 2 (Methods, "Radar-chart normalisation"):
    r_m = 1 + 2 s_m over seven axes. `df` is accepted for signature compatibility
    but the scores come from analysis/radar_profiles.py so the two figures share
    one implementation.
    """
    tbl = _radar_table()
    scores = {m: {a: float(tbl.loc[m, a]) for a in R.AXIS_ORDER}
              for m in ABL_ORDER if m in tbl.index}
    R.draw_radar(ax, scores, title=title, colors=ABL_COLORS, labels=ABL_LABELS,
                 markers=ABL_MARKERS, linestyles=ABL_DASHES)
    if panel_lbl:
        panel_label(ax, panel_lbl, x=-0.10, y=1.02)


# ═══════════════════════════════════════════════════════════════════════════
# Panel (b/c) — Virtual screening funnel (targets & molecules)
# ═══════════════════════════════════════════════════════════════════════════
FUNNEL_PREDICATES = {
    "All": None,
    "aff<-8,QED>0.5,SA<5": lambda d: (d["affinity"] < -8) & (d["qed"] > 0.5) & (d["sa_score"] < 5),
    "+SEI>0.1": lambda d: (d["affinity"] < -8) & (d["qed"] > 0.5) & (d["sa_score"] < 5) & (d["SEI"] > 0.1),
    "aff<-9,QED>0.5,SA<5,SEI>0.15": lambda d: (d["affinity"] < -9) & (d["qed"] > 0.5) & (d["sa_score"] < 5) & (d["SEI"] > 0.15),
    "aff<-9,QED>0.6,SA<4,SEI>0.2,strain<200": lambda d: (d["affinity"] < -9) & (d["qed"] > 0.6) & (d["sa_score"] < 4) & (d["SEI"] > 0.2) & (d["strain"] < 200),
}


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
        if log:
            yvals = np.where(yvals > 0, yvals, np.nan)
        ax.plot(
            x, yvals,
            marker=ABL_MARKERS.get(m, "o"), markersize=3.0, linewidth=1.1,
            color=color_for(m), label=label_for(m),
            linestyle=ABL_DASHES.get(m, "solid"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=4)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(1, 60000)
    else:
        ax.set_ylim(0, 28)
    style_axis(ax, grid_axis="y")
    panel_label(ax, panel_lbl, x=-0.02, y=1.12)


# ═══════════════════════════════════════════════════════════════════════════
# Panel (d) — Per-target docking score distribution (boxplot + strip)
# ═══════════════════════════════════════════════════════════════════════════
def plot_affinity_distribution(ax, df, panel_lbl="c"):
    per_target = {}
    for m in ABL_ORDER:
        sub = df[df["mode"] == m]
        per_target[m] = sub.groupby("pdb_id")["affinity"].median()

    common = sorted(set.intersection(*[set(per_target[m].index) for m in ABL_ORDER]))
    values = [per_target[m].reindex(common).dropna().to_numpy() for m in ABL_ORDER]
    x = np.arange(len(ABL_ORDER))

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
    for patch, m in zip(box["boxes"], ABL_ORDER):
        patch.set_facecolor("white")
        patch.set_edgecolor(color_for(m))
        patch.set_linewidth(0.9)

    for i, (m, vals) in enumerate(zip(ABL_ORDER, values)):
        if len(vals) == 0:
            continue
        rng = np.random.default_rng(2000 + i)
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(
            i + jitter,
            vals,
            s=13,
            color=lighten(color_for(m), 0.35),
            edgecolor="none",
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m) for m in ABL_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Docking score (kcal mol$^{-1}$)")
    ax.set_title("Per-target docking score", loc="left", pad=4)
    style_axis(ax, grid_axis="y")
    panel_label(ax, panel_lbl, x=-0.06)


# ═══════════════════════════════════════════════════════════════════════════
# Panel (d) — Strain energy distribution
# ═══════════════════════════════════════════════════════════════════════════
def plot_strain_distribution(ax, df, panel_lbl="d"):
    """Per-target strain energy distribution (boxplot + strip)."""
    per_target = {}
    strain = df["strain"].astype(float)
    ok = np.isfinite(strain) & (strain > 0) & (strain <= R.STRAIN_HARD_FAIL)
    df = df.assign(strain_capped=np.where(ok, strain, R.STRAIN_HARD_FAIL))
    for m in ABL_ORDER:
        sub = df[df["mode"] == m]
        per_target[m] = sub.groupby("pdb_id")["strain_capped"].median()

    common = sorted(set.intersection(*[set(per_target[m].index) for m in ABL_ORDER]))
    # Log-scale since strain spans orders of magnitude across variants
    values = [np.log10(per_target[m].reindex(common).dropna().to_numpy()) for m in ABL_ORDER]
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
        if len(vals) == 0: continue
        rng = np.random.default_rng(3000 + i)
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(i + jitter, vals, s=13,
                  color=lighten(color_for(m), 0.35), edgecolor="none", zorder=2)

    # Annotate medians in real units
    for i, m in enumerate(ABL_ORDER):
        med = np.median(per_target[m].reindex(common).dropna())
        lab = '>10,000' if med >= R.STRAIN_HARD_FAIL else f'{med:,.0f}'
        ax.text(i + 0.35, np.log10(med) if med > 0 else 0,
               lab, fontsize=5.5, va='center', ha='left', color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m) for m in ABL_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Strain energy (kcal mol⁻¹, log₁₀)")
    ax.set_title("Per-target strain energy", loc="left", pad=4)
    style_axis(ax, grid_axis="y")
    panel_label(ax, panel_lbl, x=-0.06)


def plot_docking_strain_hexbin(ax, df, panel_lbl="e"):
    """Docking score vs strain energy 2-D density (hexbin), one per config.

    Overlaid scatter is hard to read because Pretrain Only occupies an extreme
    strain region; a 2×2 small-multiple hexbin shows each config's joint
    distribution clearly with a shared colour scale.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    sub_gs = GridSpecFromSubplotSpec(
        2, 3, subplot_spec=ax.get_subplotspec(),
        wspace=0.35, hspace=0.42,
        width_ratios=[1, 1, 0.06],
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    gridsize = 24
    vmax = 0
    ims = []
    for idx, config in enumerate(ABL_ORDER):
        r, c = divmod(idx, 2)
        sub_ax = ax.figure.add_subplot(sub_gs[r, c])
        sub = df[(df["mode"] == config) & df["affinity"].notna()].copy()
        strain = sub["strain"].astype(float)
        ok = np.isfinite(strain) & (strain > 0) & (strain <= R.STRAIN_HARD_FAIL)
        sub["strain_capped"] = np.where(ok, strain, R.STRAIN_HARD_FAIL)
        if len(sub) > 20000:
            sub = sub.sample(20000, random_state=42 + idx)

        # greyscale density for the configurations the full model is compared
        # against; colour is reserved for the full model, as in the reference figure
        cmap = "YlGnBu" if config == ABL_ORDER[-1] else "Greys"
        hb = sub_ax.hexbin(
            sub["affinity"], np.log10(sub["strain_capped"]),
            gridsize=gridsize, cmap=cmap, mincnt=1,
            extent=(-12, -2, 1, 5),
        )
        vmax = max(vmax, hb.get_array().max())
        ims.append(hb)

        # Threshold lines
        sub_ax.axvline(-9, color="#555555", lw=0.7, ls="--", alpha=0.7)
        sub_ax.axhline(np.log10(200), color="#555555", lw=0.7, ls="--", alpha=0.7)

        sub_ax.set_xlim(-12, -2)
        sub_ax.set_ylim(1, 5)
        sub_ax.set_xticks([-12, -8, -4])
        sub_ax.set_yticks([2, 3, 4])
        if c == 0:
            sub_ax.set_yticklabels(["$10^2$", "$10^3$", "$10^4$"], fontsize=5.0)
            if r == 0:
                # single y-axis label shared by the left column (avoid the two
                # row labels colliding in the left margin)
                sub_ax.set_ylabel("Strain energy (kcal mol$^{-1}$)", fontsize=5.5)
        else:
            sub_ax.set_yticklabels([])
        if r == 1:
            sub_ax.set_xticklabels(["-12", "-8", "-4"], fontsize=5.0)
            sub_ax.set_xlabel("Docking (kcal mol$^{-1}$)", fontsize=5.5)
        else:
            sub_ax.set_xticklabels([])
        sub_ax.set_title(label_for(config).replace("\n", " "), fontsize=6.5, color=INK, pad=2)

    for hb in ims:
        hb.set_clim(0, vmax)

    cax = ax.figure.add_subplot(sub_gs[:, 2])
    cbar = ax.figure.colorbar(ims[0], cax=cax)
    cbar.set_label("Molecule density", fontsize=5.0)
    cbar.ax.tick_params(labelsize=4.5)
    # one density scale covers all four tiles, but only the full model is drawn in
    # colour, so say so under the bar (as the reference panel does)
    cax.text(0.5, -0.10, "Shared scale", transform=cax.transAxes, ha="center",
             va="top", fontsize=5.0, color=MUTED)

    # (the strain-aware endpoint threshold lines — docking -9 kcal mol^-1 and
    # strain < 200 kcal mol^-1 — are already visualised as dashes inside each
    # tile; an additional "favourable region" annotation is omitted to avoid
    # text collisions with the tile titles)
    panel_label(ax, panel_lbl, x=-0.05, y=1.10)


def plot_component_waterfall(ax, df, panel_lbl="f"):
    """Waterfall of strain-energy-endpoint retained molecules, showing each component's incremental gain."""
    configs = ABL_ORDER
    strain_f = lambda d: (d["affinity"] < -9) & (d["qed"] > 0.6) & (d["sa_score"] < 4) & (d["SEI"] > 0.2) & (d["strain"] < 200)

    counts = []
    for config in configs:
        sub = df[df["mode"] == config]
        filtered = sub[strain_f(sub)]
        counts.append(len(filtered))

    x = np.arange(len(configs))
    bar_heights = [counts[0]] + [counts[i] - counts[i - 1] for i in range(1, len(counts))]
    bar_bottoms = [0] + counts[:-1]

    colors = [color_for(c) for c in configs]
    labels = ["Baseline", "+Pretrain", "+Constrain", "+Contrastive\n(Full)"]

    bars = ax.bar(x, bar_heights, bottom=bar_bottoms, color=colors,
                  edgecolor="white", linewidth=0.6, width=0.6)

    for i, (bar, total, inc) in enumerate(zip(bars, counts, bar_heights)):
        ax.text(bar.get_x() + bar.get_width() / 2, total + 5,
                f"{total}", ha="center", va="bottom", fontsize=6.5,
                fontweight="bold", color=INK)
        if i == 0 or inc == 0:
            continue
        sign = "+" if inc >= 0 else ""
        if inc > 15:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar_bottoms[i] + bar_heights[i] / 2,
                    f"{sign}{inc}", ha="center", va="center",
                    fontsize=5.8, color=readable_ink(colors[i]), fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() + 0.04,
                    bar_bottoms[i] + bar_heights[i] / 2,
                    f"{sign}{inc}", ha="left", va="center",
                    fontsize=5.8, color=INK, fontweight="bold")

    ax.plot(x, counts, color=MUTED, linewidth=0.8, linestyle="--",
            marker="o", markersize=2.5, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_ylabel("Retained molecules")
    ax.set_title("Final-tier retention by configuration", loc="left", pad=4)
    style_axis(ax, grid_axis="y")
    panel_label(ax, panel_lbl, x=-0.08)


# ═══════════════════════════════════════════════════════════════════════════
# Main (original compact 3x2 gridspec layout, panels audited separately)
# ═══════════════════════════════════════════════════════════════════════════
PANELS_DIR = OUT_DIR / "panels" / "fig_fine_grained_ablation"


def render_panels(df) -> dict:
    """Render each panel as its own standalone figure for audit. Returns
    name -> list of written files (PNG, PDF)."""
    assets = {}

    # Panel a — profile table (wide, standalone)
    fig = plt.figure(figsize=(5.6, 2.7), facecolor="white")
    ax = fig.add_axes([0.16, 0.16, 0.80, 0.62], projection="polar")
    plot_profile(ax, df, panel_lbl="a")
    assets["a"] = save_panel(fig, PANELS_DIR, "a_profile")
    plt.close(fig)

    # Panel b — target retention funnel
    fig = plt.figure(figsize=(3.4, 2.6), facecolor="white")
    ax = fig.add_axes([0.13, 0.16, 0.82, 0.72])
    plot_funnel(ax, df, "targets_ge1", "Targets with >=1 candidate",
                "Target retention through strain-aware screening", "b")
    assets["b"] = save_panel(fig, PANELS_DIR, "b_funnel_targets")
    plt.close(fig)

    # Panel c — per-target docking score
    fig = plt.figure(figsize=(3.4, 2.6), facecolor="white")
    ax = fig.add_axes([0.13, 0.16, 0.82, 0.72])
    plot_affinity_distribution(ax, df, "c")
    assets["c"] = save_panel(fig, PANELS_DIR, "c_docking_dist")
    plt.close(fig)

    # Panel d — per-target strain energy
    fig = plt.figure(figsize=(3.4, 2.6), facecolor="white")
    ax = fig.add_axes([0.13, 0.16, 0.82, 0.72])
    plot_strain_distribution(ax, df, "d")
    assets["d"] = save_panel(fig, PANELS_DIR, "d_strain_dist")
    plt.close(fig)

    # Panel e — docking vs strain hexbin (small multiples; wide standalone)
    fig = plt.figure(figsize=(5.6, 2.2), facecolor="white")
    gs_e = fig.add_gridspec(1, 1, left=0.06, right=0.90, top=0.86, bottom=0.18)
    ax = fig.add_subplot(gs_e[0, 0])
    plot_docking_strain_hexbin(ax, df, "e")
    assets["e"] = save_panel(fig, PANELS_DIR, "e_dock_strain_hexbin")
    plt.close(fig)

    # Panel f — component contribution waterfall
    fig = plt.figure(figsize=(3.4, 2.6), facecolor="white")
    ax = fig.add_axes([0.13, 0.16, 0.82, 0.72])
    plot_component_waterfall(ax, df, "f")
    assets["f"] = save_panel(fig, PANELS_DIR, "f_component_waterfall")
    plt.close(fig)

    return assets


# ═══════════════════════════════════════════════════════════════════════════
# Panels (b) and (g) — recovery curve and reference-set sensitivity
# ═══════════════════════════════════════════════════════════════════════════
# Both read tables written by analysis/fig3_row3_data.py from the same per-molecule
# evaluation used by the rest of the figure, so they cannot drift from Extended Data
# Tables 6 and 7. Panel g holds the value of each metric on the targets that survive
# each reference-set definition (all actives, fine-tuning-similar actives removed,
# pretraining-library-similar actives removed); the target set shrinks from 27 to 18
# and 14, which is what moves the molecule-level medians.
FIGDATA = ROOT / "reproduction" / "figures"
DEFS = ["All actives", "FT-similar removed", "PT-similar removed"]
DEF_LABELS = ["All\n(27)", "$-$FT\n(18)", "$-$PT\n(14)"]


def plot_recovery_curve(ax, panel_lbl="b"):
    """Targets recovered versus the ECFP4-TS threshold, one line per configuration."""
    d = pd.read_csv(FIGDATA / "fig3_recovery_curves.csv")
    for m in ABL_ORDER:
        sub = d[d["mode"] == m].sort_values("tau")
        if sub.empty:
            continue
        ax.plot(sub["tau"], sub["targets_recovered"],
                color=color_for(m), linestyle=ABL_DASHES.get(m, "solid"),
                marker=ABL_MARKERS.get(m, "o"), markersize=2.4, linewidth=1.05,
                label=label_for(m).replace("\n", " "))
    ax.set_xlabel("ECFP4-TS threshold")
    ax.set_ylabel("Targets recovered (of 27)")
    ax.set_xlim(0.19, 0.51)
    ax.set_xticks([0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_title("Reference-ligand recovery", loc="left", pad=3)
    style_axis(ax, grid_axis="y")
    panel_label(ax, panel_lbl, x=-0.02, y=1.12)


METHODS_PANEL = ["NA2smol", "ED2Mol", "MolCRAFT", "Pocket2Mol", "DiffSBDD", "FLAG", "GraphBP"]
PANEL_DEFS = ["All actives", "FT-similar removed", "PT-similar removed"]
PANEL_DEF_LABELS = ["All\n(27)", "$-$FT\n(18)", "$-$PT\n(14)"]
CURVE_DEFS = {"All actives": ("-", "All actives (27)"),
              "FT-similar removed": ((0, (4, 1.6)), "$-$FT-similar (18)"),
              "PT-similar removed": ((0, (1.2, 1.2)), "$-$PT-similar (14)")}


def plot_recovery_sensitivity(ax, panel_lbl=None):
    """Reference-set sensitivity of ECFP4 recovery, four compact panels.

    1  recovery rank of each method across the three definitions (y = rank, methods coloured on
       the left axis so the axis itself is the colour key);
    2  NA2smol recovery curves on the three target sets;
    3  recovery curves of all seven methods on the fine-tuning-similar-removed subset;
    4  the same on the pretraining-library-similar-removed subset.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    sub_gs = GridSpecFromSubplotSpec(
        1, 4, subplot_spec=ax.get_subplotspec(), wspace=0.42,
        width_ratios=[1.22, 0.96, 0.96, 0.96],
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    x = np.arange(len(PANEL_DEFS))

    # ── 1: rank across definitions
    a = ax.figure.add_subplot(sub_gs[0, 0])
    d = pd.read_csv(FIGDATA / "fig3_method_reference_panel.csv")
    R = np.full((len(METHODS_PANEL), len(PANEL_DEFS)), np.nan)
    for i, m in enumerate(METHODS_PANEL):
        for j, name in enumerate(PANEL_DEFS):
            r = d[(d["method"] == m) & (d["definition"] == name)]
            if len(r):
                R[i, j] = float(r["recovery"].iloc[0])
    Rk = np.full_like(R, np.nan)
    for j in range(R.shape[1]):
        colv = R[:, j].copy()
        ok = np.isfinite(colv)
        for i in range(len(METHODS_PANEL)):
            if ok[i]:
                Rk[i, j] = 1 + int((colv[ok] > colv[i]).sum())
    order = [int(i) for i in np.argsort(np.nan_to_num(Rk[:, 0], nan=99))]
    for i in order:
        m = METHODS_PANEL[i]
        a.plot(x, Rk[i], color=METHOD_COLORS.get(m, "#606060"),
               marker=METHOD_MARKERS.get(m, "o"), markersize=2.2,
               linewidth=1.15 if m == "NA2smol" else 0.75,
               zorder=4 if m == "NA2smol" else 2, clip_on=False)
    a.set_xlim(-0.22, 2.22)
    a.set_ylim(len(METHODS_PANEL) + 0.45, 0.55)
    a.set_xticks(x)
    a.set_xticklabels(PANEL_DEF_LABELS, fontsize=4.2, linespacing=0.9)
    a.set_yticks(range(1, len(METHODS_PANEL) + 1))
    a.set_yticklabels([METHODS_PANEL[i] for i in order], fontsize=4.7)
    for tick, i in zip(a.get_yticklabels(), order):
        tick.set_color(METHOD_COLORS.get(METHODS_PANEL[i], "#202124"))
        if METHODS_PANEL[i] == "NA2smol":
            tick.set_fontweight("bold")
    a.set_title("Recovery rank\n(1 = best)", loc="left", pad=2, fontsize=5.0)
    a.grid(color="#EDEFEF", linewidth=0.35, zorder=0)
    a.set_axisbelow(True)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_linewidth(0.5)
    a.tick_params(length=0, pad=1.0)

    # ── 2: NA2smol curves on the three definitions
    c = pd.read_csv(FIGDATA / "fig3_recovery_curves_by_def.csv")

    def draw_curves(target, defs, title, with_legend=False):
        for name, (ls, label) in CURVE_DEFS.items():
            if name not in defs:
                continue
            for m in METHODS_PANEL:
                if m not in defs[name]:
                    continue
                sub = c[(c["method"] == m) & (c["definition"] == name)].sort_values("tau")
                if sub.empty:
                    continue
                target.plot(sub["tau"], 100 * sub["frac_targets"],
                            color=METHOD_COLORS.get(m, "#606060"),
                            linestyle=ls, marker=METHOD_MARKERS.get(m, "o"),
                            markersize=1.6,
                            linewidth=1.15 if m == "NA2smol" else 0.8,
                            zorder=4 if m == "NA2smol" else 2,
                            label=label if with_legend else None)
        target.set_xlim(0.19, 0.51)
        target.set_xticks([0.20, 0.30, 0.40, 0.50])
        target.set_ylim(0, 100)
        target.set_xlabel("ECFP4-TS threshold", fontsize=4.8, labelpad=1.5)
        target.set_title(title, loc="left", pad=2, fontsize=5.0)
        style_axis(target, grid_axis="y")
        target.tick_params(labelsize=4.3, pad=1)

    b = ax.figure.add_subplot(sub_gs[0, 1])
    draw_curves(b, {k: {"NA2smol"} for k in CURVE_DEFS}, "NA2smol, by definition", True)
    b.set_ylabel("Targets recovered (\%)", fontsize=4.8, labelpad=1.5)
    b.legend(frameon=False, fontsize=3.9, handlelength=1.4, labelspacing=0.12,
             loc="upper right", borderpad=0.1, handletextpad=0.25)

    for k, (name, title) in enumerate([("FT-similar removed", "$-$FT-similar (18 targets)"),
                                       ("PT-similar removed", "$-$PT-similar (14 targets)")]):
        pnl = ax.figure.add_subplot(sub_gs[0, 2 + k])
        draw_curves(pnl, {name: set(METHODS_PANEL)}, title)
        if k == 0:
            pnl.set_ylabel("Targets recovered (\%)", fontsize=4.8, labelpad=1.5)
        else:
            pnl.set_ylabel("")


def render_master(df) -> None:
    """Three-row layout: profile/recovery/screening, distributions/waterfall, then the
    reference-set sensitivity strip. Panel letters are placed in figure coordinates."""
    fig = plt.figure(figsize=(166 / 25.4, 7.05), constrained_layout=False)
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1.38, 1.0, 1.02],
        left=0.085, right=0.992, bottom=0.055, top=0.845,
        wspace=0.42, hspace=0.72,
    )
    ax_a = fig.add_subplot(gs[0, 0], projection="polar")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[2, :])

    plot_profile(ax_a, df, title_pad=14, panel_lbl=None)
    plot_recovery_curve(ax_b, "b")
    plot_funnel(ax_c, df, "targets_ge1", "Targets with >=1 candidate",
                "Target retention through screening", None)
    plot_affinity_distribution(ax_d, df, None)
    plot_strain_distribution(ax_e, df, None)
    plot_component_waterfall(ax_f, df, None)
    ax_f.set_title("Final-tier retention", loc="left", pad=3)
    ax_f.set_ylim(0, 300)
    ax_f.set_xticklabels(["Baseline", "+Pretrain", "+Constrain", "+Contrast.\n(Full)"])
    ax_f.tick_params(axis="x", labelsize=4.6)
    for t in ax_f.get_xticklabels():
        t.set_linespacing(0.9)
    # rotate the categorical labels of the two distribution panels so they do not collide
    for ax_ in (ax_d, ax_e):
        ax_.tick_params(axis="x", labelsize=4.8)
        for t in ax_.get_xticklabels():
            t.set_rotation(22)
            t.set_ha("right")
            t.set_linespacing(0.9)
    ax_f.set_title("Final-tier retention", loc="left", pad=3)
    ax_f.set_ylim(0, 300)
    ax_f.set_xticklabels(["Baseline", "+Pretrain", "+Constrain", "+Contrast.\n(Full)"])
    ax_f.tick_params(axis="x", labelsize=4.6)
    for t in ax_f.get_xticklabels():
        t.set_linespacing(0.9)
    # rotate the categorical labels of the two distribution panels so they do not collide
    for ax_ in (ax_d, ax_e):
        ax_.tick_params(axis="x", labelsize=4.8)
        for t in ax_.get_xticklabels():
            t.set_rotation(22)
            t.set_ha("right")
            t.set_linespacing(0.9)
    plot_recovery_sensitivity(ax_g, None)

    place_panel_labels(fig, gs, {
        "a": (0, 0), "b": (0, 1), "c": (0, 2),
        "d": (1, 0), "e": (1, 1), "f": (1, 2),
        "g": (2, 0),
    })

    handles = [
        Line2D([0], [0], color=color_for(m), marker=ABL_MARKERS.get(m, "o"),
               linestyle=ABL_DASHES.get(m, "solid"), linewidth=1.1,
               markersize=3.2, label=label_for(m).replace("\n", " "))
        for m in ABL_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.972),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=6.0,
    )


    detect_text_overlaps(fig)

    for out_dir in [OUT_DIR, PAPER_DIR]:
        stem = out_dir / "fig_fine_grained_ablation"
        save_all(fig, stem, fast=True)
        print(f"Wrote {stem}.pdf/.png")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Fine-grained ablation figure")
    ap.add_argument("--mode", choices=["panels", "stitch", "all"],
                    default="all", help="panels=render standalone panels; "
                    "stitch=assemble final figure from panels; all=both")
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
    # with the audited column grouping + rank semantics kept inside panels.
    if args.mode in ("stitch", "all"):
        print("Rendering compact 3x2 master figure ...")
        render_master(df)


if __name__ == "__main__":
    main()

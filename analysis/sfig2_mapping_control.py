#!/usr/bin/env python
"""Assemble an Extended Data control for MolCRAFT random residue mapping."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import wilcoxon

from style import (
    METHOD_COLORS,
    METHOD_LABELS,
    MUTED,
    lighten,
    panel_label,
    save_all,
    setup_publication_style,
    style_axis,
)


ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "data" / "benchmark" / "output" / "tables" / "dude"
RAW_DATA = ROOT / "data" / "benchmark" / "dude_stat_all.csv"
OUT_DIR = ROOT / "data" / "benchmark" / "output" / "molcraft_mapping_control"
PAPER_DIR = ROOT / "reproduction" / "figures"

METHODS = ["MolCRAFT", "MolCRAFT (Original)"]
RAW_METRICS = [
    ("affinity", "Docking"),
    ("rmsd", "RMSD"),
    ("qed", "QED"),
    ("sa_score", "SA"),
    ("SEI", "SEI"),
    ("LLE", "LLE"),
    ("max_ecfp4_tanimoto", "ECFP4"),
    ("strain", "Strain"),
]
EFFECT_METRICS = [
    ("affinity", "Docking score", -1),
    ("rmsd", "Relax. RMSD", -1),
    ("qed", "QED", 1),
    ("sa_score", "SA score", -1),
    ("SEI", "SEI", 1),
    ("LLE", "LLE", 1),
    ("max_ecfp4_tanimoto", "ECFP4", 1),
    ("strain", "Strain energy", -1),
]
FUNNEL_STEPS = [
    ("All", "All"),
    ("aff<-8,QED>0.5,SA<5", "Pharma."),
    ("+SEI>0.1", "+SEI"),
    ("aff<-9,QED>0.5,SA<5,SEI>0.15", "Strict"),
    ("aff<-9,QED>0.6,SA<4,SEI>0.2,strain<200", "+Strain"),
]


def color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#606060")


def label_for(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SEI"] = -df["affinity"] / df["tpsa"].replace(0, np.nan)
    df["LLE"] = -df["affinity"] - df["logp"]
    return df


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adjusted[i] = running
    q_values = np.empty(m, dtype=float)
    q_values[order] = np.clip(adjusted, 0.0, 1.0)
    return q_values


def paired_effect_table(raw: pd.DataFrame, n_boot: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | int | str]] = []

    for metric, label, direction in EFFECT_METRICS:
        paired = (
            raw.groupby(["mode", "pdb_id"])[metric]
            .median()
            .unstack("mode")
            .loc[:, METHODS]
            .dropna()
        )
        diff = (paired["MolCRAFT"] - paired["MolCRAFT (Original)"]) * direction
        diff_values = diff.to_numpy(dtype=float)
        effect = diff_values.mean() / diff_values.std(ddof=1)

        boot_effects = []
        indices = np.arange(len(diff_values))
        for _ in range(n_boot):
            sampled = diff_values[rng.choice(indices, size=len(indices), replace=True)]
            sd = sampled.std(ddof=1)
            boot_effects.append(sampled.mean() / sd if sd > 0 else np.nan)
        ci_low, ci_high = np.nanpercentile(boot_effects, [2.5, 97.5])

        p_value = (
            wilcoxon(diff_values, zero_method="wilcox", alternative="two-sided").pvalue
            if np.any(diff_values != 0)
            else 1.0
        )
        rows.append(
            {
                "metric": metric,
                "label": label,
                "n_targets": len(diff_values),
                "molcraft_mapped_median": paired["MolCRAFT"].median(),
                "molcraft_origin_median": paired["MolCRAFT (Original)"].median(),
                "oriented_mean_delta": diff_values.mean(),
                "oriented_median_delta": np.median(diff_values),
                "paired_effect_size_dz": effect,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wilcoxon_p": p_value,
            }
        )

    stats = pd.DataFrame(rows)
    stats["wilcoxon_q_bh"] = benjamini_hochberg(stats["wilcoxon_p"].to_numpy())
    return stats


def target_median_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for metric, label in RAW_METRICS:
        paired = (
            raw.groupby(["mode", "pdb_id"])[metric]
            .median()
            .unstack("mode")
            .loc[:, METHODS]
            .dropna()
        )
        for pdb_id, values in paired.iterrows():
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "metric": metric,
                    "label": label,
                    "molcraft_random_mapping": float(values["MolCRAFT"]),
                    "molcraft_original": float(values["MolCRAFT (Original)"]),
                }
            )
    return pd.DataFrame(rows)


def read_source_data() -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(
        RAW_DATA,
        usecols=[
            "pdb_id",
            "mode",
            "affinity",
            "rmsd",
            "qed",
            "sa_score",
            "tpsa",
            "logp",
            "max_ecfp4_tanimoto",
            "strain",
        ],
    )
    raw = add_derived_metrics(raw[raw["mode"].isin(METHODS)])
    ecfp4 = pd.read_csv(TABLE_DIR / "d2_ecfp4_target_success_by_threshold.csv")
    funnel = pd.read_csv(TABLE_DIR / "d7_virtual_screening_funnel.csv")
    return {
        "raw": raw,
        "target_medians": target_median_table(raw),
        "ecfp4": ecfp4[ecfp4["method"].isin(METHODS)].copy(),
        "funnel": funnel[funnel["method"].isin(METHODS)].copy(),
    }


def write_source_tables(tables: dict[str, pd.DataFrame], effects: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables["target_medians"].to_csv(OUT_DIR / "molcraft_mapping_target_medians_source.csv", index=False)
    tables["ecfp4"].to_csv(OUT_DIR / "molcraft_mapping_ecfp4_source.csv", index=False)
    tables["funnel"].to_csv(OUT_DIR / "molcraft_mapping_funnel_source.csv", index=False)
    selected_funnel(tables["funnel"]).to_csv(
        OUT_DIR / "molcraft_mapping_funnel_selected_source.csv",
        index=False,
    )
    effects.to_csv(OUT_DIR / "molcraft_mapping_paired_effects.csv", index=False)


def plot_target_medians(fig: plt.Figure, spec, target_medians: pd.DataFrame) -> None:
    subgs = spec.subgridspec(2, 4, wspace=0.42, hspace=0.72)
    point_color = lighten(METHOD_COLORS["MolCRAFT"], 0.30)
    for idx, (metric, label) in enumerate(RAW_METRICS):
        ax = fig.add_subplot(subgs[idx // 4, idx % 4])
        sub = target_medians[target_medians["metric"] == metric]
        x = sub["molcraft_original"].to_numpy(dtype=float)
        y = sub["molcraft_random_mapping"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        lo = float(np.nanmin(np.r_[x, y]))
        hi = float(np.nanmax(np.r_[x, y]))
        pad = (hi - lo) * 0.08 if hi > lo else 0.1
        limits = (lo - pad, hi + pad)

        ax.plot(limits, limits, color=MUTED, linewidth=0.65, linestyle="--", zorder=1)
        ax.scatter(
            x,
            y,
            s=8,
            color=point_color,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(label, loc="left", pad=2)
        ax.locator_params(axis="both", nbins=3)
        ax.tick_params(labelsize=5.0, pad=1.5)
        ax.set_ylabel("Random mapping" if idx % 4 == 0 else "", fontsize=5.6)
        ax.set_xlabel("Original" if idx >= 4 else "", fontsize=5.6)
        style_axis(ax, grid_axis=None)
        if idx == 0:
            panel_label(ax, "a", x=-0.60, y=1.25)



def plot_ecfp4(ax: plt.Axes, ecfp4: pd.DataFrame) -> None:
    for method in METHODS:
        sub = ecfp4[ecfp4["method"] == method].sort_values("tau")
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


def selected_funnel(funnel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        sub = funnel[funnel["method"] == method]
        all_count = float(sub.loc[sub["filter_step"] == "All", "mols_retained"].iloc[0])
        for step, label in FUNNEL_STEPS:
            hit = sub[sub["filter_step"] == step]
            if hit.empty:
                continue
            row = hit.iloc[0].to_dict()
            rows.append(
                {
                    "method": method,
                    "step": label,
                    "mols_retained": float(row["mols_retained"]),
                    "mols_fraction": 100.0 * float(row["mols_retained"]) / all_count,
                    "targets_ge1": float(row["targets_ge1"]),
                    "targets_ge5": float(row["targets_ge5"]),
                }
            )
    return pd.DataFrame(rows)


def plot_target_funnel(ax: plt.Axes, funnel: pd.DataFrame) -> None:
    selected = selected_funnel(funnel)
    step_labels = [label for _, label in FUNNEL_STEPS]
    x = np.arange(len(step_labels))
    for method in METHODS:
        sub = selected[selected["method"] == method].set_index("step").reindex(step_labels)
        ax.plot(
            x,
            sub["targets_ge5"],
            marker="o",
            markersize=3.0,
            linewidth=1.15,
            color=color_for(method),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=22, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Targets retained")
    ax.set_title(r"Target retention ($\geq$5 candidates)", loc="left", pad=4)
    style_axis(ax, grid_axis="y")
    panel_label(ax, "c", x=-0.12)


def plot_molecule_funnel(ax: plt.Axes, funnel: pd.DataFrame) -> None:
    selected = selected_funnel(funnel)
    step_labels = [label for _, label in FUNNEL_STEPS]
    x = np.arange(len(step_labels))
    for method in METHODS:
        sub = selected[selected["method"] == method].set_index("step").reindex(step_labels)
        ax.plot(
            x,
            sub["mols_fraction"],
            marker="o",
            markersize=3.0,
            linewidth=1.15,
            color=color_for(method),
        )
    ax.set_yscale("log")
    ax.set_ylim(1.0, 130.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}%"))
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, rotation=22, ha="right")
    ax.set_ylabel("% retained")
    ax.set_title("Molecule retention (% of valid set)", loc="left", pad=4)
    style_axis(ax, grid_axis="y")
    panel_label(ax, "d")


def plot_effects(ax: plt.Axes, effects: pd.DataFrame) -> None:
    data = effects.set_index("metric").loc[[metric for metric, _, _ in EFFECT_METRICS]].reset_index()
    short_labels = ["Dock.", "RMSD", "QED", "SA", "SEI", "LLE", "ECFP4", "Strain"]
    y = np.arange(len(data))[::-1]
    ax.axvspan(-0.2, 0.2, color="#F1F3F4", zorder=0)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    for yi, row in zip(y, data.itertuples()):
        ax.plot([row.ci95_low, row.ci95_high], [yi, yi], color=MUTED, linewidth=1.0, zorder=2)
        ax.scatter(
            row.paired_effect_size_dz,
            yi,
            s=24,
            color=METHOD_COLORS["MolCRAFT"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(short_labels)
    ax.set_xlim(-0.8, 0.8)
    ax.set_xlabel("Effect size; positive favours random mapping")
    ax.set_title("Paired effect sizes", loc="left", pad=4)
    style_axis(ax, grid_axis="x")
    panel_label(ax, "e", x=-0.12)


def main() -> None:
    setup_publication_style()
    tables = read_source_data()
    effects = paired_effect_table(tables["raw"])
    write_source_tables(tables, effects)

    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.90, 0.95, 1.28],
        left=0.105,
        right=0.985,
        bottom=0.085,
        top=0.855,
        wspace=0.44,
        hspace=0.82,
    )
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    plot_target_medians(fig, gs[0, :], tables["target_medians"])
    plot_ecfp4(ax_b, tables["ecfp4"])
    plot_target_funnel(ax_c, tables["funnel"])
    plot_molecule_funnel(ax_d, tables["funnel"])
    plot_effects(ax_e, effects)

    handles = [
        Line2D([0], [0], color=color_for(m), marker="o", linewidth=1.15, markersize=3.4, label=label_for(m))
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=len(handles),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=1.1,
    )

    for out_dir in [OUT_DIR, PAPER_DIR]:
        stem = out_dir / "molcraft_mapping_control"
        save_all(fig, stem)
        print(f"Wrote {stem}.pdf/.svg/.eps/.png/.tiff")


if __name__ == "__main__":
    main()

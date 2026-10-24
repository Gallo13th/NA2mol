"""Shared publication styling for RNA2smol manuscript figures."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.patches import Rectangle


INK = "#202124"
MUTED = "#68717A"
GRID = "#E4E7EA"

# Fig. 2 uses the same convention as Fig. 3: only the protagonist carries a colour and
# everything it is compared against takes a grey ramp. The palette below was sampled
# from the published figure, together with the marker and dash that separate the three
# greys (two of which are the same shape, so the dash does the disambiguating).
METHOD_COLORS = {
    "RNA2smol":                       "#246B70",  # protagonist teal, shared with Fig. 3
    "RNA2smol (Only General Feat)":   "#246B70",  # alias for the full model
    "RNA2smol (Baseline)":            "#505050",  # dark grey
    "MolCRAFT":                       "#A5A5A5",  # light grey
    "MolCRAFT (Original)":            "#A5A5A5",
    "Pocket2Mol":                     "#808080",  # mid grey
    "ED2Mol":                         "#505050",  # dark grey
    "ed2mol":                         "#505050",
}
METHOD_MARKERS = {
    "RNA2smol":                       "o",
    "RNA2smol (Only General Feat)":   "o",
    "RNA2smol (Baseline)":            "o",
    "MolCRAFT":                       "o",
    "MolCRAFT (Original)":            "o",
    "Pocket2Mol":                     "s",
    "ED2Mol":                         "^",
    "ed2mol":                         "^",
}
METHOD_DASHES = {
    "RNA2smol":                       "solid",
    "RNA2smol (Only General Feat)":   "solid",
    "RNA2smol (Baseline)":            "solid",
    "MolCRAFT":                       "solid",
    "MolCRAFT (Original)":            "solid",
    "Pocket2Mol":                     (0, (4, 1.8)),    # dashed
    "ED2Mol":                         (0, (1, 1.2)),    # dotted
    "ed2mol":                         (0, (1, 1.2)),
}

# Fig. 3 follows the same convention as Fig. 2: only the protagonist (the full model)
# carries a colour, and the configurations it is compared against take a grey ramp.
# Three greys alone would not be separable, so each also gets its own marker and dash
# pattern -- the same device Fig. 2 uses for the external generators.
ABL_COLORS = {
    "RNA2smol (Baseline)":          "#555555",  # grey ramp, sampled from the reference figure
    "Pretrain Only":                "#707070",
    "Pretrain+Constrain":           "#989898",
    "RNA2smol (Only General Feat)": "#246B70",  # protagonist teal, shared with Fig. 2
}
ABL_MARKERS = {
    "RNA2smol (Baseline)":          "o",
    "Pretrain Only":                "s",
    "Pretrain+Constrain":           "^",
    "RNA2smol (Only General Feat)": "o",
}
ABL_DASHES = {
    "RNA2smol (Baseline)":          (0, (1, 1.2)),              # dotted
    "Pretrain Only":                (0, (4, 1.8)),              # dashed
    "Pretrain+Constrain":           (0, (6, 1.8, 1.2, 1.8)),    # dash-dot
    "RNA2smol (Only General Feat)": "solid",
}

# ── Attrition (endpoint gate losses) palette ──────────────────────────────
# Uses a sequential GREY ramp for failure gates (colorblind-safe, visually
# distinct from all method colours which are green/red/purple/blue) and BLACK
# for Retained (maximum contrast, convention for "survived" in stacked bars).
# The caller overlays a method-coloured thick border on the Retained segment
# to connect the attrition panel to per-method identity without colour conflict.
ATTRITION_COLORS = {
    "Dock.":    "#F0F0F0",  # lightest grey — first gate
    "QED":      "#D4D4D4",
    "SA":       "#B0B0B0",
    "SEI":      "#8C8C8C",
    "Strain":   "#696969",  # darkest grey — last failure gate
    "Retained": "#000000",  # Wong black — passed all gates
}

# Sequential filter cascade used by the virtual-screening funnel plots.
# The display label includes the threshold value for incremental steps
# (+SEI, +Strain energy) so readers can interpret the step without consulting the
# caption. Multi-line labels (\n) accommodate the added criterion.
FUNNEL_STEPS = [
    ("All", "All"),
    ("aff<-8,QED>0.5,SA<5", "Pharma."),
    ("+SEI>0.1", "+SEI\n(>0.1)"),
    ("aff<-9,QED>0.5,SA<5,SEI>0.15", "Strict"),
    ("aff<-9,QED>0.6,SA<4,SEI>0.2,strain<200", "+Strain energy\n(<200 kcal/mol)"),
]

# Gates evaluated at the strain-energy endpoint for the attrition stacked-bar panel.
# Order = left-to-right stacking order; first failed gate claims the molecule.
ENDPOINT_GATES = [
    ("dock", "Dock.", lambda d: d["affinity"] < -9),
    ("qed", "QED", lambda d: d["qed"] > 0.6),
    ("sa", "SA", lambda d: d["sa_score"] < 4),
    ("sei", "SEI", lambda d: d["SEI"] > 0.2),
    ("strain", "Strain", lambda d: d["strain"] < 200),
]

METHOD_LABELS = {
    "RNA2smol": "RNA2smol",
    "RNA2smol (Only General Feat)": "RNA2smol",
    "RNA2smol (Baseline)": "RNA2smol-base",
    "MolCRAFT": "MolCRAFT",
    "MolCRAFT (Original)": "MolCRAFT orig.",
    "Pocket2Mol": "Pocket2Mol",
    "ED2Mol": "ED2Mol",
    "ed2mol": "ED2Mol",
}


def setup_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, label: str | None, x: float = -0.13, y: float = 1.08,
                ha: str = "left") -> None:
    """Draw a bold panel letter. ``label=None`` is a no-op, so a figure that
    composes several subplots can suppress the per-axes labels and place all of
    them itself in figure coordinates (see place_panel_labels)."""
    if not label:
        return
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=INK,
    )


def lighten(hex_color: str, amount: float = 0.55) -> str:
    color = hex_color.lstrip("#")
    r, g, b = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def save_all(fig, stem: Path, include_tiff: bool = True, fast: bool = False) -> None:
    """Save figure in multiple formats. If fast=True, only PDF+PNG (for iteration)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    if fast:
        return
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".eps"), bbox_inches="tight")
    if include_tiff:
        fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")


def detect_text_overlaps(fig, min_area_frac: float = 0.005, verbose: bool = True) -> list:
    """Measure every text artist's on-canvas bounding box and report overlaps.

    Uses the renderer to compute exact window extents (including ``clip_on=False``
    labels such as column headers and panel letters), then checks all text pairs
    for positive-area intersection after normalisation by the figure area.

    Returns the list of overlapping pairs ``(t1, t2, overlap_area_norm)`` so
    callers can audit 布局 collisions precisely instead of eyeballing.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_area = fig.bbox.width * fig.bbox.height

    texts = []
    fx0, fy0 = fig.bbox.x0, fig.bbox.y0
    fx1, fy1 = fig.bbox.x1, fig.bbox.y1

    def _push(t):
        if not t.get_text().strip():
            return
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:
            return
        x0, y0 = max(bb.x0, fx0), max(bb.y0, fy0)
        x1, y1 = min(bb.x1, fx1), min(bb.y1, fy1)
        if x1 <= x0 or y1 <= y0:
            return
        texts.append((t, (x0, y0, x1, y1)))

    for ax in fig.axes:
        for t in ax.texts:
            _push(t)
    for leg in fig.legends:
        for t in leg.get_texts():
            _push(t)
    for t in fig.texts:
        _push(t)

    overlaps = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            t1, b1 = texts[i]
            t2, b2 = texts[j]
            ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
            area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if area <= 0:
                continue
            frac = area / fig_area
            if frac >= min_area_frac:
                overlaps.append((t1, t2, frac))

    if verbose:
        if not overlaps:
            print("[text-overlap] none detected")
        else:
            print(f"[text-overlap] {len(overlaps)} overlapping pairs:")
            for t1, t2, frac in overlaps:
                print(f"  overlap {frac:.4f}: "
                      f"'{t1.get_text()[:40]}' ({_short(t1.get_position())}) vs "
                      f"'{t2.get_text()[:40]}' ({_short(t2.get_position())})")
    return overlaps


def _short(pos) -> str:
    return f"({pos[0]:.2f},{pos[1]:.2f})"


def save_panel(fig, panels_dir: Path, name: str) -> list[Path]:
    """Save a standalone panel figure (PNG + PDF) into ``panels_dir``.

    Returns the list of written paths (PNG first) so callers can audit the
    rendered panels before stitching.
    """
    panels_dir.mkdir(parents=True, exist_ok=True)
    out = []
    png = panels_dir / f"{name}.png"
    pdf = panels_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight")
    out.extend([png, pdf])
    return out


def stitch_panels(
    panel_assets: list[dict],
    out_stem: Path,
    figsize: tuple[float, float],
    grid: tuple[int, int],
    dpi: int = 600,
) -> None:
    """Stitch independently rendered panel PNGs into a single figure.

    ``panel_assets``: list of ``{"name": str, "png": Path, "rows": n, "cols": m}``
    entries describing each panel, its rendered PNG, and its grid span.
    ``grid``: (n_rows, n_cols) of the stitched layout. The stitched figure is
    written as PDF + PNG (+SVG/EPS/TIFF) at ``out_stem`` so downstream
    publication code can consume it exactly like ``save_all``.
    """
    from PIL import Image

    n_rows, n_cols = grid
    # Normalise: build a full grid by placing each asset at the first free slot.
    placements: list[tuple[int, int, np.ndarray, dict]] = []
    occupied: set[tuple[int, int]] = set()
    slot = 0
    for asset in panel_assets:
        img = np.asarray(Image.open(asset["png"]).convert("RGB"))
        # find top-left corner of next free slot (row-major)
        while slot in occupied:
            slot += 1
        r, c = divmod(slot, n_cols)
        placements.append((r, c, img, asset))
        for dr in range(asset.get("rows", 1)):
            for dc in range(asset.get("cols", 1)):
                occupied.add((r + dr, c + dc))
        slot += asset.get("cols", 1)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.04, hspace=0.04)
    for r, c, img, asset in placements:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img)
        ax.set_axis_off()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax, grid_axis: str | None = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def profile_value_formatter(value: float, metric_key: str) -> str:
    """Smart per-metric formatter for ``plot_profile_table``.

    - Strain values >= 10,000 use mathtext scientific notation (e.g. $2.15\\times10^{6}$)
      to keep wide-range strain columns readable.
    - Strain values < 10,000 use integer format.
    - max_ecfp4_tanimoto uses 3 decimals (small range).
    - All other metrics use 2 decimals.
    """
    if metric_key == "strain":
        if abs(value) >= 10_000:
            exp = int(np.floor(np.log10(abs(value))))
            mantissa = value / (10 ** exp)
            sign = "-" if value < 0 else ""
            return f"${sign}{mantissa:.2f}\\times10^{{{exp}}}$"
        return f"{value:.0f}"
    if metric_key == "max_ecfp4_tanimoto":
        return f"{value:.3f}"
    return f"{value:.2f}"


def group_metrics_by_direction(
    metrics: list[tuple[str, str]],
    lower_is_better: set[str],
    higher_first: bool = True,
) -> list[tuple[str, str]]:
    """Reorder metrics so that same-direction columns sit together, and append
    a direction marker ('↑' higher-is-better / '↓' lower-is-better) to each label.

    Pass ``higher_first=True`` (default) to place higher-is-better columns on
    the left and lower-is-better columns on the right; pass ``False`` to invert.
    Markers stay on the SAME line as the label so headers never rotate onto
    the cell content.
    """
    higher = [m for m in metrics if m[0] not in lower_is_better]
    lower = [m for m in metrics if m[0] in lower_is_better]
    ordered = (higher + lower) if higher_first else (lower + higher)
    out = []
    for col, lbl in ordered:
        marker = "↓" if col in lower_is_better else "↑"
        out.append((col, f"{lbl} {marker}"))
    return out


def plot_profile_table(
    ax: plt.Axes,
    profile: pd.DataFrame,
    methods: list[str],
    metrics: list[tuple[str, str]],
    title: str,
    label: str,
    method_colors: dict[str, str] | None = None,
    method_labels: dict[str, str] | None = None,
    lower_is_better: set[str] | None = None,
    value_fmt: str | dict[str, str] | Callable[[float, str], str] = "{:.2f}",
    bar_mode: str = "rank",
    title_pad: float | None = None,
) -> None:
    """Render a chart-enhanced table: raw value text over a data-bar background.

    ``bar_mode`` controls the data-bar width semantics:
      - ``"rank"`` (default): the bar width encodes the **rank-normalised score**
        (1.0 = best, ``rank_floor`` = worst) passed in ``{col}_rank_norm``
        columns. This decouples bar length from the raw value range, so columns
        with different units/scales (e.g. strain vs QED) are visually comparable.
      - ``"value"``: legacy behaviour — bar width encodes the **min-max
        normalised raw value** across methods (best fills the cell, worst has no
        bar), direction-aware via ``lower_is_better``.

    The best method per column is bolded and marked with a small dot in the
    top-right corner. Column labels should already be direction-grouped
    (e.g. via :func:`group_metrics_by_direction`).
    """
    if method_colors is None:
        method_colors = METHOD_COLORS
    if method_labels is None:
        method_labels = METHOD_LABELS
    if lower_is_better is None:
        lower_is_better = set()

    n_rows = len(methods)
    n_cols = len(metrics)

    if "method" in profile.columns:
        prof = profile.drop_duplicates("method").set_index("method")
    else:
        prof = profile

    # Best method per column (used for bolding/starring) and, for legacy
    # value-mode bars, the min-max range per column.
    best_per_col: dict[str, str] = {}
    minmax_per_col: dict[str, tuple[float, float]] = {}
    for col_key, _ in metrics:
        raw_col = f"{col_key}_raw"
        if raw_col not in prof.columns:
            continue
        present = [m for m in methods if m in prof.index]
        if not present:
            continue
        sub = prof.loc[present, raw_col]
        # Guard against duplicate-index Series expanding to a DataFrame.
        if isinstance(sub, pd.DataFrame):
            sub = sub.iloc[:, 0]
        sub = pd.to_numeric(sub, errors="coerce").dropna()
        if len(sub) == 0:
            continue
        best_per_col[col_key] = sub.idxmin() if col_key in lower_is_better else sub.idxmax()
        if len(sub) >= 2:
            minmax_per_col[col_key] = (float(sub.min()), float(sub.max()))

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()  # first method at the top
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_idx, method in enumerate(methods):
        color = method_colors.get(method, MUTED)
        row_label = method_labels.get(method, method)

        # Row label in the left margin.
        ax.text(
            -0.12,
            row_idx + 0.5,
            row_label,
            ha="right",
            va="center",
            fontsize=6.5,
            color=INK,
            clip_on=False,
        )

        for col_idx, (col_key, _) in enumerate(metrics):
            rank_col = f"{col_key}_rank_norm"
            raw_col = f"{col_key}_raw"

            rank_val = float("nan")
            raw_val = float("nan")
            if method in prof.index:
                if rank_col in prof.columns:
                    rank_val = prof.loc[method, rank_col]
                if raw_col in prof.columns:
                    raw_val = prof.loc[method, raw_col]
            # Collapse duplicate-index lookups to a scalar.
            if isinstance(rank_val, pd.Series):
                rank_val = rank_val.iloc[0]
            if isinstance(raw_val, pd.Series):
                raw_val = raw_val.iloc[0]

            # Background data bar.
            #  - rank mode: width = rank-normalised score directly (1.0 → full
            #    cell, worst → shortest bar). No min-max re-normalisation.
            #  - value mode: legacy min-max normalised raw value.
            if bar_mode == "rank":
                bar_w = float(rank_val) if rank_val == rank_val else 0.0
            else:
                bar_w = 0.0
                if raw_val == raw_val and col_key in minmax_per_col:
                    vmin, vmax = minmax_per_col[col_key]
                    if vmax - vmin > 1e-8:
                        if col_key in lower_is_better:
                            bar_w = (vmax - float(raw_val)) / (vmax - vmin)
                        else:
                            bar_w = (float(raw_val) - vmin) / (vmax - vmin)
                        bar_w = max(0.0, min(1.0, bar_w))
                    else:
                        bar_w = 0.5  # all methods identical
            if bar_w > 0.0:
                ax.add_patch(
                    Rectangle(
                        (col_idx, row_idx),
                        bar_w,
                        1,
                        facecolor=color,
                        alpha=0.55,
                        edgecolor="none",
                        zorder=1,
                    )
                )

            # Thin white cell separator drawn on top of the bar.
            ax.add_patch(
                Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    facecolor="none",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=2,
                )
            )

            # Raw value text.
            if raw_val == raw_val:
                try:
                    raw_f = float(raw_val)
                    if callable(value_fmt):
                        txt = value_fmt(raw_f, col_key)
                    elif isinstance(value_fmt, dict):
                        fmt = value_fmt.get(col_key, "{:.2f}")
                        txt = fmt.format(raw_f)
                    else:
                        txt = value_fmt.format(raw_f)
                except (TypeError, ValueError):
                    txt = str(raw_val)
            else:
                txt = "—"
            is_best = best_per_col.get(col_key) == method
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                txt,
                ha="center",
                va="center",
                fontsize=6.0,
                color=INK,
                fontweight="bold" if is_best else "normal",
                zorder=3,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white", alpha=0.92)],
            )

            # Redundant best-marker dot in the top-right corner.
            if is_best:
                ax.scatter(
                    [col_idx + 0.92],
                    [row_idx + 0.08],
                    s=14,
                    color=color,
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=4,
                    clip_on=False,
                )

    # Column labels in the top margin — always horizontal (never rotated onto
    # the cell content); they occupy the band between the title and the first
    # row of cells.
    for col_idx, (_, col_label) in enumerate(metrics):
        ax.text(
            col_idx + 0.5,
            -0.14,
            col_label,
            ha="center",
            va="top",
            fontsize=6,
            color=INK,
            rotation=0,
            linespacing=1.15,
            clip_on=False,
        )

    ax.set_title(title, fontsize=8, color=INK,
                 pad=(14 if title_pad is None else title_pad))
    panel_label(ax, label)


def place_panel_labels(fig, gridspec, cells: dict, dx: float = 0.078,
                       dy: float = 0.012, fontsize: float = 9) -> None:
    """Place panel letters at fixed figure coordinates, one per gridspec cell.

    Anchoring to the *cell* rather than the axes is what makes the letters line up:
    a polar axes is shrunk to a square inside its cell (x0 0.165 where the
    rectilinear panels of the same column sit at 0.105), so axes-relative offsets
    land in different places for different panel types. Call this after every
    subplot has been drawn.

    ``cells`` maps the letter to its (row, col) in ``gridspec``.
    """
    for letter, (row, col) in cells.items():
        box = gridspec[row, col].get_position(fig)
        fig.text(box.x0 - dx, box.y1 + dy, letter, ha="left", va="bottom",
                 fontsize=fontsize, fontweight="bold", color=INK)


def readable_ink(hex_color: str, threshold: float = 0.6) -> str:
    """White or INK, whichever stays legible on the given fill colour.

    Needed once the comparison series became a grey ramp: a white label on the
    light grey is invisible.
    """
    c = hex_color.lstrip('#')
    r, g, b = (int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if luminance > threshold else 'white'


def place_panel_labels_from_rects(fig, rects: dict, dx: float = 0.030,
                                  dy: float = 0.012, fontsize: float = 9) -> None:
    """Place panel letters from explicit ``(x0, y0, w, h)`` cell rectangles.

    The counterpart of place_panel_labels for figures laid out with add_axes instead of
    a gridspec. Anchoring to the cell rather than the axes is what makes the letters
    line up: a polar axes is inset inside its cell and reports a smaller box, so
    axes-relative offsets put its letter somewhere else.
    """
    for letter, (x0, y0, w, h) in rects.items():
        fig.text(x0 - dx, y0 + h + dy, letter, ha="left", va="bottom",
                 fontsize=fontsize, fontweight="bold", color=INK)

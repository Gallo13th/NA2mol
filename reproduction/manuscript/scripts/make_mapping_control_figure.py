"""Supplementary Fig. 1 — how much does randomising residue identity move each baseline?

Drawn at the journal column width (131 mm) so the 6 pt labels print at their nominal size
instead of being shrunk by \\includegraphics.

  a  paired effect sizes (Cohen's d_z, 95% CI) of the random-residue arm against the native arm,
     one row per metric, one point per baseline;
  b  screening retention through the manuscript's four-tier cascade, as a percentage of each
     arm's own generated pool, one small panel per baseline.

Design notes
  * palette, markers and dash patterns follow the manuscript's SI conventions
    (analysis/style.py), so this figure sits next to the others;
  * filled markers are significant after Benjamini-Hochberg correction across the eight metrics
    of that method, open markers are not;
  * the shaded band marks |d_z| < 0.2;
  * panel b uses a log axis because the cascade spans four orders of magnitude, and a point on
    the axis floor marks a tier where no molecule survived.

    python3 scripts/make_mapping_control_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
EFFECTS = ROOT / 'docs' / 'mapping_control_paired_effects_four_methods.csv'
FUNNEL = ROOT / 'docs' / 'mapping_control_funnel_four_methods.csv'
OUT = ROOT / 'figures' / 'si_fig_mapping_control.pdf'

STYLE = {
    'MolCRAFT':   dict(color='#A68979', marker='s', dash=(0, (4, 2))),
    'Pocket2Mol': dict(color='#A69879', marker='^', dash=(0, (1, 1.4))),
    'DiffSBDD':   dict(color='#7998A6', marker='D', dash=(0, (3, 1.5, 1, 1.5))),
    'FLAG':       dict(color='#A67998', marker='X', dash=(0, (2, 2))),
}
ORDER = ['FLAG', 'MolCRAFT', 'Pocket2Mol', 'DiffSBDD']       # least to most sensitive
METRICS = ['Docking score', 'Relax. RMSD', 'QED', 'SA score', 'SEI', 'LLE', 'ECFP4',
           'Strain energy']
SHORT = {'Docking score': 'Docking', 'Relax. RMSD': 'Relax. RMSD', 'QED': 'QED',
         'SA score': 'SA', 'SEI': 'SEI', 'LLE': 'LLE', 'ECFP4': 'ECFP4',
         'Strain energy': 'Strain'}
TIERS = ['Pharma.', '+SEI', 'Strict', 'Final tier']
TIER_SHORT = ['Pharma.', '+SEI', 'Strict', 'Final']
BAND = 0.2

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6.0,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.55,
    'ytick.major.width': 0.55,
    'xtick.major.size': 2.2,
    'ytick.major.size': 2.2,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})


def panel_a(ax, eff: pd.DataFrame) -> None:
    n_met = len(METRICS)
    half = 0.24
    offs = np.linspace(half, -half, len(ORDER))
    ax.axvspan(-BAND, BAND, color='0.93', zorder=0, lw=0)
    ax.axvline(0, color='0.35', lw=0.7, ls=(0, (1, 1.2)), zorder=1)

    for m_i, metric in enumerate(METRICS):
        y_base = n_met - 1 - m_i
        for meth, off in zip(ORDER, offs):
            row = eff[(eff['pair'] == meth) & (eff['label'] == metric)]
            if row.empty:
                continue
            r = row.iloc[0]
            dz, lo, hi = (float(r['paired_effect_size_dz']), float(r['ci95_low']),
                          float(r['ci95_high']))
            q = float(r['wilcoxon_q_bh'])
            st = STYLE[meth]
            y = y_base + off
            ax.plot([lo, hi], [y, y], color=st['color'], lw=1.0, solid_capstyle='butt', zorder=2)
            ax.plot(dz, y, marker=st['marker'], ms=3.2, zorder=3,
                    mfc=st['color'] if q < 0.05 else 'white', mec=st['color'], mew=0.75)

    ax.set_yticks(range(n_met))
    ax.set_yticklabels([SHORT[m] for m in reversed(METRICS)])
    ax.set_ylim(-0.75, n_met - 0.25)
    ax.set_xlim(-1.55, 0.78)
    ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5])
    ax.set_xlabel("Cohen's $d_z$, random vs native residues  (positive = random better)")
    ax.tick_params(axis='y', length=0, pad=2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    handles = [Line2D([], [], color=STYLE[m]['color'], marker=STYLE[m]['marker'], ms=3.2,
                      lw=1.0, label=m) for m in ORDER]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0),
              frameon=False, fontsize=5.6, handlelength=1.2, handletextpad=0.45,
              columnspacing=0.9, borderaxespad=0.0, ncol=4)
    ax.text(0.0, -0.235, 'shaded: $|d_z|<0.2$   ·   filled markers: $q<0.05$   ·   '
                         'open markers: not significant', transform=ax.transAxes,
            ha='left', va='top', fontsize=5.2, color='0.35')


def panel_b(axes, fun: pd.DataFrame) -> None:
    alls = fun[fun['filter_step'] == 'All'].set_index(['pair', 'arm'])['mols_retained']
    x = np.arange(len(TIERS))
    shift = 0.09
    for ax, meth in zip(axes, ORDER):
        st = STYLE[meth]
        for arm, ls, filled, dx in (('native', (0, (1, 0)), True, -shift),
                                    ('random', st['dash'], False, +shift)):
            sub = (fun[(fun['pair'] == meth) & (fun['arm'] == arm)
                       & (fun['filter_step'].isin(TIERS))]
                   .assign(tier=lambda d: d['filter_step'].map({t: i for i, t in enumerate(TIERS)}))
                   .sort_values('tier'))
            if sub.empty:
                continue
            base = float(alls.get((meth, arm), np.nan))
            y = 100.0 * sub['mols_retained'].to_numpy(float) / base
            y_plot = np.where(y > 0, y, 2.6e-4)
            ax.plot(sub['tier'] + dx, y_plot, ls=ls, color=st['color'], lw=0.9, zorder=2)
            for t, yy, val in zip(sub['tier'] + dx, y_plot, y):
                ax.plot(t, yy, ls='none', marker=st['marker'], ms=2.9, zorder=3,
                        mfc=st['color'] if (filled and val > 0) else 'white',
                        mec=st['color'], mew=0.7)
                if val <= 0:
                    ax.annotate('0', (t, yy), textcoords='offset points', xytext=(0, 2.6),
                                ha='center', fontsize=4.6, color=st['color'])
        ax.set_yscale('log')
        ax.grid(axis='y', which='major', color='0.93', lw=0.4, zorder=0)
        ax.set_ylim(1.3e-4, 60)
        ax.set_xlim(-0.45, len(TIERS) - 0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(TIER_SHORT, fontsize=5.4)
        ax.set_title(meth, fontsize=6.0, pad=2.5, color=st['color'])
        ax.tick_params(axis='x', pad=1.2, length=2.0)
        ax.tick_params(axis='y', labelsize=5.4)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel('% of pool\nretained', fontsize=5.6, labelpad=2)
    axes[2].set_ylabel('% of pool\nretained', fontsize=5.6, labelpad=2)


def main() -> int:
    eff = pd.read_csv(EFFECTS)
    fun = pd.read_csv(FUNNEL)

    width_mm = 131.0
    fig = plt.figure(figsize=(width_mm / 25.4, 6.15))
    gs_a = fig.add_gridspec(1, 1, left=0.10, right=0.99, top=0.945, bottom=0.545)
    ax_a = fig.add_subplot(gs_a[0, 0])
    panel_a(ax_a, eff)
    gs_b = fig.add_gridspec(2, 2, left=0.10, right=0.99, top=0.415, bottom=0.065,
                            hspace=0.62, wspace=0.19)
    b_axes = [fig.add_subplot(gs_b[0, 0]), fig.add_subplot(gs_b[0, 1]),
              fig.add_subplot(gs_b[1, 0]), fig.add_subplot(gs_b[1, 1])]
    panel_b(b_axes, fun)

    fig.text(0.008, 0.985, 'a', fontsize=8.0, fontweight='bold', va='top')
    fig.text(0.008, 0.50, 'b', fontsize=8.0, fontweight='bold', va='top')

    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix('.png'), dpi=600)
    print(f'wrote {OUT} and {OUT.with_suffix(".png")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

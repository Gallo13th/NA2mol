#!/usr/bin/env python3
"""Shared style for the Clivia figure panels (single source of truth)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- palette ----
C_DATA = '#9fb8d6'      # neutral data (histogram fill)
C_KEEP = '#14615c'      # retained / selected
C_HILITE = '#0a3d3a'    # dark teal: the chosen scaffold
C_DYE = '#c0392b'       # crystal-dye reference lines
C_AL1 = '#b5651d'       # compound Al-1
C_AL2 = '#8e6cb0'       # compound Al-2
C_GREY = '#cfd6d5'      # removed / background
C_TEXT = '#333333'

# ---- typography ----
FONT = ['Helvetica Neue', 'Arial', 'Helvetica', 'DejaVu Sans']
SZ_TICK, SZ_LABEL, SZ_LEGEND, SZ_NOTE = 7.0, 8.5, 7.0, 7.0

LW_AXIS, LW_THRESH, LW_REF, LW_FIT = 0.8, 0.8, 1.1, 1.1
MS_SMALL, MS_BIG = 7, 60


def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': FONT,
        'font.size': SZ_LABEL,
        'axes.labelsize': SZ_LABEL,
        'axes.titlesize': SZ_LABEL,
        'axes.linewidth': LW_AXIS,
        'xtick.labelsize': SZ_TICK,
        'ytick.labelsize': SZ_TICK,
        'xtick.major.width': LW_AXIS,
        'ytick.major.width': LW_AXIS,
        'xtick.major.size': 2.8,
        'ytick.major.size': 2.8,
        'legend.fontsize': SZ_LEGEND,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'text.color': C_TEXT,
        'axes.labelcolor': C_TEXT,
        'xtick.color': C_TEXT,
        'ytick.color': C_TEXT,
        'savefig.transparent': False,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=2.8, width=LW_AXIS)

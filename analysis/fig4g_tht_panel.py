"""KRAS ThT fluorescence competition — per-group inhibition panels.

Same design as panel_g4_f_tht: each condition is compared with its own
DMSO control (inhibition = 100*(1 - S1R1_cond / S1R1_DMSO)), drawn as
mean +/- s.d. bars with the three replicate wells as points (n=3 is far
too few for box plots, which collapse to a line). Raw S1/R1 spans only a
narrow window within each experiment, and the equilibrated second reading
is used because the 1 uM first reading is dominated by ThT aggregation
(DMSO S1/R1 ~1.2e4 vs ~450 in the second read).

Outputs (same 3.4 x 2.5 in landscape as the G4 audit panels):
  panel_kras_tht_inhib100nM.{pdf,svg,png}
  panel_kras_tht_inhib1uM.{pdf,svg,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]

# Authoritative experimental data are the two Origin projects. They are binary
# and must be exported sheet-by-sheet before this script can read them; see
# reproduction/data/tht/README.md. 荧光数据汇总.csv is NOT used: it aggregates
# all sheets into one table with an extra leading column, so the column offsets
# below (S1/R1 at index 4) do not line up.
THT_DIR = ROOT / "reproduction" / "data" / "tht"
THT_ORIGIN = [THT_DIR / "KRAS-THT-Compatative.opj",
              THT_DIR / "THT-Compatative.opju"]
THT_EXPORT = THT_DIR / "export"
OUT = ROOT / "reproduction" / "figures"

INK = "#202124"
GRID = "#E4E7EA"
BLUE = "#0072B2"     # DSF (known G4 binder)
GREEN = "#009E73"    # generated molecule
PANEL_FIGSIZE = (2.2, 3.0)   # narrower, for side-by-side placement


def load(base_dir: Path) -> dict:
    """Read raw S1/R1 from the KRAS OPJ extract (sheet 0 = 1st read,
    sheet 2 = 2nd/equilibrated). Return per-(experiment, condition) values
    of the second reading (3 wells each)."""
    out = {}
    # experiment "THT" (KRAS 100 nM + ThT 200 nM, per the experimental team's
    # protocol): ex0-5 = Buffer1, Buffer2, DMSO1, DMSO2, DSF, CPD2.
    # "10X" is the higher-concentration condition (KRAS 1 uM): ex0-4 = KONGMIN,
    # Buffer+ThT, DMSO, DSF, CPD2. Its ThT concentration is NOT stated in the
    # protocol; read it from the Origin project before quoting it anywhere.
    maps = {
        "THT": {"DMSO": [2, 3], "DSF": [4], "gen": [5]},
        "10X": {"DMSO": [2], "DSF": [3], "gen": [4]},
    }
    for pref, conds in maps.items():
        for key, exs in conds.items():
            vals = []
            for ex in exs:
                row = pd.read_csv(base_dir / f"{pref}_ex{ex}_sheet2.csv",
                                  skiprows=1, header=0)
                s1r1 = pd.to_numeric(row.iloc[:3, 4], errors="coerce").dropna().to_numpy()
                vals.extend(s1r1)
            out[(pref, key)] = np.array(vals)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = load(THT_EXPORT)



def inhib(cond_vals: np.ndarray, dmso_vals: np.ndarray) -> np.ndarray:
    dm = dmso_vals.mean()
    return 100 * (1 - cond_vals / dm)


def draw(pref: str, title: str, fname: str, stats: dict) -> None:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    groups = [("DSF", "DSF", BLUE), ("gen", "generated", GREEN)]
    x = np.arange(len(groups))
    width = 0.42
    # per-condition inhibition vs the same-experiment DMSO control (0 % line)
    dmso_mean = stats[(pref, "DMSO")].mean()
    def inh(vals):
        return 100 * (1 - vals / dmso_mean)
    for si, (key, name, color) in enumerate(groups):
        vals = inh(stats[(pref, key)])
        ax.bar(x[si], vals.mean(), yerr=vals.std(ddof=1),
               color=color, alpha=0.85, edgecolor="none",
               capsize=2.5, error_kw={"lw": 0.7, "capsize": 2.5,
                                      "color": "#333333"}, zorder=2)
        rng = np.random.default_rng(2070 + si)
        for v in vals:
            ax.scatter(x[si] + rng.uniform(-0.09, 0.09), v, s=16,
                       facecolors="white", edgecolors=color,
                       linewidths=0.6, zorder=3)
        ax.text(x[si], max(vals.mean() + 3, 3), f"{vals.mean():.0f}%",
                ha="center", va="bottom", fontsize=5.0,
                color=color, fontweight="bold")
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for _, n, _ in groups], fontsize=6.4)
    ax.set_ylabel("ThT inhibition (%)", fontsize=6.4)
    ax.set_ylim(-15, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, fontsize=7.5, pad=6)
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{fname}.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    dm = stats[(pref, 'DMSO')].mean()
    dsf_inh = 100 * (1 - stats[(pref, 'DSF')].mean() / dm)
    gen_inh = 100 * (1 - stats[(pref, 'gen')].mean() / dm)
    print(f"[panel] {fname}: DMSO_baseline={dm:.1f}  DSF_inh={dsf_inh:.1f}%  "
          f"generated_inh={gen_inh:.1f}%")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = load(THT_EXPORT)
    # 100 nM experiment: THT file (KRAS 100 nM + ThT 200 nM)
    draw("THT", "KRAS 100 nM + THT 100 nM", "panel_kras_tht_inhib100nM", stats)
    # 1 uM experiment: 10X file (KRAS 1 uM + THT 1 uM)
    draw("10X", "KRAS 1 uM + THT 1 uM", "panel_kras_tht_inhib1uM", stats)


if __name__ == "__main__":
    main()

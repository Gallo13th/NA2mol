#!/usr/bin/env python3
"""Export the values underlying the radar panels of Figures 2 and 3.

Each radar axis is a normalised score s_m in [0, 1] drawn at r = 1 + 2 s_m, so the
raw metric values are not recoverable from the figure. This script writes them out:

  * the pooled median of each of the six property metrics (the same molecule pool the
    retention-curve integrals use),
  * the ECFP4 observed recovery, its shuffled random baseline and the resulting gain,
  * and the seven normalised scores that the radar actually plots,

for both benchmarks of Fig. 2 (DUD-E protein, RNA test set) and the four
configurations of Fig. 3.

The scores are read from `radar_scores_<benchmark>.csv` (written by
radar_profiles.py) so this export never recomputes the expensive permutation null.

    python3 analysis/export_radar_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import radar as R  # noqa: E402

DATA = ROOT / 'data' / 'benchmark'
TABLES = DATA / 'output' / 'tables' / 'radar'
OUT_CSV = TABLES / 'radar_values.csv'
OUT_TEX = ROOT / 'reproduction' / 'manuscript' / 'si_table_radar_values.tex'

# benchmark -> (metrics file, actives file or None, actives smiles column)
# key -> (metrics file, actives file or None, actives smiles column)
# the table filenames keep the historical benchmark keys: smrtbench / dude
BENCH = {
    'smrtbench': (DATA / 'output' / 'smrtbench_combined_results.csv', None, None),
    'dude': (DATA / 'dude_stat_all.csv', DATA / 'DUDE_active_compounds.csv', 'canonical_smiles'),
}
# method membership per radar panel, mirroring analysis/fig2_cross_benchmark.py
# (SMRT_METHODS / DUDE_METHODS) and analysis/fig3_ablation.py (ABL_ORDER)
PANELS = [
    ('Fig. 2a', 'dude', 'DUD-E protein',
     ['RNA2smol', 'MolCRAFT', 'Pocket2Mol', 'ed2mol']),
    ('Fig. 2d', 'smrtbench', 'RNA test set',
     ['RNA2smol (Only General Feat)', 'MolCRAFT', 'Pocket2Mol', 'ED2Mol']),
    ('Fig. 3a', 'smrtbench', 'RNA test set',
     ['RNA2smol (Baseline)', 'Pretrain Only', 'Pretrain+Constrain',
      'RNA2smol (Only General Feat)']),
]

LABEL = {
    'RNA2smol (Only General Feat)': 'NA2smol',
    'RNA2smol (Baseline)': 'Baseline',
    'Pretrain Only': 'Pretrain Only',
    'Pretrain+Constrain': 'Pretrain+Constrain',
    'RNA2smol': 'NA2smol',
    'ed2mol': 'ED2Mol',
}

COLS = ['mode', 'pdb_id', 'SMILES', 'affinity', 'rmsd', 'qed', 'sa_score',
        'tpsa', 'strain']


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'pdb_id': str},
                       usecols=lambda c: c in COLS)


def main() -> int:
    frames, scores = {}, {}
    for key, (path, _, _) in BENCH.items():
        frames[key] = load(path)
        sf = TABLES / f'radar_scores_{key}.csv'
        if not sf.is_file():
            raise SystemExit(f'missing {sf}; run analysis/radar_profiles.py first')
        scores[key] = pd.read_csv(sf).set_index('method')
        scores[key].index = scores[key].index.astype(str)
        frames[key]['SEI'] = R.sei(frames[key])

    rows = []
    for panel, key, bench_label, methods in PANELS:
        frame, sc = frames[key], scores[key]
        for method in methods:
            if method not in sc.index:
                raise SystemExit(f'{panel}: {method!r} absent from radar_scores_{key}.csv')
            sub = frame[frame['mode'] == method]
            s_row = sc.loc[method]
            row = {
                'panel': panel,
                'benchmark': bench_label,
                'method': LABEL.get(method, method),
                'n_molecules': int(len(sub)),
                'n_targets': int(sub['pdb_id'].nunique()),
            }
            row.update(R.property_summaries(sub))
            row['recovery_observed'] = float(s_row['recovery_observed'])
            row['recovery_random'] = float(s_row['recovery_random'])
            for a in R.AXIS_ORDER:
                row[f's_{a}'] = float(s_row[a])
            rows.append(row)

    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format='%.4g')
    print(f'wrote {OUT_CSV.relative_to(ROOT)}  ({len(df)} rows, {len(df.columns)} cols)')
    print(df[['panel', 'method', 'n_molecules', 'median_docking', 'median_rmsd',
              'median_qed', 'median_sa', 'median_sei', 'median_strain',
              'recovery_observed', 's_ecfp4']].to_string(index=False))
    write_latex(df)
    write_main_table(df)
    return 0


# ── LaTeX (Supplementary Information style) ───────────────────────────────────
def _fmt_strain(v: float) -> str:
    """Plain for <1000 kcal/mol, mathtext scientific above that (14,290 reads badly)."""
    if v != v:
        return '--'
    if v < 1000:
        return f'{v:.0f}'
    mant, exp = f'{v:.2e}'.split('e')
    return f'${float(mant):.1f}\\times10^{{{int(exp)}}}$'


# headers carry no units: with them Panel A overflowed the text block by 50-78 pt.
# Units are stated in the caption note instead.
RAW_COLS = [
    ('median_docking', 'Dock.', '%.2f'),
    ('median_rmsd', 'RMSD', '%.2f'),
    ('median_qed', 'QED', '%.3f'),
    ('median_sa', 'SA', '%.2f'),
    ('median_sei', 'SEI', '%.3f'),
    ('median_strain', 'Strain', _fmt_strain),
    ('recovery_observed', 'ECFP4 rec.', '%.3f'),
]
SCORE_COLS = [
    ('s_qed', 'QED', '%.3f'), ('s_log10_strain', 'Strain', '%.3f'),
    ('s_sa_score', 'SA', '%.3f'), ('s_rmsd', 'RMSD', '%.3f'),
    ('s_affinity', 'Dock.', '%.3f'), ('s_ecfp4', 'ECFP4', '%.3f'),
    ('s_sei', 'SEI', '%.3f'),
]


def _grouped_table(df, cols, panel_title):
    """One table holding every figure panel, separated by full-width group rows."""
    head = ' & '.join(f'\\textbf{{{lbl}}}' for _, lbl, _ in cols)
    out = ['\\noindent\\begin{minipage}{\\textwidth}',
           f'\\textbf{{{panel_title}}}', '\\begin{center}\\footnotesize',
           '\\setlength{\\tabcolsep}{3.5pt}',
           f'\\begin{{tabular}}{{@{{}}ll{chr(99) * len(cols)}@{{}}}}', '\\toprule',
           f'\\textbf{{Method}} & \\textbf{{$n$}} & {head} \\\\', '\\midrule']
    groups = [('Fig. 2a', 'DUD-E protein'), ('Fig. 2d', 'RNA test set'),
              ('Fig. 3a', 'RNA test set, four configurations')]
    for gi, (panel, label) in enumerate(groups):
        sub = df[df['panel'] == panel]
        if sub.empty:
            continue
        if gi:
            out.append('\\midrule')
        out.append(f'\\multicolumn{{{2 + len(cols)}}}{{@{{}}l}}{{\\textit{{{panel} \u2014 {label}}}}} \\\\')
        for _, r in sub.iterrows():
            cells = []
            for key, _, fmt in cols:
                v = r[key]
                cells.append(fmt(v) if callable(fmt) else ('--' if v != v else fmt % v))
            out.append(f"{r['method']} & {int(r['n_molecules']):,} & " + ' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}', '\\end{center}', '\\end{minipage}']
    return '\n'.join(out)


def write_latex(df: pd.DataFrame) -> None:
    notes = (
        'Values underlying the radar panels of Figs.~2a, 2d and 3a. The radar axes are '
        'normalised scores $s_m$ drawn at $r_m = 1 + 2s_m$, so the raw metric values are '
        'not recoverable from the figure and are given here. Property values are pooled '
        'medians over all of the method\'s molecules, the same molecule pool used by the '
        'retention-curve integrals; the radar axes for docking, RMSD, QED, SA, SEI and '
        'strain are the normalised integral of the retention curve (log$_{10}$ scale for '
        'strain), so a median is a summary and not the plotted quantity. Dock. is in '
        'kcal mol$^{-1}$, RMSD in \\AA, and strain in kcal mol$^{-1}$; strain values above '
        '10$^3$ are written in scientific notation. ECFP4 rec. is the observed recovery '
        '(fraction of targets with at least one molecule reaching ECFP4-TS $\\geq\\tau$, '
        'averaged over $\\tau \\in [0.25,0.50]$). Panel~B gives the seven normalised scores '
        'actually plotted: on the six property axes $s=0.5$ is the mid-ring, and on the '
        'ECFP4 axis $s=0$ is the shuffled-label baseline and $s=1$ the best method. '
        'NA2smol appears under both Fig.~2d and Fig.~3a because the full model is the same '
        'output set in both.'
    )
    parts = [
        '\\section*{Supplementary Table 12 | Normalised radar scores behind the radar panels of Figs.~2 and 3}',
        notes, '', '\\bigskip', '',
        _grouped_table(df, SCORE_COLS, 'Normalised radar scores $s_m$'),
    ]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text('\n'.join(parts) + '\n')
    print(f'wrote {OUT_TEX.relative_to(ROOT)}')


# ── main-text table: ranks within each benchmark group ------------------------
# direction per column: True = lower is better
LOWER_IS_BETTER = {
    'median_docking': True, 'median_rmsd': True, 'median_qed': False,
    'median_sa': True, 'median_sei': False, 'median_strain': True,
    'recovery_observed': False,
}
MAIN_COLS = [
    ('median_docking', 'Dock.', '%.2f'), ('median_rmsd', 'RMSD', '%.2f'),
    ('median_qed', 'QED', '%.3f'), ('median_sa', 'SA', '%.2f'),
    ('median_sei', 'SEI', '%.3f'), ('median_strain', 'Strain', _fmt_strain),
    ('recovery_observed', 'ECFP4 rec.', '%.3f'),
]
MAIN_GROUPS = [('Fig. 2a', 'DUD-E protein'), ('Fig. 2d', 'RNA test set'),
               ('Fig. 3a', 'RNA test set, four configurations')]


def write_main_table(df: pd.DataFrame) -> None:
    """Best value in each benchmark group in bold, runner-up underlined."""
    out = ['\\begin{table}[!htbp]', '\\centering', '\\footnotesize',
           '\\setlength{\\tabcolsep}{4pt}',
           f'\\begin{{tabular}}{{@{{}}l{chr(99) * len(MAIN_COLS)}@{{}}}}', '\\toprule',
           '\\textbf{Method} & ' + ' & '.join(f'\\textbf{{{l}}}' for _, l, _ in MAIN_COLS) + ' \\\\',
           '\\midrule']
    for gi, (panel, group_label) in enumerate(MAIN_GROUPS):
        sub = df[df['panel'] == panel].reset_index(drop=True)
        if sub.empty:
            continue
        # rank each column inside the group; ties share the best emphasis
        marks = {}
        for key, _, _ in MAIN_COLS:
            vals = sub[key].astype(float).to_numpy()
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            ordered = np.sort(np.unique(vals[finite]))
            ranked = ordered if LOWER_IS_BETTER[key] else ordered[::-1]
            for rank, v in enumerate(ranked[:2]):
                for i, x in enumerate(vals):
                    if np.isfinite(x) and x == v:
                        marks.setdefault((i, key), rank)
        if gi:
            out.append('\\midrule')
        out.append(f'\\multicolumn{{{1 + len(MAIN_COLS)}}}{{@{{}}l}}{{\\textit{{{panel} \u2014 {group_label}}}}} \\\\')
        for i, r in sub.iterrows():
            cells = []
            for key, _, fmt in MAIN_COLS:
                v = r[key]
                txt = fmt(v) if callable(fmt) else ('--' if v != v else fmt % v)
                if txt.startswith('-'):      # math minus, not a text hyphen
                    txt = f'${txt}$'
                rank = marks.get((i, key))
                if rank == 0:
                    txt = f'\\textbf{{{txt}}}'
                elif rank == 1:
                    txt = f'\\underline{{{txt}}}'
                cells.append(txt)
            out.append(f"{r['method']} & " + ' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}',
            '\\caption{Radar-metric values for the benchmark comparison (Fig.~2a,d) and the '
            'configuration comparison (Fig.~3a). Dock., smina docking score (kcal mol$^{-1}$); '
            'RMSD (\\AA) after pose relaxation; QED; SA, synthetic accessibility score; SEI, '
            'surface efficiency index ($-$docking score / TPSA); Strain, conformational strain '
            'energy (kcal mol$^{-1}$, scientific notation above 10$^3$); ECFP4 rec., the observed '
            'recovery on the ECFP4-Tanimoto axis. Values are pooled medians over all of a '
            "method's molecules. \\textbf{Bold} marks the best value in each benchmark group and "
            '\\underline{underline} the second best; better is lower for Dock., RMSD, SA and '
            'Strain and higher for QED, SEI and ECFP4 rec. These are the single-number summaries '
            'of the distributions whose normalised retention-curve integrals are drawn in the '
            'radars; the normalised scores themselves are in Supplementary Table~12.}',
            '\\label{radar_values_tab}', '\\end{table}']
    path = ROOT / 'reproduction' / 'manuscript' / 'table_radar_values.tex'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(out) + '\n')
    print(f'wrote {path.relative_to(ROOT)}')


if __name__ == '__main__':
    raise SystemExit(main())

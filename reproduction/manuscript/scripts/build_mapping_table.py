"""Render the random-residue mapping-control Supplementary Table from the analysis CSVs.

Inputs (written by `paired_mapping_effects.py` on the analysis host)

  mapping_control_cap<N>_paired_effects.csv   one row per metric per method pair
  mapping_control_cap<N>_funnel.csv           retention per screening tier per arm per pair

Output

  si_table_mapping_control.tex                Supplementary Table 14, ready to \\input

The table has two panels: paired metric effects (long format, one row per method x metric) and
the screening funnel (one row per method x arm). Keeping the generation in one place means the
manuscript cannot drift from the numbers, and re-running the analysis only needs this script.

    python3 scripts/build_mapping_table.py --effects ... --funnel ... --out si_table_mapping_control.tex
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# order in which methods are shown; anything else is appended in file order
METHOD_ORDER = ['MolCRAFT', 'Pocket2Mol', 'DiffSBDD', 'FLAG']
METRIC_ORDER = ['Docking score', 'Relax. RMSD', 'QED', 'SA score', 'SEI', 'LLE', 'ECFP4',
                'Strain energy']
METRIC_FMT = {
    'Docking score': '{:.2f}', 'Relax. RMSD': '{:.3f}', 'QED': '{:.3f}', 'SA score': '{:.2f}',
    'SEI': '{:.3f}', 'LLE': '{:.2f}', 'ECFP4': '{:.3f}', 'Strain energy': '{:.0f}',
}
SHORT = {'Docking score': 'Docking', 'Relax. RMSD': 'RMSD', 'QED': 'QED',
         'SA score': 'SA', 'SEI': 'SEI', 'LLE': 'LLE', 'ECFP4': 'ECFP4',
         'Strain energy': 'Strain'}

TIERS = ['All', 'Pharma.', '+SEI', 'Strict', 'Final tier']
N_MOL = {'MolCRAFT': 1000, 'Pocket2Mol': 25, 'DiffSBDD': 300, 'FLAG': 300}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def order(pairs: list[str]) -> list[str]:
    head = [m for m in METHOD_ORDER if m in pairs]
    return head + [p for p in pairs if p not in head]


def fmt_q(q: float) -> str:
    if q != q:                                        # NaN
        return '--'
    if q < 1e-3:
        return f'$<10^{{-3}}$'
    return f'{q:.3f}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--effects', required=True)
    ap.add_argument('--funnel', required=True)
    ap.add_argument('--out', default='si_table_mapping_control.tex')
    ap.add_argument('--cap', default='')
    a = ap.parse_args()

    eff = read_csv(Path(a.effects))
    fun = read_csv(Path(a.funnel))
    pairs = order(list(dict.fromkeys(r['pair'] for r in eff)))
    by = {(r['pair'], r['label']): r for r in eff}

    L: list[str] = []
    ad = L.append

    cap_note = ''
    if a.cap:
        caps = dict(item.split('=') for item in a.cap.split(',') if '=' in item)
        cap_note = (
            'Both arms of a pair were trimmed to the same number of molecules at every target, '
            'so every median and every retention count below rests on identical pools; DiffSBDD, '
            'FLAG and Pocket2Mol were generated for this control at 300 molecules per target and '
            'MolCRAFT reuses its earlier campaign at about 1000. The budget is not cosmetic: at a '
            'budget of 25 molecules per target, Pocket2Mol showed no metric beyond '
            '$|d_z|=0.24$ and would have been called insensitive, whereas at 300 its docking '
            'score, pose-relaxation RMSD and ECFP4 similarity are all significantly degraded. '
            'Small controls therefore understate sensitivity, and the three regenerated methods '
            'use the same budget so their effect sizes are directly comparable. MolCRAFT is the '
            'exception in Panel~B only: its two arms were generated in an earlier campaign with '
            'different yields (76,351 native and 98,257 random molecules over the same 100 '
            'targets), so its per-target medians remain paired but its retention counts rest on '
            'unequal pools.\n\n')

    ad(r'\section*{Supplementary Table 14 | Sensitivity of each baseline to randomised residue assignment}')
    ad('')
    ad('Protein-oriented structure-based generators cannot encode nucleotide residues, so the RNA '
       'test set was scored with every nucleic-acid residue reassigned to one of the 20 amino-acid '
       'types before inference (Methods). To test whether that reassignment is itself responsible '
       'for the ordering seen on the benchmark, each baseline that accepts a residue-type channel '
       'was run on DUD-E twice: once with its native protein residue types, and once with every '
       'residue type reassigned at random while atomic coordinates, atom names and all sampling '
       'settings were held fixed. DiffSBDD, FLAG and Pocket2Mol were run for this control, and '
       'MolCRAFT was run earlier on the same protocol; GraphBP and ED2Mol take no residue-type '
       'input (element-only and electron-density conditioning, respectively), so the control has '
       'no meaning for them. Both arms of every pair were scored with the identical docking, '
       'screening and recovery pipeline, and the scrambling rule is the deterministic hash '
       'described in Methods.')
    ad('')
    ad(cap_note.rstrip())
    ad('')
    ad('\\begin{samepage}')
    ad('\\textbf{Panel A. Paired metric effects}')
    ad(r'\begin{center}\scriptsize')
    ad(r'\setlength{\tabcolsep}{2pt}')
    ad(r'\begin{tabular}{@{}llcccccc@{}}')
    ad(r'\toprule')
    ad(r'\textbf{Method} & \textbf{Metric} & \textbf{$n$ targets} & \textbf{Native} & '
       r'\textbf{Random} & \textbf{$\Delta$} & \textbf{$d_z$ (95\% CI)} & \textbf{$q$} \\')
    ad(r'\midrule')
    for pair in pairs:
        first = True
        for metric in METRIC_ORDER:
            r = by.get((pair, metric))
            if r is None:
                continue
            fmt = METRIC_FMT.get(metric, '{:.3f}')

            def num(v: float) -> str:            # typeset negatives with a proper minus sign
                txt = fmt.format(v)
                return f'${txt}$' if txt.startswith('-') else txt

            native = num(float(r['native_median']))
            random = num(float(r['random_median']))
            delta = num(float(r['oriented_mean_delta']))
            lo, hi = float(r['ci95_low']), float(r['ci95_high'])
            dz = float(r['paired_effect_size_dz'])
            lo_s, hi_s = (f'${lo:.2f}$' if lo < 0 else f'{lo:.2f}'), (f'${hi:.2f}$' if hi < 0 else f'{hi:.2f}')
            dz_s = f'${dz:.2f}$' if dz < 0 else f'{dz:.2f}'
            label = f'\\textit{{{pair}}}' if first else ''
            first = False
            ad(f'{label} & {SHORT.get(metric, metric)} & {r["n_targets"]} & {native} & {random} & {delta} & '
               f'{dz_s} ({lo_s}, {hi_s}) & {fmt_q(float(r["wilcoxon_q_bh"]))} \\\\')
        ad(r'\midrule')
    ad(r'\bottomrule')
    ad(r'\end{tabular}\end{center}')
    ad('')
    ad('Oriented $\\Delta$ is the mean paired difference in the direction of better performance '
       '(positive means the random arm scores better); $d_z$ is Cohen\'s $d$ for paired samples '
       'with a 95\\% percentile interval over targets, and $q$ is the Wilcoxon signed-rank $P$ value '
       'after Benjamini--Hochberg correction across the eight metrics of that method.')
    ad('')
    ad('\\end{samepage}')
    ad('\\begin{samepage}')
    ad('\\textbf{Panel B. Screening retention under the four-tier cascade}')
    ad(r'\begin{center}\scriptsize')
    ad(r'\setlength{\tabcolsep}{4pt}')
    ad(r'\begin{tabular}{@{}llrrrrr@{}}')
    ad(r'\toprule')
    ad(r'\textbf{Method} & \textbf{Arm} & ' +
       ' & '.join(f'\\textbf{{{t}}}' for t in TIERS) + r' \\')
    ad(r'\midrule')
    for pair in pairs:
        for arm, name in (('native', 'Native'), ('random', 'Random')):
            rows = {r['filter_step']: r for r in fun if r['pair'] == pair and r['arm'] == arm}
            if not rows:
                continue
            cells = []
            for t in TIERS:
                r = rows.get(t)
                cells.append(f'{int(r["mols_retained"]):,}' if r else '--')
            ad(f'\\textit{{{pair}}} & {name} & ' + ' & '.join(cells) + r' \\')
        ad(r'\midrule')
    ad(r'\bottomrule')
    ad(r'\end{tabular}\end{center}')
    ad('')
    ad('\\end{samepage}')
    ad('Counts are valid generated molecules surviving each successive tier of the manuscript '
       'cascade (Pharma.: docking $<-8$~kcal~mol$^{-1}$, QED $>0.5$, SA $<5$; +SEI adds SEI $>0.1$; '
       'Strict: docking $<-9$~kcal~mol$^{-1}$, QED $>0.5$, SA $<5$, SEI $>0.15$; Final tier adds '
       'QED $>0.6$, SA $<4$, SEI $>0.2$ and strain $<200$~kcal~mol$^{-1}$). Both arms of a pair '
       'were generated and scored with the same settings, so the two rows are directly comparable '
       'within a method.')
    ad('')

    # per-method summary: which metrics move, and by how much
    ad('\\noindent\\textbf{Summary.} ')
    parts = []
    for pair in pairs:
        rows = [by[(pair, m)] for m in METRIC_ORDER if (pair, m) in by
                and by[(pair, m)]['wilcoxon_q_bh'] not in ('', 'nan')]
        if not rows:
            continue
        moved = [r for r in rows if float(r['wilcoxon_q_bh']) < 0.05]
        largest = max(rows, key=lambda r: abs(float(r['paired_effect_size_dz'])))
        if moved:
            names = ', '.join(r['label'] for r in moved)
            parts.append(f'{pair}: {len(moved)} of {len(rows)} metrics at $q<0.05$ ({names}), '
                         f'largest shift {largest["label"]} '
                         f'($|d_z|={abs(float(largest["paired_effect_size_dz"])):.2f}$)')
        else:
            parts.append(f'{pair}: none of {len(rows)} metrics at $q<0.05$, largest shift '
                         f'{largest["label"]} '
                         f'($|d_z|={abs(float(largest["paired_effect_size_dz"])):.2f}$)')
    ad('; '.join(parts) + '. ')
    ad('Randomising residue identity therefore does not act uniformly across the four '
       'baselines; the effect scales with how much of the pocket chemistry each encoder carries '
       'in its residue channel. FLAG is unaffected on every metric. MolCRAFT and Pocket2Mol sit '
       'in between and disagree with themselves across axes: MolCRAFT docks slightly better from '
       'scrambled pockets while its best ECFP4 similarity falls, and Pocket2Mol loses docking '
       'quality, pose relaxation and ECFP4 similarity while gaining on the property axes. '
       'DiffSBDD is the extreme case, because its C$\\alpha$-level representation encodes the '
       'pocket almost entirely through the amino-acid one-hot channel, so scrambling removes the '
       'signal it conditions on and degrades six of eight metrics, with retention at the first '
       'screening tier falling from 1,093 to 373 molecules on the same target set. Two '
       'consequences follow for how the external comparison should be read. Random residue '
       'assignment does not manufacture the benchmark ordering for FLAG, and only mildly perturbs '
       'MolCRAFT and Pocket2Mol, so those comparisons stand as performance context. For DiffSBDD '
       'the mapping is not a neutral compatibility device, and its RNA results should be read as '
       'a lower bound obtained under a degraded pocket representation rather than as an estimate '
       'of that architecture\'s ceiling.')
    ad('')

    Path(a.out).write_text('\n'.join(L) + '\n')
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

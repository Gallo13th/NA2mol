"""Regenerate repeated summary cells from the authoritative manuscript CSV."""
from pathlib import Path
import re
import pandas as pd
from make_supp_recovery_table import rounded, fmt, NAME, main

ROOT = Path(__file__).resolve().parents[2]

def sync():
    main()
    source = pd.read_csv(ROOT / 'audit/tables/recovery_manuscript_authoritative.csv')
    ladder = pd.read_csv(ROOT / 'audit/tables/recovery_ablation_ladder.csv')
    paper = ROOT / 'reproduction'
    summary = paper / 'si_ed_table5.tex'
    text = summary.read_text()
    a, b = text.split(r'\textbf{Panel B. SMRT-Net sequential ablation configurations}')
    for bench in ['SMRT-Net', 'DUD-E']:
        for method in ['RNA2smol', 'MolCRAFT', 'Pocket2Mol', 'ED2Mol']:
            data = source[(source.panel == 'external') & (source.benchmark == bench) & (source.method == method)].set_index('filter')
            count = 27 if bench == 'SMRT-Net' else int(pd.read_csv(ROOT / 'audit/tables/recovery_target_coverage.csv').set_index('benchmark').loc['DUD-E', 'common_full'])
            name = NAME[method]
            cells = [bench, name, f'{rounded(data.iloc[0].A_original_full)} ({count})']
            for fil in ['ft_exact', 'ft_T0.5', 'pretrain_exact', 'pretrain_T0.5']:
                r = data.loc[fil]
                cells.append('$0.000$' if rounded(r.delta_filter_effect) == '0.000' else fmt(r.delta_filter_effect, r.delta_ci_lo, r.delta_ci_hi))
            replacement = ' & '.join(cells) + ' ' + chr(92)*2
            pattern = '^' + re.escape(bench + ' & ' + name) + r'\s+&.*$'
            a, n = re.subn(pattern, lambda m: replacement, a, flags=re.M)
            assert n == 1, (bench, method, n)
    manuscript = paper / 'sn-article.tex'
    maintext = manuscript.read_text()
    for method, name in [('RNA2smol', 'Full NA2smol'), ('ablation:Pretrain+Constrain', 'Pretrain+Constrain'), ('ablation:RNA2smol (Baseline)', 'no-pretraining baseline'), ('ablation:Pretrain Only', 'Pretrain Only')]:
        data = source[(source.panel == 'ablation') & (source.method == method)].set_index('filter')
        original = ladder[(ladder.config == method) & (ladder['filter'] == 'original')].iloc[0].A_original_subset
        cells = [name, rounded(original)]
        for fil in ['ft_exact', 'ft_T0.5', 'pretrain_exact', 'pretrain_T0.5']:
            r = data.loc[fil]
            cells.append('$0.000$' if rounded(r.delta_filter_effect) == '0.000' else fmt(r.delta_filter_effect, r.delta_ci_lo, r.delta_ci_hi))
        replacement = ' & '.join(cells) + ' ' + chr(92)*2
        b, n = re.subn('^'+re.escape(name)+r'\s+&.*$', lambda m: replacement, b, flags=re.M)
        assert n == 1
        mainname = r'\textbf{Full NA2smol}' if method == 'RNA2smol' else ('No-pretraining baseline' if 'Baseline' in method else name)
        value = r'\textbf{' + rounded(original) + '}' if method == 'RNA2smol' else rounded(original)
        cells = [mainname, value]
        for fil in ['ft_T0.5', 'pretrain_T0.5']:
            r = data.loc[fil]
            cells.append(fmt(r.delta_filter_effect, r.delta_ci_lo, r.delta_ci_hi))
        replacement = ' & '.join(cells) + ' ' + chr(92)*2
        maintext, n = re.subn('^'+re.escape(mainname)+r' &.*$', lambda m: replacement, maintext, flags=re.M)
        assert n == 1
    summary.write_text(a + r'\textbf{Panel B. SMRT-Net sequential ablation configurations}' + b)
    manuscript.write_text(maintext)
    print('Synchronised Table 1, Extended Data Table 5 and Supplementary Table 2.')

if __name__ == '__main__':
    sync()

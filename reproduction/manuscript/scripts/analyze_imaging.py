#!/usr/bin/env python3
"""Cell-imaging quantification from the collaborator's Prism table (93 cells x 7 conditions)."""
import numpy as np
from scipy import stats

vals = np.load('/tmp/clivia_imaging.npy')
titles = ['Mock', 'Al-1 1 µM', 'Al-1 10 µM', 'Al-2 1 µM', 'Al-2 10 µM', 'Al-3 1 µM', 'Al-3 10 µM']
print(f'cells per condition: {vals.shape[0]}\n')
print(f'{"condition":12s} {"n":>4s} {"mean":>8s} {"SD":>7s} {"median":>8s} {"IQR":>14s} {"% of Mock":>10s}')
mock = vals[:, 0]
for j, t in enumerate(titles):
    c = vals[:, j]
    q1, q3 = np.percentile(c, [25, 75])
    print(f'{t:12s} {len(c):4d} {c.mean():8.1f} {c.std(ddof=1):7.1f} {np.median(c):8.1f} '
          f'{q1:6.1f}-{q3:6.1f} {100*c.mean()/mock.mean():9.1f}%')

# normality and one-way ANOVA + multiple comparisons vs Mock
print('\nShapiro-Wilk per group:')
for j, t in enumerate(titles):
    w, p = stats.shapiro(vals[:, j])
    print(f'  {t:12s} W={w:.3f} p={p:.2e}')
F, p = stats.f_oneway(*[vals[:, j] for j in range(7)])
print(f'\nOne-way ANOVA: F={F:.1f}, p={p:.2e}')

# Kruskal-Wallis as non-parametric check
H, pk = stats.kruskal(*[vals[:, j] for j in range(7)])
print(f'Kruskal-Wallis: H={H:.1f}, p={pk:.2e}')

# Dunnett-style: each treatment vs Mock (Welch t, Holm corrected)
print('\nvs Mock (Welch t-test, Holm-corrected):')
praw = []
for j in range(1, 7):
    t, pv = stats.ttest_ind(vals[:, j], mock, equal_var=False)
    praw.append(pv)
    print(f'  {titles[j]:12s} t={t:7.2f}  p_raw={pv:.2e}')
order = np.argsort(praw)
holm = np.empty(6)
m = 6
prev = 0
for rank, idx in enumerate(order):
    adj = min(1.0, (m - rank) * praw[idx])
    adj = max(adj, prev)
    holm[idx] = adj
    prev = adj
print('\nHolm-adjusted:')
for j in range(6):
    print(f'  {titles[j+1]:12s} p_adj={holm[j]:.2e}')

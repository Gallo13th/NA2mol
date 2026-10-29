# 1FMN case study -- top-20 combined binders

Combined = 0.5 * minmax(logit) + 0.5 * minmax(-rdock_SCORE), computed over the 300 docked ligands.
FMN reference: P=1.0000, rdock SCORE=-17.157, INTER=-16.787. FMN rdock-SCORE-rank among 300 docked ~ 297 (lower=better).

| rank | gen_idx | SMILES | natoms | rnasmol P | rnasmol logit | rdock SCORE | rdock INTER | combined | rdock rank |
|------|---------|--------|--------|-----------|---------------|-------------|-------------|----------|------------|
| 1 | 221 | `COCCCCC1=CC=C([C@H](CS)C(=O)N[C@@H]2CC[C@@H](N)N2)C=CC1` | 25 | 1.0000 | +17.63 | -59.41 | -18.12 | 0.595 | 3 |
| 2 | 1897 | `CC1C=CC(N2CCC(OC(=O)O)CC2)=C2C=C(NC(=O)CCCN)N=C21` | 27 | 1.0000 | +14.21 | -64.95 | -19.12 | 0.584 | 1 |
| 3 | 1627 | `NC[C@@H](O)[C@H]1O[C@@H]([C@H]2O[C@@H](O)[C@H](N)[C@@H](O)CC2=O)[C@H](O)[C@@H](O)[C@@H]1O` | 24 | 1.0000 | +35.09 | -22.51 | -19.51 | 0.562 | 189 |
| 4 | 3498 | `NC1=CC2=NC3=CC(OC[C@H](O)C(=O)O)C(O)=C(O)C(O)=C3C=CC2=CC=C1` | 28 | 1.0000 | +29.80 | -28.93 | -23.42 | 0.523 | 52 |
| 5 | 4053 | `NC(=O)CCNC(=O)C1=NC(C/C=C/CO)=CC=C(O)C1N` | 22 | 1.0000 | +30.07 | -26.30 | -17.09 | 0.501 | 81 |
| 6 | 70 | `CSCCC1=NC(N2CCN(C(=O)O)CC2)=C(O)C2C(C)=CC=C12` | 24 | 1.0000 | +10.79 | -61.53 | -16.52 | 0.481 | 2 |
| 7 | 3189 | `[H]/N=C(/C)CCCOC1=CC=C(C(=O)NCC[C@@H](O)C=O)C(N)C=C1C` | 26 | 1.0000 | +33.50 | -17.54 | -16.65 | 0.479 | 294 |
| 8 | 3112 | `CC(NCc1ccc(N)cc1)C(C)C1=C(S)CN(CCCCC(=O)O)CC1` | 27 | 1.0000 | +15.49 | -50.40 | -11.75 | 0.460 | 5 |
| 9 | 1596 | `CC(C1=CCC=C1)C1=CC(C(=O)CC[C@H](N)C(=O)O)=CC(O)=C(O)C1` | 25 | 1.0000 | +22.15 | -35.14 | -15.92 | 0.434 | 14 |
| 10 | 467 | `[H]/N=C(\N)c1ccc(NC(=O)C2=C3C=C(C)C(O)C=C3N=C2)cc1` | 24 | 1.0000 | +26.50 | -26.44 | -22.25 | 0.431 | 76 |
| 11 | 632 | `CCN(C[C@@H]1CCN(C)C(=O)C1)C1=C(O)C=CC2C=C(N(C)C)C=C12` | 25 | 1.0000 | +10.05 | -57.58 | -16.19 | 0.425 | 4 |
| 12 | 2681 | `Cc1cc([C@H]2C[C@H](N)[C@H](O)[C@H](O)O2)nc(C2=NC(C)N=N2)c1C` | 23 | 1.0000 | +27.77 | -22.87 | -23.49 | 0.420 | 176 |
| 13 | 1628 | `NCCNC(=O)C1=CC2=C3C=CC=C3C(O)=C(O)C2C=C1` | 21 | 1.0000 | +27.28 | -22.33 | -21.63 | 0.404 | 199 |
| 14 | 1414 | `CC(N)c1ccc(-c2nnc(N)nc2-c2ccc(O)cc2)cc1` | 23 | 1.0000 | +27.36 | -21.27 | -20.77 | 0.395 | 232 |
| 15 | 2249 | `Cc1cc2nc(O)c(O)cc2cc1[C@H](S)NNC1=CC2C(N)=CC=CC2=N1` | 27 | 1.0000 | +12.67 | -48.40 | -18.46 | 0.383 | 6 |
| 16 | 1729 | `NC(O)C1=CC=c2nc3c(nc2C1)=C(OCc1ccc(O)cc1)C=CC=C3` | 27 | 1.0000 | +21.84 | -30.30 | -21.13 | 0.378 | 35 |
| 17 | 1221 | `[H]/N=C(/O)C1=CC=C2C(=NC3C(c4ccc(N)cc4)=CC=CC3=C2O)C=C1` | 27 | 1.0000 | +22.27 | -29.04 | -28.63 | 0.374 | 49 |
| 18 | 2340 | `COC1=CC2C(C)=CC(OC)=C(/C=C/C[C@H](N)C(=O)N(C)C)C2=C1` | 24 | 1.0000 | +21.79 | -29.32 | -19.02 | 0.367 | 45 |
| 19 | 2598 | `OO[C@@H]1CC2OCC(C3=C(O)N=C4C=CC=CC=C4N3)=CC2CC[C@H]1O` | 26 | 1.0000 | +19.40 | -33.64 | -23.50 | 0.364 | 18 |
| 20 | 759 | `CC1=NC2N=C(N3CCN(C(=O)O)CC3)C=C(O)C(O)=C2C(O)=N1` | 24 | 1.0000 | +10.90 | -48.02 | -21.10 | 0.343 | 7 |

## Top-100 SDF
`/home/data/users/gaoletian/rna2smol/casestudy/fmn/screening_round2/top_binders.sdf`

# 1FMN case study -- top-20 combined binders

Combined = 0.5 * minmax(logit) + 0.5 * minmax(-rdock_SCORE), computed over the 300 docked ligands.
FMN reference: P=1.0000, rdock SCORE=-17.157, INTER=-16.787. FMN rdock-SCORE-rank among 300 docked ~ 486 (lower=better).

| rank | gen_idx | SMILES | natoms | rnasmol P | rnasmol logit | rdock SCORE | rdock INTER | combined | rdock rank |
|------|---------|--------|--------|-----------|---------------|-------------|-------------|----------|------------|
| 1 | 1128 | `[H]/N=C(\O)CCCC1C=CC2=CC=CC(/N=C3\C=CC(CN)=C3)=CC2=C1` | 26 | 1.0000 | +33.50 | -27.39 | -20.76 | 0.610 | 106 |
| 2 | 8127 | `C[C@@H](CC/C=N/C(=O)N[C@@H]1C[C@H](O)[C@@H](O)[C@H](O)[C@H]1O)N[C@H](S)C1=N[C@@H](N)N=N1` | 28 | 1.0000 | +15.35 | -71.57 | -13.85 | 0.594 | 1 |
| 3 | 2454 | `NC1N=C(O)N=c2c(OCCO[C@H](O)CO)c(O)ccc2=N1` | 22 | 1.0000 | +30.16 | -27.19 | -16.11 | 0.534 | 113 |
| 4 | 1581 | `[H]/N=C1\C=CC(C(N)NNC2C[C@H](O)[C@@H](O)[C@H](O)C2=O)=C1` | 21 | 1.0000 | +32.60 | -18.33 | -17.52 | 0.510 | 472 |
| 5 | 1259 | `CC1C=CC=C(/C=C/C=C/CN)C2=C1O[C@H]1COCC[C@H]21` | 21 | 1.0000 | +30.91 | -22.06 | -21.16 | 0.505 | 344 |
| 6 | 390 | `CC1=CC(N2C[C@@H]3OCO[C@@H]3C2)=C2C=C(C(=O)O)C(O)=C2C1C` | 23 | 1.0000 | +14.63 | -62.79 | -23.88 | 0.500 | 2 |
| 7 | 3158 | `[H]/N=C/C(=N\C1C[C@H](C(=O)O)[C@@H](N)C1=O)c1ccc(N)cn1` | 22 | 1.0000 | +28.47 | -25.37 | -19.81 | 0.480 | 178 |
| 8 | 1382 | `CC1=NC(c2ccc(N)cc2)N=C2C=C(N[C@@H]3CC[C@@H](O)OC3)C=CC(O)=C12` | 28 | 1.0000 | +20.00 | -45.51 | -22.05 | 0.468 | 12 |
| 9 | 5394 | `NC(=O)[C@H](N)C/C=C/CC1C=C(CCCCC=O)C2=CC=NC2=C1C(=O)O` | 27 | 1.0000 | +29.43 | -21.44 | -15.88 | 0.467 | 368 |
| 10 | 7078 | `N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CNc2cccc3c2=CC(c2cccc(O)c2)C=3)O[C@H]1O` | 28 | 1.0000 | +28.32 | -22.89 | -25.32 | 0.455 | 300 |
| 11 | 9491 | `Cc1c2ccc3ccccc3cc-2cc1C(C)S/N=C1\CC[C@@H](N)[C@H](O)C1` | 27 | 1.0000 | +23.53 | -34.12 | -28.06 | 0.447 | 44 |
| 12 | 4607 | `N[C@H]1[C@H](Oc2ccc(C3=NC4C=CC=CC4=N3)cc2)OC[C@@H](O)[C@@H]1O` | 25 | 1.0000 | +28.37 | -21.06 | -21.08 | 0.440 | 389 |
| 13 | 3292 | `[H]/N=C(/N)c1ccc(C(OCCCCCCCCOC=O)C(=O)C(=O)CO)cc1` | 29 | 1.0000 | +29.44 | -17.80 | -5.34 | 0.435 | 479 |
| 14 | 7053 | `CN(C1=C(S)N=C2C=C(F)C=CC21)[C@@H]1C[C@H](O)[C@@H](O)CO1` | 21 | 1.0000 | +13.15 | -58.50 | -18.97 | 0.429 | 3 |
| 15 | 5074 | `CC1=CC(OC[C@H]2CC(O)(O)[C@](C)(N)CO2)=NC2C=C(O)C=C(C)C2=C1` | 26 | 1.0000 | +25.89 | -25.81 | -22.09 | 0.426 | 164 |
| 16 | 814 | `C[C@@H](/N=N\C(S)SCCCC(N)=O)c1ccc(C(N)=O)cc1` | 22 | 1.0000 | +13.94 | -55.75 | -12.36 | 0.423 | 4 |
| 17 | 5570 | `COCc1ccc2c(c1)C(/C=C/c1ccc(C=O)cc1)=NC(N)N=C2O` | 26 | 1.0000 | +24.93 | -27.53 | -18.84 | 0.420 | 103 |
| 18 | 7333 | `CC1C=C(N(C)C)C=C2C=C(N[C@H]3O[C@H](O)[C@H](O)[C@@H](O)[C@@H]3N)C=C21` | 24 | 1.0000 | +27.43 | -20.71 | -19.90 | 0.415 | 408 |
| 19 | 1271 | `CCOc1cc(O)c(O)c(C(=O)C(C(=O)O)[C@@H](O)CCCOCCN)c1` | 26 | 1.0000 | +28.62 | -15.96 | -5.46 | 0.400 | 494 |
| 20 | 7561 | `CC1=NC2N=C(C3=C(O)C4=C(C3)[C@@H](N)[C@H](O)[C@@H](O)C4=O)C(O)=NC2=N1` | 25 | 1.0000 | +26.04 | -22.04 | -22.60 | 0.396 | 345 |

## Top-100 SDF
`/home/data/users/gaoletian/rna2smol/casestudy/fmn/screening_round3/top_binders.sdf`

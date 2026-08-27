# 1FMN case study -- top-20 combined binders

Combined = 0.5 * minmax(logit) + 0.5 * minmax(-rdock_SCORE), computed over the 300 docked ligands.
FMN reference: P=1.0000, rdock SCORE=-17.157, INTER=-16.787. FMN rdock-SCORE-rank among 300 docked ~ 296 (lower=better).

| rank | gen_idx | SMILES | natoms | rnasmol P | rnasmol logit | rdock SCORE | rdock INTER | combined | rdock rank |
|------|---------|--------|--------|-----------|---------------|-------------|-------------|----------|------------|
| 1 | 3035 | `COc1ccc2c(C=O)c(C(S)[C@@H]3CC[C@@H](O[P+](O)(O)O)[C@H](N)C3)ncc2c1` | 28 | 1.0000 | +30.78 | -52.08 | -20.93 | 0.685 | 4 |
| 2 | 4401 | `OC1=NC2C=C[N+](c3ccccc3)=C2C(O)=[N+]1/C=C/CO[P+](O)(O)O` | 25 | 1.0000 | +19.82 | -72.84 | -21.03 | 0.650 | 1 |
| 3 | 821 | `CC1=NC2=C(C(=O)CCCO[P+](O)(O)O)C(C)N=C(C)C2=N1` | 22 | 1.0000 | +37.85 | -26.98 | -18.40 | 0.608 | 82 |
| 4 | 1512 | `COCC1=NC2C(NC[P+](O)(O)O)=CC(C)=C(N=O)C2=N1` | 21 | 1.0000 | +35.47 | -31.18 | -17.96 | 0.597 | 34 |
| 5 | 671 | `COC1=C(C)C2=CC=[O+]C2C(OCCO[P+](O)(O)O)=C1` | 20 | 1.0000 | +34.23 | -32.33 | -16.14 | 0.583 | 24 |
| 6 | 2704 | `C=[O+]c1c(O)c(-c2ccccc2)nc2c1=[N+](C)C(C)[N+]=2/C=C/C=[O+]/[P+](O)(O)O` | 28 | 1.0000 | +16.49 | -72.06 | -23.41 | 0.578 | 2 |
| 7 | 1682 | `CC(=O)c1nc(CO[P+](O)(O)O)n(C2=CC3=C4CC(C)NC4=CC3=C2)c1N` | 27 | 1.0000 | +35.72 | -22.78 | -21.34 | 0.530 | 214 |
| 8 | 4163 | `[H]/N=C(/N)c1ccc(C2C=c3c(C(=O)NCCCCCCO)cccc3=N2)cc1` | 29 | 1.0000 | +25.73 | -40.58 | -19.03 | 0.489 | 12 |
| 9 | 4326 | `Cc1c(NCCN2CC[C@@H](O)[C@H](O)C2)nc(C(=O)/C(Cl)=N/C(N)Cl)[n+](C)c1O` | 28 | 1.0000 | +14.24 | -65.49 | -15.08 | 0.479 | 3 |
| 10 | 631 | `CC1N=c2cc(CO[P+](O)(O)O)ccc2=[N+]1CNc1ccc(C(N)=O)cc1` | 27 | 1.0000 | +28.73 | -29.49 | -20.76 | 0.452 | 45 |
| 11 | 2811 | `CC1=C2[O+]=CC=[N+]2C(NP2COC[C@@H](O)[C@H](O)O2)=NC1Nc1cccc(N)c1` | 28 | 1.0000 | +22.43 | -43.67 | -24.16 | 0.451 | 8 |
| 12 | 2102 | `C=[O+]C1=C(O)C2C(=NC=[N+]2CCO[P+](O)(O)O)C(O)=N1` | 20 | 1.0000 | +32.05 | -21.06 | -18.29 | 0.444 | 256 |
| 13 | 440 | `C[n+]1c(C2C=C[Cl+2]=C2)nc(CO[P+](O)(O)O)c2c1C1=CC=[Cl+2]C1=N2` | 24 | 1.0000 | +31.79 | -21.58 | -20.59 | 0.444 | 244 |
| 14 | 1078 | `Cc1ccc(C2=CC=CC3N=C(CCO[P+](O)(O)O)N=C23)cc1` | 23 | 1.0000 | +28.99 | -27.29 | -21.87 | 0.438 | 76 |
| 15 | 2565 | `COC1=CC(CCO[P+](O)(O)O)=C2C(=NN=C(O)C2O)C=C1` | 22 | 1.0000 | +30.07 | -24.37 | -17.61 | 0.434 | 154 |
| 16 | 1499 | `Cc1nc2c(c(OCC[P+](O)(O)O)[n+]1C)C=C1C=[Cl+2]C=C1C2C` | 23 | 1.0000 | +28.46 | -27.82 | -22.73 | 0.432 | 65 |
| 17 | 4140 | `C=[O+]C1=[N+](CCO[P+](O)(O)O)C2N=CN=C(N)C2=C1C` | 20 | 1.0000 | +30.48 | -22.69 | -19.77 | 0.428 | 219 |
| 18 | 3888 | `C=[O+]C1=c2ccc(OC)cc2=CC(CCO[P+](O)(O)O)C=C1C` | 23 | 1.0000 | +28.58 | -26.88 | -21.56 | 0.427 | 84 |
| 19 | 446 | `C=[O+][C@@H]1CC[C@@H](NC2=CC(C(=O)c3ccccc3)=NC2N)CCO1` | 24 | 1.0000 | +27.41 | -26.63 | -16.93 | 0.402 | 93 |
| 20 | 1662 | `CC(=O)CC(=O)Nc1c(C/C=C/CC(C)O[P+](O)(O)O)c(C)c(O)c[n+]1C` | 27 | 1.0000 | +29.97 | -20.47 | -14.50 | 0.399 | 265 |

## Top-100 SDF
`/home/data/users/gaoletian/rna2smol/casestudy/fmn/top_binders.sdf`

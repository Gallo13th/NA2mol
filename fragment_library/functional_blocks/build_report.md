# Functional Blocks Build Report

Built by `build_functional_blocks.py` (template matching + sp3 extension).

Data source: `fluor_util_2/FluoDB-Full.csv`. Not using BRICS.


## Donor (D) library

- unique fragments: **2103**
- total occurrences: **30875**

### top-10

| rank | smiles | freq | src_template |
|------|--------|------|--------------|
| 1 | `CNC` | 4440 | tert_amine |
| 2 | `CCNCC` | 2059 | tert_amine |
| 3 | `c1ccc(N(c2ccccc2)c2ccccc2)cc1` | 1249 | triphenylamine |
| 4 | `Cc1ccc(N(c2ccccc2)c2ccccc2)cc1` | 823 | triphenylamine |
| 5 | `CN` | 798 | sec_amine_ar |
| 6 | `C.C` | 632 | pyrrole |
| 7 | `C1CCNCC1` | 527 | tert_amine |
| 8 | `C.CC` | 518 | pyrrole |
| 9 | `CN(C)C` | 457 | tert_amine |
| 10 | `C.CCCOC` | 445 | pyrrole |

## Acceptor (A) library

- unique fragments: **105**
- total occurrences: **23914**

### top-10

| rank | smiles | freq | src_template |
|------|--------|------|--------------|
| 1 | `CC#N` | 6737 | cyano |
| 2 | `C=C(C)C#N` | 4319 | cyano_ethylene |
| 3 | `C=C(C#N)C(=O)O` | 3428 | cyanoacrylic |
| 4 | `C#N` | 2779 | cyano |
| 5 | `O=[N+][O-]` | 2312 | nitro |
| 6 | `CC=C(C)C#N` | 527 | cyano_ethylene |
| 7 | `C=CC#N` | 516 | cyano_ethylene |
| 8 | `CC(C)=C(C)C#N` | 437 | cyano_ethylene |
| 9 | `NC(=S)S` | 375 | thiocarbonyl |
| 10 | `CN1C(=O)CSC1=S` | 346 | rhodanine |

## Bridge (pi) library

- unique fragments: **17**
- total occurrences: **150102**

### top-10

| rank | smiles | freq | src_template |
|------|--------|------|--------------|
| 1 | `c1ccccc1` | 107214 | benzene |
| 2 | `C=Cc1ccccc1` | 23010 | phenylene_vinylene |
| 3 | `c1ccsc1` | 7469 | thiophene |
| 4 | `c1ccc(-c2ccccc2)cc1` | 5533 | biphenyl |
| 5 | `c1ccc2c(c1)Cc1ccccc1-2` | 2242 | fluorene |
| 6 | `C=C` | 1961 | phenylene_vinylene |
| 7 | `C(=Cc1ccccc1)c1ccccc1` | 1657 | stilbene |
| 8 | `C(=C/c1ccccc1)\c1ccccc1` | 830 | stilbene |
| 9 | `Cc1ccccc1` | 96 | fluorene |
| 10 | `C(=C\c1ccccc1)\c1ccccc1` | 35 | stilbene |

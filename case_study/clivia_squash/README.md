# Clivia / Squash — de-novo reproduction test

## Question

Five quinoline-series compounds (mol1–mol5) were found to be fluorescently active on the
Clivia and Squash fluorogenic RNA aptamers. Does **free (unconstrained) generation** from
those aptamer pockets reproduce them?

## The five compounds

Formulas were read from the supplied structure images (`~/Documents/mol1-5`) and verified
with RDKit. mol1 and mol2 match the molecular weights printed on their images exactly
(357.41 and 356.43); mol3–mol5 carry no printed MW, so those values are computed.

| ID | Formula | MW | Structure |
|---|---|---|---|
| mol1 | C23H19NO3 | 357.41 | 2-(4-hydroxyphenyl)-4-(4-methoxyphenyl)-methoxyquinoline |
| mol2 | C23H20N2O2 | 356.43 | same, 4-aminophenyl instead of 4-hydroxyphenyl |
| mol3 | C25H24N2O2 | 384.48 | same, 4-(dimethylamino)phenyl |
| mol4 | C22H18N2O2 | 342.40 | same scaffold, pyridin-4-yl at C2 |
| mol5 | C23H19NO4 | 373.41 | 2-(3-carboxyphenyl)-4-(4-hydroxyphenyl)-methoxy-1,4-dihydroquinoline |

mol1–mol4 are a matched 2,4-diaryl-quinoline series differing only in the para substituent
(OH / NH2 / NMe2) at C2 and an aza replacement (mol4). mol5 is a different scaffold
(1,4-dihydroquinoline, extra COOH).

Exact SMILES: `target_molecules.smi`.

**Caveat on regiochemistry.** The aromatic ring methoxy could not be assigned to C6 vs C7
with certainty from the drawings alone; both are the same formula and MW. Everything below
is insensitive to that choice for formula-level comparison, but a scaffold-level match
should record it.

## Targets

| Aptamer | PDB | Resolution | Bound fluorophore | Pocket |
|---|---|---|---|---|
| Clivia | **8HZE** | 1.59 Å | NI4 | `clivia_8HZE_pocket.pdb` — 15 residues (6 A, 2 U, 3 C, 2 G, 2 Mg), 280 atoms |
| Squash | **7KVU** | 2.68 Å | 2ZY (DFHBI-1T) | `squash_7KVU_pocket.pdb` — 22 residues (7 A, 8 G, 3 U, 1 C, 3 Mg), 421 atoms |

Clivia has seven PDB entries (8HZF/8HZD/8HZE/8HZJ/8HZK/8HZL/8HZM); 8HZE is the highest
resolution. Squash has 7KVV (2.85 Å) and 7KVU (2.68 Å); 7KVU is used.

Pockets follow the BER102 convention exactly: standard nucleotides/ions whose geometric
centre is within **14 Å** of the bound fluorophore, single state.

Note that the crystallised ligands are imidazolone fluorophores (NI4, 2ZY) — chemically
unrelated to the quinoline series. So this is a genuine de-novo test, not a ligand recall.

## Pipeline

```bash
PY=/home/data/users/gaoletian/.conda/envs/rna2smol/bin/python
$PY case_study/clivia_squash/prepare_pockets.py          # PDB -> pocket + dummy ligand
CUDA_VISIBLE_DEVICES=5 $PY -u case_study/clivia_squash/run_generation.py \
    --target clivia --seed 42 --output_prefix clivia_s0 --total_target 1000
$PY case_study/clivia_squash/analyse_generation.py \
    --sdf output/clivia_s0/valid_mols.sdf --label clivia_s0
```

`run_generation.py` mirrors `case_study/kras_g4/run_generation_ber102.py`: released
checkpoint, cold start, `seed_used=False` (no fluorophore guidance), atom count drawn by
the featurizer.

`analyse_generation.py` reports, per generated molecule: best ECFP4 Tanimoto to each of
mol1–mol5, presence of the quinoline and 2,4-diaryl-quinoline scaffolds, and the closest
target.

## Status

Pockets prepared and verified. Generation launched; the model load is very slow on the
shared `/home/data` filesystem (torch + `torch.compile` over a network mount), so the first
pilot needs 20+ minutes before CUDA work begins. No generation results yet.

## Results

Atom-count distribution tightened to **26–29** heavy atoms (matching the series; benchmark
default (20, 28) preserved, overridden via `NA2SMOL_ATOM_RANGE=26,29`). 2,000 target
molecules per pocket, seed 42, cold start, no guidance.

| Run | Molecules | max ECFP4 | ≥0.5 | ≥0.4 | quinoline core | 2,4-diaryl quinoline |
|---|---|---|---|---|---|---|
| clivia, 258 mols, atoms 20–28 | 258 | 0.375 | 0 | 0 | 13 (5.0%) | **0** |
| **clivia, 2042 mols, atoms 26–29** | 2042 | **0.420** | 0 | 2 (0.1%) | 212 (10.4%) | **0** |
| **squash, 2007 mols, atoms 26–29** | 2007 | **0.491** | 0 | 11 (0.6%) | 168 (8.4%) | **0** |

**No compound reproduces the series, and neither pocket produces the 2,4-diaryl-quinoline
scaffold that mol1–mol4 share** — 0 of 4,049 molecules.

What the model does produce is chemically adjacent: methoxy-substituted aryl quinolines in
the right size range. The best Squash hit is
`COc1ccc(-c2cc3cc(C)cc-3c3ccc(C(=O)O)cc3n2)cc1` (26 heavy atoms, MW 343.4, ECFP4 0.49 to
mol1) — the same size as mol4 (26 heavy, MW 342.4). The best Clivia hit is a **2,6**-diaryl
quinoline (27 heavy, MW 355.4, ECFP4 0.42 to mol2), i.e. the correct scaffold class with the
second aryl at the wrong position.

Interpretation: the series sits near, but not inside, the model's unconstrained output
distribution for these two pockets. Reproducing it would need either far larger sampling
(ECFP4 ≥ 0.5 never occurs in ~4,000 draws) or scaffold-constrained sampling. The
regiochemistry difference (2,6 vs 2,4) is the most informative single discrepancy.

Per-run tables: `reports/comparison_{clivia,squash}_run2k.csv`.

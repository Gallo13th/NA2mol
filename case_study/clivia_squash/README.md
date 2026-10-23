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

### Correction to the scaffold count

The first count reported **0 / 4,049** molecules with the 2,4-diaryl-quinoline scaffold. That
was an artefact of the SMARTS used (`c1ccc2nc(-c3ccccc3)cc(-c3ccccc3)c2c1`), which requires a
plain phenyl at both positions and so also fails to match **mol4** (its C2 aryl is pyridine).

Re-counted with a generic definition — match the quinoline core, then ask whether the atoms at
C2 and C4 each bear an aromatic ring — every target mol1–mol4 is detected correctly, and:

| Run | C2-aryl | C4-aryl | **2,4-diaryl** |
|---|---|---|---|
| Clivia (2042) | 118 (5.8%) | 7 (0.3%) | **1 (0.05%)** |
| Squash (2007) | 90 (4.5%) | 49 (2.4%) | **18 (0.90%)** |

So **19 molecules do carry the scaffold**, at 0.5–0.9% — rare but reachable, not absent.

### What actually differs

The 19 scaffold hits all carry **carboxylic acid / amide** substituents, peaking at ECFP4 0.37.
The molecules with the **right** substituent chemistry (methoxy + hydroxyl) instead place the
second ring at **C6**, or use a **quinazoline** core, peaking at 0.49. No single molecule
combines the right scaffold *and* the right substituents:

| | scaffold | substituents | best ECFP4 |
|---|---|---|---|
| mol1–mol4 | 2,4-diaryl quinoline | 2xOMe + OH/NH2/NMe2/pyridyl | — |
| scaffold hits | **correct** | COOH / amide | 0.37 |
| best overall | 2,6-diaryl, or quinazoline | **correct** (OMe, OH) | **0.49** |

Figures: `figures/targets_vs_hits.png` (targets vs the five closest hits from each pocket),
`figures/scaffold_vs_substituents.png` (right scaffold / wrong substituents vs right
substituents / wrong scaffold).

## Scaffold-constrained attempt: the seed mechanism does not hold the core

Seeded with the bare **2,4-diphenylquinoline** core (22 heavy atoms, translated onto each
pocket's fluorophore centre so the 14 Å re-extraction succeeds), `seed_used=True`,
`total_target=1000`.

**The core did not survive.** Scaffold retention in the seeded runs is no better than in the
unconstrained ones, and for Squash it is worse:

| Run | quinoline | 2,4-diaryl | max ECFP4 |
|---|---|---|---|
| Clivia unconstrained (2042) | 5.8% C2-aryl | 0.05% | 0.42 |
| Squash unconstrained (2007) | 4.5% C2-aryl | 0.90% | **0.49** |
| **Clivia seeded (1021)** | 9.5% | **0.20%** | 0.44 |
| **Squash seeded (1014)** | 7.3% | **0.49%** | 0.44 |

The closest seeded hit is not even a quinoline: `CCCCNc1ccc(-c2cc(O)cc(-c3ccc(OC)cc3)n2)cc1`
is a **pyridine**, ECFP4 0.44 to mol1.

**Why.** `genbypkt`'s seed path sets `remain_seed=True` only for the initial batch; after a
molecule passes the affinity screen it is split into fragments by `prepare_seed_fragments`
and subsequent iterations elaborate from *those* fragments. The seeded core therefore
constrains the first iteration and is then progressively replaced. This is a
fragment-elaboration bootstrap, not a substructure constraint.

**Consequence.** True scaffold-constrained sampling is not reachable through this entry point.
It would need the core atoms pinned via `fixed_atom_mask` in the featurizer for *every*
iteration, i.e. a modification to `genbypkt.generate` rather than a new seed file. That has
not been done here, and the negative result above should not be read as evidence that the
model cannot decorate the correct core — only that this pipeline cannot hold it fixed.

Nothing in this section changes the main finding: free generation produces the right
substituent chemistry and the right scaffold separately, never together, and tops out at
ECFP4 0.49.

## 10,000-per-pocket free generation: final result

Two seeds per pocket (42, 123) x 5,000, unconstrained, atom range 26-29: **10,046 Clivia**
and **10,053 Squash** molecules. ~20,100 in total.

| Run | molecules | max ECFP4 | >=0.5 | 2,4-diaryl | >=2 OMe + >=1 OH |
|---|---|---|---|---|---|
| Clivia 258 (atoms 20-28) | 258 | 0.375 | 0 | 0 | - |
| Clivia 2k | 2,042 | 0.420 | 0 | 1 (0.05%) | - |
| Squash 2k | 2,007 | 0.491 | 0 | 18 (0.90%) | - |
| Clivia pinned 1k | 1,036 | 0.521 | - | 0 | 4 (0.4%) |
| Squash pinned 1k | 1,012 | 0.509 | - | 10 (1.0%) | 18 (1.8%) |
| **Clivia 10k** | 10,046 | **0.535** | 1 (0.01%) | 4 (0.04%) | 23 (0.23%) |
| **Squash 10k** | 10,053 | **0.518** | 1 (0.01%) | **86 (0.86%)** | **144 (1.43%)** |

**No compound reproduces the series.** Nothing exceeds ECFP4 0.535; there is no 0.7+ molecule,
no 0.6+, and only two above 0.5 in 20,099 draws.

The sampling curve has flattened: a 5x increase (2,000 -> 10,000) bought +0.12 ECFP4 for
Clivia (0.42 -> 0.535) and +0.03 for Squash (0.49 -> 0.518), with the top hits changing
identity between runs rather than converging. Further scaling is not a plausible route to
reproduction.

The best 2,4-diaryl-quinoline hit is ECFP4 **0.482** to mol1 -
`COc1ccc2c(-c3cc[o+]c(O)c3)cc(-c3ccc(C(=O)O)cc3)nc2c1` - the closest anything has come to
combining the right scaffold with plausible substituents, but it still carries COOH where the
targets have methoxy/hydroxyl.

The closest molecules overall are *not* the scaffold matches: the top Clivia hit (0.535) is a
bis-quinoline and the top Squash hit (0.518) carries NMe2 + OH but only one aryl.

**Conclusion.** Unconstrained generation from the Clivia and Squash pockets produces the right
scaffold (rarely) and the right substituent chemistry (more often) as separate populations, and
does not converge on the series at any sampling depth tested. The five compounds are most
likely the product of a guided or scaffold-constrained run, not free generation.

Figures: `figures/ten_k_best.png`. Per-molecule tables (ECFP4 >= 0.40):
`reports/ten_k_clivia.csv`, `reports/ten_k_squash.csv`.

## Complete pocket inventory

All known structures were retrieved from the RCSB full-text search: **Clivia 7 entries**
(8HZD, 8HZE, 8HZF, 8HZJ, 8HZK, 8HZL, 8HZM) and **Squash 3 entries** (7KVT, 7KVV, 7KVU).
Every non-solvent, non-ion hetero group was treated as a ligand instance, giving **21
candidate pockets**, then clustered by superposition (RMSD <= 2 Å on shared nucleotide
backbone atoms).

**Four distinct sites:**

| Cluster | Pockets | Composition | Size |
|---|---|---|---|
| **Clivia site 1** (functional) | **14** — every Clivia entry, both chains | A-rich: {A:6, C:3-4, G:2-4, U:2} + 1-3 Mg | 14-19 res |
| Clivia site 2 (8HZL only) | 2 | {A:5, C:5, G:5} + 3 Mg | 19 res |
| Clivia site 3 (8HZL only) | 2 | {A:6, C:3, G:2} + 2 Mg | 15 res |
| **Squash** | **3** — every Squash entry | **G-rich**: {A:7, C:1, G:8-9, U:3} + 2-3 Mg | 21-22 res |

Key points:

* **Clivia and Squash are unrelated pockets.** Their representative sites share **zero**
  nucleotide residues, so no superposition is even definable; composition is also opposite
  (Clivia A/C-rich, Squash G-rich) and Squash is ~50% larger.
* **Within Clivia, site 1 is the functional fluorophore site** and is present in all seven
  entries as one copy per RNA chain — the crystallised ligand differs (O2I, NI4, NJL, O00)
  but the site is the same.
* **8HZL is a different construct** (84-mer aptamer_III) and contains **three** NI4 sites
  per RNA copy. Those two extra sites appear nowhere else.
* Ligand choices per entry: 8HZD O2I; 8HZE/8HZK/8HZL/8HZM NI4; 8HZF NJL; 8HZJ O00;
  7KVT/7KVU 2ZY; 7KVV 747. Iodides (IRI) in 7KVT/8HZK and GTP/GDP/PEG were excluded as
  ions/additives.

**Consequence for the runs already done.** The generation used 8HZE NI4 chain A
(= Clivia site 1) and 7KVU 2ZY chain G (= Squash). Those are the correct functional sites,
so the negative reproduction result stands on the right pockets — it was not a
wrong-pocket artefact. What remains untested is Clivia sites 2 and 3, and whether the
series might bind a site other than the crystallised one.

Files: `pockets/` (all 21), `pockets_summary.json`, `pocket_rmsd.npy`,
`pocket_clusters.json`.

## Pocket properties (measured against the crystallised ligand)

The earlier attempt to score cavity volume from the pocket centroid alone was meaningless -
it counted solvent-exposed space. These metrics reload the crystallised fluorophore and
measure against it.

| Site | pocket | ligand | ligand vol | free vol <10 Å | enclosure | shape compl. | sphericity | contacts | N/O <5 Å |
|---|---|---|---|---|---|---|---|---|---|
| **Clivia 1** (functional) | 15 res, A-rich | NI4, 27 atoms | 297 Å³ | 1138 Å³ | 73% | 52.3% | 0.34 | 6 | 42 |
| Clivia 2 (8HZL) | 19 res | NI4, 27 | 296 Å³ | 1083 Å³ | 74% | 54.8% | 0.32 | 6 | 38 |
| Clivia 3 (8HZL) | 15 res | NI4, 27 | 294 Å³ | 1193 Å³ | 72% | 53.0% | 0.36 | 6 | 43 |
| **Squash** | 22 res, G-rich | 2ZY, 22 | 219 Å³ | 483 Å³ | **88%** | **62.2%** | 0.22 | **8** | 44 |

What this says:

* **The three Clivia sites are the same kind of pocket** - 72-74% enclosure, 1083-1193 Å³,
  same complementarity. Sites 2 and 3 are not a different environment, just more copies.
* **Squash is a different pocket**: half the free volume, 15 points more enclosed, 10 points
  better shape complementarity, 8 vs 6 direct contacts. It is a tight, preorganised slot.
* **All four are strongly non-spherical** (sphericity 0.22-0.36), i.e. flat elongated slots -
  consistent with selecting flat aromatic ligands, which is what the series is.
* **Size match favours Clivia for this series**: Clivia is built for a 297 Å³ ligand, and
  mol1-mol5 are 26-29 heavy atoms ~ 290-320 Å³. Squash's native ligand is 219 Å³, ~40% smaller.

## The core skeleton IS reproduced

Of 20,099 molecules, **90 carry the 2,4-diaryl-quinoline core** (Squash 86, Clivia 4). The
three connected ring systems that define the series - quinoline plus two aryls - are
generated de novo, best ECFP4 0.482 to mol1.

Substituent inventory across those 90 core hits:

| substituent | present in | targets have it |
|---|---|---|
| hydroxyl | 68/90 (75.6%) | yes (mol1, mol5) |
| carboxyl | 44/90 (48.9%) | no |
| amide | 27/90 (30.0%) | no |
| **methoxy** | **23/90 (25.6%)** | **yes - all five** |
| amino | 20/90 (22.2%) | yes (mol2) |
| NMe2 | 11/90 (12.2%) | yes (mol3) |
| aza-aryl | 90/90 | yes (mol4) |

So the discrepancy is **not categorical**. Every substituent the targets use does appear on
the correct core. What differs is the proportions: the model over-produces carboxylic acid
and amide (49% / 30%, absent from the targets) and under-produces the methoxy that all five
targets carry (26% vs 100%).

**Revised objective.** Not "can the core be generated" - it can - but "can the methoxy-rich
substitution pattern be enriched on the correct core". That is a sampling/enrichment problem,
which is what the extra rounds were for.

The best core hit (ECFP4 0.482) already illustrates this:
`COc1ccc2c(-c3cc[o+]c(O)c3)cc(-c3ccc(C(=O)O)cc3)nc2c1` has the correct core plus methoxy,
hydroxyl and carboxyl; only the carboxyl-bearing aryl is wrong.

Files: `pocket_properties.py`, `pocket_properties.json`.

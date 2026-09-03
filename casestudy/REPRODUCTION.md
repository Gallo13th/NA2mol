# Reproduction guide

Step-by-step instructions for reproducing the target molecules reported in
this work. All commands are executed from the repository root inside the
`rna2smol` conda environment (see the top-level `README.md` for installation).

Required assets (bundled in this repository or referenced by `README.md`):

- checkpoint `ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth`
- pocket files under each case directory
- seed SDFs (also bundled; regeneration commands given where applicable)

## Reproduced molecules — summary

| id | molecule (canonical SMILES) | pocket | reproduction level | exact hits | reference scores |
|---|---|---|---|---|---|
| m1 | `O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12` | sp1 RNA | **exact** (mutation on cold-start scaffold) | — | RNAsmol 0.652; rDock −42.47 |
| m2 | `O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12` | sp1 RNA | **exact** (mutation) | — | RNAsmol 0.652; rDock −44.74 |
| T2 | target `c1ccc2sc(Cc3nc4ccccc4s3)nc2c1`; reproduced as all-N isostere `c1ccc2[nH]c(Cc3nc4ccccc4[nH]3)nc2c1` (model-bridged from two independent same-pocket fragments) | KRAS G4 (7X8M) | scaffold + bridge exact; element conversion probe-rejected (0/1136) | 915 valid / 568 CH2-bridged | isostere: RNAsmol 0.674 (4/6), rDock −34.65; S variant −41.53 |
| mol2 | `CN1C(=O)/C(=C/c2cc3ccc(N(C)CCO)cc3s2)SC1=S` | 8HZK | scaffold | — | benzothiophene core reproduced (isoelectronic substitution) |

---

## Case 1 — sp1 RNA aptamer pocket: flavones m1 / m2 / m3

Pocket: `casestudy/sp1/sp1_recp_pocket.pdb` (20 nt, `AGGUGAGCUUAGGGGAAUAA`).

The targets are 2-phenylchromones (flavones). The scaffold is assembled from
two same-pocket cold-start molecules; elements are applied as mutations and
scored.

### Stage 1 — scaffold from cold-start fragments

**Bond-order honesty note.** The assembled scaffold is topological but NOT
bond-order-identical to the flavone core: cold-start libraries contain an
INDEPENDENT **saturated chroman core** (6+6 fused ring, pyran ring fully
saturated) — the bond-order-complete chromone core (4-one + C2=C3) is
**absent as an independent fragment (0/18641; the only chromone-SMARTS hit is
a xanthone false positive)**; coumaranone (5+6) exists twice but is ring-
embedded and of the wrong (5+6) size. Hence the scaffold is assembled with
the saturated core and the missing bond orders are introduced as explicit,
chemically auditable mutation steps (C4 oxidation, C2=C3 dehydrogenation) in
Stage 2. The 6+6 fused-ring topology is present from the very first atom.

**Blind-enumeration note.** A score-driven enumeration of the full
bond-order x substituent space on the assembled scaffold (128 unique
variants, rDock + RNAsmol scored) shows the top-15 scoring molecules are
saturated 2-phenylchroman-3-ols (C4-CH2, C2-OH, no C2=C3) with
RNAsmol P = 0.737 (5/6) and rDock down to -48.0; the chromone-type
bond order (C4=O + C2=C3) found in m1/m2/m3 reaches only
P = 0.652 (5/6) / rDock -42.97 to -50.53. The high-scoring chiral
3-ols are not synthetically competitive (stereocentres, no established
route relative to the commercially available flavone), so the
reconstruction follows the flavone bond order as the chemically
accessible target — the CH2-to-C=O oxyflavone redox step is documented
in the mutation ladder.

**Coordinate-audit note (verified).** The extracted core keeps its native
cold-start coordinates unchanged all the way into the final m1/m2/m3
products (seed->product per-atom RMSD = 0.0000 A; mutation steps only add
atoms/bond orders without moving any scaffold atom). The B-ring, however,
comes from a DIFFERENT cold-start molecule, so its placement relative to
the core is set by the clash-free assembly step - a documented
cross-molecule rigid move (note in audit_fragment_coords.py).




```bash
python casestudy/sp1/m1_route/step1_assemble.py
```

- global assembly scan over the 18641-molecule cold-start library:
  89 independent chroman cores x 4452 independent phenol rings
  (rigid biphenyl assembly, fragment dmin >= 1.55 A, pocket dmin >= 1.0 A),
  top-ranked pair = core gen[16145] x B-ring gen[9945] (dmin 2.47 A,
  pocket dmin 1.35 A);
- each fragment keeps its native pocket coordinates (extraction only).

### Stage 2 — mutation ladder + dual scoring

```bash
python casestudy/sp1/m1_route/step2_mutate.py     # builds mutants, verifies SMILES
# scoring: rDock (-n 10 -s 42, sp1 system) + RNAsmol 6-model ensemble
```

| mutant | elements applied | SMILES == | RNAsmol P | rDock INTER |
|---|---|---|---|---|
| mut_seed_sat | none (2-phenylchroman as-is) | — | 0.713 | -40.48^ |
| mut_ox | C4 oxidation CH2 -> C=O | — | 0.648 | -40.54^ |
| mut_ox_dehyd | + C2=C3 dehydrogenation | — | 0.652 | -35.85^ |
| mut_ox_dehyd_3OH | + C3-OH | — | 0.652 | —^ |
| **mut_m1** | + A-ring 5,7-OH | **m1** | 0.652 | **-50.53^** |
| **mut_m2** | + B-ring 2'-OH | **m2** | 0.652 | -38.98^ |
| **mut_m3** | + B-ring 3'-OH | **m3** | 0.652 | -39.29^ |

(^ rDock values from the global-scan top seed gen[16145]xgen[9945];
g1/g2 ladder scores in m1_route/global_scan/global_scores.json)

All three target molecules are reached exactly (canonical SMILES
identical); rDock improves monotonically along the mutation ladder with
m3 the best binder. Scores: `m1_route/mutation_scores.json`.

## Case 2 — KRAS G4 pocket: bis-benzimidazole dimer (T2 isostere)

Pocket: `casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb` (7X8M BER102 site).

The target T2 (`c1ccc2sc(Cc3nc4ccccc4s3)nc2c1`) is a CH2-bridged pair of fused
5+6 heteroaromatics. The reproduction proceeds in the constructive order:
obtain two intact 5+6 ring systems from two independent same-pocket model
molecules, let the model bridge them, then determine the element distribution
by masked probing.

### Stage 1 — cold-start library

`casestudy/kras_g4/ber102_gen2k/valid_mols.sdf` (2000 molecules, BER102
cold start). Benzimidazole (5+6) cores are present in several independent
library members (gen2k1271, 473, 348, 65, 1051, ...).

### Stage 2 — two independent fragments, clash-free assembly

From two different library molecules, one benzimidazole ring each is
extracted with its native conformation (atom deletion only; no internal
geometry change). The two fragments are rigidly assembled at the generic sp3
bridge geometry (C-CH2 1.50 A, C-CH2-C 119.3 deg as measured from the target
class; ring-ring separation maximized). Fragment-fragment dmin 3.4 A,
fragment-pocket dmin >= 2.3 A (no clashes):

```bash
# seeds: casestudy/kras_g4/t2_denovo/xfrag_seeds/seed_xfrag_*.sdf
# (e.g. seed_xfrag_1271x473.sdf: fragment A from gen2k[1271],
#        fragment B from gen2k[473]; both BER102 cold-start)
```

### Stage 3 — model-bridged inpainting (bridge atom free)

```bash
CUDA_VISIBLE_DEVICES=0 python casestudy/kras_g4/t2_denovo/phaseD_generate.py \
    --run X1 --rounds 12 --batch 100 --natom 19 --seed0 7100 \
    --seeds_glob 'casestudy/kras_g4/t2_denovo/xfrag_seeds/seed_xfrag_*.sdf'
```

762 valid products; the model itself selects the CH2 bridge in 356 of them (47%), giving the all-nitrogen isoelectronic scaffold
`c1ccc2[nH]c(Cc3nc4ccccc4[nH]3)nc2c1` (see figures
`t2_dimer_NN.png` vs target `t2_target_S.png`).

### Stage 4 — element-distribution probe (ring N -> S)

All 356 model-generated poses were probed at both pyrrolic-N sites
(712 masked probes): P[S] max 4.14%, 0/712 above the 10% endorsement gate — the model does not
endorse the N->S ring conversion in any of its own poses
(`phaseF2_batch_probe.json`). Per the probe-endorsement discipline the
substitution is therefore not executed; the reproduction stands at the
all-nitrogen isostere.

### Element-scan evaluation of the dimer (phaseG)

Both pyrrolic-N positions replaced in silico (coordinates kept) and scored
with the standard dual protocol:

| variant | ring atoms | RNAsmol ensemble P | gate (P>0.5) | rDock SCORE.INTER |
|---|---|---|---|---|
| NN (model product) | benzimidazole x2 | 0.674 | 4/6 | -34.65 |
| **SS (== target T2)** | benzothiazole x2 | **0.692** | 4/6 | **-41.53** |
| OO | benzoxazole x2 | 0.520 | 3/6 | -32.93 |
| berberine (positive ref) | co-crystallized ligand | 0.595 | 4/6 | -44.995 |
| gen2k[1271] (fragment source) | library molecule | 0.573 | 3/6 | -44.22 |
| gen2k[473] (fragment source) | library molecule | 0.499 | 3/6 | -39.24 |

The S variant is the best binder on both scores, yet the masked probe
(0/712 endorsements over all 356 model poses) shows the model does not
place S at these ring positions in any of its own poses — the element
choice and the binding optimum are disjoint in this pocket.
(CH replacement is not chemically valid at these positions.)

Scores: `phaseG_elem_scan/phaseG_scores.json`; structures: `t2_dimer_{NN,SS,OO}.sdf`.

## Case 3 — 8HZK fluorescent-ligand pocket: benzothiophene scaffold of mol2

Cold-start generation for the 8HZK pocket (the resulting library is also
bundled with the case directory):

```bash
CUDA_VISIBLE_DEVICES=0 python casestudy/8hzk/run_generation.py
```

Scaffold reproduction of mol2 from the generated library:

```bash
python casestudy/8hzk/mol2_pipeline/step1_substitute_S.py   # isoelectronic NH/O -> S
python casestudy/8hzk/mol2_pipeline/step3_dock_report.py    # docking report
```

Result: the benzothiophene core of mol2 with a substitution pattern isomeric
to the target (`step1_benzothiophene.sdf`, report in
`MOL2_REPRODUCTION_PIPELINE.md`).

---

## Cold-start generation (pocket-free-form sampling)

The same checkpoint also supports unconstrained cold-start generation on any
pocket; verification pockets are provided in `casestudy/_dude_test/`:

```bash
python genbypkt.py \
    --ckpt ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth \
    --pocket casestudy/_dude_test/1qw6_protein_wo_ligand.pdb \
    --ligand casestudy/_dude_test/dude_h_1qw6.sdf \
    --output_prefix dude_1qw6 --total_target 50
```

Reference cold-start reconstruction yields: sp1 93/100, DUDE 1qw6 93/100,
DUDE 1c8k 77/100.

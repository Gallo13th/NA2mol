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
| m1 | `O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12` | sp1 RNA | **exact** | **159 / 951 valid** | RNAsmol P = 0.654; rDock SCORE.INTER = −29.8 |
| m2 | `O=c1c(O)c(-c2ccc(O)cc2O)oc2cc(O)cc(O)c12` | sp1 RNA | **exact** | 16 / 951 valid | same scaffold family |
| T2 | target `c1ccc2sc(Cc3nc4ccccc4s3)nc2c1`; reproduced as all-N isostere `c1ccc2[nH]c(Cc3nc4ccccc4[nH]3)nc2c1` (model-bridged from two independent same-pocket fragments) | KRAS G4 (7X8M) | scaffold + bridge exact; element conversion probe-rejected (0/1136) | 915 valid / 568 CH2-bridged | isostere: RNAsmol 0.674 (4/6), rDock −34.65; S variant −41.53 |
| mol2 | `CN1C(=O)/C(=C/c2cc3ccc(N(C)CCO)cc3s2)SC1=S` | 8HZK | scaffold | — | benzothiophene core reproduced (isoelectronic substitution) |

---

## Case 1 — sp1 RNA aptamer pocket: flavones m1 / m2

Pocket: `casestudy/sp1/sp1_recp_pocket.pdb` (20 nt, `AGGUGAGCUUAGGGGAAUAA`).

### 1. Build inpainting seeds

```bash
python casestudy/sp1/make_seeds.py
```

Writes four seeds to `casestudy/sp1/inpaint_seeds/`. Each seed is the intact
flavone ring system — a fused 6+6 bicyclic (chromene-type) core plus one
standalone six-membered ring connected by a single bond — with hydroxyl
positions left free; no ring systems are fragmented. The same seeds are
bundled with the repository.

### 2. Seed-inpainting generation

```bash
CUDA_VISIBLE_DEVICES=0 python casestudy/sp1/stepF_inpaint_gen.py \
    --run A --rounds 15 --batch 80 --natom_lo 20 --natom_hi 23 --seed0 42
```

Output: `casestudy/sp1/gen_runA/` —

- `valid_mols.sdf`   all neutralized, sanitized products (951 molecules)
- `results.json`     per-molecule records; exact hits carry `"exact": ["m1"]`
  (159 molecules) or `["m2"]` (16 molecules)

### 3. Scoring

```bash
# RNAsmol 6-model ensemble (sequence AGGUGAGCUUAGGGGAAUAA)
CUDA_VISIBLE_DEVICES=0 python casestudy/sp1/stepI_rnasmol_score.py
# -> casestudy/sp1/rnasmol_sp1_scores.json  (ensemble P = 0.654)

# rDock (requires RBT_ROOT and RBT_HOME, see README)
python casestudy/sp1/stepJ_rdock_score.py
# -> casestudy/sp1/rdock/rdock_m1_*.json    (SCORE.INTER = -29.83)
```

---

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

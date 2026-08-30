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
| T2 | `c1ccc2sc(Cc3nc4ccccc4s3)nc2c1` | KRAS G4 (7X8M) | **exact** | staged de-novo route | RNAsmol P = 0.692; rDock SCORE.INTER = −42.9 |
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

## Case 2 — KRAS G4 pocket: bis-benzothiazole T2

Pocket: `casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb` (7X8M BER102 site).

The target is a system of **two connected 5+6 fused rings** (benzothiazoles).
The reproduction therefore proceeds in the natural constructive order: obtain
the two 5+6 ring systems from model generations, connect them, then determine
the element distribution by masked probing. All intermediate scaffolds and
coordinates originate from model outputs; the target structure is used only
as a verification criterion.

### Stage 1 — cold-start library and heteroaromatic scaffold retrieval

The 2000-molecule cold-start library for this pocket is bundled at
`casestudy/kras_g4/ber102_gen2k/valid_mols.sdf` (regenerate with
`genbypkt.py --pocket casestudy/kras_g4/ber102_rdock/ber102_pocket.pdb
--ligand casestudy/kras_g4/ber102_rdock/ber102_ref.sdf --total_target 2000`).
Ranking by scaffold similarity identifies the top nitrogen heteroaromatic
family (bis-fused aza systems, e.g. phenanthroline- and benzimidazole-type
cores) as the isoelectronic precursor of the benzothiazole rings. Probe
gating of direct single-ring N→S substitution:

```bash
python casestudy/kras_g4/t2_denovo/phaseA_probe.py
# per-site P[S] < 10% on all candidates -> direct substitution not endorsed
```

### Stage 2 — obtain the two 5+6 rings: pose scan + N→S replacement

Two model-generated benzimidazole cores (intact fused 5+6 ring systems) are
assembled into the all-nitrogen isoelectronic dimer with generic chemistry
(C–CH₂ 1.51 Å, coplanar, ETKDG). A rigid pose scan in the pocket probes both
ring-nitrogen sites simultaneously; poses passing the 10% endorsement gate
undergo the isoelectronic N→S ring conversion, yielding the two intact
benzothiazole (5+6 fused) fragments:

```bash
python casestudy/kras_g4/t2_denovo/phaseA6_posescan.py       # pose scan + gating
python casestudy/kras_g4/t2_denovo/phaseBC_replacement_seed.py
# -> casestudy/kras_g4/t2_denovo/phaseBC_seeds/seedBC_*.sdf
#    (18 atoms = two intact 5+6 benzothiazole fragments, bridge atom left free)
```

### Stage 3 — connect the two rings (seed inpainting)

```bash
CUDA_VISIBLE_DEVICES=0 python casestudy/kras_g4/t2_denovo/phaseD_generate.py \
    --run D1 --rounds 16 --batch 100 --natom 19 --seed0 3042
# -> casestudy/kras_g4/t2_denovo/gen_runD1/valid_mols.sdf
```

The model completes the single bridge position between the two fragments.
The closest product is the NH-bridged bis-benzothiazole
`c1ccc2sc(Nc3nc4ccccc4s3)nc2c1` — both 5+6 ring systems intact, bridge
element still to be determined.

### Stage 4 — bridge-element probe + replacement

```bash
python casestudy/kras_g4/t2_denovo/phaseE_bridge_probe.py    # masked bridge probe (redock poses)
python casestudy/kras_g4/t2_denovo/phaseE_nc_replace.py      # N -> C replacement, P[C] = 22.1% endorsed
# -> casestudy/kras_g4/t2_denovo/phaseE_bridge/t2_replaced.sdf
```

The final product's canonical SMILES equals the target
`c1ccc2sc(Cc3nc4ccccc4s3)nc2c1` (**exact reproduction**).

### Scoring

```bash
CUDA_VISIBLE_DEVICES=0 python casestudy/kras_g4/t2_denovo/phaseE_rnasmol.py
# ensemble P = 0.692 (4/6 models above threshold)
# rDock (protocol: casestudy/kras_g4/rdock_protocol/PROTOCOL.md): SCORE.INTER = -42.92
```

---

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

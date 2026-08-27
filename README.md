# rna2smol

Bayesian Flow Network (BFN) pocket-conditioned ligand generation for RNA and
protein binding sites, with editor-molecule (seed inpainting) support and a
validated dual-scoring evaluation pipeline (rDock + RNAsmol ensemble).

The model generates 3D molecular graphs conditioned on a pocket structure
(`model.sample`, cold start) or completes a partially specified seed scaffold
(`model.sample_by_seed`, inpainting). Generated trajectories are reconstructed
into valid molecules with a neutralization-aware pipeline
(`util/mol_utils.py`), minimized in-pocket with smina, and scored with rDock
and a 6-version RNAsmol ensemble.

## Repository layout

```
rna2smol/
├── genbypkt.py              # generation entry point (cold start & seed inpainting)
├── config.yaml              # model/network configuration (single canonical config)
├── model/
│   └── bfn4edmol.py         # BFN score model + samplers (knn edge construction)
├── dataloader/              # pocket/ligand featurization, datasets, atom-count tables
├── util/                    # molecule reconstruction, refinement, scoring utilities
├── train_scripts/           # pre-training / fine-tuning entry points
├── scripts/                 # shell launchers for training runs
├── ckpt/                    # released checkpoints
├── data/                    # LMDB datasets (crossdock, expdb, RNA ligands)
├── casestudy/               # reproduced case studies (see table below)
└── output/                  # generation outputs (one subdirectory per run)
```

## Setup

```bash
bash setup_env.sh            # or: conda env create -f environment (see script)
conda activate rna2smol
# optional, for rDock scoring:
export RBT_ROOT=$CONDA_PREFIX/share/rdock-2013.1-1
export RBT_HOME=$HOME        # both are required (see casestudy/*/rdock_protocol/PROTOCOL.md)
```

## Usage

Command line (cold start):

```bash
python genbypkt.py \
    --ckpt ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth \
    --pocket casestudy/sp1/sp1_recp_pocket.pdb \
    --ligand casestudy/sp1/dummy_ligand.sdf \
    --output_prefix my_run --total_target 1000
```

Library API (seed inpainting):

```python
import genbypkt
genbypkt.model, genbypkt.accelerator = model, accelerator   # inject once
genbypkt.generate((seed_sdf, pocket_pdb), output_prefix='run',
                  seed_used=True, total_target=200)
```

Masked element probing (per-position model preferences on a fixed scaffold)
and batch inpainting drivers are provided per case study, e.g.
`casestudy/sp1/step1_probe_sp1.py` and `casestudy/sp1/stepF_inpaint_gen.py`.

## Case studies

| case | target class | pocket | headline result |
|---|---|---|---|
| `casestudy/sp1` | flavone (m1–m3) | sp1 RNA aptamer site, 20 nt `AGGUGAGCUUAGGGGAAUAA` | **m1 reproduced exactly 159×** from probe-guided seed inpainting (951 valid molecules, 16.7%); m2 16×; RNAsmol P=0.654 (gate ≥4/6 on all hits); rDock SCORE.INTER −29.8 |
| `casestudy/kras_g4` | bis-benzothiazole (T2) vs cyanine (T1) | KRAS G4 7X8M/7X8O (BER102/KPT102) | T2 reproduced exactly (two compliant routes; rDock INTER −42.9, RNAsmol 0.692); T1 shown unreachable under model priors (bridge-head 98.7% rejection) — see `T1_T2_TARGETS.md` |
| `casestudy/fmn` | FMN-mimetics | 1FMN aptamer | ~62k generated, dual-tool screening, top-20 combined binders (`top20_report.md`) |
| `casestudy/8hzk` | fluorescent ligands | 8HZK | benzothiophene scaffold reproduced by probability-backed element substitution (`MOL2_REPRODUCTION_PIPELINE.md`) |

Each case directory contains its own pipeline scripts, seeds, probe results
and scoring outputs; failed exploration routes are preserved under
`_archive/` subdirectories for provenance.

## Training

Training entry points (pre-training on CrossDocked/ExpDB, fine-tuning on RNA
ligands) live in `train_scripts/` with shell launchers in `scripts/`.
Dataset split definitions are in `pocket_split.csv`,
`pocket_grouping_strict_no_smrt.csv`, and `clean_dataset_*/`.

## Methodological notes

- **Atom taxonomy**: 13 classes = (element, aromaticity) pairs; the decode
  table in `util/mol_utils.py` is aligned with `dataloader/featurizer.py`
  (earlier builds mis-decoded S/Cl/P channels).
- **Element substitution** is only accepted with masked-probe probability
  backing; per-site probes are reliable for isolated atoms, not for
  contiguous multi-atom regions (see `casestudy/kras_g4/T1_T2_TARGETS.md`).
- **Neutralization**: reconstructed ions go through a three-tier policy
  (proton transfer → hard discharge → reject) in `util/mol_utils.py`.
- **Edge construction** in the network is kNN (k=32) only; the historical
  radius/hybrid branches were removed (the radius branch referenced an
  attribute that never existed and crashed at runtime).

## Verification benchmarks

Cold-start reconstruction yield (samples producing a valid, connected,
neutralized molecule), measured with `casestudy/_dude_test/` pockets after the
2026-08-27 configuration fix (`n_heads: 16`, see below):

| pocket | type | yield |
|---|---|---|
| sp1 (20-nt RNA aptamer site) | RNA | **93/100** |
| DUDE 1qw6 | protein | **93/100** |
| DUDE 1c8k | protein | 77/100 |

Seed-inpainting chain: 15/16 valid, 2 exact m1 (flavone) reproductions in a
single 16-sample smoke round.

## Known issues (resolved)

- **Cold-start yield regression (FIXED 2026-08-27)**: an earlier hand-restored
  `config.yaml` used `n_heads: 1` while the checkpoints were trained with
  `n_heads: 16`. Because weights load with `strict=False`, the mismatch
  silently dropped 20 attention value-projection tensors (`xv_func.net.3`)
  and degraded cold-start reconstruction from ~90% to 0%. The config now
  carries the verified value; `load_compiled_model` prints the skipped-tensor
  count so any future mismatch is visible.
- rDock requires both `RBT_ROOT` and `RBT_HOME` to be exported; missing
  `RBT_HOME` causes a parameter-file resolution segfault.

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

Batch inpainting drivers are provided per case study, e.g.
`casestudy/sp1/stepF_inpaint_gen.py`; the full reproduction workflow is
documented in `casestudy/REPRODUCTION.md`.

## Case studies

| case | target class | pocket | headline result |
|---|---|---|---|
| `casestudy/sp1` | flavone (m1–m3) | sp1 RNA aptamer site, 20 nt `AGGUGAGCUUAGGGGAAUAA` | **m1 reproduced exactly 159×** by seed inpainting (951 valid molecules); m2 16×; RNAsmol P=0.654; rDock SCORE.INTER −29.8 |
| `casestudy/kras_g4` | bis-benzothiazole (T2) | KRAS G4 7X8M/7X8O (BER102/KPT102) | T2 reproduced exactly **172×**; rDock INTER −42.9, RNAsmol P=0.692 |
| `casestudy/fmn` | FMN-mimetics | 1FMN aptamer | ~62k generated, dual-tool screening, top-20 combined binders (`top20_report.md`) |
| `casestudy/8hzk` | fluorescent ligands | 8HZK | benzothiophene scaffold of mol2 reproduced by isoelectronic element substitution (`MOL2_REPRODUCTION_PIPELINE.md`) |

A step-by-step guide for reproducing the target molecules is provided in
`casestudy/REPRODUCTION.md`.

## Training

Training entry points (pre-training on CrossDocked/ExpDB, fine-tuning on RNA
ligands) live in `train_scripts/` with shell launchers in `scripts/`.
Dataset split definitions are in `pocket_split.csv`,
`pocket_grouping_strict_no_smrt.csv`, and `clean_dataset_*/`.

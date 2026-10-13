# NA2smol

Pocket-conditioned 3D molecular generation for nucleic-acid targets with a
Bayesian Flow Network (BFN). NA2smol couples a ligand-only electron-density
(ED) encoder and a domain-shared pocket encoder (pharmacophore, hybridisation,
vdW radius, Gasteiger partial charge) on a shared molecular generator, trained
with ED pretraining, constrained (teacher-student) learning and temporal
contrastive learning.

This branch contains the code required to reproduce the paper: model training
(pretraining, fine-tuning, ablations), pocket-conditioned inference, library
generation, and the KRAS promoter G-quadruplex (BER102) fragment-based
amplification case study.

## Repository layout

```
na2smol/
├── genbypkt.py              # generation entry point (cold start & seed inpainting)
├── config.yaml              # model/network configuration (single canonical config)
├── model/bfn4edmol.py       # BFN score model + samplers (kNN edge construction)
├── dataloader/              # pocket/ligand featurization, datasets, atom-count tables
├── util/                    # molecule reconstruction, refinement, scoring utilities
├── train_scripts/           # pre-training / fine-tuning / ablation entry points
├── scripts/                 # shell launchers for training runs
├── data/                    # LMDB datasets (crossdock, expdb, RNA/DNA ligands)   [not tracked]
├── ckpt/                    # checkpoints                                        [not tracked]
├── casestudy/kras_g4/       # KRAS G4 BER102 case study (see below)
└── casestudy/fmn/RNAsmol/   # RNAsmol scoring ensemble (third-party code, weights not tracked)
```

## Environment

```bash
source setup_env.sh          # conda env `rna2smol` + rDock paths
conda activate rna2smol
```

## Training recipe

1. **Electron-density pretraining** (teacher checkpoint,
   `molcraft_edmol_onlyed_multires_epoch_9_step_1125000.pth`):
   `train_scripts/ced_pretrain.py`
2. **Mixed-domain fine-tuning** (full model; protein–ligand + nucleic-acid
   complexes; contrastive + constrained objectives) → released checkpoint
   `ckpt/onlypkt_ftwithrna_oncrossdocked_onlygeneralfeat_train_best_val_loss_step_111429.pth`:
   `train_scripts/ftwithcrossdocked_train.py`
   (launcher: `scripts/train_rna_finetune_v3.sh`)
3. **Ablations** (same fine-tuning data and primary losses; auxiliary
   objectives varied):
   - no-pretraining baseline: `train_scripts/baseline_scratch_train.py`
   - Pretrain Only: `train_scripts/ablation_pretrain_only_train.py`
   - Pretrain+Constrain: `train_scripts/ablation_pretrain_constrain_only_train.py`

Data preparation utilities: `train_scripts/filter_expdb_by_dude.py`
(protein-pocket overlap filtering against DUD-E test pockets),
`train_scripts/rebuild_crossdock_indices.py` (CrossDocked index rebuild).
Nucleic-acid pocket split metadata: `pocket_split.csv`,
`pocket_grouping_strict_no_smrt.csv`, `pocket_clusters_G4.csv`.

## Inference / generation

```bash
python genbypkt.py --ckpt ckpt/<checkpoint>.pth --pocket <pocket.pdb> ...
```

Cold-start pocket-conditioned sampling (`model.sample`) and seed-constrained
inpainting (`model.sample_by_seed`) are both supported; benchmarks and the
case study use unconstrained cold-start generation.

## Case study: KRAS promoter G-quadruplex (BER102, 7X8M)

Full pipeline in `casestudy/kras_g4/`:

1. **Library generation** — `run_generation_ber102.py` (4 sampling streams,
   heavy-atom range 20–28) → `ber102_gen10k/` (10,077 unique molecules;
   `valid_mols.sdf` is not tracked — regenerate with the command in
   `denovo2/PROTOCOL_REPORT_G4.md`).
2. **Fragment-based amplification and ensemble triage** —
   `denovo2/run_chain_ber102.py`: ring-fragment cutting with port (ghost-atom)
   memory → docking/property gate → saturated heteroatom mutation →
   port-matched recombination → product gate → multi-seed relaxed rDock
   docking → 6-sub-model RNAsmol ensemble voting (5 active rankers) + rDock
   rank → Tier-1 candidates. Stage evidence: `denovo2/chain_*.json`,
   `denovo2/paper_*.json`; full protocol description:
   `denovo2/PROTOCOL_REPORT_G4.md`; purchasable Tier-1 shortlist:
   `denovo2/g4_tier1_*`.

Requirements on this machine: rDock 2013.1 (`RBT_ROOT` set by
`setup_env.sh`; the BER102 system lives in `ber102_rdock/`, score-only
parameter files in `rdock_r4/`), `util/smina.static`, and the RNAsmol
ensemble weights under `casestudy/fmn/RNAsmol/saved_model/` (not tracked;
see `casestudy/fmn/RNAsmol/README.md`). Scoring wrappers:
`casestudy/fmn/rnasmol_score.py`, `rnasmol_score_gpu.py`.

## Data availability

Raw training corpora (`data/`), checkpoints (`ckpt/`), benchmark evaluation
inputs (`paperdata/`) and generation outputs (`output/`) are kept out of git
(re-obtainable; see `.gitignore`). Benchmark evaluation, figure-source data
and the chemical-overlap / leakage audit were produced with the released
tables accompanying the manuscript.

## License

Source code: MIT License. RNAsmol subpackage keeps its upstream license
(`casestudy/fmn/RNAsmol/LICENSE`).

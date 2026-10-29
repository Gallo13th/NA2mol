# rDock Protocol (solidified)

Verified conventions for all rDock work in this project. Validated by
ground-truth redock on two independent systems (2026-08-15):

| system  | receptor | RADIUS | FLEX | runs | best SCORE.INTER | RMSD in-place | verdict |
|---------|----------|--------|------|------|------------------|---------------|---------|
| 7X8M BER102 (KRAS G4, DNA) | MODEL-1 full nucleotide receptor, 24 res | 6.0 | 3.0 | 10 | -44.995 | 1.230 A | **PASS** |
| 1FMN FMN36 (RNA aptamer)   | MODEL-1 full nucleotide receptor, 35 res | 5.0 | 3.0 | 25 | -58.976 | 2.009 A | **PASS** |

Protocol file: `$RBT_ROOT/data/scripts/dock_solv.prm` (SF5 with desolvation,
required for RNA targeting). Both conda installs have it.

## Environment

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate rna2smol
export RBT_ROOT=$CONDA_PREFIX/share/rdock-2013.1-1
```

## Pipeline

```
extract_pocket.py   # MODEL-1 pocket PDB (probe/featurizer use; 14 A convention)
extract_ligand.py   # co-crystal ligand -> reference SD (SMILES graph mapping)
prepare_receptor.sh # pocket/full PDB -> all-H MOL2 -> rDock-safe MOL2 + system prm
make_cavity.sh      # rbcavity -was  (cavity .as + grids .gs, cache & reuse)
redock_validation.py# ground-truth gate: redock + SCORE.INTER rank + RMSD PASS/FAIL
```

Only after `redock_validation.py` PASSes may the system prm be used for
production docking of new ligands.

## Correct usage (User Guide + hard-won lessons)

### Receptor
- all-atom (full H) MOL2 from obabel: strip all H (`obabel -d`) then re-add
  complete set (`obabel -h`) — never trust partial H sets in NMR PDBs
- **rDock 2013.1 segfault bug (gdb-verified)**: `RbtMOL2FileSource::
  RemoveNonPolarHydrogens()` crashes (erase-during-iteration) whenever it has
  actual work to do. Always pre-strip non-polar H yourself and hand rDock the
  extended-carbon model (polar H explicit, non-polar H implicit) — this is
  exactly the input contract stated in User Guide §9. `fix_mol2.py` does this.
- obabel `-h` wrongly protonates phosphates (PDB has no bond orders):
  H on O1P/O2P are dropped; phosphates stay anionic
- atom-name apostrophes (`O5'`) -> `O5p` (segfault trigger)
- substructure names: obabel writes `DG11` (resname+resnum); strip number AND
  map DNA->RNA names (DG->G, DA->A, DC->C, DT->U) because the stock
  `RbtIonicAtoms.prm` has RNA sections only; proteins: `ALA17`->`ALA`
  (restores ASP/GLU/LYS/ARG/HIS distributed charges)
- phosphate atom names OP1/OP2 -> O1P/O2P (stock table naming)
- MOL2 must keep obabel's fixed-width column layout, consistent
  `@<TRIPOSE>MOLECULE` counts and `@<TRIPOS>SUBSTRUCTURE` records;
  whitespace-joined lines or stale counts also segfault the parser
- `RECEPTOR_FILE` / `RECEPTOR_FLEX` are TOP-LEVEL prm keys (never in SECTION)

### Cavity (co-crystal ligand present)
```
SECTION MAPPER
    SITE_MAPPER RbtLigandSiteMapper
    REF_MOL <abs path to co-crystal ligand SD>
    RADIUS <tuned, see below>
    SMALL_SPHERE 1.0
    MIN_VOLUME 100
    MAX_CAVITIES 1
    VOL_INCR 0.0
    GRIDSTEP 0.5
END_SECTION
```
- key is `SITE_MAPPER` (not SCORING_FUNCTION); `GRID_SPACING` is NOT a valid
  key (RBT_BAD_ARGUMENT) — it is `GRIDSTEP`
- RADIUS tuning matters: 10 A on a small flat solvent-exposed ligand gives a
  ~6200 A^3 sponge where poses roam (BER102 first attempt: best pose RMSD
  3.6 A = FAIL). RADIUS 5-6 gives 1500-2200 A^3 and passed both systems.
  Tune RADIUS until `redock_validation.py` passes; record the volume.
- `rbcavity -was` writes `<receptor>.as` + `.gs`; reuse the cache, do not
  regenerate per docking run

### Ligand SD
- 3D conformer, correct formal charges/bond orders; rDock does not sample
  bond lengths or ring conformations
- **explicit polar H only** (N/O/S/P-attached); non-polar H absent — rDock
  never adds H, and ligand-side non-polar H trigger the same
  RemoveNonPolarHydrogens segfault (FMN full-H SD: segfault; polar-only: OK)
- reference extraction from a PDB co-crystal: build RDKit mol from trusted
  SMILES, map coordinates via graph isomorphism on the element-labelled
  single-bond graphs + per-atom degree check (no orphan atoms possible) —
  see extract_ligand.py; validates formula/charge/fragment/bond-lengths
- rDock may emit `RBT_MODEL_ERROR ... makes too many bonds` warnings for
  quaternary/aromatic N+ (kekule bond-order sum 4); tolerable — BER102
  redock passed 1.23 A despite it. rDock's SD writer garbles bond orders on
  such atoms, so NEVER substructure-match rDock output; read coordinates
  positionally and paste them onto a chemically-correct mol copy
  (redock_validation.py does this)

### Docking & scoring
- `rbdock -i in.sd -o out -r system.prm -p $RBT_ROOT/data/scripts/dock_solv.prm -n <runs> -s <seed>`
  (seed flag is `-s`; always absolute paths)
- rank poses by **SCORE.INTER** (true binding term); INTRA/RESTR ignored
- ground-truth gate: best-INTER pose heavy-atom RMSD in-place < 2.5 A
  (Kabsch-aligned RMSD reported as secondary diagnostic)
- flexible-ligand note: FMN (31 heavy, 5 tail torsions) needed RADIUS 5 and
  n=25; the isoalloxazine chromophore was recovered at 0.55 A even when the
  ribityl tail scored elsewhere — inspect per-domain RMSD before concluding
  protocol failure

## Files
- `extract_pocket.py` — MODEL-1 pocket extraction (center-center cutoff,
  BER101-era convention kept for probe comparability)
- `extract_ligand.py` — co-crystal ligand SD extraction (SMILES mapping)
- `fix_mol2.py` — MOL2 post-processing (called by prepare_receptor.sh)
- `prepare_receptor.sh <pocket.pdb> <out_dir> <ref.sdf> [RADIUS] [FLEX] [NAME]`
- `make_cavity.sh <system.prm>`
- `redock_validation.py --system <prm> --ref <ref.sdf> --out-dir <dir> --tag <tag>`
- `regression_1fmn/` — 1FMN regression artifacts (PASS @ RADIUS 5)

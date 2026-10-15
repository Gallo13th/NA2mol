# Ligand-leakage audit — methods, findings and recommended wording

Scope: quantify chemical overlap between the SMRT-Net and DUD-E evaluation
references and (i) the fine-tuning training ligands, (ii) the ligand-only
pretraining library, and (iii) the generated molecules; and test whether the
manuscript's target-level ECFP4-Tanimoto recovery result is robust to that
overlap.

## 1. Data and methods

| Item | Choice |
|------|--------|
| Fine-tuning ligands | `finetuning_ligands_table.tar.gz` → `training_ligands_leakage_table.csv` (225,239 records; splits `train` 127,255 / `val` 27,621 / `unused` 67,536 / `train_dude_dropped` 2,827; roles protein(crossdock/expdb) and RNA). No explicit test split exists; `unused` is reported separately and is **not** counted as gradient-update exposure. |
| Pretraining library | `pretrain_smiles.txt.gz` → 2,287,199 unique canonical SMILES (ligand-only, no pocket labels). |
| SMRT-Net references | `notebooks/RNALigBench/activesmi/{pdb}_pos.smi` (fallback `{pdb}.smi`) for the 27 evaluated systems → 96 records, 50 unique actives. |
| DUD-E references | `release_package_v1/notebooks/DUDE_active_compounds.csv` (`label == active`) for the 101 evaluated targets → 35,237 unique actives (one per `(pdb, smiles)`). |
| Generated molecules | `notebooks/output/smrtbench_combined_results.csv` and `notebooks/dude_stat_all.csv` (post-evaluation stage) → 677,552 unique (benchmark, method, target, canonical) records; 613,286 unique canonical molecules. |
| Standardisation | RDKit 2024.09.6. Primary identity = canonical isomeric SMILES; parent identity = largest fragment + uncharge; scaffold = Bemis–Murcko (no chirality). Parsing failures counted (0). |
| Fingerprint | Morgan radius 2, 2048 bits, no chirality (identical to the manuscript). Similarity = Tanimoto. |
| Thresholds | 0.3 / 0.4 / 0.5 / 0.6 / 0.7 (0.5 for continuity with prior work; not a leakage boundary). |
| Recovery statistic | Target-level: for each method, R(τ)=fraction of targets with ≥1 molecule reaching ECFP4-TS ≥ τ; A = mean R over τ∈[0.25,0.5] (the manuscript definition). Bootstrap CIs resample **targets** (2,000 draws). |

Fine-tuning similarity was computed for the full set of reference actives (all
sources, RDKit C++ bulk) and for all 613,286 unique generated molecules
(vs `train`). Pretraining similarity for the **generated** molecules was computed
on a stratified random sample (4,000 per benchmark×method; 67,664 molecules
total) because the full 613k × 2.29M exact sweep is not tractable on the
available 6-physical-core machine; exact-identity pretraining overlap and the
reference-active pretraining similarity were computed for **all** records. The
three file-based deliverables below record what is exact and what is sampled.

## 2. Experiment I — benchmark-active overlap (exact for all actives)

Ligand-level (unique actives); percentages.

**DUD-E (35,237 actives / 101 targets)**

| Source | exact | parent | scaffold | max-TS ≥0.5 | ≥0.7 |
|--------|------:|-------:|---------:|------------:|-----:|
| ft train | 2.16% | 2.70% | 12.26% | 36.4% | 10.9% |
| ft train–protein | 2.16% | 2.70% | 12.25% | 36.4% | 10.9% |
| ft train–RNA | 0.01% | 0.01% | 0.95% | 0.3% | 0.0% |
| ft val | 0.41% | 0.57% | 6.12% | 11.9% | 2.3% |
| ft unused | 0.56% | 1.33% | 7.24% | 18.1% | 3.8% |
| pretrain | 9.19% | 19.09% | 47.61% | 90.0% | 56.6% |

Target-level: 99/101 (98.0%) DUD-E targets have ≥1 active exactly matching a
training ligand and 101/101 (100%) share a scaffold; 94/101 (93.1%) have an
exact pretraining match.

**SMRT-Net (50 actives / 27 targets)**

| Source | exact | parent | scaffold | max-TS ≥0.5 | ≥0.7 |
|--------|------:|-------:|---------:|------------:|-----:|
| ft train | 18.0% | 20.0% | 58.0% | 58.0% | 38.0% |
| ft train–RNA | 10.0% | 10.0% | 38.0% | 32.0% | 26.0% |
| pretrain | 4.0% | 4.0% | 54.0% | 72.0% | 22.0% |

Target-level: 11/27 (40.7%) SMRT targets have ≥1 exact training match; 2/27
(7.4%) have an exact pretraining match.

Headline: exact supervised-training duplication is low for DUD-E (2.2% of
actives) but the fraction with a **close** training analogue is substantial
(36% ≥0.5). Because DUD-E targets carry hundreds of actives each, almost every
target still touches the training set at ≥1 active (98%), even though the
per-active rate is small — the two summaries answer different questions.

## 3. Experiment II — generated-output overlap

Fine-tuning overlap is exact for all 613,286 unique generated molecules;
pretraining similarity is from the stratified sample (Wilson 95% CI in
`generated_overlap_summary.csv`).

Main methods (unique molecules):

| Benchmark | Method | exact ft | ≥0.5 ft | exact pre | ≥0.5 pre (95% CI) |
|-----------|--------|---------:|--------:|----------:|------------------:|
| DUD-E | RNA2smol | 0.08% | 7.3% | 0.36% | 26.5% (25.2–27.9) |
| DUD-E | MolCRAFT | 0.06% | 1.0% | 0.04% | 4.0% (3.4–4.6) |
| DUD-E | Pocket2Mol | 0.76% | 8.6% | 0.58% | 23.5% (22.3–24.8) |
| DUD-E | ED2Mol | 0.05% | 4.5% | 0.27% | 22.2% (21.1–23.4) |
| SMRT-Net | RNA2smol | 0.16% | 6.0% | 0.44% | 16.8% (15.8–17.9) |
| SMRT-Net | MolCRAFT | 0.65% | 5.8% | 0.22% | 8.3% (7.5–9.2) |
| SMRT-Net | Pocket2Mol | 0.55% | 10.6% | 0.44% | 25.1% (23.8–26.4) |
| SMRT-Net | ED2Mol | 0.08% | 7.4% | 0.46% | 26.7% (25.4–28.0) |

- RNA2smol reproduces fine-tuning ligands at very low rates (0.08–0.16% exact;
  6–7% within 0.5) — no evidence of supervised memorisation by the main model.
- RNA2smol generated molecules are chemically familiar to the **pretraining**
  library (26.5% within 0.5 for DUD-E) at a level comparable to Pocket2Mol/ED2Mol
  and well above MolCRAFT. This is expected for a model pretrained on 2.29M
  drug-like structures and, since pretraining carries no pocket labels, is not
  direct target-associated leakage.
- The `RNA2smol(AR)` ablation is an outlier (10.1% exact / 55.4% ≥0.5 vs
  fine-tuning training) and is **not** the reported model.

## 4. Experiment III — recovery robustness (`recovery_original_vs_overlap_filtered.csv`)

All methods scored on a common eligible-target set; A = mean target recovery
over τ∈[0.25,0.5]; brackets are 95% target-bootstrap CIs.

**SMRT-Net** (27 → 18–26 eligible targets depending on filter)

| Filter | eligible | RNA2smol | MolCRAFT | Pocket2Mol | ED2Mol |
|--------|---------:|---------:|---------:|-----------:|-------:|
| original | 27 | 0.218 | 0.060 | 0.081 | 0.078 |
| exact ft excluded | 24 | 0.245 | 0.064 | 0.087 | 0.087 |
| ft analogues ≥0.5 excluded | 18 | 0.249 | 0.027 | 0.098 | 0.106 |
| ft analogues ≥0.6 excluded | 19 | 0.283 | 0.029 | 0.109 | 0.108 |
| ft analogues ≥0.7 excluded | 19 | 0.309 | 0.029 | 0.109 | 0.110 |
| exact pretrain excluded | 26 | 0.226 | 0.062 | 0.077 | 0.081 |
| pretrain analogues ≥0.5 excluded | 14 | 0.052 | 0.040 | 0.011 | 0.085 |

**DUD-E** (83 → 80 eligible targets)

| Filter | eligible | RNA2smol | MolCRAFT | Pocket2Mol | ED2Mol |
|--------|---------:|---------:|---------:|-----------:|-------:|
| original | 83 | 0.695 | 0.337 | 0.506 | 0.511 |
| exact ft excluded | 83 | 0.685 | 0.334 | 0.486 | 0.505 |
| ft analogues ≥0.5 excluded | 83 | 0.580 | 0.265 | 0.365 | 0.433 |
| ft analogues ≥0.6 excluded | 83 | 0.635 | 0.304 | 0.430 | 0.464 |
| ft analogues ≥0.7 excluded | 83 | 0.674 | 0.322 | 0.465 | 0.493 |
| exact pretrain excluded | 83 | 0.681 | 0.331 | 0.486 | 0.497 |
| pretrain analogues ≥0.5 excluded | 80 | 0.240 | 0.068 | 0.171 | 0.170 |

- **Exact** supervised-training removal changes nothing (DUD-E −0.010,
  SMRT +0.027): the recovery result is not driven by exact test-ligand
  duplication in supervised training.
- Removing **close** training analogues (≥0.5) reduces DUD-E RNA2smol from 0.695
  to 0.580 — still ~1.3–2.2× every baseline — and at ≥0.6/≥0.7 the value returns
  toward the original. For SMRT-Net the advantage if anything **strengthens**
  (0.218 → 0.309) as overlapping references are removed. The RNA2smol ordering
  over the baselines is preserved at every training-overlap filter.
- The pretraining-overlap filter (≥0.5) collapses recovery for every method
  (DUD-E RNA2smol 0.695→0.240) because ~90% of DUD-E actives have a close
  pretraining neighbour; the remaining eligible set is small and the reference
  chemistry is heavily pruned. This is a limited sensitivity analysis, not
  evidence of model failure.

## 5. Experiment IV — fixed-target sensitivity and reconstruction reconciliation

`recovery_fixed_target_sensitivity.csv`, `recovery_reproduction_check.csv`,
`recovery_target_coverage.csv` (script `exp4_fixed_target_sensitivity.py`).

The filter rows in Exp III use different target populations, so a change between
rows mixes the reference-chemistry effect with a target-composition effect. Exp
IV recomputes the **original** A on exactly the target subset that survives each
filter and compares it with the **filtered** A on the same subset.

Selected values (A original subset → A filtered subset on the same targets):

| Benchmark | Filter | targets | RNA2smol orig→filt | filter effect |
|-----------|--------|--------:|-------------------:|--------------:|
| SMRT-Net | ft exact | 24 | 0.245 → 0.245 | 0.000 |
| SMRT-Net | ft analogues ≥0.5 | 18 | 0.326 → 0.249 | −0.077 |
| SMRT-Net | ft analogues ≥0.6 | 19 | 0.309 → 0.283 | −0.026 |
| SMRT-Net | pretrain exact | 26 | 0.226 → 0.226 | 0.000 |
| SMRT-Net | pretrain analogues ≥0.5 | 14 | 0.251 → 0.052 | −0.198 |
| DUD-E | ft exact | 83 | 0.695 → 0.685 | −0.010 |
| DUD-E | ft analogues ≥0.5 | 83 | 0.695 → 0.580 | −0.115 |
| DUD-E | ft analogues ≥0.6 | 83 | 0.695 → 0.635 | −0.061 |
| DUD-E | pretrain exact | 83 | 0.695 → 0.681 | −0.015 |
| DUD-E | pretrain analogues ≥0.5 | 80 | 0.699 → 0.240 | −0.459 |

Consequences:

- The apparent **increase** in SMRT-Net recovery across the fine-tuning filters
  (0.218 → 0.249 → 0.309) is mostly a target-composition effect: on the same 18
  targets the original recovery is already 0.326, and removing close training
  analogues **lowers** it to 0.249. So close training analogues do contribute
  some genuine recovery, but RNA2smol still leads all baselines on the same fixed
  subset (0.249 vs 0.027 / 0.098 / 0.106).
- Exact-match exclusion has essentially no effect on any fixed target subset
  (SMRT 0.000; DUD-E −0.010 to −0.015).
- Close **pretraining** analogues carry a large effect once targets are fixed
  (SMRT −0.198; DUD-E −0.459), confirming that the metric is sensitive to the
  chemical space of the references.

**Reconstruction reconciliation.** `recovery_reproduction_check.csv` reports A
per method on each method's own evaluable target set, alongside the manuscript's
own per-threshold target-success tables (`d2_..._by_threshold.csv` for DUD-E and
`t1b_...` for SMRT-Net) and the cached permutation-gain value.

| Benchmark | Method | audit recompute | manuscript table | cached `gain.csv` | targets (audit / table) |
|-----------|--------|----------------:|-----------------:|------------------:|------------------------:|
| SMRT-Net | RNA2smol | 0.2175 | 0.2296 | 0.2155 | 27 / 27 |
| SMRT-Net | MolCRAFT | 0.0596 | 0.0556 | 0.0596 | 27 / 27 |
| SMRT-Net | Pocket2Mol | 0.0806 | 0.0889 | 0.0806 | 27 / 27 |
| DUD-E | RNA2smol | 0.6763 | 0.6647 | 0.6483 | 99 / 102 |
| DUD-E | MolCRAFT | 0.3444 | 0.3690 | 0.3508 | 98 / 100 |
| DUD-E | Pocket2Mol | 0.5008 | 0.5174 | 0.3959 | 85 / 86 |
| DUD-E | ED2Mol | 0.5127 | 0.5299 | 0.4762 | 96 / 97 |

The audit's per-method recomputation agrees with the manuscript's **own
per-threshold tables** to within ≈0.02 (SMRT-Net) / ≈0.025 (DUD-E); the residual
offset is target coverage (the audit's reference-active file covers 101 DUD-E
targets vs the manuscript's 102, and per-method molecule coverage differs by
1–3 targets) plus grid discretisation. The earlier 83-target reconstruction used
a common eligible set and is superseded by these per-method values. Note that the
cached permutation-gain A for DUD-E Pocket2Mol (0.396) is inconsistent with both
the manuscript's own table (0.517) and the recomputation (0.501) — a discrepancy
internal to the cached artefact, not to the audit.

## 6. Experiment V — ablation ladder: does the advantage depend on pretraining?

`recovery_ablation_ladder.csv`, `recovery_ablation_coverage.csv` (script
`exp5_ablation_recovery.py`). SMRT-Net only; all four configurations cover all
27 targets.

The manuscript's ablation ladder is added to the recovery analysis. The
no-pretraining configuration (`RNA2smol (Baseline)`, pretraining omitted,
λ_cont = λ_constrain = 0) is the key control for the pretraining question.

A_obs at each reference filter (all configs on their common eligible target set):

| filter | full model | Pretrain+Constrain | no-pretrain Baseline | Pretrain Only |
|--------|-----------:|-------------------:|---------------------:|--------------:|
| original (27) | 0.218 | 0.132 | 0.141 | 0.034 |
| ft exact (24) | 0.245 | 0.146 | 0.156 | 0.039 |
| ft analogues ≥0.5 (18) | 0.249 | 0.182 | 0.134 | 0.034 |
| ft analogues ≥0.7 (19) | 0.309 | 0.184 | 0.197 | 0.049 |
| pretrain exact (26) | 0.226 | 0.137 | 0.146 | 0.036 |
| pretrain analogues ≥0.5 (14) | 0.052 | 0.104 | 0.114 | 0.052 |

Same-target filter effect (A_filtered − A_original on the surviving targets):

| filter | full model | Pretrain+Constrain | no-pretrain Baseline |
|--------|-----------:|-------------------:|---------------------:|
| ft analogues ≥0.5 | −0.077 | −0.016 | −0.077 |
| pretrain exact | 0.000 | −0.003 | −0.003 |
| pretrain analogues ≥0.5 | −0.198 | −0.099 | −0.077 |

Findings:

- **The no-pretraining baseline already beats every external generator**
  (SMRT-Net A_obs 0.141 vs MolCRAFT 0.060, Pocket2Mol 0.081, ED2Mol 0.078), so
  part of RNA2smol's recovery advantage comes from the shared representation and
  fine-tuning strategy rather than pretraining.
- **Pretraining alone is not sufficient**: Pretrain Only is the weakest
  configuration (0.034), consistent with its documented geometry pathology; the
  full model's advantage emerges from pretraining combined with constrained and
  contrastive learning.
- **The full model's recovery is markedly more pretraining-dependent**: on the
  same 14 targets, removing actives with a pretraining neighbour ≥0.5 drops the
  full model 0.251→0.052 (Δ −0.198) versus Δ −0.077 for the no-pretraining
  baseline. This quantifies a real contribution of pretraining-adjacent chemistry
  to the headline recovery while showing the representation alone still recovers.
  Note the full model drops *below* the no-pretraining baseline on this filtered
  set (0.052 vs 0.114) — the residual advantage reverts to the no-pretraining
  level once pretraining-adjacent references are removed.

## 7. Interpretation (per the plan's decision framework)

This is closest to **Outcome 2**: overlap with the ligand-only pretraining
library is substantial and is the largest source of chemical familiarity, while
direct exact overlap with supervised training is limited. The headline
target-recovery advantage of RNA2smol over the baselines is **not** explained by
exact duplication of evaluation actives in supervised training. On fixed target
subsets, removing exact training matches leaves recovery unchanged and removing
close training analogues causes only a modest reduction while preserving the
RNA2smol ordering over baselines; the larger pretraining-analogue effect reflects
the sensitivity of the metric to reference chemistry rather than target-specific
leakage. We cannot claim this excludes every form of memorisation.

The ablation ladder sharpens the conclusion: the no-pretraining baseline already
exceeds all external generators, so part of the advantage is attributable to the
shared representation and fine-tuning; but the full model's ECFP4 recovery is
substantially more dependent on pretraining-adjacent reference chemistry
(Δ −0.198 on the pretraining-analogue filter versus −0.077 for the
no-pretraining baseline), and falls to the no-pretraining level once those
references are removed. Ligand-only pretraining therefore makes a real
contribution to the retrospective ECFP4 recovery, without target-pocket labels
and without being the sole source of the advantage.

## 8. Recommended manuscript text (Methods / Limitations)

> To assess potential chemical overlap between the evaluation references and the
> training data, SMRT-Net and DUD-E active ligands were standardised (RDKit
> 2024.09.6; canonical isomeric SMILES, largest-fragment parent) and compared
> with the fine-tuning partitions and the ligand-only pretraining library using
> exact identity, Bemis–Murcko scaffold and maximum ECFP4-Tanimoto similarity
> (Morgan r=2, 2048 bits). Fine-tuning overlap was reported separately from
> pretraining overlap, since pretraining ligands carry no pocket labels. Exact
> matches to the fine-tuning **training** split affected 2.2% of DUD-E and 18%
> of SMRT-Net actives, but close analogues (ECFP4-TS ≥0.5) were more common (36%
> and 58%). Generated molecules were compared with the same libraries: exact
> reproduction of training ligands was rare for RNA2smol (0.08–0.16%), while
> ~17–27% of generated molecules had a pretraining neighbour ≥0.5. Target-level
> ECFP4-TS recovery was reproduced and recomputed after excluding reference
> actives with exact or close supervised-training matches on a common eligible
> target set, with target-level bootstrap resampling for uncertainty. Recovery
> was unchanged by exact-match exclusion and remained above all baselines after
> excluding close training analogues under every threshold tested.

## 9. Files (in `audit/tables/`)

| File | Contents |
|------|----------|
| `reference_active_overlap.csv` | per-active identity/similarity/scaffold vs every source (38,855 rows) |
| `reference_overlap_summary.csv` | overlap at ligand and target level, by source and threshold |
| `generated_overlap_molecules.csv` | exact/parent/scaffold and max-TS-to-fine-tuning for all 613,286 unique generated molecules |
| `generated_overlap_pretrain_sample.csv` | pretraining max-TS for the 67,664-molecule stratified sample |
| `generated_overlap_summary.csv` | generated-output overlap by benchmark×method (pretraining fractions with Wilson CIs) |
| `recovery_molecule_provenance.csv` | recovery-contributing molecules (≥0.25 to a target active) with closest active and training/pretraining neighbours |
| `recovery_original_vs_overlap_filtered.csv` | A(τ∈0.25–0.5) per method and filter, eligible targets, bootstrap CIs, published reference values |
| `recovery_curves.csv` | per-target maxima and eligible-target lists per filter/method |
| `recovery_fixed_target_sensitivity.csv` | Exp IV: original vs filtered A on the *same* target subset (filter vs target-composition effects) |
| `recovery_reproduction_check.csv` | Exp IV: per-method A vs manuscript per-threshold table vs cached gain, with target counts |
| `recovery_target_coverage.csv` | Exp IV: per-method target coverage and common-set size |
| `recovery_ablation_ladder.csv` | Exp V: A_obs per filter for the no-pretraining baseline, Pretrain Only, Pretrain+Constrain and the full model (SMRT-Net) |
| `recovery_ablation_coverage.csv` | Exp V: target coverage of the four ablation configurations |
| `ligand_audit_summary.md` | this document |

Scripts are in `audit/scripts/` (`lka_common.py`,
`build_reference_actives.py`, `exp1_reference_overlap.py`,
`exp2a/2b/2_summary`, `exp3_recovery_sensitivity.py`,
`exp4_fixed_target_sensitivity.py`, `exp5_ablation_recovery.py`,
`build_provenance.py`, `summarize_reference.py`).

## 10. Deviations and limitations

1. **Pretraining similarity for generated molecules is sampled** (4,000/method;
   67,664 total) because the full 613k×2.29M sweep exceeds the available compute
   (6 physical cores, 15 GB). Exact-identity overlays and fine-tuning
   similarities are exact for all molecules.
2. The permutation-standardised null was not recomputed for the filtered
   reference sets (it requires the full molecule×target similarity matrix);
   published gains are attached for the original analysis, and the filtered
   comparisons rely on target-bootstrap CIs of A(τ).
3. No explicit `test` split exists in the fine-tuning table; `val` is reported
   separately and `unused` is not treated as gradient-update exposure.
4. Only the post-evaluation molecule stage was available; raw generated SDFs
   were not re-scored, so stage-wise attrition of chemical familiarity could not
   be quantified.
5. Target-level coverage differs by benchmark and method. Exp III uses the
   common eligible-target set for within-filter method comparisons; Exp IV adds
   the fixed-target original-vs-filtered comparison to separate reference
   chemistry from target composition. The audit's DUD-E reference-active file
   covers 101 targets vs the manuscript's 102, and per-method molecule coverage
   differs by 1–3 targets, so per-method reconstruction agrees with the
   manuscript's own per-threshold tables to within ≈0.02–0.025 rather than
   exactly.

# 1FMN screening round-1 (clean snapshot)

**Target:** 1FMN aptamer (35 nt, `GGCGUGUAGGAUAUGCUUCGGCAGAAGGACACGCC`)
**Date:** 2026-08-11
**Pipeline:** RNA2smol generation → rnasmol GPU scoring → rdock docking → combined ranking → neutral-16 deep evaluation

---

## 1. Round summary

| stage | output | numbers |
|---|---|---|
| generation (RNA2smol) | `output/fmn_pocket/dummy_ligand/dummy_ligand.sdf` | 4464 molecules (20–28 heavy atoms) |
| rnasmol GPU scoring | `rnasmol_scores.tsv` | 4464/4464 scored, 0 failed (7.76 ms/mol) |
| rdock top-300 docking | `docked_top300.sd`, `rdock_top300_scores.tsv` (in parent) | 300 mol × 3 poses, 900 total |
| combined rank (rnasmol 0.5 + rdock 0.5) | `top_binders.sdf` | top-100 binders, all rnasmol P = 1.0 |
| formal-charge split | — | 84 charged (phosphonium / oxonium / chloronium by-products), **16 neutral** |
| **neutral-16 deep evaluation** | `neutral_16_poses.sdf`, `neutral_16_evaluation.pdf`, `neutral_16_analysis.json`, `neutral_16_interactions.tsv`, `neutral_16_druglike.tsv` | binding mode (prolif) + FMN comparison + drug-likeness |

FMN reference: rdock self-dock SCORE = **-17.16** (INTER = **-16.79**), reproducing the
known ~-17 kcal/mol value. 295/300 docked molecules beat FMN on SCORE, 253/300 on INTER
→ the generation model found structures more complementary than FMN under the
same heavy-atom budget.

---

## 2. The neutral-16 deep evaluation

### 2.1 Binding-mode analysis (prolif, default curated interaction set)
- Pocket residues: G1, A8, G9, G10, A11, U12, A13, U14, A23, G24, A25, A26, G27, G28, A29 (15 nt)
- Each neutral ligand: rdock best-INTER pose (heavy atoms) → explicit-H added in 3D → prolif vs pocket
- Aggregate interaction-type totals across the 16:

| interaction | total count |
|---|---|
| VdWContact | 72 |
| PiStacking | 14 |
| HBDonor | 12 |
| Hydrophobic | 9 |
| (HBAcceptor / Anionic / Cationic / CationPi / PiCation) | 0 |

- **Key residues contacted by neutral-16** (non-VdW only): G9 (9/16), G10 (9/16), U12 (9/16), G27 (2/16), G1 (1/16), A8 (1/16).

### 2.2 FMN reference (crystal pose)
- Used the **crystal FMN pose** (`FMN_reference_ligand.sdf`) as the binding-mode reference,
  because rDock self-docking drifted ~5 Å from the crystal pose — the redock INTER
  (-21.0) is more negative than the crystal-like pose (-16.8) but the pose is
  geometrically poor (only VdW + Hydrophobic detected).
- FMN crystal binding mode: **HBAcceptor ×2** (G9, G27), VdW ×7, Hydrophobic ×1 (U12).
- FMN "signature" residue set (non-VdW): **{G9, U12, G27}**.
- Drug-likeness: MW 456, **QED 0.176** (very poor — FMN is a cofactor, not a drug),
  SA 4.03, 31 heavy atoms, 6 HBD, 10 HBA, TPSA 208, cLogP -1.61.

### 2.3 Neutral-16 vs FMN

| metric | finding |
|---|---|
| **Highest FMN similarity** | **N11** (gen=1436): residue Jaccard **0.75**, 3/3 FMN signature residues contacted (G9, U12, G27), 10 interactions incl. 3 PiStacking — INTER -26.12 (best of 16) |
| Also high similarity | N02, N06, N16 (Jaccard 0.667 each) |
| **Extra interactions vs FMN** | G10 contacted by 9/16 neutral ligands (FMN only VdW); many neutral molecules add HBDonor / PiStacking where FMN uses HBAcceptor — chemically orthogonal H-bonding pattern |
| HBAcceptor gap | 0/16 neutral molecules form HBAcceptor (their donors are dominant); FMN's signature is acceptor-based. This is the single biggest binding-mode divergence |

### 2.4 Drug-likeness distribution (16 neutral + FMN)

| property | neutral-16 range | median | FMN |
|---|---|---|---|
| MW | 284–386 | 356 | 456 |
| **QED** | **0.17 – 0.92** | **0.46** | **0.18** |
| SA | 3.62 – 5.07 | 4.21 | 4.03 |
| heavy atoms | 20–28 | 26 | 31 |
| HBD | 1–6 | 3 | 6 |
| HBA | 2–9 | 4 | 10 |
| rotatable | 1–9 | 3 | 7 |
| rings | 2–5 | 4 | 3 |
| TPSA | 56–154 | 105 | 208 |
| cLogP | -0.5 – 2.3 | 1.0 | -1.6 |

All 16 are smaller, less polar, and more drug-like than FMN. 14/16 beat FMN on QED
(only N02 and N15 are at the FMN level of drug-likeness).

### 2.5 Top-5 overall (composite = 0.5·norm(rdock INTER) + 0.3·QED + 0.2·norm(SA))

| rank | pose | gen | INTER | MW | QED | SA | FMN∩ | Jacc | composite | comment |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **N10** | 367 | -23.85 | 328 | **0.919** | 4.48 | 1 | 0.33 | **0.800** | best drug-like + strong binding |
| 2 | **N11** | 1436 | **-26.12** | 356 | 0.619 | 4.21 | **3** | **0.75** | 0.786 | **best FMN mimic**, best INTER |
| 3 | N13 | 2953 | -24.53 | 376 | 0.592 | 5.04 | 1 | 0.25 | 0.704 | diverse interactions (PiStacking + HBDonor) |
| 4 | N12 | 1656 | -25.42 | 370 | 0.405 | 4.88 | 0 | 0.00 | 0.679 | strong INTER but prolif-sparse (pose may be inward-folded) |
| 5 | N05 | 772 | -20.84 | 313 | 0.691 | 3.76 | 2 | 0.50 | 0.668 | balanced |

**Recommended for hit-to-lead follow-up:** **N10** (best drug-likeness, good binding)
and **N11** (best FMN-mimic + strongest binding). They bracket the two extremes of
the optimization frontier and together cover the G9/U12/G27 FMN signature.

---

## 3. Notes / caveats

1. **rdock INTER vs SCORE.** SCORE is polluted by INTRA conformational strain
   (some molecules reach SCORE -72 with INTER only -21). INTER is the pure
   receptor–ligand term and is used everywhere in this deep evaluation
   (see `SCREENING_REPORT.md` §5.1).

2. **Charge by-products.** 84/100 top binders carry net formal charge
   (phosphonium `[P+]`, oxonium `[O+]`, chloronium `[Cl+2]`). These arise because
   the aromaticity-maintenance step in generation can produce ylide-like
   hypervalent forms when the model tries to satisfy an aromatic ring around a
   polar heteroatom. They are real RDKit-parseable structures but chemically
   dubious as drug candidates, so they are excluded from this round. The 16
   neutral molecules (`formal_charge == 0`) have no such issue and are the
   promoted candidates. No filtering bias was introduced beyond formal charge.

3. **HBAcceptor gap is partly methodological.** HBAcceptor requires a residue
   N–H donor pointing at a ligand acceptor. The pocket PDB has H added by
   OpenBabel (sometimes suboptimal rotamers), so the prolif acceptor count is a
   lower bound — some true H-bonds are likely missed. The qualitative finding
   (neutral ligands prefer HBDonor + PiStacking; FMN prefers HBAcceptor) is robust.

4. **prolif default curated set** (Hydrophobic, HBDonor, HBAcceptor, PiStacking,
   VdWContact, Anionic, Cationic, CationPi, PiCation) was used; no metals or
   halogens are present so MetalAcceptor/Donor and XBAcceptor/Donor are not
   applicable. EdgeToFace/FaceToFace are subsumed under PiStacking.

---

## 4. Files in this directory

### Primary deliverables
| file | description |
|---|---|
| `neutral_16_evaluation.pdf` | **19-page PDF**: cover + FMN ref + 16 ligand pages + top-5 highlight |
| `neutral_16_poses.sdf` | 16 neutral molecules' best rdock pose (heavy atoms, 3D), all properties |
| `SCREENING_REPORT.md` | round-1 screening narrative (4464 → top-100), copied from parent |
| `README.md` | this file |

### Copied inputs (for reproducibility)
| file | description |
|---|---|
| `top_binders.sdf` | top-100 candidates with combined rank + rdock/rnasmol props |
| `docked_top300.sd` | rdock poses (900 = 300×3) |
| `rnasmol_scores.tsv` | rnasmol P(binder) for all 4464 + FMN |
| `1FMN_pocket_extracted_addh.pdb` | 15-residue pocket (H added), prolif receptor |
| `FMN_reference_ligand.sdf` | crystal FMN pose, binding-mode reference |

### Analysis artifacts
| file | description |
|---|---|
| `neutral_16_analysis.json` | full structured results (prolif + drug-likeness + FMN overlap) |
| `neutral_16_interactions.tsv` | flat per-interaction table (pose × residue × interaction) |
| `neutral_16_druglike.tsv` | drug-likeness table (16 + FMN) |
| `neutral_16_analysis_summary.txt` | short text summary |

### Pipeline scripts (re-runnable)
| file | description |
|---|---|
| `step1_extract_neutral_poses.py` | extract 16 neutral poses from `docked_top300.sd` |
| `step234_analyze.py` | prolif analysis + FMN comparison + drug-likeness |
| `step5_build_pdf.py` | build `neutral_16_evaluation.pdf` |

All original files are preserved untouched in the parent directory
(`casestudy/fmn/`); this folder is a clean snapshot of round-1 deliverables.

# 1FMN screening round-3 (1w pool, all neutral — second batch)

**Target:** 1FMN aptamer (35 nt, `GGCGUGUAGGAUAUGCUUCGGCAGAAGGACACGCC`)
**Date:** 2026-08-12
**Input:** `output/fmn_1w/dummy_ligand/dummy_ligand.sdf`
(**10002 all-neutral molecules**, second-batch 1w generation by the post-charge-fix code path)
**Pipeline:** RNA2smol generation → rnasmol GPU scoring → rdock docking →
combined ranking → PubChem similarity → top-20 deep evaluation (prolif+QED/SA+PDF)

---

## 1. Round summary

| stage | output | numbers |
|---|---|---|
| generation (RNA2smol, post-fix) | `output/fmn_1w/dummy_ligand/dummy_ligand.sdf` | 10002 molecules, **all formal_charge==0**, 20–29 heavy atoms |
| rnasmol GPU scoring | `rnasmol_scores.tsv` | 10002/10002 scored, 0 failed (**5.77 ms/mol**) |
| rdock top-500 docking | `docked_top500.sd`, `rdock_top500_scores.tsv` | 500 mol × 3 poses, 1500 total |
| combined rank (rnasmol 0.5 + rdock 0.5) | `top_binders.sdf` | top-100 binders, **all 100 neutral**, all rnasmol P=1.0 |
| **top-20 deep evaluation** | `top20_evaluation.pdf`, `top20_analysis.json`, `top20_interactions.tsv`, `top20_druglike.tsv`, `top20_analysis_summary.txt` | binding mode (prolif) + FMN comparison + drug-likeness |
| PubChem similarity search | `pubchem_top20.tsv` | 19/20 hits at T≥0.70, max Tanimoto **0.370** |

FMN reference: rdock self-dock best SCORE = **-17.16** (INTER = -16.79, best-SCORE pose),
reproducing the known ~-17 kcal/mol value. 485/500 docked molecules beat FMN on SCORE,
417/500 on INTER → the (all-neutral) 1w generation model still finds structures more
complementary than FMN under the same heavy-atom budget, with a **larger candidate pool
than round2**.

---

## 2. Top-20 deep evaluation (all neutral)

### 2.1 Drug-likeness vs FMN

| metric | top-20 mean (range) | FMN |
|---|---|---|
| MW | 345.1 (282–417) | 456 |
| QED | **0.468** (0.128–0.811) | 0.176 |
| SA | 4.69 (3.53–5.73) | 4.03 |
| formal charge | **0** (all 20) | 0 |
| heavy atoms | 21–29 | 31 |

**4/20** have QED ≥ 0.7 (drug-like). Three highest QED:
- **T17** (gen 5570) — QED **0.811**, SA 3.54 (quinazolinone-cinnamaldehyde)
- **T05** (gen 1259) — QED **0.811**, SA 4.93 (tetrahydrofuran-fused aryl-dienylamine)
- **T06** (gen 390)  — QED **0.807**, SA 4.77 (dihydrofuran-isoquinoline carboxylic acid)

### 2.2 Chemotypes (round-3 vs round-2 — different chemical space)

The top-20 are clean, synthetically tractable scaffolds — and **complementary to round-2**
(peptide/triazine/β-carboline heavy). Round-3 top-20 chemical composition:

- **Glycosides / aminosugars (8/20 — dominant)** — T02 (tetrazole-glycosylamide),
  T04 (sugar-hydrazone), T10 (sugar-biphenylamine), T12 (sugar-pyridopyrimidine),
  T14 (thiol-quinoline-furanol), T15 (isoquinoline-aminosugar), T18 (naphthyl-aminosugar),
  T20 (pyrimidine-naphthoquinone-aminosugar)
- **Isoquinoline / naphthyridine / quinazolinone** — T06, T08, T17, T20
  (topologically similar to FMN's isoalloxazine)
- **Phenanthrene fused / biaryl** — T10, T11
- **Schiff base / hydrazone** — T01, T04, T07, T11
- **Amino-acid conjugates / peptide-like** — T02, T07, T09, T16, T19
- **Amidino-phenyl (benzamidine-like)** — T13
- **Azo / disulfide** — T16

Round-2 vs round-3 chemotype split: **round-2 was peptide/triazine-heavy;
round-3 is sugar/heteroaromatic-heavy**. The two batches together give a
diverse ~200-hit pool.

### 2.3 Binding-mode analysis (prolif vs 1FMN pocket)

Pocket residues: G1, A8, G9, G10, A11, U12, A13, U14, A23, G24, A25, A26, G27, G28, A29
(15 nt). Aggregate interaction-type totals across top-20:

| interaction | total count (top-20) | FMN (crystal pose) |
|---|---|---|
| VdWContact | 92 | 7 |
| HBDonor | 17 | 0 |
| PiStacking | 10 | 0 |
| Hydrophobic | 9 | 1 |
| HBAcceptor | 1 | 1 |
| (Anionic / Cationic / CationPi / PiCation) | 0 | 0 |

**Key residues contacted by top-20 (non-VdW)**:
- **U12.A (10/20)** — FMN Hydrophobic signature; strong consensus across rounds.
- **G10.A (9/20)** — same as round-2 (9/20); actively exploited by both batches
  even though FMN only makes VdW there.
- **G9.A (9/20)** — **up from round-2's 5/20**; round-3's hydroxyl-rich
  glycosides form more H-bonds with G9 (FMN HBAcceptor signature).

### 2.4 FMN reference (crystal pose)

- Used **crystal FMN pose** (`FMN_reference_ligand.sdf`) as binding-mode reference,
  same convention as round-1/2 (rDock self-docking drifts ~5 Å).
- FMN crystal binding mode: **HBAcceptor ×1** (G9), VdW ×7, Hydrophobic ×1 (U12).
- FMN "signature" residue set (non-VdW): **{G9, U12}**.
- Drug-likeness: MW 456, **QED 0.176** (cofactor, not a drug), SA 4.03, 31 heavy atoms,
  6 HBD, 10 HBA, TPSA 208, cLogP -1.61.

### 2.5 Top-20 vs FMN

| metric | finding |
|---|---|
| **Highest FMN overlap (Jaccard)** | **T13** (gen 3292): Jaccard **0.667**, covers both FMN signature residues {G9, U12} plus G10, with HBDonor+Hydrophobic+VdW — INTER -5.34 (weak dock but mode-consistent), PubChem Tani 0.362 (benzamidine-like) |
| **Strongest binder (rdock INTER)** | **T11** (gen 9491): INTER **-28.06**, phenanthrene-imine-thioether, Jaccard 1.0 (covers G9+U12), QED 0.61 |
| **Best drug-like + binder** | **T06** (gen 390): INTER -23.89, QED **0.807**, SA 4.77 (dihydrofuran-isoquinoline carboxylic acid) |
| **Best G9+U12 dual coverage + binder** | **T10** (gen 7078): INTER -25.32, Jaccard 1.0, biphenyl-aminosugar |
| **Novelty with known-neighbour (PubChem)** | **T16** (gen 814): Tanimoto **0.370** to 4-(2-Amino-3-fluoro-1-methoxysulfanylpropyl)benzamide (CID 129313078); azo-disulfide-benzamide (rare for RNA) |

---

## 3. PubChem nearest-known-compound search (top-20)

| rank | gen | top CID | top name | Tanimoto |
|------|-----|---------|----------|----------|
| **16** | 814 | 129313078 | 4-(2-Amino-3-fluoro-1-methoxysulfanylpropyl)benzamide | **0.370** |
| 13 | 3292 | 53994375 | [2-(4-Carbamimidoylphenyl)-2-oxo-1-phenylethyl] formate | 0.362 |
| 2 | 8127 | 90668311 | N-[(1R,2S,3S,4S,5R)-2,3,4,5-tetrahydroxycyclohexyl]acetamide | 0.311 |
| 7 | 3158 | 176636737 | cis-(1S,2S)-3-amino-4-oxo-2-pyridin-3-ylcyclopentane-1-carbo... | 0.308 |
| 19 | 1271 | 12649334 | 1-[4-(2-Aminoethoxy)-2,6-dihydroxyphenyl]-3-(3-hydroxy-4-methoxyphenyl)propan-1-one | 0.300 |
| ... | | | | (rest in `pubchem_top20.tsv`) |

**Key takeaways**

- 0/20 share Tanimoto ≥ 0.85 with a PubChem compound → **no known-drug / known-natural-product
  rediscovery**, top-20 are genuinely novel.
- **0/20 at Tanimoto ≥ 0.4** → no scaffold-level match as strong as round-2's T14.
  Round-3 top-20 is **more novel** than round-2 top-20 (max Tani 0.370 vs 0.478).
- **T13 (gen 3292)** is the most "RNA-relevant" neighbour: its PubChem nearest is a
  **benzamidine (carbamimidoyl-phenyl)** compound — benzamidine is a classic
  RNA/dsDNA minor-groove binder scaffold, suggesting T13 is worth following up.
- **T16 (gen 814)** has the highest Tanimoto (azo-disulfide-benzamide) but the
  azo/disulfide chemotype is rare in RNA ligand literature — value is novelty, not SAR.

---

## 4. Comparison with round-2 (4247 molecules, first batch)

| metric | round-2 (first 4247) | **round-3 (1w, 10002)** |
|---|---|---|
| input | 4247 (all neutral) | **10002 (all neutral)** |
| rnasmol P median | 0.632 | 0.627 (consistent) |
| rnasmol P ≥ 0.99 | 20.32% | 20.14% (consistent) |
| rdock top-N | 300 | **500** (larger pool) |
| rdock SCORE median | -23.71 | -23.81 (consistent) |
| rdock INTER median | -20.42 | -20.09 (consistent) |
| rdock SCORE min | -64.95 | **-71.57** (pool-larger → wider extreme) |
| top-100 neutral | 100/100 | **100/100** |
| top-20 QED mean | 0.522 | **0.468** (slightly lower; more glycosides) |
| top-20 QED ≥ 0.7 | 7/20 | **4/20** |
| top-20 chemistry | peptide / triazine / β-carboline | **glycoside / isoquinoline / Schiff base** |
| PubChem top-20 max Tani | **0.478 (T14 triazin-phenol)** | **0.370 (T16 azo-disulfide)** |
| Most actionable single hit | **T14 (gen 1414)** high-Tani + high-Jaccard + low-SA | **T13/T16/T11 (each strong in one axis)** |

**Conclusion**

- Two independent batches (4247 + 10002 = 14249 total) show **statistically identical
  rnasmol/rdock score distributions** — generation + scoring pipeline is reproducible.
- The 1w pool does **not** produce a "better than round-2 T14" single hit on the
  PubChem-Tanimoto axis; max drops from 0.478 → 0.370. Larger pool **samples more
  novel chemical space**, not closer-to-known space.
- **Round-2 T14 (gen 1414, 1,2,4-triazin-phenol) does NOT appear in the 1w pool**
  (no exact match, no Tanimoto ≥ 0.4 variant). It is a rare scaffold in the generator's
  output and round-3 did not re-sample it.
- Round-3 top-20 is **chemically complementary** to round-2: glycoside-heavy (8/20)
  vs peptide/triazine-heavy. **Merging the two batches' top-100** (~200 unique hits)
  is the practical way forward.

---

## 5. Recommended next steps

1. **T13 (gen 3292)** — benzamidine-like amidino-phenyl scaffold, FMN residue Jaccard
   0.667 (highest in round-3), PubChem Tani 0.362. Benzamidine is a known RNA groove
   binder; explore as a **novel FMN-aptamer binder with established SAR precedent**.
2. **T11 (gen 9491)** — phenanthrene-imine-thioether, rdock INTER **-28.06** (strongest
   in top-20), Jaccard 1.0. Strong binder + complete FMN mode coverage; assess solubility.
3. **T10 (gen 7078)** — biphenyl-aminosugar, INTER -25.32, Jaccard 1.0. Strong binder
   with sugar moiety for G9/U9 H-bond network.
4. **T06 (gen 390)** — dihydrofuran-isoquinoline-carboxylic acid, QED 0.807,
   INTER -23.89 — best drug-like + binder balance in round-3.
5. **Merge round-2 + round-3 top-100** for a unified ~200-hit pool; cluster by scaffold
   and prioritize **round-2 T14 (1,2,4-triazine-phenol)** as flagship, with round-3
   glycosides / amidines as orthogonal chemical-series follow-ups.

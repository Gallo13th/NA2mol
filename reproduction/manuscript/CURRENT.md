# CURRENT — single source of truth

**Consolidated:** 2026-11-06. One manuscript version only. Everything superseded is under
`archive/` and is not part of the build.

A clean, self-contained copy of the current package is in `clean/` (15 MB, 57 files); it was also
mirrored to the biocourse project folder on the shared server.

## Manuscript (build inputs + output)

| File / dir | Role |
|---|---|
| `sn-article-revised.tex` | the manuscript (only source) |
| title | **Bridging protein and nucleic-acid pockets via shared physicochemical representations for 3D molecular generation with NA2smol** |
| `sn-article-revised.pdf` | compiled PDF, 34 pp (main text; SI is separate) |
| `table_radar_values.tex`, `table_overlap.tex` | main Tables 1 and 2 |
| `supplementary_notes_revised.tex`, `si_*.tex` | Supplementary Information (appended to the manuscript PDF) |
| `sn-bibliography.bib`, `sn-jnl.cls`, `bst/`, `figures/` | bibliography, class, style, figures |

Build (keeps the source tree clean):

```bash
./scripts/build_paper.sh      # -> sn-article-revised.pdf
```

## Presubmission package

| File | Role |
|---|---|
| `presubmission_supplement.pdf/.docx` | supplement, 8 pp |
| `presubmission_supplement.tex` | supplement source (generated) |
| `presub_table1.tex`, `presub_table2.tex` | table exhibits for the supplement (generated — do not edit by hand) |
| `presubmission_enquiry.docx` | presubmission enquiry — corresponding author's Word letter, maintained outside the build |
| `presubmission_enquiry.pdf` | rendered enquiry letter, 1 p |
| `NA2smol_presubmission_NMET.pdf` | submission PDF for Nature Methods: enquiry letter + supplement, 9 pp |

Rebuild sequence:

```bash
python3 scripts/build_supplement_tex.py   # regenerate presub tables + presubmission_supplement.tex
./scripts/build_presub.sh                 # -> presubmission_supplement.pdf
python3 scripts/build_docx.py             # -> presubmission_supplement.docx only
# the enquiry letter (presubmission_enquiry.docx) is author-maintained and is NOT regenerated
# requires python-docx (pip install python-docx)
```

`build_supplement_tex.py` strips Supplementary-Table pointers from the exhibit captions and
drops whole sentences that only point at material absent from the excerpt.

## Cover letter

`cover_letter_ncs.tex/.pdf/.docx` — target journal is **Nature Computational Science**.
The Nature Methods variant is in `archive/superseded_2026-11-06/old_versions/`
(`cover_letter_natmethods.*`), recoverable if the target changes.

Rebuild: `python3 scripts/build_cover_letter_docx.py` (requires python-docx; it writes both journal
variants — move the non-target NatMethods docx back into the archive after building). The Re: line
tracks the manuscript title and must be updated in both `cover_letter_ncs.tex` and the script when the
title changes.

## Current documentation

- `docs/review_writing_2026-11-05.md` — writing/format review report (P0/P1/P2).
- `docs/polish_changes_2026-11-05.md` — full change log: Gemini polish rounds, SI renumbering,
  2026-11-06 statistics recalculation and its verification.
- `docs/polish_round_2026-11-06.md` — final author-style language pass (Gemini 4) and Lead corrections.
- `docs/enquiry_and_title_2026-11-06.md` — enquiry polish, chosen title and alternatives.
- `docs/title_discussion_2026-11-06.md` — 22-paper title reference set and the Candidate A/B/C/D decision.
- `docs/STYLE_BRIEF.md`, `docs/nature_style_reference.md` — house-style reference.
- `docs/FIGURES_README.md` — figure provenance.
- `docs/tht_assay_protocol_2026-10-15.md` — collaborator assay protocol.

## Data

`data/` — THT Origin projects (`*.opj`, `*.opju`) and `荧光数据汇总.csv` (raw fluorescence data).

## Archive (do not use for building)

- `archive/superseded_2026-11-06/` — files moved during this consolidation:
  `old_versions/` (original draft `sn-article.tex`, old journal variants, duplicate PDF),
  `historical_docs/` (earlier review rounds, handoff notes, repo-doc snapshots),
  `old_scripts/`, `build_artifacts/`.
- `archive/` (rest) — earlier rounds: superseded drafts, figure archives, build snapshots.

No manuscript or source file was deleted; only OS/temp caches (.DS_Store, Word lock file, __pycache__)
were removed. Everything else can be moved back from the archive if needed.

## Verified state

- Main PDF: 43 pp, 0 undefined reference/citation, 2 pre-existing overfull boxes
  (main equation 6.5 pt; SI table 20.65 pt).
- Supplement PDF: 8 pp, 0 undefined, no overfull.
- SI tables numbered 10 → 11 → 12 without duplicates; 7 cross-references in the main text match.
- Strain statistics: four definitions (Table 1 ≤10⁴ pool; SI Table 8 evaluated; Fig. 3d
  per-target medians; SI Table 12 coverage) are documented as non-interchangeable, and the
  numbers were independently re-derived from `audit/tables/*.csv` (repo).

## 2026-11-22 — Clivia aptamer case study added

- New Results subsection **between the ablation study and the KRAS case study** (KRAS remains the final case
  study): *NA2smol generation defines a combinatorial library around the reproducible scaffold of the Clivia aptamer*.
- New **Figure 4** (`figures/figure5_clivia.pdf`, 9 panels a–i; source `docs/clivia_8hze/figures/main_figure.pdf`);
  the KRAS figure is renumbered to **Figure 5** and all cross-references updated.
- New Methods subsubsection: *Clivia aptamer case study: pockets, enumeration, scoring and assays*, placed before
  the KRAS post-processing methods.
- Abstract, introduction and Discussion reordered to match (Clivia first, KRAS last).
- Build: `pdflatex → bibtex → pdflatex × 2` (0 undefined references, 34 pp).

## 2026-11-22 — Figure 5 (KRAS) rebuilt

- Rebuilt the KRAS figure as `figures/figure5_kras.pdf` (180 × 232 mm) with
  `scripts/make_kras_figure5.py`.
- Panels now run in reading order **a** workflow, **b** fragment gate, **c** product gate,
  **d** relaxed docking, **e** ensemble triage, **f** candidate grid, **g** ThT, **h** qPCR;
  panel letters share one baseline above each panel.
- Panel letters are added once, at composition time, all at the same offset above each panel. The original
  per-panel letters were deleted from the vector source (`figures/figure4.svg` → `figures/figure4_noletters.svg`,
  converted with `rsvg-convert`), so no masking is used and no duplicate letters can survive a crop.
- Panel f is an independent artwork: 5 × 7 cells of 23 × 16 mm, every structure rendered at one common bond
  length (120 px on a 2200 × 1400 canvas), downscaled by a single factor and centred in its cell. No per-cell
  text; the highest-vote candidate is framed.
- Panel f (candidate grid) redrawn at print size: larger structures, readable CID/votes text,
  box marking the tested candidate 01.
- Panel g: DC50 annotation moved to the lower-left inside the axes so it no longer crosses the curve.
- Manuscript caption and the `Fig. 5f` cross-reference updated to the new panel order.

Rebuild chain for Figure 5: `python3 scripts/make_kras_figure5.py` (redraws f and g, re-emits
`docs/clivia_8hze/figures/figure5_kras.tex`, compiles `figures/figure5_kras.pdf`).

## 2026-11-22 — Figure 5 refinements (panels e, f)

- Panel f: structures now use shorter, thicker bonds and a bold font for heteroatom labels; every cell
  carries its vote count (`n/5 votes`), with the four-vote candidate set in the panel colour; the framed
  first cell is candidate 01, the only four-vote candidate and the one tested.
- Panel e: the inline "Tier-1: teal / Other: grey" text was deleted from the vector source and replaced by a
  two-entry legend (filled dots + labels) drawn as vector text in the panel's top-right corner.

## 2026-11-22 — Length surgery; leakage table folded into Fig 3

- **Main Text** 3,763 → **2,782 words** (limit 3,000; margin 218). Trimming was done paragraph
  by paragraph, keeping every number, citation and figure pointer; no content was removed that is
  not still stated in an exhibit. Section split now: Introduction 471, Results 1,729, Discussion 614.
- **Abstract** 163 → **148 words** (limit 150).
- **Display items** 5 figures + 2 tables = 7 → **5 figures + 1 table = 6**: the recovery-sensitivity
  table (`table_overlap.tex`, former main Table 2) is deleted and its content is now **Fig. 3g**
  (left: ECFP4 recovery of the full model on DUD-E and RNA across the three reference sets with
  eligible-target counts; right: the resulting change for the four training configurations).
  `table_overlap.tex` is no longer `\input` anywhere; the file is kept for provenance.
- **Fig. 3 rebuilt** at exactly 180 mm wide (200 mm tall) with the new panel g:
  `analysis/fig3_ablation.py` on the analysis host gained `plot_overlap_sensitivity()` and a fourth
  gridspec row; the script now saves at 180 mm directly, so no Ghostscript rescale is involved.
  The manuscript includes it at `width=180mm` (was `0.92\textwidth`), so its 7 pt labels print at 7 pt.
- **Clivia pocket-superposition numbers corrected** in the same pass: the old sentence read
  "C4′/P RMSD 0.22–1.00 Å, median 1.09 Å" (median > maximum). Per `docs/clivia_8hze/POCKET_RMSD.md`
  it is 14 of the 16 comparable pockets forming one site at **0.22–0.81 Å (median 0.53 Å)**; the
  1.09 Å figure is the dye-centroid offset of 8HZK_B110, not an RMSD. The paragraph now also states
  that the two remaining copies bind elsewhere.
- **Cross-references repaired** after the ED/Supplementary split: Discussion "Table 14" →
  "Extended Data Table 9"; main-text recovery/overlap pointers → Fig. 3g plus Extended Data
  Tables 6 and 7; Table 1 caption "Supplementary Table 10/8/12" → Extended Data Tables 6/4/8;
  Table 2's own caption pointed at a non-existent "Supplementary Table 13" and is gone with the table.
- Build: `./scripts/build_submission_files.sh` → manuscript **31 pp**, Extended Data **10 pp**,
  Supplementary **7 pp**; 0 undefined reference/citation; 1 pre-existing 6.5 pt overfull equation.

## 2026-11-22 — Fig. 3g redesigned (baseline-anchored) and configuration filter fixed

- **Panel g is now baseline-anchored.** Left: a Fig. 2-style rank grid of ECFP4 recovery for the
  seven benchmarked methods across the three reference sets (RNA test set; 27, 18 and 14 eligible
  targets, same targets for every method within a column), with the value printed in each cell and
  bubble size and shade encoding the rank within that column. Right: the change in recovery for the
  four training configurations on the matched eligible subset, drawn as a dot plot.
  DUD-E values stay in Extended Data Table 6. `analysis/fig3_ablation.py` gained
  `_rank_bubbles()` and the redrawn right-hand panel.
- **The pretraining-analogue filter is no longer applied to the no-pretraining baseline**, which
  never saw the pretraining library: its PT entry is a cross marked "n/a" in the figure and "—" in
  Extended Data Table 7, whose caption now states the restriction and points to the external
  generators (none of which saw that library) as the control. The Discussion sentence was narrowed
  from "the full model showed the largest drop of the four training configurations" to the three
  pretrained configurations ($-0.198$ against $-0.099$ and $-0.014$).
- **Fig. 3 refitted to the page.** The float was silently taller than the page (LaTeX "Float too
  large … by 254 pt"), which clipped the end of its caption. Canvas is now 180 × 173 mm and the
  caption was condensed to ~270 words: figure and caption now fit on one page with nothing clipped.
- **Open, same defect:** Fig. 4 (Clivia) and Fig. 5 (KRAS) are also reported too large by LaTeX
  (63 pt and 89 pt); their captions end inside the bottom margin. They need the same treatment
  (shorter caption or smaller typeset height) before submission.

## 2026-11-22 — Fig. 3 relaid out as a 3 × 3 figure

- Layout now mirrors Fig. 2: **row 1** ablation radar (a), ECFP4 reference-ligand recovery curve
  (b), screening retention curve (c); **row 2** per-target docking (d), per-target strain (e),
  final-tier retention waterfall (f); **row 3** reference-set sensitivity strip (g).
- The old docking-versus-strain hexbin panel was dropped to make room; panel letters in the
  caption and in the text were updated (strain is now e, retention f).
- **New row 3 (g)**: one small multiple per metric — ECFP4 recovery, QED, SA score and strain
  (log10) — with the value on the targets that survive each reference-set definition
  (all actives 27, fine-tuning-similar actives removed 18, pretraining-library-similar actives
  removed 14). It shows that the ranking of the four configurations survives a change of
  reference definition, and it puts the molecule-level metrics on the same footing as recovery.
  Values come from `analysis/fig3_row3_data.py`, which writes
  `reproduction/figures/fig3_row3_metrics.csv` and `fig3_recovery_curves.csv` from the same
  per-molecule evaluation as Extended Data Tables 6 and 7.
- The figure is now 179 mm × 176 mm and its caption (~200 words) fits on the same page.
- Still open: Fig. 4 and Fig. 5 exceed the float height (LaTeX reports 63 pt and 89 pt), so their
  captions end inside the bottom margin.

## 2026-11-22 — Fig. 3g now a cross-method bubble chart; panels drawn letter-free

- **Panel g no longer shows the ablation configurations.** It is now a cross-method bubble chart:
  NA2smol plus the six external generators on the target subsets that survive each reference-set
  definition (27 / 18 / 14 targets), with one grid per metric — ECFP4 recovery, QED, SA score and
  strain (log10). Columns are the three definitions, so a row shows the shift of a method under a
  change of reference chemistry and a column shows the method ranking; bubble size and shade
  encode the rank within a column (SA and strain ranked with lower values better). This also
  restores the between-method anchoring that the configuration-only version lacked.
- Data: `analysis/fig3_row3_data.py::method_reference_panel()` writes
  `reproduction/figures/fig3_method_reference_panel.csv` (recovery from the overlap audit;
  QED/SA/strain as pooled medians over the surviving targets' molecules, combining the four
  original benchmark runs with DiffSBDD/GraphBP/FLAG).
- **Every panel is now drawn letter-free** and the letters are placed in a single final pass on
  the shared gridspec, so a/d/g, b/e and c/f share an x position and the letters clear the axis
  labels. Figure is 179 × 175 mm.

## 2026-11-22 — Fig. 3g becomes a rank bump chart

- Panel g is now a **rank bump chart**, one small panel per metric (ECFP4 recovery, QED, SA score,
  strain). The y axis is the rank within a reference-set subset (1 = best), so a rising line marks a
  method that gains places when the reference chemistry is redefined. Colours and markers follow the
  method (the `METHOD_COLORS`/`METHOD_MARKERS` palette used by Fig. 2), the left labels list the
  methods in their 'All actives' order, and the right axis repeats the rank numbers.
- Strain is no longer tagged log10 in panel g: the panel encodes rank, not the energy itself. The
  raw strain axis in panel e keeps its log scale.
- Caption updated accordingly. Figure 179 × 175 mm, panel letters placed once in the final pass.

## 2026-11-22 — Fig. 3g simplified to ECFP4 recovery only

- Since the other benchmark metrics barely move when the reference chemistry is redefined, panel g
  now shows **ECFP4 recovery only**: left, recovery of NA2smol and the six external generators on
  the target sets that survive each definition (27 / 18 / 14); right, NA2smol's recovery curves on
  the same three target sets (fraction of eligible targets recovered versus the ECFP4-TS threshold).
  QED, SA score and strain panels were removed; the caption states that they shift by far less.
- New data table: `reproduction/figures/fig3_recovery_curves_by_def.csv` (curve per method and
  definition). The seven-method colour legend moved to a figure-level row under panel g.

## 2026-11-22 — Fig. 3g left panel back to a rank line chart; Figs 2, 4, 5 refitted

- **Fig. 3g left** is again a rank line chart (recovery rank on the y axis, methods named on the left
  axis in their 'All actives' order, rank numbers repeated on the right), so the seven-method
  colour legend under the figure was removed. A rank-computation bug (ranking a column in place
  while reading it) was fixed at the same time; the left-axis order now matches the actual ranks.
- **Figure widths** were re-derived from the NBT rule (≤180 mm wide, 5–7 pt labels at print size)
  and from the need to keep the caption inside the page:
  Fig. 2 `0.90\textwidth` → **180 mm** (its 7 pt labels now print at 7 pt);
  Fig. 4 (Clivia, 179 mm canvas) → **140 mm** (labels ≈5.5 pt);
  Fig. 5 (KRAS, 183 mm canvas) → **140 mm** (labels ≈5.4 pt).
  Both tall figures now end their captions above the bottom margin (Fig. 4 at ~265 mm, Fig. 5 at
  ~257 mm on a 297 mm page) instead of running into it.
- LaTeX still reports "Float too large for page" for Figs 2–5 because the sn-jnl text block is only
  195 mm tall; the floats are placed on float pages and nothing is clipped (verified page by page).
- **Still open:** Fig. 1 (designer artwork, canvas 180 mm) carries text at ~4.2 pt at print size,
  below the 5 pt floor; it needs to be redrawn or split.

## 2026-11-22 — Fig. 3g becomes four compact panels; Figs 2/4/5 refitted

- Panel g is now a four-panel strip: (i) recovery rank of the seven methods across the three
  reference sets, with the method colours keyed on the left axis so no separate legend is needed;
  (ii) NA2smol recovery curves on the three target sets; (iii) recovery curves of all seven methods
  on the fine-tuning-similar-removed subset (18 targets); (iv) the same on the
  pretraining-similar-removed subset (14 targets). The two curve panels are the pairs the reviewer
  asked for: NA2smol against the baselines under each redefinition.
- The per-molecule ECFP4-TS for DiffSBDD, GraphBP and FLAG is not in the exported benchmark table,
  so `analysis/fig3_curves_all_methods.py` recomputes it from the reference actives
  (`audit/tables/reference_active_overlap.csv`) and writes
  `reproduction/figures/fig3_recovery_curves_by_def.csv` for all seven methods.
- Figure widths re-derived from the NBT rules: Fig. 2 at 180 mm (7 pt labels print at 7 pt),
  Fig. 4 and Fig. 5 at 140 mm (labels ≈5.5 and ≈5.4 pt). Their captions now end above the bottom
  margin (Fig. 4 ≈265 mm, Fig. 5 ≈257 mm of a 297 mm page) instead of running into it.
- The sn-jnl text block is only 195 mm tall, so LaTeX still reports "Float too large" for Figs 2–5;
  the floats are placed on float pages and no content is clipped (checked page by page). If that is
  ever unacceptable, the alternative is to move all figures after the reference list and suppress
  page numbers, which is what the journal's PDF-only initial submission allows.
- **Still open:** Fig. 1 (designer artwork) carries text at ≈4.2 pt at print size, below the 5 pt floor.

## 2026-11-22 — Fig. 3 spacing pass

- Panels widened by shrinking the side margins and column gutters (`left` 0.105 → 0.085,
  `wspace` 0.62 → 0.42, panel-g strip `wspace` 0.55 → 0.42), and the shared four-configuration
  legend moved down from the very top edge (anchor y 0.995 → 0.972) with the panel area lowered
  (`top` 0.885 → 0.845) so the legend no longer sits on panel a's title.
- Figure is now 179 × 173 mm; Fig. 3 sits on p8 with its caption intact.

## 2026-11-22 — Text polish with gemini-4-argon (main text + captions)

Polished in three rounds with `api88/gemini-4-argon`, under three rules: keep the story, narrative
logic and claim strength; delete defensive/hedged/meta rhetoric and state what was done; read the
whole text between rounds.

- **Round 1 (main text, 9 paragraphs)**: removed reviewer-pre-empting caveats — the ECFP4 proxy
  disclaimer, "the no-pretraining baseline is not a matched comparison", "does not isolate
  nucleic-acid-specific conditioning", "a lower bound under a degraded representation",
  "affinity, selectivity and binding mode were not measured", "evaluates the integrated workflow
  rather than unconstrained generation alone", and signposting such as "told a complementary story"
  and "we next asked whether".
- **Round 2 (14 proposed, 11 accepted)**: removed the administrative sentence in the training
  paragraph, "nevertheless", "suggesting that" (kept where it is the correct epistemic status of an
  interpretation), and negative constructions ("not dye-specific", "scored without re-docking").
  **Rejected three proposals** because they would have changed claim strength or dropped content:
  "far fewer" → "fewer", "stayed within the range" → "matched", and removing the sentence
  "The high strain was therefore widespread rather than outlier-driven", the phrase "have no
  nucleotide analogue" and the design-rationale clause in the model paragraph. Those passages were
  restored in a curated merge.
- **Round 3 (captions, 3 of 6 changed)**: removed the meta sentence in Fig. 3g ("ECFP4 recovery is
  the one this redefinition moves …"), "retrospectively" and the log-axis justification in Fig. 5,
  and the "rather than as an exact value" construction in Table 1.
- Effect: main text **2,801 → 2,509 words** (intro 462, results 1,588, discussion 459); captions
  −61 words. 31 pp, 0 undefined references, same figure/table numbering.
- Left untouched deliberately: the dedicated *Limitations and scope* subsection (a journal
  requirement) and the Methods' statistical caveats.

## 2026-11-22 — Text polish with gemini-4-argon, rounds 4–5 (Methods, SI/ED)

- **Round 4 (Methods + back matter)**: 14 paragraphs proposed, **9 applied**: removed the
  promotional sentence opening the evaluation subsection, "heuristic filters rather than
  ground-truth measurements", "Scores are screening indications rather than affinity estimates",
  the self-justifying clauses around smina-minimised strain and the failed-pose denominator,
  "Because identical identifiers do not imply identical pockets", "rather than by unusual
  heteroatom geometries", and the meta remark "Two universes therefore appear in the analysis";
  *Limitations and scope* was rewritten as factual statements with the same content.
  **5 rejected** because they removed methodological transparency or a deliberate response to
  reviewer zwb1: the distinction between heavy-atom count and sampling steps, the justification
  of the fixed recovery window over a single-threshold hit rate, the disclosure that no resampling
  interval is reported and no family-wide multiplicity control is applied, the explanation that the
  two radar/rank forms carry the same score, and the statement that filtering removes reference
  actives only.
- **Round 5 (Supplementary Notes + ED table notes)**: 16 proposed, **15 applied**, **3 reverted**
  after review because they dropped content rather than rhetoric: the power argument for the
  300-molecule control budget, the per-method effect sizes in the mapping-control summary, and the
  target-set definition plus Extended Data Table cross-reference in the strain note.
- **Corrections made while reading**: the k-NN graph sentence now states which conditioning cloud is
  concatenated (pocket atoms during fine-tuning, electron-density points during pretraining);
  `si_note5_chemical_overlap.tex` pointed at the deleted main "Table 2" and at
  "Supplementary Tables 10 and 11" and now points at Supplementary Table 5 and Extended Data
  Tables 6 and 7.
- State: main text **2,509 words**, Methods **5,218 words**, manuscript **30 pp**, ED 10 pp, SI 7 pp,
  0 undefined references.
- **Open for the authors**: the ThT compound is attributed to "Meck Company (cat. no.
  BDPH9BCD844A)", which looks like a Bidepharm catalogue number; and the pretraining corpus is
  described as "approximately 2.4 million molecules" in Methods but as 2,287,199 unique canonical
  SMILES later — both may be right, but the wording should be made comparable.

## 2026-11-22 — Remaining hedges removed; pre-submission enquiry updated

- Deleted the three passages that had been kept in the previous round, on the authors' instruction:
  (i) "Each comparison summarises the complete set of evaluable targets and generated molecules, so no
  resampling interval is reported" and ", reported without family-wide multiplicity control";
  (ii) the justification of the fixed recovery window against a single-threshold hit rate
  ("This window integral is used instead of the conventional single-threshold hit rate … by which
  threshold is reported");
  (iii) "and is distinct from the number of sampling steps \(N\)" after the heavy-atom count.
  Manuscript stays at 30 pp.
- `presubmission_enquiry_nature_biotechnology.docx` rebuilt from the 10 November version: date
  22 November 2026; benchmark paragraph now cites the closed-pipeline retention (239/17/27, 0.68% of
  35,358) plus recovery on both benchmarks and the component analysis; a new Clivia paragraph
  (448-molecule library, IC50 3.29 and 3.16 µM, cellular fluorescence); the KRAS paragraph moved to
  Fig. 5 and trimmed; the closing paragraph rewritten to name two complementary workflows. One page.

## 2026-11-22 — Nature Methods enquiry and format check

- New `presubmission_enquiry_nature_methods.docx` (one page, same letterhead as the NBT version but
  written for a methods audience): the obstacle is stated as representational and data-related, the
  advance as unified physicochemical pocket conditioning plus complex-free density pretraining and
  teacher–student alignment, the evidence as the seven-method paired benchmark with component
  analysis and the analogue-removal audit, the validation as the two experimental case studies, and
  the closing paragraph names the three methodological contributions and code release. Request line
  now asks for Nature Methods.
- Format check against the Nature Methods Article specification: main text 3,000 words (up to 5,000
  at discretion) — ours 2,509; abstract ≤150 words, unreferenced — ours 148 and citation-free;
  ≤6 display items — ours 6 (Figs 1–5 + Table 1); ≤50 references — 42 cited; Methods after the main
  text with no length limit — 5,218 words. **No separate Nature Methods manuscript build is needed**;
  the same PDF satisfies both journals, and no journal name appears anywhere in the manuscript source.
- Both journals require the Reporting Summary and Editorial Policy Checklist; still to be prepared.

## 2026-11-22 — Presubmission supplement regenerated from the current figures

- `scripts/build_supplement_tex.py` and `scripts/build_docx.py` were updated to the post-revision
  exhibits: five main figures (Fig. 1 `figure1_nbt.pdf`, Fig. 2 `figure2_nbt.pdf`, Fig. 3
  `figure3_nbt.pdf`, Fig. 4 `figure5_clivia.pdf`, Fig. 5 `figure5_kras.pdf`) and the single main
  table; Table 2 (`table_overlap.tex`) was dropped with the main-text table.
- The generator also now strips Extended Data pointers as well as Supplementary pointers, because
  the exhibit is self-contained; the `\exhibit` macro starts a new paragraph so the caption no
  longer wraps around the figure; figure height is capped at 0.60 text height; and a cover page
  lists the six exhibits.
- `presubmission_supplement.tex/.pdf` = 8 pp (cover, Figs 1–5, Table 1, references);
  `presubmission_supplement.docx` = 7 pp (cover, Figs 1–5, Table 1; no reference list by design).
- Dates in both the supplementary exhibits and the enquiry letters are now 22 November 2026.

## 2026-11-22 — Reporting Summary draft and polished enquiry letters

- `reporting_summary_draft.docx` (+ PDF): section-by-section draft answers to the Nature Portfolio
  Life Sciences Reporting Summary — sample size, data exclusions, replication, randomisation,
  blinding; statistics (two-sided Wilcoxon signed-rank, Welch t with Holm correction, four-parameter
  logistic fits, figure-level summary-statistic definitions); software (PyTorch/PyG, RDKit, PyMOL,
  cctbx, smina, rDock, PoseCheck, RNAsmol, Origin); Data and Code availability; and the
  materials-reporting block (cell lines, antibodies, animals, human participants, oligonucleotides,
  small molecules, computational reproducibility). Six items are flagged for author input: HeLa
  authentication/mycoplasma/passage, vendor and catalogue numbers, whether the imaging experiment was
  repeated, blinding statement, commit hash, and the funding/contributions/ORCID block.
- Both enquiry letters were polished by gemini-4-argon in the voice of the 10 November reference
  version: the Nature Biotechnology letter unchanged in argument (translational lead, two case
  studies), the Nature Methods letter re-sequenced around the methodological contribution; each fits
  one page at 453 and 457 body words after one slack line was removed before the signature block.

## 2026-11-22 — Final enquiry texts (after gemini polish)

- The gemini-4-argon pass tightened both letters: Nature Biotechnology 443 words, Nature Methods
  421 words of body text (paragraphs 8–14), one page each after LibreOffice rendering. All
  quantitative anchors verified present in both: 239 / 17 of 27 / 0.68% / 35,358 / 448 / 3.29 and
  3.16 µM / 10,077 / 22.4 µM, and the 22 November 2026 date line.
- One redundancy introduced by the polish ("complex-free … requiring no pocket–ligand complexes") was
  removed from the Nature Methods closing paragraph.
- PDFs regenerated from the final .docx for both letters.

## 2026-11-22 — Repository sync

- The whole submission package was copied into the repository at
  `reproduction/manuscript/` (LaTeX sources, the six figure PDFs actually included, `bst/`,
  `scripts/`, the three built PDFs, both enquiry letters and the Reporting Summary draft) and
  committed as `80c2ceb`; the Data and Code availability statement now cites that commit, and the
  follow-up commit `31ef4a0` carries the rebuilt PDFs.
- `~/projects/NA2smol_paper` was refreshed from the same package so the mirror is no longer stale.
- The model, dataloader, inpainting scripts and the case-study scripts are in the same commit, so the
  released code and the manuscript describe the same thing.
- Not pushed to GitHub: the analysis host has no SSH key for `Gallo13th/NA2mol`
  (`git ls-remote origin` → permission denied). The commits exist on the host only.
- Building on the analysis host is not possible as-is: its TeX Live 2021 install lacks the
  sn-jnl class dependencies (`cuted`, `stfloats`, `threeparttable`, `appendix`, `lineno`, `natbib`
  and others), and mixing a current `hyperref` into a user texmf tree raises a version-mismatch
  error. The PDFs in the repository were therefore built on the Mac and committed.

## 2026-11-24 — Local clone and GitHub push

- Cloned the branch `paper-release` from the analysis host into
  `repo_NA2mol/` (session workspace, 498 MB, HEAD `1bc797c`); the clone carries the full
  114-commit history and is byte-identical to the workspace for `sn-article-revised.tex`,
  `CURRENT.md` and both enquiry letters.
- Added the GitHub remote `github` (git@github.com:Gallo13th/NA2mol.git) and pushed
  `paper-release` as a new branch; the local branch now tracks `github/paper-release`.
- GitHub `main` (56ec5c3) is an unrelated squashed snapshot from before the branch history and was
  left untouched — repointing it would be a history rewrite and needs a decision.
- Push emitted two size warnings from history (`property_landscape_audit.tiff` 71 MB,
  `molcraft_mapping_control.tiff` 66 MB, plus `chemical_space_audit.tiff` 47 MB): under GitHub's
  100 MB hard limit, but candidates for LFS or removal if the repo is ever made public.
- Host, Mac clone and GitHub `paper-release` were unified at `3a03b97` (host `git reset --hard` to
  the pushed commit after transferring it through a temporary ref). From now on the Mac is the push
  point: its SSH key authenticates as `Gallo13th`.

## 2026-11-24 — History cleanup: no oversized files, images only PDF/PNG

- The repository history was rewritten with git-filter-repo to remove every image file that is not
  PDF or PNG (5 TIFF, 19 EPS, 85 SVG, all JPG/BMP/GIF if any) and the one remaining oversized blob
  (`casestudy/sp1/library/cold_start_18641.sdf`, 38.7 MB). `.git` shrank from 151 MB to 104 MB and
  the largest remaining blob is a 10.3 MB PDF.
- Because the rewrite changes every commit hash, the Data and Code availability statement was
  repointed from `80c2ceb` to the rewritten equivalent `5bd3a6b`, and the three PDFs were rebuilt.
- GitHub `paper-release` and the analysis-host checkout were force-updated to the rewritten history.

## 2026-11-25 — Authors, affiliations and experimental protocols

- Author list updated: Wenjin Wan (affiliations 2 and 8, wanwenjin@him.cas.cn, ORCID 0009-0008-4628-528X)
  and Yuxuan Mo (affiliation 7, moyuxuan25@mails.ucas.ac.cn, ORCID 0009-0001-8521-845X) added as
  regular authors after Jiahao Chen, before the corresponding authors; Yi Li now carries affiliations
  2 and 7 (Institute of Zoology, CAS and University of Chinese Academy of Sciences); new affiliations
  7 (UCAS, 100049) and 8 (Hangzhou Institute of Medicine, CAS, 310022).
- Methods gained the displacement-assay protocol (ThT 200 nM with the KRAS G-quadruplex 100 nM;
  NBSI618 200 nM with the Clivia aptamer 100 nM; 0–20 µM compound; 425/490 nm and 510/618 nm;
  DC50 in GraphPad Prism 9), the RNA-imaging protocol (Clivia plasmid in HEK293T cells, Hoechst
  33342, NBSI618 0.1 µM, 561 nm excitation / 640 nm emission, Olympus SpinSR10 with live-cell
  module, 20× objective, Prime BSI camera, Fiji quantification) and a qRT-PCR protocol (A549 cells,
  2×10^5 per well, 8 h treatment, TransGen RNA kit, NanoDrop One, HiScript Q RT SuperMix, LightCycler
  480 II with SYBR Green, GAPDH reference, triplicates over three biological replicates).
- New **Supplementary Table 6** lists the KRAS G-quadruplex and qRT-PCR primer sequences.
- The imaging cell line is now HEK293T throughout (Results, Fig. 4 caption, Methods) and the KRAS
  readout is described as a transcript level measured by qRT-PCR (previously "KRAS DNA level");
  the Reporting Summary was updated to match.
- To confirm with the authors: the previous draft named HeLa for the imaging, gave the KRAS
  G-quadruplex as a 35-nt sequence, quoted 451/481 nm for ThT, 0–1 mM compound and Origin 2021 for
  the fits, and described the KRAS readout as DNA — all superseded here by the protocol text supplied.
- Follow-up (same day): the authors confirmed that the supplied protocol is authoritative. Panel h of
  Fig. 5 was redrawn with the axis label "KRAS transcript level (% of control)", the caption now reads
  "three independent biological replicates, each measured in triplicate" instead of "n = 5-6 from two
  independent experiments", and the Results sentence names the qRT-PCR readout next to the ThT
  result. The figure was rebuilt from `scripts/make_kras_qpcr_bar.py` and
  `scripts/make_kras_figure5.py`, and `scripts/make_kras_qpcr_bar.py` was updated with the new label.

#!/usr/bin/env python
"""Assemble the single ligand-leakage-audit document: summary statistics tables
verbatim + full source of every audit script (+ the audit summary prose)."""
from datetime import date
from pathlib import Path

ROOT = (ROOT / "audit")
SCRIPTS = ROOT / "scripts"
TABLES = ROOT / "tables"
OUT = ROOT / "ligand_audit_full_document.md"

STAT_TABLES = [
    ("reference_overlap_summary.csv", "Experiment I - reference-active overlap summary (ligand & target level, by source and threshold)"),
    ("generated_overlap_summary.csv", "Experiment II - generated-output overlap by benchmark x method (pretraining fractions use the stratified sample with Wilson CIs)"),
    ("recovery_original_vs_overlap_filtered.csv", "Experiment III - target-level recovery A(tau in [0.25,0.5]) per method and reference filter, with target-bootstrap CIs and published reference values"),
    ("recovery_curves.csv", "Experiment III - per-target maxima and eligible-target lists for every filter/method"),
    ("recovery_reproduction_check.csv", "Experiment IV - per-method A_obs vs the manuscript's own per-threshold tables and the cached permutation gain"),
    ("recovery_fixed_target_sensitivity.csv", "Experiment IV - original vs filtered A_obs on the same eligible-target subset (filter vs target-composition effects)"),
    ("recovery_target_coverage.csv", "Experiment IV - per-method evaluable-target coverage"),
    ("recovery_ablation_ladder.csv", "Experiment V - recovery A_obs per reference filter for the no-pretraining baseline, Pretrain Only, Pretrain+Constrain and the full model"),
    ("recovery_ablation_coverage.csv", "Experiment V - target coverage of the four ablation configurations"),
]

CODE_FILES = [
    ("lka_common.py", "Shared utilities: standardisation, ECFP4, packed fingerprints, RDKit bulk similarity, inverted bit index"),
    ("build_lib.py", "Build a packed ECFP4 fingerprint library from a SMILES file"),
    ("build_index.py", "Build the inverted bit index over a packed fingerprint library"),
    ("build_reference_actives.py", "Build the SMRT-Net + DUD-E reference-active table"),
    ("build_sources.py", "Build per-source identity/parent/scaffold sets and packed fingerprints"),
    ("build_generated_table.py", "Build the generated-molecule table from the evaluation CSVs"),
    ("summarize_reference.py", "Reference-overlap summary at ligand and target level"),
    ("exp1_reference_overlap.py", "Experiment I - quantify benchmark-active overlap"),
    ("exp2a_generated_ft_overlap.py", "Experiment IIa - generated-overlap, full pass (fine-tuning)"),
    ("exp2b_generated_pretrain_sample.py", "Experiment IIb - pretraining similarity on the stratified sample"),
    ("exp2_summary.py", "Combine IIa + IIb into the generated-overlap summary"),
    ("build_provenance.py", "Recovery-contributing molecule provenance"),
    ("exp3_recovery_sensitivity.py", "Experiment III - recovery robustness to overlap"),
    ("exp4_fixed_target_sensitivity.py", "Experiment IV - reconstruction reconciliation and fixed-target sensitivity"),
    ("exp5_ablation_recovery.py", "Experiment V - ablation-ladder recovery (no-pretraining baseline vs pretraining configs)"),
    ("run_chain.sh", "Sequential driver for Exp III, IIa, IIb, combine, provenance"),
    ("assemble_document.py", "Assemble this single document from the tables and code"),
]

BIG_ARTEFACTS = [
    "reference_active_overlap.csv", "generated_overlap_molecules.csv",
    "generated_overlap_pretrain_sample.csv", "reference_actives.csv",
    "generated_long.csv", "recovery_molecule_provenance.csv",
]


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def fence(text, lang):
    return f"```{lang}\n{text.rstrip()}\n```\n"


def main():
    L = []
    L.append("# Ligand-leakage audit - raw statistics and code")
    L.append("")
    L.append(f"Generated {date.today().isoformat()}. Companion to "
             "`tables/ligand_audit_summary.md`.")
    L.append("")
    L.append("This single document contains (A) the audit's summary statistics "
             "tables verbatim and (B) the full source of every script used to "
             "produce them. The very large per-record tables are not inlined "
             "(their paths and sizes are listed at the end); their aggregate "
             "results are in Part A and in the companion summary.")
    L.append("")
    L.append("## Environment")
    L.append("")
    L.append("- Python 3.10.18, RDKit 2024.09.6, NumPy 1.26.4, pandas")
    L.append("- Hardware: Intel i5-12400F, 6 physical cores / 12 threads, 15 GB RAM")
    L.append("- Fingerprint: Morgan r=2, 2048 bits, no chirality (ECFP4)")
    L.append("- Fine-tuning table: `training_ligands_leakage_table.csv` "
             "(225,239 records) from `finetuning_ligands_table.tar.gz`")
    L.append("- Pretraining library: `pretrain_smiles.txt.gz` "
             "(2,287,199 unique canonical SMILES)")
    L.append("- Reference actives: RNALigBench `activesmi` (SMRT-Net) and "
             "`DUDE_active_compounds.csv` (DUD-E)")
    L.append("- Generated molecules: `notebooks/output/smrtbench_combined_results.csv` "
             "and `notebooks/dude_stat_all.csv`")
    L.append("")
    L.append("---")
    L.append("")
    L.append("# Part A - Raw statistics")
    L.append("")
    L.append("## A0. Coverage")
    L.append("")
    L.append("| Quantity | Value |")
    L.append("|---|---|")
    L.append("| Reference actives (rows / unique ligands / targets) | "
             "38,855 / DUD-E 35,237 + SMRT 50 / 101 + 27 |")
    L.append("| Fine-tuning partitions (train/val/unused/drop) | "
             "127,255 / 27,621 / 67,536 / 2,827 records |")
    L.append("| Pretraining library (unique canonical) | 2,287,199 |")
    L.append("| Unique generated molecules (union) | 613,286 |")
    L.append("| Generated (benchmark, method, target, molecule) rows | 677,552 |")
    L.append("| Shared evaluable targets | SMRT-Net 27, DUD-E 83 |")
    L.append("")
    for i, (name, title) in enumerate(STAT_TABLES, 1):
        p = TABLES / name
        if not p.exists():
            continue
        L.append(f"## A{i}. {name}")
        L.append("")
        L.append(title + ".")
        L.append("")
        L.append(fence(p.read_text(), "csv"))
    L.append("---")
    L.append("")
    L.append("# Part B - Raw code")
    L.append("")
    for i, (name, desc) in enumerate(CODE_FILES, 1):
        p = SCRIPTS / name
        if not p.exists():
            continue
        lang = "bash" if name.endswith(".sh") else "python"
        L.append(f"## B{i}. `scripts/{name}`")
        L.append("")
        L.append(desc + ".")
        L.append("")
        L.append(fence(p.read_text(), lang))
    L.append("---")
    L.append("")
    L.append("# Appendix - audit summary")
    L.append("")
    sm = TABLES / "ligand_audit_summary.md"
    if sm.exists():
        L.append(sm.read_text())
    L.append("---")
    L.append("")
    L.append("# End matter - large per-record artefacts (not inlined)")
    L.append("")
    L.append("| File | Size | Rows |")
    L.append("|---|---:|---:|")
    for name in BIG_ARTEFACTS:
        p = TABLES / name
        if p.exists():
            with open(p) as f:
                n = sum(1 for _ in f)
            L.append(f"| tables/{name} | {human(p.stat().st_size)} | {n - 1:,} |")
    L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT, human(OUT.stat().st_size),
          f"{OUT.read_text().count(chr(10)):,} lines")


if __name__ == "__main__":
    main()

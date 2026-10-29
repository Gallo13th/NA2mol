#!/usr/bin/env python3
"""Export available audited tables to canonical release paths, with checksums.

This does not pretend to recompute unavailable source metrics. Every exported
file retains its legacy source path, SHA256 and original method/data identifiers.
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproduction/results'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def main():
    mapping={
      'audit/tables/recovery_manuscript_authoritative.csv':'leakage/recovery_manuscript_authoritative.csv',
      'audit/tables/recovery_fixed_target_sensitivity.csv':'leakage/recovery_fixed_target_sensitivity.csv',
      'audit/tables/recovery_ablation_ladder.csv':'leakage/recovery_ablation_ladder.csv',
      'audit/tables/recovery_ablation_paired_ci.csv':'leakage/recovery_ablation_paired_ci.csv',
      'audit/tables/reference_overlap_summary.csv':'leakage/reference_overlap_summary.csv',
      'audit/tables/generated_overlap_summary.csv':'leakage/generated_overlap_summary.csv',
      'case_study/kras_g4/denovo2/paper_funnel.json':'kras_g4/paper_funnel.json',
      'case_study/kras_g4/denovo2/paper_stage1_fragments.json':'kras_g4/paper_stage1_fragments.json',
      'case_study/kras_g4/denovo2/paper_stage2_variants.json':'kras_g4/paper_stage2_variants.json',
      'case_study/kras_g4/denovo2/paper_stage4_products.json':'kras_g4/paper_stage4_products.json',
      'case_study/kras_g4/denovo2/chain_tiers.json':'kras_g4/chain_tiers.json',
      'case_study/kras_g4/denovo2/g4_tier1_order_table.csv':'kras_g4/g4_tier1_order_table.csv',
    }
    records=[]
    for source,target in mapping.items():
        src=ROOT/source;dst=OUT/target
        if not src.is_file():raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        sha=digest(src)
        if digest(dst)!=sha:raise RuntimeError('Copy checksum mismatch: '+str(dst))
        records.append({'source':source,'output':str(dst.relative_to(ROOT)),'sha256':sha,'bytes':src.stat().st_size})
        print('EXPORTED',target)
    p=ROOT/'reproduction/manifests/exported_sources.json'
    p.write_text(json.dumps(records,indent=2)+'\n')
    print(f'{len(records)} source artifacts exported; metrics were not recalculated.')
if __name__=='__main__':main()

#!/usr/bin/env bash
set -e
cd /home/glt/github_official/DiffEDMol/audit
source /home/glt/miniconda3/etc/profile.d/conda.sh
conda activate diffedmol
python scripts/exp3_recovery_sensitivity.py > logs/exp3.log 2>&1
python scripts/exp2a_generated_ft_overlap.py > logs/exp2a.log 2>&1
python scripts/exp2b_generated_pretrain_sample.py > logs/exp2b.log 2>&1
python scripts/exp2_summary.py > logs/exp2_summary.log 2>&1
python scripts/build_provenance.py > logs/provenance.log 2>&1
echo DONE > logs/chain_done.txt

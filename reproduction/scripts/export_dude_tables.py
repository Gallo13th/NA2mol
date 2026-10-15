#!/usr/bin/env python3
"""Export DUD-E recovery and screening tables from per-molecule source rows.

Scope: observed outputs, without assuming a reference-target universe. The fixed
101-target recovery universe must be supplied separately for the main figure.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=ROOT/'data/benchmark/dude_stat_all.csv'
    cols=['mode','pdb_id','affinity','qed','sa_score','tpsa','strain','max_ecfp4_tanimoto']
    df=pd.read_csv(source,usecols=cols)
    df['SEI']=-df.affinity/df.tpsa.replace(0,np.nan)
    df['SEI']=df.SEI.replace([np.inf,-np.inf],np.nan).fillna(0)
    out=ROOT/'reproduction/results/benchmark_tables/dude';out.mkdir(parents=True,exist_ok=True)
    recovery=[];funnel=[]
    for method,sub in df.groupby('mode',sort=True):
        n=sub.pdb_id.nunique()
        for tau in [.20,.25,.30,.35,.40,.50]:
            counts=sub.assign(hit=sub.max_ecfp4_tanimoto.ge(tau)).groupby('pdb_id').hit.sum()
            recovery.append(dict(method=method,tau=tau,successful_targets_ge1_molecule=int(counts.ge(1).sum()),successful_targets_ge5_molecules=int(counts.ge(5).sum()),total_evaluable_targets=n,hit_mols_total=int(counts.sum())))
        a=sub.affinity;q=sub.qed;s=sub.sa_score;e=sub.SEI
        pharma=(a<-8)&(q>.5)&(s<5)
        masks=[('All',pd.Series(True,index=sub.index)),('aff<-8,QED>0.5,SA<5',pharma),('+SEI>0.1',pharma&(e>.1)),('aff<-9,QED>0.5,SA<5,SEI>0.15',(a<-9)&(q>.5)&(s<5)&(e>.15)),('aff<-9,QED>0.6,SA<4,SEI>0.2,strain<200',(a<-9)&(q>.6)&(s<4)&(e>.2)&(sub.strain<200))]
        for step,mask in masks:
            kept=sub.loc[mask];counts=kept.groupby('pdb_id').size()
            funnel.append(dict(method=method,filter_step=step,mols_retained=len(kept),targets_ge1=int(counts.ge(1).sum()),targets_ge5=int(counts.ge(5).sum())))
    pd.DataFrame(recovery).to_csv(out/'d2_ecfp4_target_success_by_threshold.csv',index=False)
    pd.DataFrame(funnel).to_csv(out/'d7_virtual_screening_funnel.csv',index=False)
    status={'stage':'export-dude','complete':True,'input':str(source.relative_to(ROOT)),'rows':len(df),'methods':sorted(df['mode'].unique()),'scope':'observed-output target counts only; fixed 101-target reference universe not assumed'}
    (ROOT/'reproduction/manifests/dude_export_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2),flush=True)
if __name__=='__main__':main()

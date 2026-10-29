#!/usr/bin/env python3
"""Validate actual paper inputs; missing prerequisites are failures, not skips."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def validate(items=None):
    manifest = json.loads((ROOT/'reproduction/manifests/inputs.json').read_text())
    results=[]
    for spec in manifest['inputs']:
        if items and not set(items).intersection(spec['items']):
            continue
        p=ROOT/spec['path']; errors=[]; info={}
        if not p.is_file():
            errors.append('missing file')
        elif spec.get('required_columns'):
            with p.open(newline='',encoding='utf-8-sig') as f:
                reader=csv.DictReader(f); columns=reader.fieldnames or []
                missing=sorted(set(spec['required_columns'])-set(columns))
                if missing: errors.append('missing columns: '+', '.join(missing))
                modes=set(); rows=0
                for row in reader:
                    rows+=1
                    if row.get('mode'): modes.add(row['mode'])
            info={'rows':rows,'columns':columns,'methods':sorted(modes)}
            absent=sorted(set(spec.get('required_methods',[]))-modes)
            if absent: errors.append('missing method/configuration: '+', '.join(absent))
        results.append({'id':spec['id'],'path':spec['path'],'items':spec['items'],
                        'ok':not errors,'errors':errors,**info})
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items',nargs='+',help='Only validate named display items/stages')
    parser.add_argument('--report',type=Path,help='Write JSON status report')
    args=parser.parse_args(); results=validate(args.items)
    if not results:
        parser.error('No inputs match the requested item IDs')
    for row in results:
        print(('PASS' if row['ok'] else 'FAIL')+' '+row['id']+': '+row['path'])
        for error in row['errors']: print('     '+error)
    report={'complete':all(r['ok'] for r in results),'inputs':results,
            'excluded_by_design':['pretraining LMDB']}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(f"\n{sum(r['ok'] for r in results)}/{len(results)} prerequisites pass; pretraining LMDB excluded.")
    return 0 if report['complete'] else 2
if __name__=='__main__':
    raise SystemExit(main())
